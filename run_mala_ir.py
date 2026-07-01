import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import (load_imagereward, tokenize_prompt,
                   grad_and_log_posterior_ir, log_posterior_ir,
                   compute_w2, compute_diversity, compute_diversity_cov)

PROMPT   = "a bald man"
DT_MALA  = 0.05   # from step-size sweep (56.8% accept, closest to 57% optimal)
N_CHAINS = 100
N_STEPS  = 1000
BURNIN   = 200
THIN_K   = 80

parser = argparse.ArgumentParser()
parser.add_argument('--n_trials',    type=int, default=5)
parser.add_argument('--seed',        type=int, default=321)
parser.add_argument('--rs_path',     type=str, default='experiments/bald_ir/results_rs_ir.pt')
parser.add_argument('--output_path', type=str, default='experiments/bald_ir/results_mala_ir.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
stylegan.G = torch.compile(stylegan.G)
reward_model = load_imagereward(device)
print('Models loaded.', flush=True)

noise_scale = np.sqrt(2 * DT_MALA)
n_kept      = (N_STEPS - BURNIN) // THIN_K

rs_data       = torch.load(args.rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

samples_list, w2_values, avg_ir_score, accept_rates = [], [], [], []

for trial in range(args.n_trials):
    print(f'\n=== Trial {trial+1}/{args.n_trials} ===', flush=True)
    torch.manual_seed(args.seed + trial)

    # max prompt length = N_CHAINS (chunked inside grad fn)
    prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, N_CHAINS)

    z = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    z_grad, log_p_z = grad_and_log_posterior_ir(z, stylegan, reward_model, prompt_ids, prompt_mask)

    n_accept = 0
    kept = []

    for step in tqdm(range(1, N_STEPS + 1), desc=f'MALA trial {trial+1}'):
        noise  = torch.randn_like(z)
        z_prop = z + DT_MALA * z_grad + noise_scale * noise

        z_prop_grad, log_p_prop = grad_and_log_posterior_ir(
            z_prop, stylegan, reward_model, prompt_ids, prompt_mask)

        log_q_fwd = -((z_prop - (z + DT_MALA * z_grad)) ** 2).sum(1) / (4 * DT_MALA)
        log_q_bwd = -((z - (z_prop + DT_MALA * z_prop_grad)) ** 2).sum(1) / (4 * DT_MALA)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)

        accept   = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask     = accept.unsqueeze(1)
        z        = torch.where(mask, z_prop,      z)
        z_grad   = torch.where(mask, z_prop_grad, z_grad)
        log_p_z  = torch.where(accept, log_p_prop, log_p_z)
        n_accept += accept.sum().item()

        if step > BURNIN and (step - BURNIN) % THIN_K == 0:
            kept.append(z.cpu())

    accept_rate = n_accept / (N_STEPS * N_CHAINS)
    accept_rates.append(accept_rate)
    print(f'  accept rate: {accept_rate:.1%}', flush=True)

    trial_samples = torch.cat(kept, dim=0)   # (n_kept * N_CHAINS, 512)
    samples_list.append(trial_samples)
    w2_values.append(compute_w2(trial_samples, rs_samples_list[trial]))

    prompt_ids_eval, prompt_mask_eval = tokenize_prompt(reward_model, PROMPT, device, len(trial_samples))
    ir_log_p = log_posterior_ir(trial_samples.to(device), stylegan, reward_model,
                                 prompt_ids_eval, prompt_mask_eval)
    avg_ir_score.append(ir_log_p.mean().item())
    print(f'  W2={w2_values[-1]:.4f}  avg_log_p={avg_ir_score[-1]:.4f}', flush=True)

print(f'\nMean accept rate: {np.mean(accept_rates):.1%}')
print(f'Mean W2:          {np.mean(w2_values):.4f}')
print(f'Mean avg_log_p:   {np.mean(avg_ir_score):.4f}')

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save({
    'samples':       samples_list,
    'w2_values':     w2_values,
    'avg_ir_score':  avg_ir_score,
    'accept_rates':  accept_rates,
    'prompt':        PROMPT,
    'dt':            DT_MALA,
}, args.output_path)
print(f'Saved to {args.output_path}')
