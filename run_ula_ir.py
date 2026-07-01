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
                   compute_w2)

PROMPT   = "a bald man"
DT_ULA   = 0.01
N_CHAINS = 100
N_STEPS  = 3000
BURNIN   = 1000
THIN_K   = 200

parser = argparse.ArgumentParser()
parser.add_argument('--n_trials',    type=int, default=5)
parser.add_argument('--seed',        type=int, default=321)
parser.add_argument('--rs_path',     type=str, default='experiments/bald_ir/results_rs_ir.pt')
parser.add_argument('--output_path', type=str, default='experiments/bald_ir/results_ula_ir.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
stylegan.G = torch.compile(stylegan.G)
reward_model = load_imagereward(device)
print('Models loaded.', flush=True)

noise_scale = np.sqrt(2 * DT_ULA)

rs_data          = torch.load(args.rs_path, weights_only=False)
rs_samples_list  = rs_data['samples']

samples_list, w2_values, avg_ir_score = [], [], []

for trial in range(args.n_trials):
    print(f'\n=== Trial {trial+1}/{args.n_trials} ===', flush=True)
    torch.manual_seed(args.seed + trial)

    prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, N_CHAINS)

    z = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    kept = []

    for step in tqdm(range(1, N_STEPS + 1), desc=f'ULA trial {trial+1}'):
        z_grad, _ = grad_and_log_posterior_ir(z, stylegan, reward_model, prompt_ids, prompt_mask)
        noise = torch.randn_like(z)
        z = z + DT_ULA * z_grad + noise_scale * noise

        if step > BURNIN and (step - BURNIN) % THIN_K == 0:
            kept.append(z.detach().cpu())

    trial_samples = torch.cat(kept, dim=0)
    samples_list.append(trial_samples)
    w2_values.append(compute_w2(trial_samples, rs_samples_list[trial]))

    prompt_ids_eval, prompt_mask_eval = tokenize_prompt(reward_model, PROMPT, device, len(trial_samples))
    ir_log_p = log_posterior_ir(trial_samples.to(device), stylegan, reward_model,
                                 prompt_ids_eval, prompt_mask_eval)
    avg_ir_score.append(ir_log_p.mean().item())
    print(f'  W2={w2_values[-1]:.4f}  avg_log_p={avg_ir_score[-1]:.4f}', flush=True)

print(f'\nMean W2:        {np.mean(w2_values):.4f}')
print(f'Mean avg_log_p: {np.mean(avg_ir_score):.4f}')

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save({
    'samples':      samples_list,
    'w2_values':    w2_values,
    'avg_ir_score': avg_ir_score,
    'prompt':       PROMPT,
    'dt':           DT_ULA,
}, args.output_path)
print(f'Saved to {args.output_path}')
