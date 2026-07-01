import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch, numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import load_imagereward, tokenize_prompt, grad_and_log_posterior_ir

PROMPT         = "a bald man"
STEP_SIZES     = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
SNAPSHOT_STEPS = {0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000}
N_CHAINS       = 3
N_STEPS        = 3000
SEED           = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
stylegan.G     = torch.compile(stylegan.G)
reward_model   = load_imagereward(device)
print('Models loaded.', flush=True)

prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, N_CHAINS)


def run_mala(dt):
    torch.manual_seed(SEED)
    z           = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    noise_scale = np.sqrt(2 * dt)
    snaps       = {}

    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    z_grad, log_p_z = grad_and_log_posterior_ir(z, stylegan, reward_model, prompt_ids, prompt_mask)

    for step in tqdm(range(1, N_STEPS + 1), desc=f'MALA-IR dt={dt}'):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise

        z_prop_grad, log_p_prop = grad_and_log_posterior_ir(
            z_prop, stylegan, reward_model, prompt_ids, prompt_mask)

        log_q_fwd = -((z_prop - (z + dt * z_grad)) ** 2).sum(1) / (4 * dt)
        log_q_bwd = -((z - (z_prop + dt * z_prop_grad)) ** 2).sum(1) / (4 * dt)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)

        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

        if step in SNAPSHOT_STEPS:
            snaps[step] = z.detach().cpu().clone()

    return snaps


snapshots = {}
for dt in STEP_SIZES:
    print(f'\n=== MALA-IR dt={dt} ===', flush=True)
    snapshots[dt] = run_mala(dt)

out_dir  = os.path.join('experiments', 'bald', 'trajectory', 'imagereward')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'stepsize_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)
