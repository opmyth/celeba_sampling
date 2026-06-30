import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm

import argparse
from model_loader import load_models
from utils import grad_and_log_posterior_celeba

STEP_SIZES     = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
SNAPSHOT_STEPS = {0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000}
N_CHAINS       = 3
N_STEPS        = 3000
SEED           = 42

parser = argparse.ArgumentParser()
parser.add_argument('--attribute', required=True, choices=['smile', 'eyeglasses', 'bald'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)

stylegan, clf, _ = load_models(args.attribute, device)
print('Compiling models...', flush=True)
clf        = torch.compile(clf)
stylegan.G = torch.compile(stylegan.G)
print('Done.', flush=True)


def run_mala(dt):
    """MALA for N_STEPS with fixed random init; returns snapshots at SNAPSHOT_STEPS."""
    torch.manual_seed(SEED)
    z           = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    noise_scale = np.sqrt(2 * dt)
    snaps       = {}

    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    z_grad, log_p_z = grad_and_log_posterior_celeba(z, stylegan, clf)

    for step in tqdm(range(1, N_STEPS + 1), desc=f'MALA dt={dt}'):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise

        z_prop_grad, log_p_prop = grad_and_log_posterior_celeba(z_prop, stylegan, clf)

        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad))**2,      dim=1) / (4 * dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad))**2, dim=1) / (4 * dt)
        log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)

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
    print(f'\n=== MALA dt={dt} ===', flush=True)
    snapshots[dt] = run_mala(dt)

out_dir  = os.path.join('experiments', args.attribute, 'trajectory')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'stepsize_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)
