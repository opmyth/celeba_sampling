import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import grad_log_posterior_celeba

STEP_SIZES     = [0.001, 0.01, 0.03, 0.1]
SNAPSHOT_STEPS = {0, 200, 500, 1000, 2000, 3000}
N_CHAINS       = 3
N_STEPS        = 3000
ATTRIBUTE      = 'eyeglasses'
SEED           = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)

stylegan, clf, _ = load_models(ATTRIBUTE, device)
print('Compiling models...', flush=True)
clf        = torch.compile(clf)
stylegan.G = torch.compile(stylegan.G)
print('Done.', flush=True)


def run_ula(dt):
    """ULA for N_STEPS with fixed random init; returns snapshots at SNAPSHOT_STEPS."""
    torch.manual_seed(SEED)
    z           = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    noise_scale = np.sqrt(2 * dt)
    snaps       = {}

    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    for step in tqdm(range(1, N_STEPS + 1), desc=f'ULA dt={dt}'):
        grad = grad_log_posterior_celeba(z, stylegan, clf)
        z    = z + dt * grad + noise_scale * torch.randn_like(z)

        if step in SNAPSHOT_STEPS:
            snaps[step] = z.detach().cpu().clone()

    return snaps


snapshots = {}
for dt in STEP_SIZES:
    print(f'\n=== ULA dt={dt} ===', flush=True)
    snapshots[dt] = run_ula(dt)

out_dir  = os.path.join('experiments', ATTRIBUTE, 'trajectory')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'stepsize_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)
