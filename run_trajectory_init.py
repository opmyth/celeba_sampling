import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import grad_and_log_posterior_celeba

# MALA step sizes matching the main pipeline per attribute (from run_all.sh / project.md)
MALA_DT = {
    'smile':      0.1,
    'eyeglasses': 0.05,
    'bald':       0.1,
}

SNAPSHOT_STEPS = {0, 200, 500, 1000, 2000, 3000}
N_CHAINS    = 3
N_STEPS     = 3000
N_CANDIDATES = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--attribute', required=True, choices=list(MALA_DT.keys()))
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)

stylegan, clf, _ = load_models(args.attribute, device)
print('Compiling models...', flush=True)
clf        = torch.compile(clf)
stylegan.G = torch.compile(stylegan.G)
print('Done.', flush=True)

dt          = MALA_DT[args.attribute]
noise_scale = np.sqrt(2 * dt)


def get_init_z(init_type):
    if init_type == 'random':
        return torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    print(f'  scanning {N_CANDIDATES} candidates for {init_type} init...', flush=True)
    batch = 512
    scores, zs = [], []
    for start in range(0, N_CANDIDATES, batch):
        size   = min(batch, N_CANDIDATES - start)
        z_cand = torch.randn(size, stylegan.latent_dim, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(clf(stylegan(z_cand))).squeeze()
        scores.append(probs.cpu())
        zs.append(z_cand.cpu())
    all_scores = torch.cat(scores)
    all_z      = torch.cat(zs)
    idx        = torch.argsort(all_scores)
    selected   = idx[:N_CHAINS] if init_type == 'cold' else idx[-N_CHAINS:]
    rng = all_scores[selected]
    print(f'  score range of selected: [{rng.min():.4f}, {rng.max():.4f}]', flush=True)
    return all_z[selected].to(device)


def run_mala(z_init):
    """MALA for N_STEPS; returns snapshots at SNAPSHOT_STEPS."""
    z = z_init.clone()
    snaps = {}

    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    z_grad, log_p_z = grad_and_log_posterior_celeba(z, stylegan, clf)

    for step in tqdm(range(1, N_STEPS + 1), desc='MALA'):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise

        z_prop_grad, log_p_prop = grad_and_log_posterior_celeba(z_prop, stylegan, clf)

        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad))**2,       dim=1) / (4 * dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad))**2,  dim=1) / (4 * dt)
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
for init_type in ['random', 'cold', 'warm']:
    print(f'\n=== init: {init_type} ===', flush=True)
    torch.manual_seed(args.seed)
    z0 = get_init_z(init_type)
    torch.manual_seed(args.seed) 
    snapshots[init_type] = run_mala(z0)

out_dir  = os.path.join('experiments', args.attribute, 'trajectory')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'init_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)
