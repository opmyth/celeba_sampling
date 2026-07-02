import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from model_loader import load_models

STEP_SIZES     = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
SNAPSHOT_STEPS = {0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000}
N_CHAINS       = 3
N_STEPS        = 3000
N_CANDIDATES   = 10000
SEED           = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, clf_eye, clf_male = load_models('eyeglasses', device)
clf_eye  = torch.compile(clf_eye)
clf_male = torch.compile(clf_male)
# stylegan.G not compiled: avoid requires_grad mismatch between cold-init scan and MALA
print('Models loaded.', flush=True)


def grad_log_p(z):
    z = z.detach().requires_grad_(True)
    imgs  = stylegan.G(z, None)
    log_p = (-0.5 * (z ** 2).sum(1)
             + F.logsigmoid(clf_male(imgs).squeeze(-1))
             + F.logsigmoid(clf_eye(imgs).squeeze(-1)))
    log_p.sum().backward()
    return z.grad.clone(), log_p.detach()


def get_cold_init():
    """Start from female-without-glasses: lowest p_male × p_eye over N_CANDIDATES."""
    torch.manual_seed(SEED)
    scores, zs = [], []
    for start in range(0, N_CANDIDATES, 64):
        size   = min(64, N_CANDIDATES - start)
        z_cand = torch.randn(size, stylegan.latent_dim, device=device)
        with torch.no_grad():
            imgs  = stylegan.G(z_cand, None)
            score = (torch.sigmoid(clf_male(imgs)).squeeze(-1)
                     * torch.sigmoid(clf_eye(imgs)).squeeze(-1))
        scores.append(score.cpu())
        zs.append(z_cand.cpu())
    all_scores = torch.cat(scores)
    all_z      = torch.cat(zs)
    idx        = torch.argsort(all_scores)[:N_CHAINS]
    print(f'Cold init score range: [{all_scores[idx].min():.4f}, {all_scores[idx].max():.4f}]', flush=True)
    return all_z[idx].to(device)


z_cold = get_cold_init()


def run_mala(dt):
    z           = z_cold.clone()
    noise_scale = np.sqrt(2 * dt)
    snaps       = {}

    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    z_grad, log_p_z = grad_log_p(z)

    for step in tqdm(range(1, N_STEPS + 1), desc=f'MALA dt={dt}'):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_log_p(z_prop)
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
    print(f'\n=== dt={dt} ===', flush=True)
    snapshots[dt] = run_mala(dt)

out_dir  = os.path.join('experiments', 'male_eye', 'trajectory')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'stepsize_cold_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)

from plot_trajectory_male_eye import plot_stepsize_grid
plot_stepsize_grid(stylegan, clf_male, clf_eye, snap_file='stepsize_cold_snapshots.pt')
