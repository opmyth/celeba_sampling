import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from model_loader import load_models

MALA_DT        = 0.05    # from sweep: 57.1% accept
SNAPSHOT_STEPS = {0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000}
N_CHAINS       = 3
N_STEPS        = 3000
N_CANDIDATES   = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--seed',  type=int, default=42)
parser.add_argument('--noise', choices=['same', 'indep'], default='same')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, clf_eye, clf_male = load_models('eyeglasses', device)
clf_eye  = torch.compile(clf_eye)
clf_male = torch.compile(clf_male)
# stylegan.G intentionally NOT compiled: compile caches requires_grad state,
# causing crashes when switching between grad_log_p (requires_grad=True) and
# log_p_no_grad (no_grad context) for cold/warm init scanning.
print('Models loaded.', flush=True)

noise_scale = np.sqrt(2 * MALA_DT)


def log_p_no_grad(z):
    with torch.no_grad():
        imgs = stylegan.G(z, None)
        return (-0.5 * (z ** 2).sum(1)
                + F.logsigmoid(clf_male(imgs).squeeze(-1))
                + F.logsigmoid(clf_eye(imgs).squeeze(-1)))


def grad_log_p(z):
    z = z.detach().requires_grad_(True)
    imgs  = stylegan.G(z, None)
    log_p = (-0.5 * (z ** 2).sum(1)
             + F.logsigmoid(clf_male(imgs).squeeze(-1))
             + F.logsigmoid(clf_eye(imgs).squeeze(-1)))
    log_p.sum().backward()
    return z.grad.clone(), log_p.detach()


def get_init_z(init_type):
    if init_type == 'random':
        return torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    torch.cuda.empty_cache()
    print(f'  scanning {N_CANDIDATES} candidates for {init_type} init...', flush=True)
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
    idx        = torch.argsort(all_scores)
    selected   = idx[:N_CHAINS] if init_type == 'cold' else idx[-N_CHAINS:]
    print(f'  score range: [{all_scores[selected].min():.4f}, {all_scores[selected].max():.4f}]', flush=True)
    return all_z[selected].to(device)


def run_mala(z_init):
    z     = z_init.clone()
    snaps = {}
    if 0 in SNAPSHOT_STEPS:
        snaps[0] = z.detach().cpu().clone()

    z_grad, log_p_z = grad_log_p(z)

    for step in tqdm(range(1, N_STEPS + 1), desc='MALA'):
        noise  = torch.randn_like(z)
        z_prop = z + MALA_DT * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_log_p(z_prop)
        log_q_fwd = -((z_prop - (z + MALA_DT * z_grad)) ** 2).sum(1) / (4 * MALA_DT)
        log_q_bwd = -((z - (z_prop + MALA_DT * z_prop_grad)) ** 2).sum(1) / (4 * MALA_DT)
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
for init_type in ['random', 'cold', 'warm']:
    print(f'\n=== init: {init_type} ===', flush=True)
    torch.manual_seed(args.seed)
    z0 = get_init_z(init_type)
    if args.noise == 'same':
        torch.manual_seed(args.seed)
    snapshots[init_type] = run_mala(z0)

noise_dir = 'same_noise' if args.noise == 'same' else 'indep_noise'
out_dir   = os.path.join('experiments', 'male_eye', 'trajectory', noise_dir)
os.makedirs(out_dir, exist_ok=True)
out_path  = os.path.join(out_dir, 'init_snapshots.pt')
torch.save(snapshots, out_path)
print(f'\nSaved to {out_path}', flush=True)

from plot_trajectory_male_eye import plot_init_grid
plot_init_grid(stylegan, clf_male, clf_eye, noise=args.noise)
