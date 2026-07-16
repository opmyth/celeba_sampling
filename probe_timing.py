"""One-off timing probe for N_CHAINS=100 trajectory runs (2026-07-16).

Measures the real per-step MALA cost at n_chains=100 for both posterior
kinds - imagereward (bald_ir) and classifier (notmale) - using the exact
code path run_trajectory.py uses (same sampler, same init scan, same
load_imagereward incl. BLIP torch.compile), WITHOUT touching any files in
experiments/ (run_trajectory.py --mode stepsize has no single-dt option and
would overwrite the real saved trajectory .pt files with probe garbage).

Usage (interactive srun session, after `source scripts/env.sh`):
    python probe_timing.py [--n_chains 100] [--n_steps 300]

Extrapolation the numbers feed (per experiment):
    stepsize sweep = 9 x (3000-step run) + 1 init scan
    init sweep     = 2 noise settings x 3 x (init scan + 3000-step run)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_stylegan, load_classifier
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from init_scan import get_init_z
from samplers import latent_MALA_celeba
from utils import load_imagereward

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=300)
args = parser.parse_args()

device = torch.device('cuda')
N, STEPS = args.n_chains, args.n_steps
print(f'Probe: n_chains={N}, n_steps={STEPS}', flush=True)
print(torch.cuda.get_device_name(0), flush=True)

stylegan = load_stylegan(device)


def probe(tag, posterior, dt):
    t0 = time.time()
    init_gen = rng_mod.make_generator(42, device)
    z0 = get_init_z('cold', posterior.reward_only_fn, N, stylegan.latent_dim,
                     device, init_gen, n_candidates=10000)
    t_init = time.time() - t0
    print(f'[{tag}] init scan 10000 candidates: {t_init:.1f}s', flush=True)

    t0 = time.time()
    latent_MALA_celeba(posterior, N, STEPS, dt, stylegan.latent_dim, device,
                        generator=rng_mod.make_generator(43, device),
                        burnin=0, thin_k=1, z_init=z0)
    el = time.time() - t0
    per_3000 = el * (3000 / STEPS)
    print(f'[{tag}] MALA {STEPS} steps @ n_chains={N}: {el:.1f}s '
          f'-> {STEPS/el:.3f} it/s -> {per_3000/60:.1f} min per 3000-step run', flush=True)
    return t_init, per_3000


# --- probe 1: imagereward (bald_ir, default prompt, dt=cfg.dt_mala) ---
cfg = EXPERIMENTS['bald_ir']
reward_model = load_imagereward(device)
post_ir = imagereward_posterior(stylegan, reward_model, cfg.prompt, device, load_r_max(cfg.prompt))
ir_init, ir_run = probe('IR', post_ir, cfg.dt_mala)

# --- probe 2: classifier (notmale, dt=cfg.dt_mala) ---
cfg2 = EXPERIMENTS['notmale']
post_clf = classifier_posterior(stylegan, [load_classifier(n, device) for n in cfg2.clf_names])
clf_init, clf_run = probe('CLF', post_clf, cfg2.dt_mala)

print('\n=== extrapolation (this GPU, 3000 steps) ===', flush=True)
for tag, t_init, per_run in [('IR', ir_init, ir_run), ('CLF', clf_init, clf_run)]:
    stepsize = (9 * per_run + t_init) / 3600
    init_sweep = (2 * 3 * (t_init + per_run)) / 3600
    print(f'[{tag}] stepsize sweep (9 dt): {stepsize:.1f}h | '
          f'init sweep (3 types x 2 noise): {init_sweep:.1f}h | '
          f'both: {stepsize + init_sweep:.1f}h', flush=True)
