"""Precompute the ImageReward clipping bound M (= r_max) for each bald_ir
prompt via a large prior scan.

Per Sanghyeok's fix (replacing the old empirical-headroom RS bound): redefine
the reward everywhere - RS, ULA, MALA, G_MH - as r_tilde(z) = min(IR(z), M),
so r_tilde/M <= 1 holds exactly by construction instead of RS relying on an
unverified "max of 2000 samples + 0.5 headroom" estimate that could in
principle be exceeded somewhere in the 512-dim latent space the scan never
visited. M is just the observed max itself, no headroom added - see
posteriors.py's estimate_r_max/load_r_max/imagereward_posterior for how it's
used downstream.

Usage: python estimate_r_max.py [--n_samples 50000]

Saves experiments/bald_ir/r_max.pt: {prompt: r_max}, one entry per prompt in
config.EXPERIMENTS['bald_ir'].prompts. Every script that builds an
imagereward_posterior for bald_ir (run_rs.py, run_prior.py, run_sampler.py,
run_trajectory.py, plot_trajectory.py, sweep_hyperparams.py,
run_prior_metrics.py) loads this file via posteriors.load_r_max(prompt)
rather than recomputing the scan itself - recomputing per script/per
invocation would be enormously wasteful (the whole point of this script is
to run the big scan exactly once per prompt) and would make M vary run to
run depending on each script's own RNG stream, when it should be one fixed
number shared by every sampler being compared.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_stylegan
from posteriors import estimate_r_max
from utils import load_imagereward

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=50000)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--force', action='store_true',
                     help='recompute prompts that already have a cached r_max instead of skipping them')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan = load_stylegan(device)
reward_model = load_imagereward(device)
print('Models loaded.', flush=True)

cfg = EXPERIMENTS['bald_ir']
out_path = 'experiments/bald_ir/r_max.pt'
os.makedirs('experiments/bald_ir', exist_ok=True)
r_max_by_prompt = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}

for prompt in cfg.prompts:
    if prompt in r_max_by_prompt and not args.force:
        print(f'\n=== prompt: "{prompt}" === already cached (M = {r_max_by_prompt[prompt]:.4f}), skipping '
              f'(--force to recompute)', flush=True)
        continue
    print(f'\n=== prompt: "{prompt}" ===', flush=True)
    generator = rng_mod.make_generator(args.seed, device)
    r_max = estimate_r_max(stylegan, reward_model, prompt, device, args.n_samples, generator)
    print(f'  M = {r_max:.4f}  (scanned {args.n_samples} samples)', flush=True)
    r_max_by_prompt[prompt] = r_max
    # saved after every prompt (not just once at the end) - a scan is ~50000
    # forward passes per prompt, expensive enough that losing an already-
    # completed prompt to a later prompt's crash/timeout would be wasteful.
    torch.save(r_max_by_prompt, out_path)
    print(f'  Saved to {out_path}', flush=True)

print(f'\nDone. {out_path} has: {list(r_max_by_prompt.keys())}', flush=True)
