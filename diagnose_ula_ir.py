"""Diagnostic (2026-07-17): does ULA diverge on the ImageReward posterior
because of THIS WEEK's r_max clipping, or was it always unstable?

Runs ULA at the OLD settings (dt=0.01, 1000 steps, the config the deleted
run_ula_ir.py used) on the SAME posterior two ways:
  - CLIPPED   : r_tilde = min(IR, M)   (this week's reward)
  - UNCLIPPED : raw IR                 (old-style, r_max=inf so clamp never binds)
and reports, for each, the step at which the first chain goes NaN and how
many chains are NaN at the end. Writes NOTHING to experiments/.

Reading it:
  both NaN, similar onset  -> ULA was always unstable on IR; clipping irrelevant.
                              Old ULA-IR results (if they existed) were a
                              lucky/partial run, and dropping ULA is correct.
  unclipped survives, clipped NaNs -> the clip introduced the instability;
                              worth understanding before reporting.
  clipped survives, unclipped NaNs -> old raw-IR ULA was WORSE; new setup
                              is the more stable one.

Usage: python diagnose_ula_ir.py [--dt 0.01] [--n_steps 1000] [--prompt "a bald man"]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch

import rng as rng_mod
from model_loader import load_stylegan
from posteriors import imagereward_posterior, load_r_max
from samplers import latent_ULA_celeba
from utils import load_imagereward, maybe_enable_tf32

parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--prompt', type=str, default='a bald man')
parser.add_argument('--seed', type=int, default=321)
args = parser.parse_args()
maybe_enable_tf32()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
stylegan = load_stylegan(device)
reward_model = load_imagereward(device)
r_max_clip = load_r_max(args.prompt)
print(f'clipped r_max (M) = {r_max_clip:.4f}', flush=True)
print(f'\n=== ULA dt={args.dt}, {args.n_chains} chains, {args.n_steps} steps, '
      f'prompt="{args.prompt}" ===', flush=True)


def run(label, r_max):
    post = imagereward_posterior(stylegan, reward_model, args.prompt, device, r_max)
    gen = rng_mod.make_generator(args.seed, device)
    # thin_k huge + burnin 0 => keep almost no samples (memory); the diagnostic
    # signal is log_p_trace (n_steps, n_chains), returned via return_diagnostics.
    _, _, _, trace = latent_ULA_celeba(
        post, args.n_chains, args.n_steps, args.dt, stylegan.latent_dim, device,
        generator=gen, burnin=0, thin_k=args.n_steps, return_diagnostics=True)
    nan_step = torch.isnan(trace).any(dim=1)            # (n_steps,) any chain NaN
    first = int(nan_step.float().argmax().item()) if bool(nan_step.any()) else None
    final_nan = int(torch.isnan(trace[-1]).sum().item())
    if first is None:
        print(f'[{label}] survived all {args.n_steps} steps, 0 NaN chains', flush=True)
    else:
        print(f'[{label}] first NaN at step {first}, {final_nan}/{args.n_chains} chains NaN by end', flush=True)


run('CLIPPED (this week)', r_max_clip)
run('UNCLIPPED (old-style raw IR)', float('inf'))
print('\nDone.', flush=True)
