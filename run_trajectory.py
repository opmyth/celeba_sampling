"""Unified trajectory/diagnostics runner - replaces the 6 duplicated
run_trajectory_{stepsize,init}[_ir|_male_eye].py scripts.

  python run_trajectory.py --experiment <name> --mode stepsize [--init cold]
  python run_trajectory.py --experiment <name> --mode init [--noise same|indep]

Both modes call the generalized MALA sampler with thin_k=1, burnin=0, so the
full per-step latent + log_p trace is saved alongside the snapshot grid -
that trace is what plot_trajectory.py's jump_distance/log_reward plots read,
and it's also literally "save every accepted latent", not a separate feature.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior
from init_scan import get_init_z
from samplers import latent_MALA_celeba
from utils import load_imagereward

STEP_SIZES     = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
N_CHAINS       = 1
N_CANDIDATES   = 10000
INIT_TYPES     = ['random', 'cold', 'warm']

# Fractions of n_steps to snapshot - reproduces the original hardcoded
# {0,50,100,200,300,500,750,1000,2000,3000} exactly at n_steps=3000, and
# scales proportionally (same log-ish, burn-in-dense shape) for any other
# n_steps instead of needing new hardcoded values per step count.
_SNAPSHOT_FRACS = [0, 1/60, 1/30, 1/15, 1/10, 1/6, 1/4, 1/3, 2/3, 1.0]


def _snapshot_steps(n_steps):
    return {round(f * n_steps) for f in _SNAPSHOT_FRACS}


def _slug(s):
    return s.lower().replace(' ', '_')


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--mode', required=True, choices=['stepsize', 'init'])
parser.add_argument('--noise', default='same', choices=['same', 'indep'],
                     help='(init mode only) same: all init types share step noise; '
                          'indep: each init type gets its own noise stream')
parser.add_argument('--init', default='cold', choices=INIT_TYPES,
                     help='(stepsize mode only) which init to sweep step sizes from')
parser.add_argument('--prompt', type=str, default=None,
                     help='override the config default prompt (imagereward experiments only)')
parser.add_argument('--n_steps', type=int, default=3000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

SNAPSHOT_STEPS = _snapshot_steps(args.n_steps)

cfg = EXPERIMENTS[args.experiment]
prompt = args.prompt or cfg.prompt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, clfs, _ = load_models(cfg.clf_names or [], device)
if cfg.kind == 'classifier':
    posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
else:
    reward_model = load_imagereward(device)
    posterior = imagereward_posterior(stylegan, reward_model, prompt, device)
# stylegan.G intentionally NOT compiled here: compile caches requires_grad
# state, and every mode below toggles between the no-grad init scan and
# grad-requiring MALA steps, which crashes a compiled G. These are small
# (N_CHAINS=3) diagnostic runs, so compiling wouldn't pay off anyway.

base_dir = os.path.join('experiments', args.experiment, 'trajectory')
# imagereward experiments always get a prompt subdirectory (even for the
# config default prompt) so runs for different prompts never collide.
prompt_slug = f'prompt_{_slug(prompt)}' if cfg.kind == 'imagereward' else None


def run_mala_with_trace(dt, z_init, generator):
    samples, log_p_kept, accept_rate, _ = latent_MALA_celeba(
        posterior, N_CHAINS, args.n_steps, dt, stylegan.latent_dim, device,
        generator=generator, burnin=0, thin_k=1, z_init=z_init)
    snaps = {step: samples[min(step, len(samples) - 1)].cpu()
             for step in SNAPSHOT_STEPS if step <= args.n_steps}
    trace = {'z': torch.stack(samples).cpu(), 'log_p': torch.stack(log_p_kept).cpu()}
    print(f'  accept_rate={accept_rate:.1%}' if accept_rate is not None else '', flush=True)
    return snaps, trace


if args.mode == 'stepsize':
    out_dir = os.path.join(base_dir, prompt_slug) if prompt_slug else base_dir
    os.makedirs(out_dir, exist_ok=True)

    init_gen = rng_mod.make_generator(args.seed, device)
    z0 = get_init_z(args.init, posterior.reward_only_fn, N_CHAINS, stylegan.latent_dim,
                     device, init_gen, n_candidates=N_CANDIDATES)

    snapshots, traces = {}, {}
    for dt in STEP_SIZES:
        print(f'\n=== dt={dt} ===', flush=True)
        # same generator seed every dt => identical noise/accept draws across
        # the whole sweep, so only dt itself differs between rows.
        gen = rng_mod.make_generator(args.seed + 1, device)
        snapshots[dt], traces[dt] = run_mala_with_trace(dt, z0, gen)

    torch.save(snapshots, os.path.join(out_dir, 'stepsize_snapshots.pt'))
    torch.save(traces, os.path.join(out_dir, 'stepsize_trace.pt'))
    print(f'\nSaved to {out_dir}', flush=True)

else:  # init
    snapshots, traces = {}, {}
    for i, init_type in enumerate(INIT_TYPES):
        print(f'\n=== init: {init_type} ===', flush=True)
        # always the same seed for the candidate scan itself, so cold/warm
        # are picked from an identical pool regardless of --noise.
        init_gen = rng_mod.make_generator(args.seed, device)
        z0 = get_init_z(init_type, posterior.reward_only_fn, N_CHAINS, stylegan.latent_dim,
                         device, init_gen, n_candidates=N_CANDIDATES)

        noise_seed = args.seed if args.noise == 'same' else args.seed + i + 1
        noise_gen = rng_mod.make_generator(noise_seed, device)
        snapshots[init_type], traces[init_type] = run_mala_with_trace(cfg.dt_mala, z0, noise_gen)

    noise_dir = 'same_noise' if args.noise == 'same' else 'indep_noise'
    sub_dir = os.path.join(base_dir, noise_dir)
    if prompt_slug:
        sub_dir = os.path.join(sub_dir, prompt_slug)
    os.makedirs(sub_dir, exist_ok=True)
    torch.save(snapshots, os.path.join(sub_dir, 'init_snapshots.pt'))
    torch.save(traces, os.path.join(sub_dir, 'init_trace.pt'))
    print(f'\nSaved to {sub_dir}', flush=True)

print('Done.', flush=True)
