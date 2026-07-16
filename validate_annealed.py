"""Validation gate for annealed MALA (samplers.latent_annealed_MALA_celeba),
2026-07-16. Run this BEFORE using annealing on Bald/Hat.

Runs annealed MALA on an experiment with known-good non-annealed MALA/RS
agreement (default: eyeglasses, see EXPERIMENTS.md - published MALA
W2=30.12+-0.09, AvgLogR=-0.57+-0.03 vs RS W2 baseline 30.23+-0.10,
AvgLogR=-0.63+-0.15) and reports the same W2/AvgLogR metrics against the
SAME RS reference samples the published numbers used. Since the T=1 tail
delegates to the plain MALA sampler (bit-exact at annealing_steps=0,
verified), a correct annealing phase should land the tail in the same
region plain MALA reaches from random init - reproducing the published
agreement. If it doesn't match reasonably, STOP: a weird Bald/Hat result
under annealing would be indistinguishable from an annealing bug.

Writes NOTHING to experiments/ - results print + append to
validate_annealed_results.txt (repo root).

Usage:
  python validate_annealed.py --probe          # quick timing probe first (no metrics)
  python validate_annealed.py                  # full 5-trial validation (~3h on A6000)
"""
import sys, os, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import numpy as np

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from samplers import latent_annealed_MALA_celeba
from utils import compute_w2, load_imagereward

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='eyeglasses', choices=list(EXPERIMENTS))
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--n_chains', type=int, default=None)
parser.add_argument('--n_trials', type=int, default=None)
parser.add_argument('--n_steps', type=int, default=None)
parser.add_argument('--burnin', type=int, default=None)
parser.add_argument('--thin_k', type=int, default=None)
parser.add_argument('--dt', type=float, default=None)
parser.add_argument('--n_temps', type=int, default=6)
parser.add_argument('--annealing_steps', type=int, default=700)
parser.add_argument('--dt_anneal', type=str, default=None,
                     help='comma-separated per-temperature dt list (n_temps+1 values, largest T first)')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--probe', action='store_true',
                     help='timing probe only: 1 trial, 300-step tail, short annealing, no metrics')
parser.add_argument('--chunk_size', type=int, default=None,
                     help='override StyleGAN2Wrapper.max_batch_size (default 64) - smaller for '
                          'low-VRAM GPUs; numerically equivalent, only affects memory/speed')
parser.add_argument('--out', type=str, default='validate_annealed_results.txt')
args = parser.parse_args()

from utils import maybe_enable_tf32
maybe_enable_tf32()

cfg = EXPERIMENTS[args.experiment]
prompt = args.prompt or cfg.prompt
n_chains = args.n_chains or cfg.n_chains
n_trials = args.n_trials or cfg.n_trials
n_steps = args.n_steps or cfg.n_steps
burnin = args.burnin if args.burnin is not None else cfg.burnin
thin_k = args.thin_k or cfg.thin_k
dt = args.dt if args.dt is not None else cfg.dt_mala
dt_anneal = [float(v) for v in args.dt_anneal.split(',')] if args.dt_anneal else None
if args.probe:
    n_trials, n_steps, burnin, thin_k = 1, 300, 0, 1
    annealing_steps = (args.n_temps + 1) * 20
else:
    annealing_steps = args.annealing_steps

_lines = []


def log(msg=''):
    print(msg, flush=True)
    _lines.append(msg)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'=== validate_annealed {datetime.datetime.now().isoformat(timespec="seconds")} ===')
log(f'experiment={args.experiment}, probe={args.probe}, n_chains={n_chains}, n_trials={n_trials}, '
    f'tail n_steps={n_steps} (burnin={burnin}, thin_k={thin_k}), dt={dt}, '
    f'n_temps={args.n_temps} (+1 temps), annealing_steps={annealing_steps}, dt_anneal={dt_anneal}, '
    f'node={os.environ.get("SLURMD_NODENAME", "local")}')

stylegan, clfs, _ = load_models(cfg.clf_names or [], device)
if args.chunk_size:
    # must happen BEFORE building the posterior - classifier_posterior reads
    # model.max_batch_size once at construction time
    stylegan.max_batch_size = args.chunk_size
    log(f'chunk_size override: {args.chunk_size}')
if cfg.kind == 'classifier':
    posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
else:
    reward_model = load_imagereward(device)
    posterior = imagereward_posterior(stylegan, reward_model, prompt, device, load_r_max(prompt))

# RS reference: same samples the published numbers were computed against
rs_samples_list = None
if not args.probe:
    for candidate in (f'experiments/{args.experiment}/results_rs.pt',
                      f'experiments/{args.experiment}/results_stylegan.pt'):
        if os.path.exists(candidate):
            data = torch.load(candidate, weights_only=False, map_location='cpu')
            rs_samples_list = data['samples'] if 'w2_baseline' in data else \
                data[list(data.keys())[0]]['samples']['RS']
            log(f'RS reference: {candidate} ({len(rs_samples_list)} trials x {rs_samples_list[0].shape[0]} samples)')
            break
    if rs_samples_list is None:
        sys.exit(f'No RS reference found for {args.experiment} - need results_rs.pt or results_stylegan.pt')

w2_values, alr_values, accept_rates = [], [], []
for trial in range(n_trials):
    generator = rng_mod.make_generator(args.seed + trial, device)
    t0 = time.time()
    samples, log_p_kept, accept_rate, _ = latent_annealed_MALA_celeba(
        posterior, n_chains, n_steps, dt, stylegan.latent_dim, device,
        generator=generator, burnin=burnin, thin_k=thin_k,
        n_temps=args.n_temps, annealing_steps=annealing_steps, dt_anneal=dt_anneal)
    el = time.time() - t0
    total_steps = annealing_steps // (args.n_temps + 1) * (args.n_temps + 1) + n_steps
    log(f'trial {trial + 1}/{n_trials}: {el:.1f}s ({total_steps} steps -> {total_steps/el:.2f} it/s), '
        f'tail accept={accept_rate:.1%}')
    accept_rates.append(accept_rate)

    if args.probe:
        full_cfg_steps = annealing_steps + (cfg.n_steps)
        log(f'probe extrapolation: full config ({args.annealing_steps} anneal + {cfg.n_steps} tail) x '
            f'{cfg.n_trials} trials = ~{(args.annealing_steps + cfg.n_steps) * cfg.n_trials / (total_steps/el) / 3600:.1f}h on this GPU')
        break

    z_all = torch.cat(samples, dim=0).cpu()
    lp_all = torch.cat(log_p_kept, dim=0).cpu()
    w2_values.append(compute_w2(z_all, rs_samples_list[trial]))
    alr_values.append((lp_all + 0.5 * (z_all ** 2).sum(1)).mean().item())
    log(f'  W2 vs RS={w2_values[-1]:.3f}, AvgLogR={alr_values[-1]:.3f}')

if not args.probe:
    log()
    log(f'=== annealed MALA on {args.experiment}: {n_trials} trials ===')
    log(f'W2:      {np.mean(w2_values):.2f}+-{np.std(w2_values, ddof=1):.2f}')
    log(f'AvgLogR: {np.mean(alr_values):.2f}+-{np.std(alr_values, ddof=1):.2f}')
    log(f'tail accept: {np.mean(accept_rates):.1%}')
    if args.experiment == 'eyeglasses':
        log('published non-annealed (EXPERIMENTS.md): MALA W2=30.12+-0.09, AvgLogR=-0.57+-0.03 '
            '(RS baseline W2=30.23+-0.10, AvgLogR=-0.63+-0.15, MALA accept ~unrecorded)')
        log('PASS criterion: W2 and AvgLogR within ~2 combined std of the published MALA row.')

with open(args.out, 'a') as f:
    f.write('\n'.join(_lines) + '\n\n')
print(f'\nResults appended to {args.out}', flush=True)
