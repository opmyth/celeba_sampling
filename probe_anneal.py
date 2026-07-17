"""Accept-rate probe for annealed MALA (2026-07-18), before committing the
full 5-trial runs. For each target posterior, runs the ANNEALING PHASE ONLY
(no tail - accept rate per temperature is the signal) two ways:
  - flat        : dt_T = dt_mala at every temperature (current default)
  - inv_sqrt_t  : dt_T = dt_mala / sqrt(T)
and prints per-temperature accept rates side by side, so we can pick the
schedule per posterior before the overnight runs.

What to look for: at T=1 (the last row) the accept rate should be in line
with that posterior's normal MALA accept rate (~45-65%). If flat leaves T=1
very low (chain effectively stuck as the target sharpens), inv_sqrt_t should
recover it. If BOTH leave T=1 very low, flag it - annealing_steps or the
schedule needs adjusting before committing.

Writes NOTHING to experiments/; appends a timestamped table to
probe_anneal_results.txt.

Usage:
  python probe_anneal.py                       # all target posteriors, in one process
  python probe_anneal.py --experiment bald     # just one (+ --prompt for bald_ir)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, datetime, torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_stylegan, load_classifier
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from samplers import anneal_to_T1, anneal_dt_schedule, anneal_temps
from utils import load_imagereward, maybe_enable_tf32

# (experiment, prompt) targets - the narrow posteriors annealing is aimed at
DEFAULT_TARGETS = [
    ('bald', None), ('wearing_hat', None), ('male_hat', None), ('notmale_hat', None),
    ('bald_ir', 'a bald man'), ('bald_ir', 'a person with a shaved head'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default=None, choices=list(EXPERIMENTS))
parser.add_argument('--prompt', default=None)
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--annealing_steps', type=int, default=350,
                     help='cheaper than the real 700 - enough steps/temp to estimate accept rate')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--out', type=str, default='probe_anneal_results.txt')
args = parser.parse_args()
maybe_enable_tf32()

targets = [(args.experiment, args.prompt)] if args.experiment else DEFAULT_TARGETS

_lines = []
def log(msg=''):
    print(msg, flush=True)
    _lines.append(msg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'=== probe_anneal {datetime.datetime.now().isoformat(timespec="seconds")} ===')
log(f'device={device}, node={os.environ.get("SLURMD_NODENAME","local")}, '
    f'n_chains={args.n_chains}, annealing_steps={args.annealing_steps}')

stylegan = load_stylegan(device)
_clf_cache = {}
_reward_model = None


def build_posterior(exp, prompt):
    global _reward_model
    cfg = EXPERIMENTS[exp]
    if cfg.kind == 'classifier':
        for n in cfg.clf_names:
            _clf_cache.setdefault(n, load_classifier(n, device))
        return classifier_posterior(stylegan, [_clf_cache[n] for n in cfg.clf_names])
    if _reward_model is None:
        _reward_model = load_imagereward(device)
    return imagereward_posterior(stylegan, _reward_model, prompt or cfg.prompt, device,
                                 load_r_max(prompt or cfg.prompt))


for exp, prompt in targets:
    cfg = EXPERIMENTS[exp]
    label = exp + (f' ["{prompt or cfg.prompt}"]' if cfg.kind == 'imagereward' else '')
    log(f'\n{"#"*70}\n### {label}  (dt_mala={cfg.dt_mala}, n_temps={cfg.anneal_n_temps})\n{"#"*70}')
    posterior = build_posterior(exp, prompt)

    results = {}
    for mode in ('flat', 'inv_sqrt_t'):
        z = torch.randn(args.n_chains, stylegan.latent_dim, device=device,
                        generator=rng_mod.make_generator(args.seed, device))
        _, per_temp, _ = anneal_to_T1(
            posterior, z, args.n_chains, stylegan.latent_dim, device,
            rng_mod.make_generator(args.seed + 1, device),
            n_temps=cfg.anneal_n_temps, annealing_steps=args.annealing_steps,
            dt=cfg.dt_mala, dt_anneal=anneal_dt_schedule(cfg.dt_mala, cfg.anneal_n_temps, mode))
        results[mode] = per_temp

    # side-by-side table
    log(f'  {"T":>7}  {"flat dt":>8} {"flat acc":>9}   {"1/sqrtT dt":>10} {"acc":>7}')
    for (T, dtf, af), (_, dti, ai) in zip(results['flat'], results['inv_sqrt_t']):
        log(f'  {T:>7.3f}  {dtf:>8.4g} {af:>8.1%}   {dti:>10.4g} {ai:>7.1%}')

with open(args.out, 'a') as f:
    f.write('\n'.join(_lines) + '\n\n')
print(f'\nDone. Results appended to {args.out}', flush=True)
