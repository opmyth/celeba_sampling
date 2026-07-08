"""Real, importable hyperparameter sweep - replaces the 4 near-identical
Python heredocs embedded in submit_sweep.sh / submit_sweep_male.sh /
submit_sweep_ir.sh / submit_sweep_beta_ir.sh.

  python sweep_hyperparams.py --experiment eyeglasses --sweep dt_mala --values 0.05,0.07,0.08,0.09,0.10
  python sweep_hyperparams.py --experiment bald_ir     --sweep beta   --values 10,50,100,200
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, math
import numpy as np
import torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior
from samplers import latent_MALA_celeba, latent_ULA_celeba, latent_Gaussian_MH_celeba
from utils import load_imagereward

SAMPLER_FOR_SWEEP = {
    'dt_mala': latent_MALA_celeba,
    'dt_ula': latent_ULA_celeba,
    'sigma_gmh': latent_Gaussian_MH_celeba,
    'beta': latent_MALA_celeba,
}
TARGET_ACCEPT = {'dt_mala': (0.45, 0.70), 'sigma_gmh': (0.15, 0.32), 'beta': (0.45, 0.70)}


def log_p_stats(trace):
    arr = np.asarray(trace)
    n = min(50, len(arr))
    early, late = arr[:n].mean(), arr[-n:].mean()
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    return early, late, slope


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--sweep', required=True, choices=list(SAMPLER_FOR_SWEEP))
parser.add_argument('--values', required=True, help='comma-separated values to sweep')
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_warmup', type=int, default=5)
parser.add_argument('--n_steps', type=int, default=300)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

cfg = EXPERIMENTS[args.experiment]
prompt = args.prompt or cfg.prompt
values = [float(v) for v in args.values.split(',')]

if args.sweep == 'beta' and cfg.kind != 'imagereward':
    raise ValueError("--sweep beta only applies to imagereward experiments")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, clfs, _ = load_models(cfg.clf_names or [], device)
reward_model = load_imagereward(device) if cfg.kind == 'imagereward' else None
posterior = (classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names]) if cfg.kind == 'classifier'
             else imagereward_posterior(stylegan, reward_model, prompt, device))

sampler_fn = SAMPLER_FOR_SWEEP[args.sweep]


def run_trial(value):
    if args.sweep == 'beta':
        post = imagereward_posterior(stylegan, reward_model, prompt, device, beta=value)
        step_param = cfg.dt_mala
    else:
        post = posterior
        step_param = value

    warmup_gen = rng_mod.make_generator(args.seed, device)
    sampler_fn(post, args.n_chains, args.n_warmup, step_param, stylegan.latent_dim, device,
               generator=warmup_gen)

    gen = rng_mod.make_generator(args.seed, device)
    samples, log_p_kept, accept_rate, log_p_trace = sampler_fn(
        post, args.n_chains, args.n_steps, step_param, stylegan.latent_dim, device,
        generator=gen, burnin=0, thin_k=1, return_diagnostics=True)

    if args.sweep == 'beta':
        # track raw (untempered) IR score, not the beta-scaled log_p
        trace = np.array([post.reward_only_fn(z).mean().item() for z in samples])
    else:
        trace = log_p_trace.mean(dim=1).numpy()

    return accept_rate, *log_p_stats(trace)


label = {'dt_mala': 'DT', 'dt_ula': 'DT', 'sigma_gmh': 'sigma', 'beta': 'beta'}[args.sweep]
print(f'\n{"=" * 65}', flush=True)
print(f'  {args.experiment} - {args.sweep} sweep  ({args.n_chains} chains x {args.n_steps} steps)', flush=True)
if args.sweep == 'sigma_gmh':
    print(f'  Theory: sigma_opt = 2.38/sqrt(512) = {2.38 / math.sqrt(512):.4f}', flush=True)
print(f'{"=" * 65}', flush=True)
print(f'  {label:>8}  {"accept%":>8}  {"trace early":>12}  {"trace late":>10}  {"trend/step":>10}', flush=True)
print(f'  {"-" * 56}', flush=True)

for value in values:
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    accept_rate, early, late, slope = run_trial(value)
    accept_str = f'{accept_rate:>7.1%}' if accept_rate is not None else '     N/A'
    target = TARGET_ACCEPT.get(args.sweep)
    flag = ' <- target' if target and accept_rate is not None and target[0] < accept_rate < target[1] else ''
    print(f'  {value:>8.4g}  {accept_str}  {early:>12.4f}  {late:>10.4f}  {slope:>+10.4f}{flag}', flush=True)

print('\nDone.', flush=True)
