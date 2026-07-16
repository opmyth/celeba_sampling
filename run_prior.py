#!/usr/bin/env python3
"""Evaluate the prior z ~ N(0, I) as a baseline. Samples directly - no MCMC."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")

import argparse, torch
import numpy as np
import wandb

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from utils import (compute_w2, compute_diversity, compute_diversity_cov,
                    compute_male_fraction, load_imagereward)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--n_trials', type=int, default=None)
parser.add_argument('--prompt', type=str, default=None,
                     help='override the config default prompt (imagereward experiments only)')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--rs_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

cfg = EXPERIMENTS[args.experiment]
n_trials = args.n_trials or cfg.n_trials
prompt = args.prompt or cfg.prompt
rs_path = args.rs_path or f'experiments/{args.experiment}/results_rs.pt'
output_path = args.output_path or f'experiments/{args.experiment}/results_prior.pt'

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"Prior-{args.experiment}-trials{n_trials}",
    group=f"{args.experiment}_t{n_trials}",
    job_type="Prior",
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, clfs, male_clf = load_models(cfg.clf_names or [], device)

if cfg.kind == 'classifier':
    posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
else:
    reward_model = load_imagereward(device)
    posterior = imagereward_posterior(stylegan, reward_model, prompt, device, load_r_max(prompt))

rs_data = torch.load(rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

# Prior samples are drawn on CPU (matching RS's/downstream's expected placement);
# .to(device) happens per-call when they're fed to the models.
cpu_generator = rng_mod.make_generator(args.seed, 'cpu')
samples_list = [torch.randn(cfg.rs_target, stylegan.latent_dim, generator=cpu_generator)
                 for _ in range(n_trials)]

w2_values = [compute_w2(samples_list[i], rs_samples_list[i]) for i in range(n_trials)]
avg_log_reward = [posterior.reward_only_fn(samples_list[i].to(device)).mean().item() for i in range(n_trials)]
diversity = [compute_diversity(samples_list[i]) for i in range(n_trials)]
diversity_trace_cov = [compute_diversity_cov(samples_list[i]) for i in range(n_trials)]
male_fraction = [compute_male_fraction(stylegan, male_clf, samples_list[i].to(device)) for i in range(n_trials)]

wandb.log({
    "w2_values_mean":           np.mean(w2_values),
    "w2_values_std":            np.std(w2_values, ddof=1),
    "avg_log_reward_mean":      np.mean(avg_log_reward),
    "avg_log_reward_std":       np.std(avg_log_reward, ddof=1),
    "diversity_mean":           np.mean(diversity),
    "diversity_std":            np.std(diversity, ddof=1),
    "diversity_trace_cov_mean": np.mean(diversity_trace_cov),
    "diversity_trace_cov_std":  np.std(diversity_trace_cov, ddof=1),
    "male_fraction_mean":       np.mean(male_fraction),
    "male_fraction_std":        np.std(male_fraction, ddof=1),
})

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save({
    'samples':             samples_list,
    'w2_values':            w2_values,
    'avg_log_reward':      avg_log_reward,
    'diversity':           diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction':       male_fraction,
}, output_path)

print(f'Prior results saved to {output_path}')
wandb.finish()
