#!/usr/bin/env python3
"""
Evaluate the prior z ~ N(0, I) as a baseline sampler.
Samples directly — no MCMC needed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")

import argparse, torch, time
import numpy as np
import torch.nn.functional as F
import wandb

from model_loader import load_models
from utils import compute_w2, compute_diversity, compute_diversity_cov, compute_male_fraction

parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str, required=True)
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_trials', type=int, default=5)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--rs_path', type=str, default='results_rs.pt')
parser.add_argument('--output_path', type=str, default='results_prior.pt')
args = parser.parse_args()

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"Prior-{args.clf_name}-chains{args.n_chains}-trials{args.n_trials}",
    group=f"{args.clf_name}_n{args.n_chains}_t{args.n_trials}",
    job_type="Prior",
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, clf, male_clf = load_models(args.clf_name, device)
torch.manual_seed(args.seed)

rs_data = torch.load(args.rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

latent_dim = stylegan.latent_dim
samples_list = [
    torch.randn(args.n_chains, latent_dim)
    for _ in range(args.n_trials)
]

t = time.time()
w2_values = [compute_w2(samples_list[i], rs_samples_list[i]) for i in range(args.n_trials)]
print(f"Prior W2 done: {time.time()-t:.2f}s", flush=True)

t = time.time()
with torch.no_grad():
    avg_log_reward = [
        F.logsigmoid(clf(stylegan(samples_list[i].to(device)))).mean().item()
        for i in range(args.n_trials)
    ]
print(f"avg_log_reward done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity = [compute_diversity(samples_list[i]) for i in range(args.n_trials)]
print(f"diversity done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity_trace_cov = [compute_diversity_cov(samples_list[i]) for i in range(args.n_trials)]
print(f"diversity_trace_cov done: {time.time()-t:.2f}s", flush=True)

t = time.time()
male_fraction = [compute_male_fraction(stylegan, male_clf, samples_list[i].to(device)) for i in range(args.n_trials)]
print(f"male_fraction done: {time.time()-t:.2f}s", flush=True)

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

torch.save({
    'samples':            samples_list,
    'w2_values':          w2_values,
    'avg_log_reward':     avg_log_reward,
    'diversity':          diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction':      male_fraction,
}, args.output_path)

print(f'Prior results saved to {args.output_path}')
wandb.finish()
