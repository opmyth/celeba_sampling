#!/usr/bin/env python3
"""
Patch Prior (z ~ N(0,I)) results into existing results_stylegan.pt files.

Reads RS samples already stored in each merged file for W2 comparison,
so no separate results_rs.pt needed.

Usage: python patch_prior.py
Run from the celeba_sampling directory on the cluster.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import torch.nn.functional as F

from model_loader import load_models
from utils import compute_w2, compute_diversity, compute_diversity_cov, compute_male_fraction

EXPS = [
    ('smile',      'experiments/smile/results_stylegan.pt'),
    ('eyeglasses', 'experiments/eyeglasses/results_stylegan.pt'),
    ('bald',       'experiments/bald/results_stylegan.pt'),
    ('smile',      'experiments/smile_cold/results_stylegan.pt'),
    ('smile',      'experiments/smile_warm/results_stylegan.pt'),
]

N_CHAINS = 100
N_TRIALS = 5
SEED     = 321

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

loaded_models = {}

for clf_name, path in EXPS:
    print(f'\n=== {path} ===')
    data = torch.load(path, weights_only=False, map_location='cpu')
    r = data['stylegan']

    if clf_name not in loaded_models:
        print(f'Loading models for {clf_name}...')
        stylegan, clf, male_clf = load_models(clf_name, device)
        loaded_models[clf_name] = (stylegan, clf, male_clf)
    stylegan, clf, male_clf = loaded_models[clf_name]

    rs_samples_list = r['samples']['RS']   # list of tensors, one per trial
    n_trials = len(rs_samples_list)
    latent_dim = rs_samples_list[0].shape[1]

    torch.manual_seed(SEED)
    prior_samples = [torch.randn(N_CHAINS, latent_dim) for _ in range(n_trials)]

    w2_values = [compute_w2(prior_samples[i], rs_samples_list[i]) for i in range(n_trials)]
    print(f'  W2: {[round(v,3) for v in w2_values]}')

    with torch.no_grad():
        avg_log_reward = [
            F.logsigmoid(clf(stylegan(prior_samples[i].to(device)))).mean().item()
            for i in range(n_trials)
        ]
    print(f'  AvgLogR: {[round(v,3) for v in avg_log_reward]}')

    diversity = [compute_diversity(prior_samples[i]) for i in range(n_trials)]
    diversity_trace_cov = [compute_diversity_cov(prior_samples[i]) for i in range(n_trials)]
    print(f'  Diversity: {[round(v,2) for v in diversity]}')

    male_fraction = [compute_male_fraction(stylegan, male_clf, prior_samples[i].to(device)) for i in range(n_trials)]

    # patch into existing dicts
    r['samples']['Prior']            = prior_samples
    r['w2_values']['Prior']          = w2_values
    r['avg_log_reward']['Prior']     = avg_log_reward
    r['diversity']['Prior']          = diversity
    r['diversity_trace_cov']['Prior'] = diversity_trace_cov
    r['male_fraction']['Prior']      = male_fraction

    torch.save(data, path)
    print(f'  -> patched and saved')

print('\nDone.')
