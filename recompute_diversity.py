#!/usr/bin/env python3
"""
Recompute diversity metrics on already-saved samples and update results files in place.

Usage: python recompute_diversity.py
Run from the celeba_sampling directory on the cluster.
"""

import torch
from utils import compute_diversity, compute_diversity_cov

RESULTS = [
    'experiments/smile/results_stylegan.pt',
    'experiments/eyeglasses/results_stylegan.pt',
    'experiments/bald/results_stylegan.pt',
    'experiments/smile_cold/results_stylegan.pt',
    'experiments/smile_warm/results_stylegan.pt',
]

SAMPLERS = ['Prior', 'RS', 'ULA', 'MALA', 'G_MH']

for path in RESULTS:
    print(f'\n{path}')
    data = torch.load(path, weights_only=False, map_location='cpu')
    r = data['stylegan']
    samples = r['samples']  # dict: sampler -> list of z tensors (one per trial)

    diversity       = {}
    diversity_cov   = {}

    for s in SAMPLERS:
        if s not in samples:
            continue
        trial_list = samples[s]
        div_vals = [compute_diversity(z) for z in trial_list]
        cov_vals = [compute_diversity_cov(z) for z in trial_list]
        diversity[s]     = div_vals
        diversity_cov[s] = cov_vals
        print(f'  {s}: diversity={[f"{v:.2f}" for v in div_vals]}')
        print(f'       div_cov  ={[f"{v:.2f}" for v in cov_vals]}')

    r['diversity']           = diversity
    r['diversity_trace_cov'] = diversity_cov
    torch.save(data, path)
    print(f'  -> saved')

print('\nDone.')
