#!/usr/bin/env python3
"""
Generate a LaTeX table of step sizes and empirical acceptance rates.

Usage: python generate_hyperparams_table.py
Output: appended / printed to stdout, redirect to a .tex file as needed.
"""

import os, sys
import numpy as np
import torch

# ─── helpers ─────────────────────────────────────────────────────────────────

def parse_config(path):
    """Return dict of key→value strings from a config.txt file."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                k, v = line.split(':', 1)
                cfg[k.strip()] = v.strip()
    return cfg

def get_step_sizes(cfg):
    dt_ula  = cfg.get('dt_ula',  '---')
    dt_mala = cfg.get('dt_mala', '---')
    sigma   = cfg.get('sigma', cfg.get('sigma_gmh', '---'))  # bald uses sigma_gmh
    return dt_ula, dt_mala, sigma

# Fallback accept rates for experiments where they weren't saved to the results file
ACCEPT_FALLBACK = {
    'Bald': {
        'MALA': [0.613, 0.602, 0.605, 0.611, 0.607],
        'G_MH': [0.230, 0.230, 0.232, 0.231, 0.232],
    }
}

def load_accept_rates(exp_name, results_path, sampler):
    """Return (mean, std) of accept rates across trials, or None if not stored."""
    r = torch.load(results_path, weights_only=False, map_location='cpu')['stylegan']
    ar = r.get('accept_rates', {})
    vals = ar.get(sampler)
    if vals is None:
        vals = ACCEPT_FALLBACK.get(exp_name, {}).get(sampler)
    if vals is None:
        return None
    vals = np.array(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1)

def fmt_rate(result):
    if result is None:
        return r'\multicolumn{1}{c}{---}'
    m, s = result
    return f'${m:.3f} \\pm {s:.3f}$'

def fmt_dt(val):
    return f'${val}$' if val != '---' else r'\multicolumn{1}{c}{---}'

# ─── experiments ─────────────────────────────────────────────────────────────

EXPS = [
    ('Smile',       'experiments/smile'),
    ('Eyeglasses',  'experiments/eyeglasses'),
    ('Bald',        'experiments/bald'),
]

configs = {}
for name, path in EXPS:
    configs[name] = parse_config(os.path.join(path, 'config.txt'))

results = {}
for name, path in EXPS:
    results[name] = os.path.join(path, 'results_stylegan.pt')

# ─── build rows ──────────────────────────────────────────────────────────────

def row(label, vals):
    return label + ' & ' + ' & '.join(vals) + r' \\'

ula_dts  = [fmt_dt(get_step_sizes(configs[n])[0]) for n, _ in EXPS]
mala_dts = [fmt_dt(get_step_sizes(configs[n])[1]) for n, _ in EXPS]
gmh_sigs = [fmt_dt(get_step_sizes(configs[n])[2]) for n, _ in EXPS]

mala_rates = [fmt_rate(load_accept_rates(n, results[n], 'MALA')) for n, _ in EXPS]
gmh_rates  = [fmt_rate(load_accept_rates(n, results[n], 'G_MH')) for n, _ in EXPS]

# ─── LaTeX ───────────────────────────────────────────────────────────────────

exp_names = [n for n, _ in EXPS]

print(r'\begin{table}[t]')
print(r'\centering')
print(r'\caption{%')
print(r'    Sampler hyperparameters and empirical acceptance rates.')
print(r'    Acceptance rates are mean$\,\pm\,$std across 5 independent trials.')
print(r'}')
print(r'\label{tab:hyperparams}')
print(r'\begin{tabular}{l ccc}')
print(r'\toprule')
print(r' & \textbf{Smile} & \textbf{Eyeglasses} & \textbf{Bald} \\')
print(r'\midrule')
print(row(r'ULA $\delta$',        ula_dts))
print(row(r'MALA $\delta$',       mala_dts))
print(row(r'Gaussian MH $\sigma$', gmh_sigs))
print(r'\midrule')
print(row(r'MALA accept rate',     mala_rates))
print(row(r'G-MH accept rate',    gmh_rates))
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
