#!/usr/bin/env python3
"""
Generate LaTeX tables from StyleGAN2 MCMC posterior-sampling results.

Requirements (on cluster):
    pip install scipy  (already present in the venv)

Usage:
    python generate_tables.py

LaTeX preamble requirements:
    \\usepackage{booktabs}
    \\usepackage{bm}        % bold math for significant cells
    \\usepackage{makecell}  % stacked cell content
"""

import sys, os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from scipy import stats

# ─── loaders & stats helpers ─────────────────────────────────────────────────

def load_exp(path):
    return torch.load(path, weights_only=False, map_location='cpu')['stylegan']

def ms(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=1)

def ttest_p(a, b):
    _, p = stats.ttest_ind(np.array(a, dtype=float), np.array(b, dtype=float))
    return float(p)

def fmt_ms(m, s):
    return f'${m:.2f} \\pm {s:.2f}$'

ALPHA = 0.05 / 3   # Bonferroni threshold: 3 comparisons per metric

def fmt_cell(m, s, p_raw):
    """mean±std stacked over raw p-value; bold p-value if significant."""
    top = f'${m:.2f} \\pm {s:.2f}$'
    p_str = f'{p_raw:.3f}'
    if p_raw < ALPHA:
        bot = f'{{\\footnotesize$\\mathbf{{({p_str})}}$}}'
    else:
        bot = f'{{\\footnotesize$({p_str})$}}'
    return f'\\makecell{{{top} \\\\ {bot}}}'

# ─── load all experiments ────────────────────────────────────────────────────

smile       = load_exp('experiments/smile/results_stylegan.pt')
eyeglasses  = load_exp('experiments/eyeglasses/results_stylegan.pt')
bald        = load_exp('experiments/bald/results_stylegan.pt')
smile_cold  = load_exp('experiments/smile_cold/results_stylegan.pt')
smile_warm  = load_exp('experiments/smile_warm/results_stylegan.pt')

SAMPLERS    = ['Prior', 'ULA', 'MALA', 'G_MH']
SAM_LABELS  = ['Prior', 'ULA', 'MALA', 'Gaussian MH']

# ─── metric accessors ────────────────────────────────────────────────────────

def rs_vals(exp, metric):
    if metric == 'w2':
        return np.array(exp['w2_baseline'], dtype=float)
    return np.array(exp[metric]['RS'], dtype=float)

def sam_vals(exp, metric, sampler):
    if metric == 'w2':
        return np.array(exp['w2_values'][sampler], dtype=float)
    return np.array(exp[metric][sampler], dtype=float)

# ═════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Main results  (Smile / Eyeglasses / Bald)
# ═════════════════════════════════════════════════════════════════════════════

EXP_T1 = [
    ('Smile',       smile),
    ('Eyeglasses',  eyeglasses),
    ('Bald',        bald),
]

METRICS_T1 = [
    ('$W_2$',         'w2'),
    ('AvgLogR',       'avg_log_reward'),
    ('Diversity',     'diversity_trace_cov'),
]

# collect raw p-values: 3 exps × 3 metrics × 3 samplers = 27 tests
raw_p1 = {}
for ei, (_, exp) in enumerate(EXP_T1):
    for mi, (_, mkey) in enumerate(METRICS_T1):
        rv = rs_vals(exp, mkey)
        for si, s in enumerate(SAMPLERS):
            raw_p1[(ei, mi, si)] = ttest_p(sam_vals(exp, mkey, s), rv)

adj_p1 = raw_p1   # display raw p; significance tested against ALPHA = 0.05/3

def t1_rs_row():
    cells = ['RS']
    for _, exp in EXP_T1:
        for _, mkey in METRICS_T1:
            m, s = ms(rs_vals(exp, mkey))
            cells.append(fmt_ms(m, s))
    return ' & '.join(cells) + r' \\'

def t1_sam_row(si, label):
    cells = [label]
    is_prior = SAMPLERS[si] == 'Prior'
    for ei, (_, exp) in enumerate(EXP_T1):
        for mi, (_, mkey) in enumerate(METRICS_T1):
            m, s = ms(sam_vals(exp, mkey, SAMPLERS[si]))
            cells.append(fmt_ms(m, s) if is_prior else fmt_cell(m, s, adj_p1[(ei, mi, si)]))
    return ' & '.join(cells) + r' \\'

n_exp1 = len(EXP_T1)
n_met1 = len(METRICS_T1)

OUT = 'tables.tex'
_stdout = sys.stdout
sys.stdout = open(OUT, 'w')

print('%' + '=' * 72)
print('% TABLE 1 — Main Results')
print('%' + '=' * 72)
print()
print(r'\begin{table*}[t]')
print(r'\centering')
print(r'\footnotesize')
print(r'\setlength{\tabcolsep}{4pt}')
print(r'\caption{%')
print(r'    MCMC samplers vs.\ rejection sampling (RS) across three posterior')
print(r'    distributions of increasing difficulty.')
print(r'    Metrics: Wasserstein-2 distance ($W_2$), average log-reward (AvgLogR),')
print(r'    and all-pairs sample diversity.')
print(r'    Raw $p$-values shown in parentheses (two-sided $t$-test vs.\ RS).')
print(r'    \textbf{Bold} indicates $p < 0.05/3 \approx 0.017$ (Bonferroni over')
print(r'    3 comparisons per metric).')
print(r'}')
print(r'\label{tab:main_results}')
print(r'\resizebox{\linewidth}{!}{%')
print(r'\begin{tabular}{l' + ' ccc' * n_exp1 + '}')
print(r'\toprule')

# experiment-level column headers
top = ['']
for exp_name, _ in EXP_T1:
    top.append(r'\multicolumn{' + str(n_met1) + r'}{c}{\textbf{' + exp_name + r'}}')
print(' & '.join(top) + r' \\')

# cmidrules under experiment headers
rules = []
for i in range(n_exp1):
    c1 = 2 + i * n_met1
    c2 = c1 + n_met1 - 1
    rules.append(f'\\cmidrule(lr){{{c1}-{c2}}}')
print(' '.join(rules))

# metric sub-headers
sub = ['Sampler']
for _ in EXP_T1:
    for met_name, _ in METRICS_T1:
        sub.append(met_name)
print(' & '.join(sub) + r' \\')

print(r'\midrule')
print(t1_rs_row())
print(r'\midrule')
for si, label in enumerate(SAM_LABELS):
    print(t1_sam_row(si, label))
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'}')
print(r'\end{table*}')


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Initialization robustness  (Smile: random / cold / warm)
# ═════════════════════════════════════════════════════════════════════════════

INITS_T2 = [
    ('Random', smile),
    ('Cold',   smile_cold),
    ('Warm',   smile_warm),
]

METRICS_T2 = [
    ('$W_2$',    'w2'),
    ('AvgLogR',  'avg_log_reward'),
]

# RS reference is always the smile (random-init) RS — init-agnostic
rs_ref_t2 = {mkey: rs_vals(smile, mkey) for _, mkey in METRICS_T2}

# collect raw p-values: 3 inits × 2 metrics × 3 samplers = 18 tests
raw_p2 = {}
for ii, (_, exp) in enumerate(INITS_T2):
    for mi, (_, mkey) in enumerate(METRICS_T2):
        for si, s in enumerate(SAMPLERS):
            raw_p2[(ii, mi, si)] = ttest_p(sam_vals(exp, mkey, s), rs_ref_t2[mkey])

adj_p2 = raw_p2   # display raw p; significance tested against ALPHA = 0.05/3

def t2_rs_row():
    cells = ['RS']
    for ii, (_, exp) in enumerate(INITS_T2):
        for _, mkey in METRICS_T2:
            if ii == 0:
                m, s = ms(rs_vals(exp, mkey))
                cells.append(fmt_ms(m, s))
            else:
                cells.append(r'\multicolumn{1}{c}{---}')
    return ' & '.join(cells) + r' \\'

def t2_sam_row(si, label):
    cells = [label]
    is_prior = SAMPLERS[si] == 'Prior'
    for ii, (_, exp) in enumerate(INITS_T2):
        for mi, (_, mkey) in enumerate(METRICS_T2):
            m, s = ms(sam_vals(exp, mkey, SAMPLERS[si]))
            cells.append(fmt_ms(m, s) if is_prior else fmt_cell(m, s, adj_p2[(ii, mi, si)]))
    return ' & '.join(cells) + r' \\'

n_init2 = len(INITS_T2)
n_met2  = len(METRICS_T2)

print()
print()
print('%' + '=' * 72)
print('% TABLE 2 — Initialization Robustness')
print('%' + '=' * 72)
print()
print(r'\begin{table}[t]')
print(r'\centering')
print(r'\footnotesize')
print(r'\caption{%')
print(r'    Effect of chain initialisation on sampling quality (Smile attribute).')
print(r'    \emph{Cold}: chains start from the lowest-scoring 100 of 10\,000 candidates;')
print(r'    \emph{Warm}: from the highest-scoring 100.')
print(r'    RS is initialisation-agnostic and appears only in the Random column.')
print(r'    Raw $p$-values shown in parentheses (two-sided $t$-test vs.\ RS).')
print(r'    \textbf{Bold} indicates $p < 0.05/3 \approx 0.017$ (Bonferroni over')
print(r'    3 comparisons per metric).')
print(r'}')
print(r'\label{tab:init_robustness}')
print(r'\begin{tabular}{l' + ' cc' * n_init2 + '}')
print(r'\toprule')

# init-level column headers
top2 = ['']
for init_name, _ in INITS_T2:
    top2.append(r'\multicolumn{' + str(n_met2) + r'}{c}{\textbf{' + init_name + r'}}')
print(' & '.join(top2) + r' \\')

# cmidrules
rules2 = []
for i in range(n_init2):
    c1 = 2 + i * n_met2
    c2 = c1 + n_met2 - 1
    rules2.append(f'\\cmidrule(lr){{{c1}-{c2}}}')
print(' '.join(rules2))

# metric sub-headers
sub2 = ['Sampler']
for _ in INITS_T2:
    for met_name, _ in METRICS_T2:
        sub2.append(met_name)
print(' & '.join(sub2) + r' \\')

print(r'\midrule')
print(t2_rs_row())
print(r'\midrule')
for si, label in enumerate(SAM_LABELS):
    print(t2_sam_row(si, label))
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')

sys.stdout.close()
sys.stdout = _stdout
print(f'Written to {OUT}')
