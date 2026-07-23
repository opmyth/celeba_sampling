"""Deterministic-G test: does making StyleGAN2 deterministic (noise_mode=
'const') stop the repeated collapse/escape cycling seen with the default
stochastic G (noise_mode='random')?

Prediction: with a deterministic reward landscape, once two same-noise MALA
chains collapse together their acceptance probabilities become identical
(the generator no longer re-randomises the reward each evaluation), so the
shared accept/reject uniform can never split them - escapes should vanish and
the chains lock permanently. This mirrors the smooth-GMM / rippled-Gaussian
toy result (ripple_coupling.py): on ANY deterministic target, exact
coalescence is absorbing.

Reads two same-noise init traces for `notmale` and compares them:
  random: experiments/notmale/trajectory/same_noise/init_trace.pt        (existing)
  const:  experiments/notmale/trajectory_const/same_noise/init_trace.pt  (new run)

Escape/mismatch detection is imported unchanged from analyze_escape_mismatch
(same COLLAPSE/ESCAPE/HORIZON thresholds and the same accept-flag recovery),
so the two conditions are scored identically and comparably to the earlier
real-data diagnostic. Pure offline post-processing - no cluster, no rerun.
"""
import os

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from analyze_escape_mismatch import (PAIRS, PAIR_COLOR, COLLAPSE, ESCAPE,
                                     HORIZON, analyze_pair)

EXP = 'notmale'

CONDITIONS = [
    ('random', os.path.join('experiments', EXP, 'trajectory',
                            'same_noise', 'init_trace.pt')),
    ('const', os.path.join('experiments', EXP, 'trajectory_const',
                           'same_noise', 'init_trace.pt')),
]


def _load_pairs(path):
    """path -> {pair: analyze_pair(...)} for the 3 cross-init pairs, chain 0."""
    trace = torch.load(path, weights_only=False, map_location='cpu')
    z = {k: trace[k]['z'][:, 0, :].numpy() for k in ['random', 'cold', 'warm']}
    return {p: analyze_pair(z[p[0]], z[p[1]]) for p in PAIRS}


def _panel(ax, result, title):
    d = result['d']
    T = len(d)

    ax.plot(np.arange(T), d, color='#2e86ab', lw=0.9, alpha=0.9)
    ax.axhline(COLLAPSE, color='0.6', ls=':', lw=0.8)
    ax.axhline(ESCAPE, color='0.6', ls='--', lw=0.8)

    mismatch_steps = np.flatnonzero(result['mismatch'])
    ax.plot(mismatch_steps, np.full(mismatch_steps.size, -1.2), '|',
            color='0.3', ms=4, alpha=0.5,
            label=f'mismatch (n={mismatch_steps.size})')

    for start, hit in zip(result['escapes'], result['esc_hit']):
        ax.axvline(start, color=('green' if hit else 'red'), lw=0.9, alpha=0.6)

    ax.set_ylim(-2, 34)
    ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.margins(x=0)


def main():
    loaded = {}
    for name, path in CONDITIONS:
        if not os.path.exists(path):
            print(f'[missing] {name}: {path}')
            if name == 'const':
                print('\nRun the const trajectory first (on the cluster):\n'
                      '  sbatch scripts/submit_trajectory.sh notmale init same '
                      '--noise_mode const\n'
                      'or, interactively / whatever your srun wrapper is:\n'
                      '  python run_trajectory.py --experiment notmale --mode init '
                      '--noise same --noise_mode const\n')
            return
        loaded[name] = _load_pairs(path)

    # ---- summary table ----
    lines = [
        f'# Deterministic-G test - {EXP}, same-noise (random vs const)',
        f'(COLLAPSE<{COLLAPSE}, ESCAPE>{ESCAPE}, horizon={HORIZON}; identical '
        f'detection for both conditions)\n',
        '| Pair | condition | n_escapes | escapes w/ mismatch at onset | '
        'mismatch-while-collapsed | min dist | last-500 mean |',
        '|---|---|---|---|---|---|---|',
    ]
    totals = {name: 0 for name, _ in CONDITIONS}
    for p in PAIRS:
        for name, _ in CONDITIONS:
            r = loaded[name][p]
            n = len(r['escapes'])
            totals[name] += n
            hit = int(sum(r['esc_hit']))
            frac = f'{hit}/{n} = {hit / n:.0%}' if n else '0/0'
            lines.append(
                f"| {p[0]}-{p[1]} | {name} | {n} | {frac} | "
                f"{r['n_mm_collapsed']} | {r['d'].min():.2f} | "
                f"{r['d'][-500:].mean():.2f} |")

    lines.append(
        f'\n**Total escapes - random: {totals["random"]}, '
        f'const: {totals["const"]}.**')

    predicted = totals['const'] < totals['random']
    if totals['const'] == 0:
        verdict = ('CONFIRMED as predicted: escapes vanish entirely under a '
                   'deterministic G - the chains lock permanently once collapsed.')
    elif predicted:
        verdict = (f'PARTIALLY as predicted: escapes drop sharply under const '
                   f'({totals["random"]} -> {totals["const"]}) but do not fully '
                   f'vanish - some residual non-determinism or a genuinely rough '
                   f'landscape remains.')
    else:
        verdict = ('NOT as predicted: const does not reduce escapes - the '
                   'cycling is not driven by the generator\'s evaluation noise.')
    lines.append(f'\n**Verdict: {verdict}**')

    summary = '\n'.join(lines) + '\n'
    print(summary)
    out_md = f'experiments/{EXP}_const_vs_random_summary.md'
    open(out_md, 'w').write(summary)

    # ---- figure: 3 pairs x 2 conditions ----
    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(15, 9),
                             sharex=True, sharey=True)
    for row, p in enumerate(PAIRS):
        for col, (name, _) in enumerate(CONDITIONS):
            r = loaded[name][p]
            _panel(axes[row, col],
                   r, f'{p[0]}-{p[1]}   noise_mode={name}   '
                      f'({len(r["escapes"])} escapes)')
        axes[row, 0].set_ylabel(r'$\|z_t^{(i)}-z_t^{(j)}\|_2$')
    axes[-1, 0].set_xlabel('MALA step $t$')
    axes[-1, 1].set_xlabel('MALA step $t$')
    fig.suptitle(
        f'{EXP} same-noise: stochastic G (random) vs deterministic G (const) - '
        'distance, escapes (green=mismatch at onset, red=none), mismatch ticks',
        fontsize=12)
    fig.tight_layout()
    out_png = f'experiments/{EXP}_const_vs_random.png'
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'saved {out_png}')
    print(f'saved {out_md}')


if __name__ == '__main__':
    main()
