"""Composite 2x2 writeup figures for the same-noise convergence phenomenon.

Per posterior, one image:
    row 0 = same-noise    | row 1 = indep-noise
    col 0 = decoded init-trajectory face grid (random/cold/warm rows)
    col 1 = the pairwise-distance panel for that condition

Left cells reuse the existing saved face grids
(experiments/<exp>/trajectory/<cond>_noise/init_grid_0.png); right cells are
re-rendered fresh from init_trace.pt (single condition each, not the 2-panel
figure) via the helpers in analyze_pairwise_distance.py. Pure offline
post-processing - reads only saved .pt/.png files.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from analyze_pairwise_distance import (PAIRS, PAIR_COLOR, INDEP_REF, LAST_N,
                                       ROLL_W, _rolling, _dist_curve)

EXPERIMENTS = ['male', 'eyeglasses']


def _curves(trace):
    return {pair: _dist_curve(trace[pair[0]]['z'], trace[pair[1]]['z']) for pair in PAIRS}


def _distance_panel(ax, curves, T, title, legend=False):
    for pair in PAIRS:
        ax.plot(np.arange(T), _rolling(curves[pair]), color=PAIR_COLOR[pair],
                lw=1.7, label=f'{pair[0]}–{pair[1]}')
    ax.axhline(INDEP_REF, color='0.35', ls=':', lw=1.0)
    ax.axvspan(T - LAST_N, T, color='0.92', zorder=0)         # last-500 window
    ax.set_ylim(0, INDEP_REF * 1.12)
    ax.set_xlim(0, T - 1)
    ax.set_ylabel(r'$\|z_t^{(i)}-z_t^{(j)}\|_2$')
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    ax.margins(x=0)
    if legend:
        ax.text(0.012, INDEP_REF - 0.6, r'indep. baseline  $\sqrt{2\cdot512}\approx32$',
                transform=ax.get_yaxis_transform(), va='top', ha='left',
                fontsize=8, color='0.35')
        ax.legend(ncol=3, fontsize=8, loc='upper right', framealpha=0.9,
                  title=f'rolling mean (w={ROLL_W})')


def _face_cell(ax, png_path, title):
    ax.imshow(mpimg.imread(png_path))          # native aspect, undistorted
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    ax.axis('off')


def build(exp):
    tdir = os.path.join('experiments', exp, 'trajectory')
    traces = {c: torch.load(os.path.join(tdir, f'{c}_noise', 'init_trace.pt'),
                            weights_only=False, map_location='cpu')
              for c in ['same', 'indep']}
    curves = {c: _curves(traces[c]) for c in traces}
    T = len(curves['same'][PAIRS[0]])

    fig, ax = plt.subplots(2, 2, figsize=(20, 8),
                           gridspec_kw={'width_ratios': [1.7, 1]})

    _face_cell(ax[0, 0], os.path.join(tdir, 'same_noise', 'init_grid_0.png'),
               'same-noise — init trajectories (random / cold / warm)')
    _distance_panel(ax[0, 1], curves['same'], T,
                    'same-noise — cross-init pairwise distance', legend=True)

    _face_cell(ax[1, 0], os.path.join(tdir, 'indep_noise', 'init_grid_0.png'),
               'indep-noise — init trajectories (random / cold / warm)')
    _distance_panel(ax[1, 1], curves['indep'], T,
                    'indep-noise — cross-init pairwise distance')
    ax[1, 1].set_xlabel('MALA step $t$')

    fig.suptitle(f'{exp} — same-noise vs independent-noise coupling', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f'{exp}_comparison.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  saved {out}', flush=True)


if __name__ == '__main__':
    for exp in EXPERIMENTS:
        build(exp)
