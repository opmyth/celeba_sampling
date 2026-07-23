"""Offline post-processing (no reruns, no cluster): cross-init-type pairwise
distance ||z_t^(i) - z_t^(j)||_2 at every step, under same-noise vs indep-noise.

Reads the existing trajectory init traces only:
  experiments/<exp>/trajectory/{same_noise,indep_noise}/init_trace.pt
each { 'random'|'cold'|'warm': {'z': (T, C, 512), 'log_p': (T, C)} }. Current
saved traces are single-chain (C=1); this pairs chain index 0 across init types
(under same-noise they share the per-step noise, so the distance is the
synchronous-coupling contraction). If C>1 it averages the per-chain-index
distance over the C paired chains.

Outputs:
  - one PNG per experiment: experiments/<exp>/trajectory/pairwise_distance.png
    3 pairs x 2 conditions = 6 curves, shared (linear, from 0) y-axis so
    same vs indep are directly comparable.
  - a summary table (stdout + experiments/pairwise_distance_summary.md):
    mean pairwise distance over the last 500 steps per experiment/condition.
"""
import os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAIRS = [('random', 'cold'), ('random', 'warm'), ('cold', 'warm')]
PAIR_COLOR = {('random', 'cold'): '#d1495b',
              ('random', 'warm'): '#2e86ab',
              ('cold', 'warm'): '#e0a458'}
LAST_N = 500
ROLL_W = 50
INDEP_REF = (2 * 512) ** 0.5             # ~32.0: dist between two independent N(0,I) 512-vecs


def _dist_curve(z_i, z_j):
    """z_*: (T, C, 512) -> (T,) per-step distance, averaged over the C paired
    chain indices."""
    d = torch.norm(z_i - z_j, dim=2)      # (T, C)
    return d.mean(dim=1).numpy()          # (T,)


def _rolling(y, w=ROLL_W):
    """Centered moving average, edge-normalized (window shrinks at the ends
    rather than ramping toward 0), same length as y."""
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    c = np.ones(w)
    return np.convolve(y, c, mode='same') / np.convolve(np.ones_like(y), c, mode='same')


def _label_from_path(same_path):
    # experiments/<...>/trajectory/same_noise/init_trace.pt -> "<...>"
    return same_path.split('experiments/')[-1].split('/trajectory/')[0]


def _plot_experiment(label, curves, C, T, out):
    """Two panels: same-noise (top) and indep-noise (bottom), rolling-mean (w)
    lines only, shared y so the same-noise collapses read against the flat
    indep baseline."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    x = np.arange(T)
    for ax, cond in zip(axes, ['same', 'indep']):
        for pair in PAIRS:
            ax.plot(x, _rolling(curves[(cond, pair)]), color=PAIR_COLOR[pair],
                    lw=1.7, label=f'{pair[0]}–{pair[1]}')
        ax.axhline(INDEP_REF, color='0.35', ls=':', lw=1.0)
        ax.axvspan(T - LAST_N, T, color='0.92', zorder=0)           # last-500 window
        ax.set_ylabel(r'$\|z_t^{(i)}-z_t^{(j)}\|_2$')
        ax.set_title(f'{cond}-noise', loc='left', fontweight='bold', fontsize=10)
        ax.margins(x=0)
    axes[0].set_ylim(0, INDEP_REF * 1.12)
    # ref-line label at the LEFT so it never sits under the upper-right legend
    axes[0].text(0.012, INDEP_REF - 0.6, r'independent baseline  $\sqrt{2\cdot512}\approx32$',
                 transform=axes[0].get_yaxis_transform(), va='top', ha='left',
                 fontsize=8, color='0.35')
    axes[0].legend(ncol=3, fontsize=8, loc='upper right', framealpha=0.9,
                   title=f'rolling mean (w={ROLL_W})')
    axes[-1].set_xlabel('MALA step $t$')
    fig.suptitle(f'{label} — cross-init pairwise distance')
    # chain count kept out of the title, as a small corner caption
    fig.text(0.995, 0.005, f'C={C} chain{"s" if C > 1 else ""}',
             ha='right', va='bottom', fontsize=7, color='0.5')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  saved {out}', flush=True)


def make_grid(grid_data):
    """Small-multiples: same-noise rolling-mean panel for all posteriors on a
    shared 0-35 y-axis, so coupling strength is comparable at a glance."""
    ncols, nrows = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 12), sharey=True)
    axes = axes.ravel()
    for ax, (label, roll, T) in zip(axes, grid_data):
        for pair in PAIRS:
            ax.plot(np.arange(T), roll[pair], color=PAIR_COLOR[pair], lw=1.3)
        ax.axhline(INDEP_REF, color='0.4', ls=':', lw=0.9)
        ax.set_ylim(0, 35)
        ax.set_xlim(0, T - 1)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(grid_data):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color=PAIR_COLOR[p], lw=2, label=f'{p[0]}–{p[1]}')
               for p in PAIRS]
    fig.legend(handles=handles, ncol=3, loc='lower center', fontsize=10,
               bbox_to_anchor=(0.5, 0.01),
               title=r'same-noise rolling mean (w=%d); dotted = independent baseline $\approx32$' % ROLL_W)
    fig.suptitle('Same-noise cross-init pairwise distance — all posteriors (shared y 0–35)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.98])
    out = 'experiments/pairwise_distance_grid.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  saved {out}', flush=True)


def analyze():
    same_paths = sorted(glob.glob('experiments/**/trajectory/same_noise/init_trace.pt',
                                  recursive=True))
    rows = []          # (label, condition, pair, last500_mean)
    agg = []           # (label, same_mean_over_pairs, indep_mean_over_pairs)
    grid_data = []     # (label, {pair: same-noise rolling mean}, T) for the grid figure

    for sp in same_paths:
        label = _label_from_path(sp)
        traces = {}
        for cond, path in [('same', sp), ('indep', sp.replace('same_noise', 'indep_noise'))]:
            if not os.path.exists(path):
                print(f'  [skip {label} / {cond}] missing {path}')
                continue
            traces[cond] = torch.load(path, weights_only=False, map_location='cpu')

        if 'same' not in traces or 'indep' not in traces:
            continue

        C = traces['same']['random']['z'].shape[1]

        # ---- compute all curves ----
        curves = {}  # (cond, pair) -> (T,) array
        for cond, d in traces.items():
            for pair in PAIRS:
                i, j = pair
                curves[(cond, pair)] = _dist_curve(d[i]['z'], d[j]['z'])

        # ---- record last-500 means (numbers unchanged) ----
        for (cond, pair), y in curves.items():
            rows.append((label, cond, f'{pair[0]}-{pair[1]}', float(np.mean(y[-LAST_N:]))))

        # ---- per-experiment figure + collect grid data ----
        T = len(curves[('same', PAIRS[0])])
        out = os.path.join(os.path.dirname(os.path.dirname(sp)), 'pairwise_distance.png')
        _plot_experiment(label, curves, C, T, out)
        grid_data.append((label, {p: _rolling(curves[('same', p)]) for p in PAIRS}, T))

        # aggregate: mean over the 3 pairs of the last-500 mean, per condition
        def _cond_agg(cond):
            return float(np.mean([np.mean(curves[(cond, p)][-LAST_N:]) for p in PAIRS]))
        agg.append((label, _cond_agg('same'), _cond_agg('indep')))

    return rows, agg, grid_data


def write_summary(rows, agg):
    lines = []
    lines.append(f'# Cross-init pairwise distance — mean over last {LAST_N} steps\n')
    lines.append('Synchronous-coupling contraction: under **same** noise, chains from '
                 'different init types should collapse toward each other (small distance); '
                 'under **indep** noise they should not. `ratio = same / indep` near 0 = '
                 'strong contraction, near 1 = no coupling effect.\n')

    # headline: one number per experiment/condition (mean over the 3 pairs)
    lines.append('## Headline (mean over the 3 init-pairs)\n')
    lines.append('| Experiment | same | indep | ratio (same/indep) |')
    lines.append('|---|---|---|---|')
    for label, s, i in agg:
        ratio = s / i if i else float('nan')
        lines.append(f'| {label} | {s:.2f} | {i:.2f} | {ratio:.3f} |')

    # detail: per pair
    lines.append('\n## Per-pair detail\n')
    lines.append('| Experiment | Pair | same | indep | ratio |')
    lines.append('|---|---|---|---|---|')
    by = {}
    for label, cond, pair, m in rows:
        by.setdefault((label, pair), {})[cond] = m
    for (label, pair), d in by.items():
        s, i = d.get('same', float('nan')), d.get('indep', float('nan'))
        ratio = s / i if i else float('nan')
        lines.append(f'| {label} | {pair} | {s:.2f} | {i:.2f} | {ratio:.3f} |')

    text = '\n'.join(lines) + '\n'
    out = 'experiments/pairwise_distance_summary.md'
    open(out, 'w').write(text)
    print('\n' + text)
    print(f'summary written to {out}')


if __name__ == '__main__':
    rows, agg, grid_data = analyze()
    make_grid(grid_data)
    write_summary(rows, agg)
