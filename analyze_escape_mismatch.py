"""Test the accept/reject-mismatch escape mechanism on real coupled-chain data.

Hypothesis: when two same-noise MALA chains have collapsed together, they later
"escape" because the shared accept/reject uniform u_t falls between their two
acceptance probabilities -> one chain accepts its proposal, the other rejects,
so their states separate.

No rerun needed: a MALA reject leaves the state EXACTLY unchanged (verified), so
accept_t = (z[t+1] != z[t]) recovers each chain's per-step accept flag bit-for-bit
from the already-saved init_trace.pt. mismatch_t = accept_i,t != accept_j,t.

For notmale (strongest coupling), same-noise, all 3 init pairs:
  1. detect escape events (d dips < COLLAPSE, then rises > ESCAPE within HORIZON)
  2. check whether a mismatch occurs at/near each escape start
  3. report fraction of escapes preceded/coincident with a mismatch, and the
     reverse (fraction of while-collapsed mismatches that lead to an escape)
  4. plot the distance trace with mismatch + escape-start markers
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = 'notmale'
PAIRS = [('random', 'cold'), ('random', 'warm'), ('cold', 'warm')]
PAIR_COLOR = {('random', 'cold'): '#d1495b', ('random', 'warm'): '#2e86ab',
              ('cold', 'warm'): '#e0a458'}
COLLAPSE = 2.0        # "close" threshold
ESCAPE = 10.0         # "separated" threshold
HORIZON = 20          # steps within which a collapse must rise to ESCAPE
NEAR = (1, 2)         # mismatch counts if within [start-1, start+2] of escape start


def accept_flags(z):
    """z: (T, 512) -> (T-1,) bool, True where the chain accepted (state moved)."""
    return (z[1:] != z[:-1]).any(axis=1)


def find_escapes(d):
    """Escape start = the last step with d<COLLAPSE immediately before d first
    rises above ESCAPE (within HORIZON). Returns sorted unique start indices."""
    collapsed = d < COLLAPSE
    big = d > ESCAPE
    starts = []
    for t in range(1, len(d)):
        if big[t] and not big[t - 1]:                 # first crossing above ESCAPE
            lo = max(0, t - HORIZON)
            idx = np.where(collapsed[lo:t])[0]
            if idx.size:
                starts.append(lo + idx[-1])
    return sorted(set(starts))


def analyze_pair(z_i, z_j):
    d = np.linalg.norm(z_i - z_j, axis=1)             # (T,)
    ai, aj = accept_flags(z_i), accept_flags(z_j)     # (T-1,)
    mismatch = ai != aj                               # (T-1,)  aligned to transition t->t+1
    escapes = find_escapes(d)

    # (2) each escape: any mismatch within [start-NEAR0, start+NEAR1]?
    def mismatch_near(s):
        lo, hi = max(0, s - NEAR[0]), min(len(mismatch), s + NEAR[1] + 1)
        return bool(mismatch[lo:hi].any())
    esc_hit = [mismatch_near(s) for s in escapes]

    # (3 reverse) of while-collapsed mismatches, fraction followed by an escape
    collapsed_at = d[:-1] < COLLAPSE                  # align to transition index
    mm_steps = np.where(mismatch)[0]
    mm_collapsed = [m for m in mm_steps if collapsed_at[m]]

    def leads_to_escape(m):
        hi = min(len(d), m + HORIZON + 1)
        return bool((d[m:hi] > ESCAPE).any())
    mm_coll_escape = [leads_to_escape(m) for m in mm_collapsed]

    return dict(d=d, mismatch=mismatch, escapes=escapes, esc_hit=esc_hit,
                n_mismatch=int(mismatch.sum()), n_mm_collapsed=len(mm_collapsed),
                n_mm_coll_escape=int(sum(mm_coll_escape)))


def main():
    tp = os.path.join('experiments', EXP, 'trajectory', 'same_noise', 'init_trace.pt')
    tr = torch.load(tp, weights_only=False, map_location='cpu')
    Z = {k: tr[k]['z'][:, 0, :].numpy() for k in ['random', 'cold', 'warm']}

    results = {p: analyze_pair(Z[p[0]], Z[p[1]]) for p in PAIRS}

    # ---- summary ----
    lines = [f'# Escape / accept-reject-mismatch test — {EXP}, same-noise',
             f'(COLLAPSE<{COLLAPSE}, ESCAPE>{ESCAPE}, horizon={HORIZON} steps; '
             f'accept flags recovered from saved z, no rerun)\n',
             '| Pair | n_escapes | escapes w/ mismatch near start | '
             'n_mismatch(total) | mismatch-while-collapsed | of those -> escape |',
             '|---|---|---|---|---|---|']
    tot_esc = tot_hit = 0
    for p in PAIRS:
        r = results[p]
        n = len(r['escapes']); hit = int(sum(r['esc_hit']))
        tot_esc += n; tot_hit += hit
        frac_e = f'{hit}/{n} = {hit / n:.0%}' if n else '0/0'
        frac_r = (f"{r['n_mm_coll_escape']}/{r['n_mm_collapsed']} = "
                  f"{r['n_mm_coll_escape'] / r['n_mm_collapsed']:.0%}") if r['n_mm_collapsed'] else '0/0'
        lines.append(f"| {p[0]}–{p[1]} | {n} | {frac_e} | {r['n_mismatch']} | "
                     f"{r['n_mm_collapsed']} | {frac_r} |")
    lines.append(f'\n**Overall: {tot_hit}/{tot_esc} = '
                 f'{(tot_hit / tot_esc if tot_esc else 0):.0%} of escapes have a mismatch at their onset.**')
    summary = '\n'.join(lines) + '\n'
    print(summary)
    open(f'experiments/{EXP}_escape_mismatch_summary.md', 'w').write(summary)

    # ---- plot: distance trace + mismatch ticks + escape starts ----
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(12, 8), sharex=True)
    for ax, p in zip(axes, PAIRS):
        r = results[p]
        T = len(r['d'])
        ax.plot(np.arange(T), r['d'], color=PAIR_COLOR[p], lw=1.0, alpha=0.9)
        ax.axhline(COLLAPSE, color='0.6', ls=':', lw=0.8)
        ax.axhline(ESCAPE, color='0.6', ls='--', lw=0.8)
        # mismatch ticks along the bottom
        mm = np.where(r['mismatch'])[0]
        ax.plot(mm, np.full_like(mm, -1.2, dtype=float), '|', color='0.3',
                ms=4, alpha=0.5, label=f'mismatch (n={len(mm)})')
        # escape starts as vertical guides, colored by whether a mismatch is near
        for s, hit in zip(r['escapes'], r['esc_hit']):
            ax.axvline(s, color=('green' if hit else 'red'), lw=0.9, alpha=0.6)
        ax.set_ylim(-2, 34)
        ax.set_ylabel(r'$\|z_t^{(i)}-z_t^{(j)}\|_2$')
        ax.set_title(f'{p[0]}–{p[1]}   (green escape=mismatch near onset, red=none)',
                     loc='left', fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.margins(x=0)
    axes[-1].set_xlabel('MALA step $t$')
    fig.suptitle(f'{EXP} same-noise: distance, escape events, and accept/reject mismatches',
                 fontsize=12)
    fig.tight_layout()
    out = f'experiments/{EXP}_escape_mismatch.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
