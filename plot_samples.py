"""Save the qualitative sample grids that celeba_sampling.ipynb only draws
inline (plt.show()), so EXPERIMENTS.md can link them - the same role
plot_trajectory.py plays for the trajectory diagnostics.

One combined PNG per experiment (per prompt for bald_ir): a stacked block per
sampler present (RS/ULA/MALA/ANNEALED_MALA/G_MH), each showing the SAME
fixed-seed representative n_show samples the notebook displays - a random draw
over the sampler's full pooled output, NOT the first rows (those are all the
earliest, least-mixed kept step). Each tile is titled with its own score
(classifier P(attr), both scores for joint posteriors, or the clipped
ImageReward r_tilde); each block header is "<Sampler> | Accept=<rate>".

Usage:
  python plot_samples.py                        # every experiment
  python plot_samples.py --experiment bald      # one
  python plot_samples.py --n_show 20 --seed 0
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import EXPERIMENTS
from model_loader import load_stylegan, load_classifier
from utils import compute_diversity_cov

ORDER = ['RS', 'ULA', 'MALA', 'ANNEALED_MALA', 'G_MH']
TITLES = {'ANNEALED_MALA': 'Annealed MALA', 'G_MH': 'Gaussian MH'}
LABELS = {'male': 'M', 'not_male': 'NotM', 'eyeglasses': 'E',
          'WearingHat': 'Hat', 'smile': 'Smile', 'bald': 'Bald'}

# bald's merged file predates the accept_rates field (per-stage files are gone);
# these come from its job logs, same values the notebook/EXPERIMENTS.md use.
ACCEPT_FALLBACK = {'bald': {'MALA': 0.611, 'ANNEALED_MALA': 0.611, 'G_MH': 0.231}}


def results_path(name, prompt_slug=None):
    base = os.path.join('experiments', name)
    if prompt_slug:
        base = os.path.join(base, f'prompt_{prompt_slug}')
    for fn in ('results_stylegan.pt', 'results_merged.pt'):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no results file under {base}')


def pick(viz_full, n_show, seed):
    """Same representative draw as the notebook: fixed-seed random n_show of the
    sampler's full pooled output (all kept steps x chains)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(viz_full.shape[0], generator=g)[:n_show]
    return viz_full[idx]


def accept_str(d, sampler, trial, exp_name):
    if sampler == 'ULA':
        return None                      # no accept/reject step
    if sampler == 'RS':
        r = d.get('rs_accept_rate')
        return f'{r:.1%}' if r is not None else None
    ar = (d.get('accept_rates') or {}).get(sampler)
    if ar is not None:
        return f'{np.mean(ar[trial]) if np.ndim(ar[trial]) else ar[trial]:.1%}'
    fb = ACCEPT_FALLBACK.get(exp_name, {}).get(sampler)
    return f'{fb:.1%}' if fb is not None else None


def score_tiles(gen, clfs, ir=None, prompt=None, r_max=None, device='cpu'):
    """Returns list of (text, value) per image plus a colouring rule."""
    if ir is not None:
        from utils import _preprocess_for_blip, tokenize_prompt
        imgs_blip = _preprocess_for_blip(gen, device)
        pid, pmask = tokenize_prompt(ir, prompt, device, gen.shape[0])
        raw = ir.score_gard(pid, pmask, imgs_blip).squeeze(-1)
        s = torch.clamp(raw, max=r_max).cpu().numpy()
        return [[(f'{v:.2f}', 'green' if v > 0.5 else 'red' if v < -0.5 else 'orange')] for v in s]
    out = []
    per_clf = []
    for cname, clf in clfs.items():
        p = torch.sigmoid(clf(gen)).squeeze(-1).cpu().numpy()
        per_clf.append((LABELS.get(cname, cname), p))
    n = gen.shape[0]
    for i in range(n):
        tiles = []
        for lab, p in per_clf:
            v = float(p[i])
            c = 'green' if v > 0.7 else 'red' if v < 0.3 else 'orange'
            txt = f'{v:.2f}' if len(per_clf) == 1 else f'{lab}:{v:.2f}'
            tiles.append((txt, c))
        out.append(tiles)
    return out


def build_figure(d, samplers, stylegan, clfs, device, n_show, seed, trial,
                 exp_name, ir=None, prompt=None, r_max=None, cols=10):
    rows = (n_show + cols - 1) // cols
    fig = plt.figure(figsize=(1.5 * cols, (1.7 * rows + 0.5) * len(samplers)))
    subfigs = fig.subfigures(len(samplers), 1)
    if len(samplers) == 1:
        subfigs = [subfigs]

    for sf, s in zip(subfigs, samplers):
        viz = pick(d['samples'][s][trial], n_show, seed)
        with torch.no_grad():
            gen = stylegan(viz.to(device))
            imgs = ((gen + 1) / 2).clamp(0, 1).cpu()
            tiles = score_tiles(gen, clfs, ir, prompt, r_max, device)

        acc = accept_str(d, s, trial, exp_name)
        head = TITLES.get(s, s) + (f'  |  Accept={acc}' if acc else '')
        sf.suptitle(head, fontsize=11, y=0.99)

        axes = sf.subplots(rows, cols)
        axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            if i < n_show:
                ax.imshow(imgs[i].permute(1, 2, 0))
                t = tiles[i]
                ax.set_title(t[0][0], fontsize=7, color=t[0][1], pad=2)
                if len(t) > 1:
                    ax.text(0.5, -0.08, t[1][0], transform=ax.transAxes,
                            ha='center', fontsize=7, color=t[1][1])
            ax.axis('off')
    return fig


def run_one(exp_name, device, n_show, seed, trial):
    cfg = EXPERIMENTS[exp_name]
    stylegan = load_stylegan(device)
    outputs = []

    if cfg.kind == 'imagereward':
        from utils import load_imagereward
        ir = load_imagereward(device)
        for prompt in (cfg.prompts or [cfg.prompt]):
            slug = prompt.lower().replace(' ', '_')
            path = results_path(exp_name, slug)
            d = torch.load(path, weights_only=False, map_location='cpu')
            d = d[list(d)[0]]
            samplers = [s for s in ORDER if s in d['samples']
                        and np.isfinite(np.mean(d['avg_log_reward'][s]))]
            fig = build_figure(d, samplers, stylegan, {}, device, n_show, seed,
                               trial, exp_name, ir=ir, prompt=prompt,
                               r_max=d['rs_r_max'])
            out = os.path.join(os.path.dirname(path), 'samples_grid.png')
            fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
            outputs.append(out)
        return outputs

    clfs = {n: load_classifier(n, device) for n in (cfg.clf_names or [])}
    path = results_path(exp_name)
    d = torch.load(path, weights_only=False, map_location='cpu')
    d = d[list(d)[0]]
    samplers = [s for s in ORDER if s in d['samples']
                and np.isfinite(np.mean(d['avg_log_reward'][s]))]
    fig = build_figure(d, samplers, stylegan, clfs, device, n_show, seed,
                       trial, exp_name)
    out = os.path.join(os.path.dirname(path), 'samples_grid.png')
    fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
    return [out]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default=None, choices=list(EXPERIMENTS))
    ap.add_argument('--n_show', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trial', type=int, default=1)
    args = ap.parse_args()

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    names = [args.experiment] if args.experiment else list(EXPERIMENTS)
    for name in names:
        try:
            for out in run_one(name, device, args.n_show, args.seed, args.trial):
                print(f'  saved {out}', flush=True)
        except Exception as e:
            print(f'  SKIP {name}: {type(e).__name__}: {e}', flush=True)
