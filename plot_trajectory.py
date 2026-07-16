"""Unified trajectory plotting - replaces plot_trajectory.py/_ir.py/_male_eye.py.

  python plot_trajectory.py --experiment <name> --plot stepsize
  python plot_trajectory.py --experiment <name> --plot init [--noise same|indep|both]
  python plot_trajectory.py --experiment <name> --plot jump_distance --mode stepsize|init
  python plot_trajectory.py --experiment <name> --plot log_reward   --mode stepsize|init
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import matplotlib.pyplot as plt
import numpy as np

from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from utils import load_imagereward

INIT_TYPES     = ['random', 'cold', 'warm']


def _slug(s):
    return s.lower().replace(' ', '_')


def _prompt_slug(cfg, prompt):
    return f'prompt_{_slug(prompt)}' if cfg.kind == 'imagereward' else None


def _stepsize_dir(cfg, experiment, prompt):
    base = os.path.join('experiments', experiment, 'trajectory')
    slug = _prompt_slug(cfg, prompt)
    return os.path.join(base, slug) if slug else base


def _init_dir(cfg, experiment, prompt, noise):
    base = os.path.join('experiments', experiment, 'trajectory')
    noise_dir = 'same_noise' if noise == 'same' else 'indep_noise'
    d = os.path.join(base, noise_dir)
    slug = _prompt_slug(cfg, prompt)
    return os.path.join(d, slug) if slug else d


def _build_posterior(cfg, stylegan, clfs, prompt, device):
    if cfg.kind == 'classifier':
        return classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
    reward_model = load_imagereward(device)
    return imagereward_posterior(stylegan, reward_model, prompt, device, load_r_max(prompt))


def _decode(z_batch, stylegan, posterior):
    device = next(stylegan.parameters()).device
    with torch.no_grad():
        imgs = stylegan.G(z_batch.to(device), None)
        reward = posterior.reward_only_fn(z_batch.to(device))
    imgs_np = ((imgs.clamp(-1, 1) + 1) / 2).cpu().numpy()
    return imgs_np.transpose(0, 2, 3, 1), reward.cpu().tolist()


def _annotate(ax, cfg, raw_reward):
    """classifier (single or joint): raw_reward is a log-prob (or log of a
    product of probs for joint experiments) - exp() recovers the displayed
    probability. imagereward: raw_reward is already the raw IR score."""
    if cfg.kind == 'imagereward':
        color = 'green' if raw_reward > -0.5 else ('red' if raw_reward < -2.0 else 'orange')
        ax.set_title(f'{raw_reward:.2f}', fontsize=9, color=color, pad=3, fontweight='bold')
    else:
        prob = float(np.exp(raw_reward))
        color = 'green' if prob > 0.7 else ('red' if prob < 0.3 else 'orange')
        ax.set_title(f'{prob:.2f}', fontsize=10, color=color, pad=3, fontweight='bold')


def _make_grid(n_rows, n_cols, row_labels, col_labels, title, cell_size=2.4):
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols + 1.0, cell_size * n_rows + 0.6),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for c, label in enumerate(col_labels):
        axes[0, c].text(0.5, 1.13, label, transform=axes[0, c].transAxes,
                         ha='center', va='bottom', fontsize=8)
    for r, label in enumerate(row_labels):
        axes[r, 0].text(-0.06, 0.5, label, va='center', ha='right', rotation=90,
                         transform=axes[r, 0].transAxes, fontsize=8, fontweight='bold')

    fig.suptitle(title, fontsize=10, y=1.01)
    return fig, axes


def _load_models_and_posterior(experiment, prompt):
    cfg = EXPERIMENTS[experiment]
    prompt = prompt or cfg.prompt
    device = torch.device('mps' if torch.backends.mps.is_available() else
                           'cuda' if torch.cuda.is_available() else 'cpu')
    stylegan, clfs, _ = load_models(cfg.clf_names or [], device)
    posterior = _build_posterior(cfg, stylegan, clfs, prompt, device)
    return cfg, prompt, stylegan, posterior


def plot_stepsize_grid(experiment, prompt=None):
    cfg, prompt, stylegan, posterior = _load_models_and_posterior(experiment, prompt)
    out_dir = _stepsize_dir(cfg, experiment, prompt)
    snapshots = torch.load(os.path.join(out_dir, 'stepsize_snapshots.pt'), weights_only=False)

    step_sizes = sorted(snapshots.keys())
    steps_present = sorted(snapshots[step_sizes[0]].keys())
    col_labels = [f'Step {s}' for s in steps_present]
    row_labels = [f'dt={dt}' for dt in step_sizes]
    title = f'{experiment} - MALA trajectory by step size'

    fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels, title)
    for r, dt in enumerate(step_sizes):
        for c, step in enumerate(steps_present):
            imgs, rewards = _decode(snapshots[dt][step][:1], stylegan, posterior)
            axes[r, c].imshow(imgs[0])
            axes[r, c].axis('off')
            _annotate(axes[r, c], cfg, rewards[0])

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'stepsize_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_init_grid(experiment, noise='same', prompt=None):
    cfg, prompt, stylegan, posterior = _load_models_and_posterior(experiment, prompt)
    snap_dir = _init_dir(cfg, experiment, prompt, noise)
    snapshots = torch.load(os.path.join(snap_dir, 'init_snapshots.pt'), weights_only=False)

    steps_present = sorted(snapshots[INIT_TYPES[0]].keys())
    col_labels = [f'Step {s}' for s in steps_present]
    title = f'{experiment} - MALA trajectory by init ({noise} noise)'
    n_chains = next(iter(snapshots[INIT_TYPES[0]].values())).shape[0]

    # Capped at 2 grids no matter how many chains ran: each grid costs a
    # StyleGAN2 decode per cell, and at N_CHAINS=100 rendering one per chain
    # would be 100 near-identical images nobody inspects. Two gives a
    # cross-chain visual comparison; the full-population view is the averaged
    # trace plots, not image grids.
    for chain_idx in range(min(2, n_chains)):
        fig, axes = _make_grid(len(INIT_TYPES), len(col_labels), INIT_TYPES, col_labels,
                                f'{title} (chain {chain_idx})')
        for r, init_type in enumerate(INIT_TYPES):
            for c, step in enumerate(steps_present):
                z_snap = snapshots[init_type][step]
                imgs, rewards = _decode(z_snap[chain_idx:chain_idx + 1], stylegan, posterior)
                axes[r, c].imshow(imgs[0])
                axes[r, c].axis('off')
                _annotate(axes[r, c], cfg, rewards[0])

        plt.tight_layout()
        out_path = os.path.join(snap_dir, f'init_grid_{chain_idx}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')


def plot_trace(experiment, mode, metric, noise='same', chain_idx=0, prompt=None):
    """metric: 'jump_distance' (||z_t+1 - z_t||_2) or 'log_reward', read
    straight from the trace .pt saved by run_trajectory.py (no re-sampling,
    no model needed). One subplot per swept value (dt or init type), each
    with its own y-range - overlaying all of them on one axes made
    small-dt/less-active lines invisible next to large-dt ones.

    chain_idx: an int plots that single chain's trace (spiky by construction
    for jump_distance - MALA rejects leave the chain exactly still - hence
    the rolling-mean overlay). The string 'mean' instead averages the metric
    across ALL chains at each step and plots that as one line - a convergence
    trend for the whole chain population rather than one noisy walker. No
    spread band: this is a convergence sanity check, not a statistical
    comparison like the 5-trial pipeline tables, so the mean trend alone is
    the deliverable. No rolling overlay either - cross-chain averaging is
    already the smoother (and at n_chains=1 the mean just degenerates to
    chain 0's raw trace)."""
    cfg = EXPERIMENTS[experiment]
    prompt = prompt or cfg.prompt
    mean_mode = chain_idx == 'mean'

    if mode == 'stepsize':
        sub_dir = _stepsize_dir(cfg, experiment, prompt)
        traces = torch.load(os.path.join(sub_dir, 'stepsize_trace.pt'), weights_only=False)
        keys = sorted(traces.keys())
        label_fn = lambda k: f'dt={k}'
    else:
        sub_dir = _init_dir(cfg, experiment, prompt, noise)
        traces = torch.load(os.path.join(sub_dir, 'init_trace.pt'), weights_only=False)
        keys = INIT_TYPES
        label_fn = lambda k: k

    ROLLING_WINDOW = 50
    n_chains = traces[keys[0]]['z'].shape[1]

    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False, sharey=(metric == 'jump_distance'))
    axes_flat = axes.flatten()

    for i, k in enumerate(keys):
        ax = axes_flat[i]
        if mean_mode:
            z = traces[k]['z']            # (n_steps, n_chains, latent_dim)
            log_p = traces[k]['log_p']    # (n_steps, n_chains)
            if metric == 'jump_distance':
                y = torch.norm(z[1:] - z[:-1], dim=2).mean(dim=1).numpy()
                ax.set_yscale('log')
            else:
                y = (log_p + 0.5 * (z ** 2).sum(2)).mean(dim=1).numpy()   # reward = log_p + prior term
            ax.plot(y, linewidth=0.8)
        else:
            z = traces[k]['z'][:, chain_idx, :]        # (n_steps, latent_dim)
            log_p = traces[k]['log_p'][:, chain_idx]   # (n_steps,)
            if metric == 'jump_distance':
                y = torch.norm(z[1:] - z[:-1], dim=1).numpy()
                ax.set_yscale('log')
                ax.plot(y, linewidth=0.5, alpha=0.4, color='C0')
                if len(y) >= ROLLING_WINDOW:
                    smoothed = np.convolve(y, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode='valid')
                    x_smooth = np.arange(ROLLING_WINDOW - 1, len(y))
                    ax.plot(x_smooth, smoothed, linewidth=1.3, color='C1')
            else:
                y = (log_p + 0.5 * (z ** 2).sum(1)).numpy()   # reward = log_p + prior term
                ax.plot(y, linewidth=0.8)
        ax.set_title(label_fn(k), fontsize=9)
        ax.set_xlabel('step', fontsize=8)
        ax.tick_params(labelsize=7)

    for i in range(len(keys), len(axes_flat)):
        axes_flat[i].axis('off')

    ylabel = r'$\|z_{t+1} - z_t\|_2$' if metric == 'jump_distance' else 'log r(z)'
    if mean_mode:
        ylabel += f'  (mean over {n_chains} chains)'
    fig.supylabel(ylabel, fontsize=9)
    which = f'mean over {n_chains} chains' if mean_mode else f'chain {chain_idx}'
    fig.suptitle(f'{experiment} - {metric} ({mode}, {which})', fontsize=11)
    plt.tight_layout()

    suffix = 'mean' if mean_mode else f'chain{chain_idx}'
    out_path = os.path.join(sub_dir, f'{metric}_{mode}_{suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
    parser.add_argument('--plot', required=True,
                         choices=['stepsize', 'init', 'jump_distance', 'log_reward'])
    parser.add_argument('--mode', choices=['stepsize', 'init'], default=None,
                         help='required for jump_distance/log_reward')
    parser.add_argument('--noise', choices=['same', 'indep', 'both'], default='both')
    parser.add_argument('--chain', type=str, default='0',
                         help="chain index for jump_distance/log_reward, or 'mean' "
                              "for the across-chain average trend")
    parser.add_argument('--prompt', type=str, default=None)
    args = parser.parse_args()
    chain = args.chain if args.chain == 'mean' else int(args.chain)

    if args.plot == 'stepsize':
        plot_stepsize_grid(args.experiment, prompt=args.prompt)
    elif args.plot == 'init':
        noises = ['same', 'indep'] if args.noise == 'both' else [args.noise]
        for n in noises:
            plot_init_grid(args.experiment, noise=n, prompt=args.prompt)
    else:
        if not args.mode:
            parser.error('--mode is required for jump_distance/log_reward')
        if args.mode == 'init':
            noises = ['same', 'indep'] if args.noise == 'both' else [args.noise]
            for n in noises:
                plot_trace(args.experiment, 'init', args.plot, noise=n,
                           chain_idx=chain, prompt=args.prompt)
        else:
            plot_trace(args.experiment, 'stepsize', args.plot,
                       chain_idx=chain, prompt=args.prompt)

    print('Done.')
