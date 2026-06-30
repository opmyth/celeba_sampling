import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import matplotlib.pyplot as plt
import numpy as np

from model_loader import load_models

SNAPSHOT_STEPS = [0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000]
INIT_TYPES     = ['random', 'cold', 'warm']


def _decode(z_batch, stylegan, clf=None):
    """(N, 512) → images (N, H, W, 3) in [0,1] and optional probs list."""
    device = next(stylegan.parameters()).device
    with torch.no_grad():
        imgs = stylegan.G(z_batch.to(device), None)       # (N, 3, H, W) in [-1, 1]
        probs = torch.sigmoid(clf(imgs)).squeeze(-1).cpu().tolist() if clf is not None else None
    imgs_np = ((imgs.clamp(-1, 1) + 1) / 2).cpu().numpy()
    return imgs_np.transpose(0, 2, 3, 1), probs           # (N, H, W, 3), list|None


def _annotate_prob(ax, prob):
    color = 'green' if prob > 0.7 else 'red' if prob < 0.3 else 'orange'
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
        axes[r, 0].text(
            -0.06, 0.5, label,
            va='center', ha='right', rotation=90,
            transform=axes[r, 0].transAxes,
            fontsize=8, fontweight='bold',
        )

    fig.suptitle(title, fontsize=10, y=1.01)
    return fig, axes


def plot_init_grid(attribute, noise='same'):
    noise_dir = 'same_noise' if noise == 'same' else 'indep_noise'
    base_dir  = os.path.join('experiments', attribute, 'trajectory', noise_dir)
    snap_path = os.path.join(base_dir, 'init_snapshots.pt')
    snapshots = torch.load(snap_path, weights_only=False)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stylegan, clf, _ = load_models(attribute, device)

    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    row_labels = INIT_TYPES
    title      = f'{attribute} — MALA trajectory by initialisation'

    n_chains = next(iter(next(iter(snapshots.values())).values())).shape[0]

    for chain_idx in range(n_chains):
        fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels,
                               f'{title} (chain {chain_idx})')

        for r, init_type in enumerate(INIT_TYPES):
            for c, step in enumerate(SNAPSHOT_STEPS):
                z_snap       = snapshots[init_type][step]  # (n_chains, 512)
                imgs, probs  = _decode(z_snap[chain_idx:chain_idx+1], stylegan, clf)
                axes[r, c].imshow(imgs[0])
                axes[r, c].axis('off')
                _annotate_prob(axes[r, c], probs[0])

        plt.tight_layout()
        out_path = os.path.join(base_dir, f'init_grid_{chain_idx}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved to {out_path}')


def plot_stepsize_grid():
    attribute = 'eyeglasses'
    base_dir  = os.path.join('experiments', attribute, 'trajectory')
    snap_path = os.path.join(base_dir, 'stepsize_snapshots.pt')
    snapshots = torch.load(snap_path, weights_only=False)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stylegan, clf, _ = load_models(attribute, device)

    step_sizes = sorted(snapshots.keys())
    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    row_labels = [f'dt={dt}' for dt in step_sizes]
    title      = 'eyeglasses — ULA trajectory by step size'

    fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels, title)

    for r, dt in enumerate(step_sizes):
        for c, step in enumerate(SNAPSHOT_STEPS):
            z_snap      = snapshots[dt][step]       # (3, 512)
            imgs, probs = _decode(z_snap[:1], stylegan, clf)  # chain 0
            axes[r, c].imshow(imgs[0])
            axes[r, c].axis('off')
            _annotate_prob(axes[r, c], probs[0])

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'stepsize_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',      required=True, choices=['init', 'stepsize'])
    parser.add_argument('--attribute', help='required for --plot init')
    parser.add_argument('--noise',     choices=['same', 'indep'], default='same',
                        help='which noise subdirectory to load from (same_noise / indep_noise)')
    args = parser.parse_args()

    if args.plot == 'init':
        if not args.attribute:
            parser.error('--attribute is required for --plot init')
        plot_init_grid(args.attribute, noise=args.noise)
    else:
        plot_stepsize_grid()
