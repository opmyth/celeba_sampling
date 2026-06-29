import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import matplotlib.pyplot as plt
import numpy as np

from model_loader import load_models

SNAPSHOT_STEPS = [0, 200, 500, 1000, 2000, 3000]
INIT_TYPES     = ['random', 'cold', 'warm']


def _decode(z_batch, stylegan):
    """(N, 512) → (N, H, W, 3) numpy float32 in [0, 1]."""
    device = next(stylegan.parameters()).device
    with torch.no_grad():
        imgs = stylegan.G(z_batch.to(device), None)   # (N, 3, H, W) in [-1, 1]
    imgs = ((imgs.clamp(-1, 1) + 1) / 2).cpu().numpy()
    return imgs.transpose(0, 2, 3, 1)                 # (N, H, W, 3)


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
        axes[0, c].set_title(label, fontsize=8, pad=4)

    for r, label in enumerate(row_labels):
        axes[r, 0].text(
            -0.06, 0.5, label,
            va='center', ha='right', rotation=90,
            transform=axes[r, 0].transAxes,
            fontsize=8, fontweight='bold',
        )

    fig.suptitle(title, fontsize=10, y=1.01)
    return fig, axes


def plot_init_grid(attribute):
    base_dir  = os.path.join('experiments', attribute, 'trajectory')
    snap_path = os.path.join(base_dir, 'init_snapshots.pt')
    snapshots = torch.load(snap_path, weights_only=False)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stylegan, _, _ = load_models(attribute, device)

    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    row_labels = INIT_TYPES
    title      = f'{attribute} — MALA trajectory by initialisation'

    fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels, title)

    for r, init_type in enumerate(INIT_TYPES):
        for c, step in enumerate(SNAPSHOT_STEPS):
            z_snap = snapshots[init_type][step]  # (3, 512)
            img    = _decode(z_snap[:1], stylegan)[0]  # chain 0
            axes[r, c].imshow(img)
            axes[r, c].axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'init_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


def plot_stepsize_grid():
    attribute = 'eyeglasses'
    base_dir  = os.path.join('experiments', attribute, 'trajectory')
    snap_path = os.path.join(base_dir, 'stepsize_snapshots.pt')
    snapshots = torch.load(snap_path, weights_only=False)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stylegan, _, _ = load_models(attribute, device)

    step_sizes = sorted(snapshots.keys())
    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    row_labels = [f'dt={dt}' for dt in step_sizes]
    title      = 'eyeglasses — ULA trajectory by step size'

    fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels, title)

    for r, dt in enumerate(step_sizes):
        for c, step in enumerate(SNAPSHOT_STEPS):
            z_snap = snapshots[dt][step]  # (3, 512)
            img    = _decode(z_snap[:1], stylegan)[0]  # chain 0
            axes[r, c].imshow(img)
            axes[r, c].axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'stepsize_grid.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',      required=True, choices=['init', 'stepsize'])
    parser.add_argument('--attribute', help='required for --plot init')
    args = parser.parse_args()

    if args.plot == 'init':
        if not args.attribute:
            parser.error('--attribute is required for --plot init')
        plot_init_grid(args.attribute)
    else:
        plot_stepsize_grid()
