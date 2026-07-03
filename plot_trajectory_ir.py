import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import matplotlib.pyplot as plt
import numpy as np

from model_loader import load_models
from utils import load_imagereward, tokenize_prompt, _preprocess_for_blip

PROMPT         = "a bald man"
SNAPSHOT_STEPS = [0, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000]
INIT_TYPES     = ['random', 'cold', 'warm']
BASE_DIR       = os.path.join('experiments', 'bald', 'trajectory', 'imagereward')


def _decode(z_batch, stylegan, reward_model, prompt_ids, prompt_mask):
    device = next(stylegan.parameters()).device
    with torch.no_grad():
        imgs      = stylegan.G(z_batch.to(device), None)
        imgs_blip = _preprocess_for_blip(imgs, device)
        B         = imgs.size(0)
        scores    = reward_model.score_gard(
            prompt_ids[:B], prompt_mask[:B], imgs_blip
        ).squeeze(-1)
    imgs_np = ((imgs.clamp(-1, 1) + 1) / 2).cpu().numpy()
    return imgs_np.transpose(0, 2, 3, 1), scores.cpu().tolist()


def _annotate(ax, score):
    color = 'green' if score > -0.5 else ('red' if score < -2.0 else 'orange')
    ax.set_title(f'{score:.2f}', fontsize=9, color=color, pad=3, fontweight='bold')


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


def plot_stepsize_grid(stylegan, reward_model, prompt_ids, prompt_mask, snap_file='stepsize_snapshots.pt'):
    snap_path = os.path.join(BASE_DIR, snap_file)
    snapshots = torch.load(snap_path, weights_only=False)

    step_sizes = sorted(snapshots.keys())
    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    row_labels = [f'dt={dt}' for dt in step_sizes]
    title      = 'Bald-IR — MALA trajectory by step size'

    fig, axes = _make_grid(len(row_labels), len(col_labels), row_labels, col_labels, title)

    for r, dt in enumerate(step_sizes):
        for c, step in enumerate(SNAPSHOT_STEPS):
            z_snap      = snapshots[dt][step]
            imgs, scores = _decode(z_snap[:1], stylegan, reward_model, prompt_ids, prompt_mask)
            axes[r, c].imshow(imgs[0])
            axes[r, c].axis('off')
            _annotate(axes[r, c], scores[0])

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, snap_file.replace('.pt', '_grid.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_init_grid(stylegan, reward_model, prompt_ids, prompt_mask, noise='same'):
    noise_dir = 'same_noise' if noise == 'same' else 'indep_noise'
    snap_dir  = os.path.join(BASE_DIR, noise_dir)
    snapshots = torch.load(os.path.join(snap_dir, 'init_snapshots.pt'), weights_only=False)

    col_labels = [f'Step {s}' for s in SNAPSHOT_STEPS]
    title      = f'Bald-IR — MALA trajectory by init ({noise} noise)'

    n_chains = next(iter(next(iter(snapshots.values())).values())).shape[0]

    for chain_idx in range(n_chains):
        fig, axes = _make_grid(len(INIT_TYPES), len(col_labels), INIT_TYPES, col_labels,
                               f'{title} (chain {chain_idx})')
        for r, init_type in enumerate(INIT_TYPES):
            for c, step in enumerate(SNAPSHOT_STEPS):
                z_snap       = snapshots[init_type][step]
                imgs, scores = _decode(z_snap[chain_idx:chain_idx+1], stylegan,
                                       reward_model, prompt_ids, prompt_mask)
                axes[r, c].imshow(imgs[0])
                axes[r, c].axis('off')
                _annotate(axes[r, c], scores[0])

        plt.tight_layout()
        out_path = os.path.join(snap_dir, f'init_grid_{chain_idx}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',  required=True, choices=['stepsize', 'init', 'all'])
    parser.add_argument('--noise', choices=['same', 'indep', 'both'], default='both')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    stylegan, _, _ = load_models('bald', device)
    stylegan.G     = torch.compile(stylegan.G)
    reward_model   = load_imagereward(device)
    print('Models loaded.', flush=True)

    prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, 1)

    if args.plot in ('stepsize', 'all'):
        plot_stepsize_grid(stylegan, reward_model, prompt_ids, prompt_mask)

    if args.plot in ('init', 'all'):
        noises = ['same', 'indep'] if args.noise == 'both' else [args.noise]
        for n in noises:
            plot_init_grid(stylegan, reward_model, prompt_ids, prompt_mask, noise=n)

    print('Done.')
