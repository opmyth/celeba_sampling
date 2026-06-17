import sys, os
sys.path.insert(0, os.path.abspath('stylegan2-ada-pytorch'))

import torch
import pickle
import subprocess
from tqdm import tqdm
from models import StyleGAN2Wrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- load StyleGAN2 ---
with open('stylegan2_checkpoints/stylegan2_celeba.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
stylegan = StyleGAN2Wrapper(G).to(device)
stylegan.eval()

# --- load results ---
r = torch.load('results_stylegan.pt', weights_only=False, map_location=device)
samples = r['stylegan']['samples']  # dict: {'RS':..., 'ULA':..., 'MALA':..., 'G_MH':...}

def compute_diversity_all_pairs(model, z_samples, device):
    z_samples = z_samples.to(device)
    with torch.no_grad():
        imgs = model(z_samples)
        imgs_flat = imgs.view(imgs.size(0), -1)

    N = imgs_flat.size(0)
    sq_norms = (imgs_flat ** 2).sum(dim=1)
    dot = imgs_flat @ imgs_flat.T
    sq_dists = (sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * dot).clamp(min=0)
    dists = torch.sqrt(sq_dists)

    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=dists.device), diagonal=1)
    return dists[mask].mean().item()

# --- recompute diversity for every sampler, every trial ---
diversity_all_pairs = {}
for sampler_name, trial_list in samples.items():
    vals = []
    for z in tqdm(trial_list, desc=f"{sampler_name}"):
        d = compute_diversity_all_pairs(stylegan, z, device)
        vals.append(d)
    diversity_all_pairs[sampler_name] = vals
    print(f"{sampler_name}: {vals}")

torch.save(diversity_all_pairs, 'diversity_all_pairs_results.pt')
print("Saved to diversity_all_pairs_results.pt")
print(diversity_all_pairs)
