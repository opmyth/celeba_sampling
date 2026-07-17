import os
import torch

import numpy as np
import torch.nn.functional as F
import ot

from itertools import combinations
from scipy.stats import wasserstein_distance_nd
from scipy.stats import ttest_rel


def maybe_enable_tf32():
    """Opt-in TF32 matmuls via the TF32=1 env var (torch's default is off).

    ~1.3-2x on transformer-heavy workloads (BLIP dominates ImageReward steps)
    at ~1e-3 relative matmul noise. Deliberately an env flag, NOT enabled
    globally: the classifier experiments' published numbers were produced
    without it, so silent global enablement would make future reruns subtly
    non-comparable. Set TF32=1 per job where the speed matters and the
    experiment's results are being regenerated anyway (e.g. bald_ir's
    2026-07-16 from-scratch rerun). Prints a provenance line so job logs
    record which mode produced any given result file."""
    if os.environ.get('TF32') == '1':
        torch.set_float32_matmul_precision('high')
        print('TF32 matmul enabled (TF32=1)', flush=True)


def compute_w2(samples_1, samples_2):
    samples_1_np = samples_1.detach().cpu().numpy()
    samples_2_np = samples_2.detach().cpu().numpy()

    # ot.emd2 silently returns 0.0 when the cost matrix contains NaN - a
    # diverged sampler (e.g. bald_ir ULA at dt=0.01, 2026-07-17: 992/1000
    # NaN samples) would otherwise show up in results tables as a *perfect*
    # W2 of 0.0. Return NaN loudly instead: unmistakable in any table, and
    # doesn't crash the pipeline stage after its expensive sampling is done.
    if np.isnan(samples_1_np).any() or np.isnan(samples_2_np).any():
        print(f'WARNING compute_w2: NaN in input samples '
              f'({np.isnan(samples_1_np).any(axis=1).sum()}/{len(samples_1_np)} and '
              f'{np.isnan(samples_2_np).any(axis=1).sum()}/{len(samples_2_np)} rows) - '
              f'returning NaN, not a real distance', flush=True)
        return float('nan')

    n1, n2 = samples_1_np.shape[0], samples_2_np.shape[0]
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    M = ot.dist(samples_1_np, samples_2_np, metric='sqeuclidean')
    w2_squared = ot.emd2(a, b, M)
    return np.sqrt(w2_squared)

# def compute_sliced_w2(samples_1, samples_2, n_projections=200):
#     projections = torch.randn(n_projections, samples_1.size(1)).to(samples_1.device)
#     projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
#     samples_1_projections = samples_1 @ projections.T
#     samples_2_projections = samples_2 @ projections.T
    
#     samples_1_sorted = torch.sort(samples_1_projections, dim=0).values
#     samples_2_sorted = torch.sort(samples_2_projections, dim=0).values

#     return torch.sqrt(torch.mean((samples_1_sorted - samples_2_sorted)**2)).item()
    
def compute_stats(w2_values):
    return {name: {'mean': np.mean(w2_values[name]), 'std':np.std(w2_values[name], ddof=1)} for name in w2_values}

def compute_ttest(values_per_sampler, baseline, alpha=0.05, pairwise=False):
    result = {}
    sampler_names = list(values_per_sampler.keys())
    combs = list(combinations(sampler_names, 2)) if pairwise else []
    
    corrected_alpha = alpha / (len(sampler_names) + len(combs))
    
    for sampler_name in sampler_names:
        t, p = ttest_rel(values_per_sampler[sampler_name], baseline).statistic, \
               ttest_rel(values_per_sampler[sampler_name], baseline).pvalue
        result[f'{sampler_name} vs RS'] = {
            't': t, 'p-value': p,
            'significant': "Yes" if p < corrected_alpha else "No"
        }
    
    for c in combs:
        t, p = ttest_rel(values_per_sampler[c[0]], values_per_sampler[c[1]]).statistic, \
               ttest_rel(values_per_sampler[c[0]], values_per_sampler[c[1]]).pvalue
        result[f'{c[0]} vs {c[1]}'] = {
            't': t, 'p-value': p,
            'significant': "Yes" if p < corrected_alpha else "No"
        }
    return result

def compute_diversity(z_samples):
    N = z_samples.size(0)
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    distances = torch.norm(z_samples[idx_i] - z_samples[idx_j], dim=1) ** 2
    return distances.mean().item()

def compute_diversity_cov(z_samples):
    return torch.trace(torch.cov(z_samples.T)).item()

def load_imagereward(device):
    from types import ModuleType
    import sys
    sys.modules.setdefault('ImageReward.ReFL', ModuleType('ImageReward.ReFL'))
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device=str(device))
    model.eval()
    model.requires_grad_(False)
    model.blip = torch.compile(model.blip)
    return model

_BLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_BLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def _preprocess_for_blip(imgs, device):
    imgs_01 = (imgs + 1) / 2
    imgs_224 = F.interpolate(imgs_01, size=(224, 224), mode='bicubic', align_corners=False)
    return (imgs_224 - _BLIP_MEAN.to(device)) / _BLIP_STD.to(device)

def tokenize_prompt(reward_model, prompt, device, n):
    inp = reward_model.blip.tokenizer(
        prompt, padding='max_length', truncation=True, max_length=35, return_tensors='pt'
    ).to(device)
    return inp.input_ids.expand(n, -1).contiguous(), inp.attention_mask.expand(n, -1).contiguous()

def compute_male_fraction(model, male_clf, z_samples):
    with torch.no_grad():
        imgs = model(z_samples)
        logits = male_clf(imgs).squeeze()
        preds = (logits > 0).float()
    return preds.mean().item()
