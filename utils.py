import torch

import numpy as np
import torch.nn.functional as F
import ot

from itertools import combinations
from scipy.stats import wasserstein_distance_nd
from scipy.stats import ttest_rel


def log_posterior_celeba(z, model, clf):
    chunk_size = model.max_batch_size
    log_p_list = []
    with torch.no_grad():
        for start in range(0, z.size(0), chunk_size):
            z_chunk = z[start:start + chunk_size]
            imgs = model.G(z_chunk, None)
            logits = clf(imgs).squeeze()
            log_p_list.append(-0.5 * torch.sum(z_chunk**2, dim=1) + F.logsigmoid(logits))
    return torch.cat(log_p_list)

def grad_log_posterior_celeba(z, model, clf):
    z = z.detach().requires_grad_(True)
    chunk_size = model.max_batch_size
    for start in range(0, z.size(0), chunk_size):
        z_chunk = z[start:start + chunk_size]
        imgs = model.G(z_chunk, None)
        logits = clf(imgs).squeeze()
        posterior_chunk = (-0.5 * torch.sum(z_chunk**2, dim=1) + F.logsigmoid(logits)).sum()
        posterior_chunk.backward()
    return z.grad.clone()

def grad_and_log_posterior_celeba(z, model, clf):
    z = z.detach().requires_grad_(True)
    log_p_list = []
    chunk_size = model.max_batch_size  # 64: backward() called per chunk so activations freed immediately
    for start in range(0, z.size(0), chunk_size):
        z_chunk = z[start:start + chunk_size]
        imgs = model.G(z_chunk, None)
        logits = clf(imgs).squeeze()
        log_p_chunk = -0.5 * torch.sum(z_chunk**2, dim=1) + F.logsigmoid(logits)
        log_p_chunk.sum().backward()
        log_p_list.append(log_p_chunk.detach())
    return z.grad.clone(), torch.cat(log_p_list)

def compute_w2(samples_1, samples_2):
    samples_1_np = samples_1.detach().cpu().numpy()
    samples_2_np = samples_2.detach().cpu().numpy()
    
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

def compute_male_fraction(model, male_clf, z_samples):
    with torch.no_grad():
        imgs = model(z_samples)
        logits = male_clf(imgs).squeeze()
        preds = (logits > 0).float()
    return preds.mean().item()
