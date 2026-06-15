import torch

import numpy as np
import torch.nn.functional as F
import ot

from itertools import combinations
from scipy.stats import wasserstein_distance_nd
from scipy.stats import ttest_rel


def log_posterior_celeba(z, model, clf):
    model.eval(); clf.eval()
    with torch.no_grad():
        imgs = model(z) #imgs.shape (B, 3, 64, 64)
        logits = clf(imgs).squeeze() #logits shape(B, )
    return -0.5 * torch.sum(z**2, dim=1) + F.logsigmoid(logits)

def grad_log_posterior_celeba(z, model, clf):
    z = z.detach().requires_grad_(True)
    imgs = model(z)
    logits = clf(imgs).squeeze()
    log_probs_smiling = F.logsigmoid(logits)
    posterior = (-0.5 * torch.sum(z**2, dim=1) + log_probs_smiling).sum()
    posterior.backward()
    return z.grad.clone()

def compute_w2(samples_1, samples_2):
    samples_1_np = samples_1.detach().cpu().numpy()
    samples_2_np = samples_2.detach().cpu().numpy()
    
    n1, n2 = samples_1_np.shape[0], samples_2_np.shape[0]
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2
    
    M = ot.dist(samples_1_np, samples_2_np, metric='sqeuclidean')
    w2_squared = ot.emd2(a, b, M)
    return np.sqrt(w2_squared)

def compute_sliced_w2(samples_1, samples_2, n_projections=100):
    projections = torch.randn(n_projections, samples_1.size(1)).to(samples_1.device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    samples_1_projections = samples_1 @ projections.T
    samples_2_projections = samples_2 @ projections.T
    
    samples_1_sorted = torch.sort(samples_1_projections, dim=0).values
    samples_2_sorted = torch.sort(samples_2_projections, dim=0).values

    return torch.sqrt(torch.mean((samples_1_sorted - samples_2_sorted)**2)).item()
    
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

def compute_diversity(model, z_samples, n_pairs=500):

    with torch.no_grad():
        imgs = model(z_samples)
        imgs_flat = imgs.view(imgs.size(0), -1)

    N = imgs_flat.size(0)
    idx_i = torch.randint(0, N, (n_pairs,))
    idx_j = torch.randint(0, N, (n_pairs,))

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    distances = torch.norm(imgs_flat[idx_i] - imgs_flat[idx_j], dim=1)
    return distances.mean().item()

def compute_diversity_cov(z_samples):
    return torch.trace(torch.cov(z_samples.T)).item()

def compute_male_fraction(model, male_clf, z_samples):
    with torch.no_grad():
        imgs = model(z_samples)
        logits = male_clf(imgs).squeeze()
        preds = (logits > 0).float()
    return preds.mean().item()
