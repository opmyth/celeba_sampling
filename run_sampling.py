import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

import argparse
import torch
import numpy as np
import torch.nn.functional as F
import time
import pickle

from tqdm import tqdm
from models import classifier, StyleGAN2Wrapper
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba, rejection_sampling
from utils import compute_sliced_w2, compute_diversity, compute_male_fraction, compute_diversity_cov

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=800)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--output_path', type=str, default='results.pt')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

smile_clf = classifier().to(device)
smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf.pth', weights_only=False))
smile_clf.eval()

male_clf = classifier().to(device)
male_clf.load_state_dict(torch.load('clf_checkpoints/male_clf.pth', weights_only=False))
male_clf.eval()

with open('stylegan2_checkpoints/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
    
stylegan = StyleGAN2Wrapper(G).to(device)
stylegan.eval()

def run_trials(model, clf, male_clf, dt, sigma, n_chains, n_steps, device, n_trials=10, seed=321):
    torch.manual_seed(seed)
    
    w2_values = {'ULA': [], 'MALA': [], 'G_MH': []}
    w2_baseline = []

    samples = {'RS': None, 'ULA': None, 'MALA': None, 'G_MH': None}
    avg_log_reward = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}
    diversity = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}
    diversity_trace_cov = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}
    male_fraction = {'RS': [], 'ULA': [], 'MALA': [], 'G_MH': []}

    t = time.time()
    RS_samples = rejection_sampling(model, clf, n_chains * n_trials * 2, device=device)
    print(f"RS done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    chunks = torch.chunk(RS_samples, n_trials*2, dim=0)
    w2_baseline = [compute_sliced_w2(chunks[2*i], chunks[2*i + 1]) for i in range(n_trials)]
    samples['RS'] = list(chunks[::2])
    print(f"RS W2 baseline done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    ULA_samples = latent_ULA_celeba(model, clf, n_chains * n_trials, n_steps, dt, device=device)[-1]
    print(f"ULA done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    ULA_chunks = torch.chunk(ULA_samples, n_trials, dim=0)
    w2_values['ULA'] = [compute_sliced_w2(ULA_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['ULA'] = list(ULA_chunks)
    print(f"ULA W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    MALA_samples = latent_MALA_celeba(model, clf, n_chains * n_trials, n_steps, dt, device=device)[-1]
    print(f"MALA done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    MALA_chunks = torch.chunk(MALA_samples, n_trials, dim=0)
    w2_values['MALA'] = [compute_sliced_w2(MALA_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['MALA'] = list(MALA_chunks)
    print(f"MALA W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    gaussianMH_samples = latent_Gaussian_MH_celeba(model, clf, n_chains * n_trials, n_steps, sigma, device=device)[-1]
    print(f"G_MH done: {time.time()-t:.2f}s", flush=True)
    t = time.time()
    gaussianMH_chunks = torch.chunk(gaussianMH_samples, n_trials, dim=0)
    w2_values['G_MH'] = [compute_sliced_w2(gaussianMH_chunks[i], chunks[2*i]) for i in range(n_trials)]
    samples['G_MH'] = list(gaussianMH_chunks)
    print(f"G_MH W2 done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in avg_log_reward:
        with torch.no_grad():
            avg_log_reward[name] = [F.logsigmoid(clf(model(samples[name][i]))).mean().item() for i in range(n_trials)]
    print(f"avg_log_reward done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in diversity:
        diversity[name] = [compute_diversity(model, samples[name][i]) for i in range(n_trials)]
    print(f"diversity done: {time.time()-t:.2f}s", flush=True)
    
    t = time.time()
    for name in diversity_trace_cov:
        diversity_trace_cov[name] = [compute_diversity_cov( samples[name][i]) for i in range(n_trials)]
    print(f"diversity trace covariance done: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    for name in male_fraction:
        male_fraction[name] = [compute_male_fraction(model, male_clf, samples[name][i]) for i in range(n_trials)]
    print(f"male_fraction done: {time.time()-t:.2f}s", flush=True)

    return w2_values, w2_baseline, samples, avg_log_reward, diversity, diversity_trace_cov, male_fraction

start = time.time()
stylegan_w2_values, stylegan_w2_baseline, stylegan_samples, stylegan_avg_log_reward, stylegan_diversity, stylegan_diversity_trace_cov, stylegan_male_fraction = run_trials(
    stylegan, smile_clf, male_clf, dt=args.dt, sigma=args.sigma, n_chains=args.n_chains, n_steps=args.n_steps, n_trials=args.n_trials, device=device
)
print(f"Total time: {time.time() - start:.2f}s")
torch.save({
    'stylegan': {
        'w2_values': stylegan_w2_values,
        'w2_baseline': stylegan_w2_baseline,
        'samples': stylegan_samples,
        'avg_log_reward': stylegan_avg_log_reward,
        'diversity': stylegan_diversity,
        'diversity_trace_cov': stylegan_diversity_trace_cov,
        'male_fraction': stylegan_male_fraction,
    }
}, args.output_path)

print(f'Results saved to {args.output_path}')
