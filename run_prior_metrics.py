"""Higher-precision prior baseline (N_PER_TRIAL=1000, vs the 100 used in
results_prior.pt for W2 comparability with RS) - produces the 'Prior AvgLogR'
numbers reported at the top of each EXPERIMENTS.md section. Iterates
config.EXPERIMENTS instead of a hardcoded per-attribute list, so male_eye's
joint reward and bald_ir's IR posterior fall out of the same loop as the
single-classifier experiments."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_stylegan, load_classifier
from posteriors import classifier_posterior, imagereward_posterior
from utils import compute_w2, compute_diversity, compute_diversity_cov, load_imagereward

N_TRIALS    = 5
N_PER_TRIAL = 1000
BATCH_SIZE  = 64
SEED        = 321

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan = load_stylegan(device)
print('StyleGAN loaded.', flush=True)

clf_names = sorted({name for cfg in EXPERIMENTS.values() if cfg.kind == 'classifier' for name in cfg.clf_names})
clfs = {name: load_classifier(name, device) for name in clf_names}
print(f'Classifiers loaded: {clf_names}', flush=True)

reward_model = None
if any(cfg.kind == 'imagereward' for cfg in EXPERIMENTS.values()):
    reward_model = load_imagereward(device)
    print('ImageReward loaded.', flush=True)

results = {}

print('\n=== [1/2] Prior z-space metrics (W2, Diversity) ===', flush=True)
w2_p, div_p, div_cov_p, z_trials = [], [], [], []
cpu_gen = rng_mod.make_generator(SEED, 'cpu')
for t in range(N_TRIALS):
    z_all = torch.randn(N_PER_TRIAL * 2, stylegan.latent_dim, generator=cpu_gen)
    z_a, z_b = z_all[:N_PER_TRIAL], z_all[N_PER_TRIAL:]
    z_trials.append(z_a)
    w2_p.append(compute_w2(z_a, z_b))
    div_p.append(compute_diversity(z_a))
    div_cov_p.append(compute_diversity_cov(z_a))
    print(f'  trial {t + 1}/{N_TRIALS}: W2={w2_p[-1]:.2f}, Div={div_p[-1]:.1f}', flush=True)
results['shared'] = {'w2': w2_p, 'div': div_p, 'div_cov': div_cov_p}

print('\n=== [2/2] Prior AvgLogR per experiment ===', flush=True)
for name, cfg in EXPERIMENTS.items():
    if cfg.kind == 'classifier':
        posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
    else:
        posterior = imagereward_posterior(stylegan, reward_model, cfg.prompt, device)

    reward_vals = []
    for z_a in z_trials:
        vals = [posterior.reward_only_fn(z_a[i:i + BATCH_SIZE].to(device)).cpu()
                for i in range(0, N_PER_TRIAL, BATCH_SIZE)]
        reward_vals.append(torch.cat(vals))

    if cfg.kind == 'classifier':
        alr = [r.mean().item() for r in reward_vals]
        results[name] = {'alr': alr}
        print(f'  {name}: {sum(alr) / len(alr):.4f}', flush=True)
    else:
        log_post = [(r + (-0.5) * (z_a ** 2).sum(1)).mean().item()
                     for r, z_a in zip(reward_vals, z_trials)]
        raw_ir = [r.mean().item() for r in reward_vals]
        results[name] = {'alr': log_post, 'raw_ir': raw_ir}
        print(f'  {name}: log_post={sum(log_post) / len(log_post):.3f}  '
              f'raw_ir={sum(raw_ir) / len(raw_ir):.3f}', flush=True)

os.makedirs('experiments', exist_ok=True)
out_path = 'experiments/prior_metrics.pt'
torch.save(results, out_path)
print(f'\nSaved to {out_path}', flush=True)
