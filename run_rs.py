import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")

import argparse, torch, time
import numpy as np
import torch.nn.functional as F
import wandb

from model_loader import load_models
from samplers import rejection_sampling
from utils import compute_w2, compute_diversity, compute_male_fraction, compute_diversity_cov

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--output_path', type=str, default='results_rs.pt')
args = parser.parse_args()

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"RS-chains{args.n_chains}-trials{args.n_trials}",
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, smile_clf, male_clf = load_models(device)
torch.manual_seed(args.seed)

t = time.time()
RS_samples = rejection_sampling(stylegan, smile_clf, args.n_chains * args.n_trials * 2, device=device)
print(f"RS done: {time.time()-t:.2f}s", flush=True)

t = time.time()
chunks = torch.chunk(RS_samples, args.n_trials * 2, dim=0)
w2_baseline = [compute_w2(chunks[2*i], chunks[2*i+1]) for i in range(args.n_trials)]
rs_samples_list = list(chunks[::2])
print(f"RS W2 baseline done: {time.time()-t:.2f}s", flush=True)

t = time.time()
with torch.no_grad():
    avg_log_reward = [F.logsigmoid(smile_clf(stylegan(rs_samples_list[i]))).mean().item() for i in range(args.n_trials)]
print(f"avg_log_reward done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity = [compute_diversity(stylegan, rs_samples_list[i]) for i in range(args.n_trials)]
print(f"diversity done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity_trace_cov = [compute_diversity_cov(rs_samples_list[i]) for i in range(args.n_trials)]
print(f"diversity_trace_cov done: {time.time()-t:.2f}s", flush=True)

t = time.time()
male_fraction = [compute_male_fraction(stylegan, male_clf, rs_samples_list[i]) for i in range(args.n_trials)]
print(f"male_fraction done: {time.time()-t:.2f}s", flush=True)

wandb.log({
    "w2_baseline_mean": np.mean(w2_baseline),
    "w2_baseline_std": np.std(w2_baseline, ddof=1),
    "avg_log_reward_mean": np.mean(avg_log_reward),
    "avg_log_reward_std": np.std(avg_log_reward, ddof=1),
    "diversity_mean": np.mean(diversity),
    "diversity_std": np.std(diversity, ddof=1),
    "diversity_trace_cov_mean": np.mean(diversity_trace_cov),
    "diversity_trace_cov_std": np.std(diversity_trace_cov, ddof=1),
    "male_fraction_mean": np.mean(male_fraction),
    "male_fraction_std": np.std(male_fraction, ddof=1),
})

torch.save({
    'samples': rs_samples_list,
    'w2_baseline': w2_baseline,
    'avg_log_reward': avg_log_reward,
    'diversity': diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction': male_fraction,
}, args.output_path)

print(f'RS results saved to {args.output_path}')
wandb.finish()
