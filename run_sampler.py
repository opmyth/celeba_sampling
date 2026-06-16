import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

import warnings
warnings.filterwarnings("ignore")
import argparse, torch, time
import torch.nn.functional as F

from model_loader import load_models
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba
from utils import compute_w2, compute_diversity, compute_male_fraction, compute_diversity_cov

SAMPLER_FNS = {
    'ULA': latent_ULA_celeba,
    'MALA': latent_MALA_celeba,
    'G_MH': latent_Gaussian_MH_celeba,
}

parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, required=True, choices=['ULA', 'MALA', 'G_MH'])
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=800)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--rs_path', type=str, default='results_rs.pt')
parser.add_argument('--output_path', type=str, default='results_sampler.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, smile_clf, male_clf = load_models(device)
torch.manual_seed(args.seed)

rs_data = torch.load(args.rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

sampler_fn = SAMPLER_FNS[args.sampler]
param = args.sigma if args.sampler == 'G_MH' else args.dt

def run_sampler_minibatched(sampler_fn, model, clf, total_chains, n_steps, param, batch_size, device):
    results = []
    for start in range(0, total_chains, batch_size):
        size = min(batch_size, total_chains - start)
        out = sampler_fn(model, clf, size, n_steps, param, device=device)[-1]
        results.append(out)
        del out
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)

t = time.time()
all_samples = run_sampler_minibatched(sampler_fn, stylegan, smile_clf, args.n_chains * args.n_trials, args.n_steps, param, batch_size=args.batch_size, device=device)
print(f"{args.sampler} done: {time.time()-t:.2f}s", flush=True)

t = time.time()
samples_list = list(torch.chunk(all_samples, args.n_trials, dim=0))
w2_values = [compute_w2(samples_list[i], rs_samples_list[i]) for i in range(args.n_trials)]
print(f"{args.sampler} W2 done: {time.time()-t:.2f}s", flush=True)

t = time.time()
with torch.no_grad():
    avg_log_reward = [F.logsigmoid(smile_clf(stylegan(samples_list[i]))).mean().item() for i in range(args.n_trials)]
print(f"avg_log_reward done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity = [compute_diversity(stylegan, samples_list[i]) for i in range(args.n_trials)]
print(f"diversity done: {time.time()-t:.2f}s", flush=True)

t = time.time()
diversity_trace_cov = [compute_diversity_cov(samples_list[i]) for i in range(args.n_trials)]
print(f"diversity_trace_cov done: {time.time()-t:.2f}s", flush=True)

t = time.time()
male_fraction = [compute_male_fraction(stylegan, male_clf, samples_list[i]) for i in range(args.n_trials)]
print(f"male_fraction done: {time.time()-t:.2f}s", flush=True)

torch.save({
    'samples': samples_list,
    'w2_values': w2_values,
    'avg_log_reward': avg_log_reward,
    'diversity': diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction': male_fraction,
}, args.output_path)

print(f'{args.sampler} results saved to {args.output_path}')
