import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")

import argparse, torch, time
import numpy as np
import wandb

import rng as rng_mod
from config import EXPERIMENTS
from model_loader import load_models
from posteriors import classifier_posterior, imagereward_posterior, load_r_max
from samplers import rejection_sampling
from utils import (compute_w2, compute_diversity, compute_diversity_cov,
                    compute_male_fraction, load_imagereward)

def _slug(s):
    return s.lower().replace(' ', '_')


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--n_trials', type=int, default=None)
parser.add_argument('--prompt', type=str, default=None,
                     help='override the config default prompt (imagereward experiments only)')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

from utils import maybe_enable_tf32
maybe_enable_tf32()

cfg = EXPERIMENTS[args.experiment]
n_trials = args.n_trials or cfg.n_trials
prompt = args.prompt or cfg.prompt
# imagereward experiments nest by prompt so different prompts' runs never
# collide (bald_ir has 3) - classifier experiments are unaffected (no prompt).
expr_dir = f'experiments/{args.experiment}'
if cfg.kind == 'imagereward':
    expr_dir = os.path.join(expr_dir, f'prompt_{_slug(prompt)}')
output_path = args.output_path or os.path.join(expr_dir, 'results_rs.pt')

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"RS-{args.experiment}-trials{n_trials}",
    group=f"{args.experiment}_t{n_trials}",
    job_type="RS",
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, clfs, male_clf = load_models(cfg.clf_names or [], device)

if cfg.kind == 'classifier':
    posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
    r_max = None
else:
    reward_model = load_imagereward(device)
    r_max = load_r_max(prompt)
    posterior = imagereward_posterior(stylegan, reward_model, prompt, device, r_max)
    print(f'r_max = {r_max:.4f}', flush=True)

t = time.time()
generator = rng_mod.make_generator(args.seed, device)
RS_samples, accept_rate = rejection_sampling(
    posterior, cfg.rs_target * n_trials * 2, stylegan.latent_dim, device, generator=generator)
print(f"RS done: {time.time()-t:.2f}s  accept_rate={accept_rate:.4f}", flush=True)

chunks = torch.chunk(RS_samples, n_trials * 2, dim=0)
w2_baseline = [compute_w2(chunks[2 * i], chunks[2 * i + 1]) for i in range(n_trials)]
rs_samples_list = list(chunks[::2])

avg_log_reward = [posterior.reward_only_fn(z.to(device)).mean().item() for z in rs_samples_list]
diversity = [compute_diversity(z) for z in rs_samples_list]
diversity_trace_cov = [compute_diversity_cov(z) for z in rs_samples_list]
male_fraction = [compute_male_fraction(stylegan, male_clf, z.to(device)) for z in rs_samples_list]

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
    "accept_rate": accept_rate,
})

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save({
    'samples': rs_samples_list,
    'w2_baseline': w2_baseline,
    'avg_log_reward': avg_log_reward,
    'diversity': diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction': male_fraction,
    'accept_rate': accept_rate,
    'prompt': prompt if cfg.kind == 'imagereward' else None,
    'r_max': r_max,
}, output_path)

print(f'RS results saved to {output_path}')
wandb.finish()
