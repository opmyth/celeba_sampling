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
from init_scan import get_init_z
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba
from utils import (compute_w2, compute_diversity, compute_diversity_cov,
                    compute_male_fraction, load_imagereward)

SAMPLER_FNS = {
    'ULA': latent_ULA_celeba,
    'MALA': latent_MALA_celeba,
    'G_MH': latent_Gaussian_MH_celeba,
}


def _slug(s):
    return s.lower().replace(' ', '_')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
parser.add_argument('--sampler', required=True, choices=['ULA', 'MALA', 'G_MH'])
parser.add_argument('--init', type=str, default='random', choices=['random', 'cold', 'warm'])
parser.add_argument('--prompt', type=str, default=None,
                     help='override the config default prompt (imagereward experiments only)')
parser.add_argument('--n_chains', type=int, default=None)
parser.add_argument('--n_trials', type=int, default=None)
parser.add_argument('--n_steps', type=int, default=None)
parser.add_argument('--dt', type=float, default=None)
parser.add_argument('--sigma', type=float, default=None)
parser.add_argument('--burnin', type=int, default=None)
parser.add_argument('--thin_k', type=int, default=None)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--rs_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

cfg = EXPERIMENTS[args.experiment]
if args.sampler not in cfg.samplers:
    raise ValueError(f"'{args.experiment}' does not run {args.sampler} "
                      f"(configured samplers: {cfg.samplers})")

n_chains = args.n_chains or cfg.n_chains
n_trials = args.n_trials or cfg.n_trials
n_steps  = args.n_steps or cfg.n_steps
burnin   = args.burnin if args.burnin is not None else cfg.burnin
thin_k   = args.thin_k or cfg.thin_k
prompt   = args.prompt or cfg.prompt
# imagereward experiments nest by prompt so different prompts' runs never
# collide (bald_ir has 3) - classifier experiments are unaffected (no prompt).
prompt_dir = f'experiments/{args.experiment}'
if cfg.kind == 'imagereward':
    prompt_dir = os.path.join(prompt_dir, f'prompt_{_slug(prompt)}')
rs_path     = args.rs_path or os.path.join(prompt_dir, 'results_rs.pt')
output_path = args.output_path or os.path.join(prompt_dir, f'results_{args.sampler.lower()}.pt')

if args.sampler == 'MALA':
    param = args.dt if args.dt is not None else cfg.dt_mala
elif args.sampler == 'ULA':
    param = args.dt if args.dt is not None else cfg.dt_ula
else:
    param = args.sigma if args.sigma is not None else cfg.sigma_gmh

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"{args.sampler}-{args.experiment}-{args.init}-chains{n_chains}-trials{n_trials}",
    group=f"{args.experiment}_{args.init}_n{n_chains}_t{n_trials}",
    job_type=args.sampler,
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, clfs, male_clf = load_models(cfg.clf_names or [], device)

if cfg.kind == 'classifier':
    posterior = classifier_posterior(stylegan, [clfs[n] for n in cfg.clf_names])
else:
    reward_model = load_imagereward(device)
    posterior = imagereward_posterior(stylegan, reward_model, prompt, device, load_r_max(prompt))

z_init_tensor = None
if args.init != 'random':
    init_gen = rng_mod.make_generator(args.seed, device)
    z_init_tensor = get_init_z(args.init, posterior.reward_only_fn, n_chains,
                                stylegan.latent_dim, device, init_gen)

rs_data = torch.load(rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

sampler_fn = SAMPLER_FNS[args.sampler]

t = time.time()
samples_list, log_p_kept_list, accept_rates, z_final_list = [], [], [], []
for trial in range(n_trials):
    generator = rng_mod.make_generator(args.seed + trial, device)
    samples, log_p_kept, accept_rate, _ = sampler_fn(
        posterior, n_chains, n_steps, param, stylegan.latent_dim, device,
        generator=generator, burnin=burnin, thin_k=thin_k, z_init=z_init_tensor)

    if accept_rate is not None:
        accept_rates.append(accept_rate)
        print(f"  trial {trial + 1}/{n_trials} accept_rate={accept_rate:.1%}", flush=True)

    z_final_list.append(samples[-1].cpu())
    samples_list.append(torch.cat(samples, dim=0).cpu())
    log_p_kept_list.append(torch.cat(log_p_kept, dim=0).cpu())
print(f"{args.sampler} done: {time.time() - t:.2f}s", flush=True)
if accept_rates:
    print(f"Mean acceptance rate: {np.mean(accept_rates):.1%}", flush=True)

t = time.time()
w2_values = [compute_w2(samples_list[i], rs_samples_list[i]) for i in range(n_trials)]
print(f"{args.sampler} W2 done: {time.time() - t:.2f}s", flush=True)

# avg reward = log_p + 0.5||z||^2, derived from log_p already computed during
# sampling (log_p_kept_list) - no second forward pass through the model needed.
t = time.time()
avg_log_reward = [
    (log_p_kept_list[i] + 0.5 * (samples_list[i] ** 2).sum(1)).mean().item()
    for i in range(n_trials)
]
print(f"avg_log_reward done: {time.time() - t:.2f}s", flush=True)

t = time.time()
diversity = [compute_diversity(samples_list[i]) for i in range(n_trials)]
diversity_trace_cov = [compute_diversity_cov(samples_list[i]) for i in range(n_trials)]
print(f"diversity done: {time.time() - t:.2f}s", flush=True)

t = time.time()
male_fraction = [compute_male_fraction(stylegan, male_clf, samples_list[i].to(device)) for i in range(n_trials)]
print(f"male_fraction done: {time.time() - t:.2f}s", flush=True)

wandb_log = {
    "w2_values_mean": np.mean(w2_values),
    "w2_values_std": np.std(w2_values, ddof=1),
    "avg_log_reward_mean": np.mean(avg_log_reward),
    "avg_log_reward_std": np.std(avg_log_reward, ddof=1),
    "diversity_mean": np.mean(diversity),
    "diversity_std": np.std(diversity, ddof=1),
    "diversity_trace_cov_mean": np.mean(diversity_trace_cov),
    "diversity_trace_cov_std": np.std(diversity_trace_cov, ddof=1),
    "male_fraction_mean": np.mean(male_fraction),
    "male_fraction_std": np.std(male_fraction, ddof=1),
}
if accept_rates:
    wandb_log["accept_rate_mean"] = np.mean(accept_rates)
    wandb_log["accept_rate_std"] = np.std(accept_rates, ddof=1)
wandb.log(wandb_log)

expr_dir = os.path.dirname(os.path.abspath(output_path))
os.makedirs(expr_dir, exist_ok=True)
if z_init_tensor is not None:
    torch.save(z_init_tensor, os.path.join(expr_dir, 'z_init.pt'))
    print(f'z_init saved to {expr_dir}/z_init.pt', flush=True)

z_final = z_final_list[-1] if z_final_list else None
if z_final is not None:
    torch.save(z_final, os.path.join(expr_dir, f'z_final_{args.sampler.lower()}.pt'))
    print(f'z_final saved to {expr_dir}/z_final_{args.sampler.lower()}.pt', flush=True)

torch.save({
    'samples': samples_list,
    'w2_values': w2_values,
    'avg_log_reward': avg_log_reward,
    'diversity': diversity,
    'diversity_trace_cov': diversity_trace_cov,
    'male_fraction': male_fraction,
    'accept_rates': accept_rates if accept_rates else None,
    'z_init': z_init_tensor,
    'z_final': z_final,
    'prompt': prompt if cfg.kind == 'imagereward' else None,
    'param': param,
}, output_path)

print(f'{args.sampler} results saved to {output_path}')
wandb.finish()
