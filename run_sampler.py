import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch, time
import numpy as np
import torch.nn.functional as F
import wandb

from model_loader import load_models
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba
from utils import compute_w2, compute_diversity, compute_male_fraction, compute_diversity_cov

SAMPLER_FNS = {
    'ULA': latent_ULA_celeba,
    'MALA': latent_MALA_celeba,
    'G_MH': latent_Gaussian_MH_celeba,
}

parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type=str, required=True)
parser.add_argument('--sampler', type=str, required=True, choices=['ULA', 'MALA', 'G_MH'])
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_steps', type=int, default=800)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--burnin', type=int, default=0,
                    help='Steps to discard before collecting samples (thinning mode only)')
parser.add_argument('--thin_k', type=int, default=1,
                    help='Keep 1 every thin_k post-burnin steps (thinning mode only)')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--init', type=str, default='random', choices=['random', 'cold', 'warm'])
parser.add_argument('--rs_path', type=str, default='results_rs.pt')
parser.add_argument('--output_path', type=str, default='results_sampler.pt')
args = parser.parse_args()

wandb.init(
    project="dissertation-stylegan-sampling",
    name=f"{args.sampler}-{args.clf_name}-{args.init}-chains{args.n_chains}-trials{args.n_trials}",
    group=f"{args.clf_name}_{args.init}_n{args.n_chains}_t{args.n_trials}",
    job_type=args.sampler,
    config=vars(args),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

stylegan, clf, male_clf = load_models(args.clf_name, device)
print('Compiling models...', flush=True)
clf = torch.compile(clf)
male_clf = torch.compile(male_clf)
stylegan.G = torch.compile(stylegan.G)
print('Done.', flush=True)
torch.manual_seed(args.seed)

def get_init_z(init_type, n_chains, model, clf, device, n_candidates=10000, batch_size=512):
    if init_type == 'random':
        return None
    print(f'Computing {init_type} init from {n_candidates} candidates...', flush=True)
    scores, zs = [], []
    for start in range(0, n_candidates, batch_size):
        size = min(batch_size, n_candidates - start)
        z_batch = torch.randn(size, model.latent_dim, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(clf(model(z_batch))).squeeze()
        scores.append(probs.cpu())
        zs.append(z_batch.cpu())
    all_scores = torch.cat(scores)
    all_z = torch.cat(zs)
    idx = torch.argsort(all_scores)
    selected = idx[:n_chains] if init_type == 'cold' else idx[-n_chains:]
    print(f'  score range of selected: [{all_scores[selected].min():.4f}, {all_scores[selected].max():.4f}]', flush=True)
    return all_z[selected]

z_init_tensor = get_init_z(args.init, args.n_chains, stylegan, clf, device)

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
    return torch.cat(results, dim=0)

HAS_ACCEPT = {'MALA', 'G_MH'}

t = time.time()
accept_rates = []
z_final_list = []
if args.burnin == 0 and args.thin_k == 1:
    all_samples = run_sampler_minibatched(sampler_fn, stylegan, clf, args.n_chains * args.n_trials, args.n_steps, param, batch_size=args.batch_size, device=device)
    samples_list = list(torch.chunk(all_samples, args.n_trials, dim=0))
else:
    kept_per_chain = (args.n_steps - args.burnin) // args.thin_k
    print(f"Thinning: {args.n_chains} chains x {args.n_steps} steps, burnin={args.burnin}, thin_k={args.thin_k} -> {kept_per_chain} samples/chain -> {args.n_chains * kept_per_chain} total/trial", flush=True)
    samples_list = []
    z_final_list = []
    for trial in range(args.n_trials):
        torch.manual_seed(args.seed + trial)
        result = sampler_fn(stylegan, clf, args.n_chains, args.n_steps, param, device=device,
                            burnin=args.burnin, thin_k=args.thin_k, return_diagnostics=True,
                            z_init=z_init_tensor)
        if args.sampler in HAS_ACCEPT:
            chain, accept_rate, _ = result
            accept_rates.append(accept_rate)
            print(f"  trial {trial+1}/{args.n_trials} accept_rate={accept_rate:.1%}", flush=True)
        else:
            chain, _ = result
        z_final_list.append(chain[-1].cpu())
        samples_list.append(torch.cat(chain, dim=0))
print(f"{args.sampler} done: {time.time()-t:.2f}s", flush=True)
if accept_rates:
    print(f"Mean acceptance rate: {np.mean(accept_rates):.1%}", flush=True)

t = time.time()
w2_values = [compute_w2(samples_list[i], rs_samples_list[i]) for i in range(args.n_trials)]
print(f"{args.sampler} W2 done: {time.time()-t:.2f}s", flush=True)

t = time.time()
with torch.no_grad():
    avg_log_reward = [F.logsigmoid(clf(stylegan(samples_list[i]))).mean().item() for i in range(args.n_trials)]
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

expr_dir = os.path.dirname(os.path.abspath(args.output_path))
if z_init_tensor is not None:
    torch.save(z_init_tensor, os.path.join(expr_dir, 'z_init.pt'))
    print(f'z_init saved to {expr_dir}/z_init.pt', flush=True)
if z_final_list:
    z_final = z_final_list[-1]
    torch.save(z_final, os.path.join(expr_dir, f'z_final_{args.sampler.lower()}.pt'))
    print(f'z_final saved to {expr_dir}/z_final_{args.sampler.lower()}.pt', flush=True)
else:
    z_final = None

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
}, args.output_path)

print(f'{args.sampler} results saved to {args.output_path}')
wandb.finish()
