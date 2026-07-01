import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import compute_w2

SIGMA    = 2.38 / (512 ** 0.5)
N_CHAINS = 100
N_STEPS  = 3000
BURNIN   = 1000
THIN_K   = 200

parser = argparse.ArgumentParser()
parser.add_argument('--n_trials',    type=int, default=5)
parser.add_argument('--seed',        type=int, default=321)
parser.add_argument('--rs_path',     type=str, default='experiments/male_eye/results_rs.pt')
parser.add_argument('--output_path', type=str, default='experiments/male_eye/results_gmh.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  sigma={SIGMA:.4f}', flush=True)

stylegan, clf_eye, clf_male = load_models('eyeglasses', device)
clf_eye    = torch.compile(clf_eye)
clf_male   = torch.compile(clf_male)
stylegan.G = torch.compile(stylegan.G)
print('Models loaded.', flush=True)

rs_data         = torch.load(args.rs_path, weights_only=False)
rs_samples_list = rs_data['samples']

def log_p(z):
    with torch.no_grad():
        imgs = stylegan.G(z, None)
        return (-0.5 * (z ** 2).sum(1)
                + F.logsigmoid(clf_male(imgs).squeeze(-1))
                + F.logsigmoid(clf_eye(imgs).squeeze(-1)))

samples_list, w2_values, avg_log_reward, accept_rates = [], [], [], []

for trial in range(args.n_trials):
    print(f'\n=== Trial {trial+1}/{args.n_trials} ===', flush=True)
    torch.manual_seed(args.seed + trial)

    z       = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    log_p_z = log_p(z)
    n_accept = 0
    kept     = []

    for step in tqdm(range(1, N_STEPS + 1), desc=f'G_MH trial {trial+1}'):
        z_prop     = z + SIGMA * torch.randn_like(z)
        log_p_prop = log_p(z_prop)
        log_alpha  = (log_p_prop - log_p_z).clamp(max=0)
        accept     = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask       = accept.unsqueeze(1)
        z          = torch.where(mask, z_prop,      z)
        log_p_z    = torch.where(accept, log_p_prop, log_p_z)
        n_accept  += accept.sum().item()
        if step > BURNIN and (step - BURNIN) % THIN_K == 0:
            kept.append(z.detach().cpu())

    accept_rate = n_accept / (N_STEPS * N_CHAINS)
    accept_rates.append(accept_rate)
    trial_samples = torch.cat(kept, dim=0)
    samples_list.append(trial_samples)
    w2_values.append(compute_w2(trial_samples, rs_samples_list[trial]))
    avg_log_reward.append(log_p(trial_samples.to(device)).mean().item())
    print(f'  accept={accept_rate:.1%}  W2={w2_values[-1]:.4f}  avg_log_p={avg_log_reward[-1]:.4f}', flush=True)

print(f'\nMean accept: {np.mean(accept_rates):.1%}')
print(f'Mean W2:     {np.mean(w2_values):.4f}')

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save({'samples': samples_list, 'w2_values': w2_values,
            'avg_log_reward': avg_log_reward, 'accept_rates': accept_rates,
            'sigma': SIGMA}, args.output_path)
print(f'Saved to {args.output_path}')
