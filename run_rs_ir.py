import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import numpy as np
from tqdm import tqdm

from model_loader import load_models
from utils import load_imagereward, tokenize_prompt, log_posterior_ir

PROMPT     = "a bald man"
BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains', type=int, default=100)
parser.add_argument('--n_trials', type=int, default=5)
parser.add_argument('--seed',     type=int, default=321)
parser.add_argument('--output_path', type=str, default='experiments/bald_ir/results_rs_ir.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
stylegan.G = torch.compile(stylegan.G)
reward_model  = load_imagereward(device)
print('Models loaded.', flush=True)

prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, BATCH_SIZE)
torch.manual_seed(args.seed)

# Estimate r_max = max IR score from 2000 prior samples. Used as the RS upper bound:
# accept prob = exp(IR(z) - r_max) <= 1 for all z.
print('Estimating r_max from 2000 prior samples...', flush=True)
scan_scores = []
for _ in range(2000 // BATCH_SIZE):
    z_scan = torch.randn(BATCH_SIZE, stylegan.latent_dim, device=device)
    log_p  = log_posterior_ir(z_scan, stylegan, reward_model, prompt_ids, prompt_mask)
    scan_scores.append((log_p + 0.5 * (z_scan**2).sum(1)).cpu())
r_max = torch.cat(scan_scores).max().item() + 0.5   # +0.5 headroom
print(f'r_max = {r_max:.4f}', flush=True)

n_needed = args.n_chains * args.n_trials
accepted  = []
attempted = 0

pbar = tqdm(total=n_needed, desc='RS-IR')
while len(accepted) < n_needed:
    z_batch = torch.randn(BATCH_SIZE, stylegan.latent_dim, device=device)
    log_p   = log_posterior_ir(z_batch, stylegan, reward_model, prompt_ids, prompt_mask)
    ir      = log_p + 0.5 * (z_batch**2).sum(1)      # recover raw IR score from log_p
    log_acc = (ir - r_max).clamp(max=0)               # log exp(IR - r_max)
    accept  = torch.log(torch.rand(BATCH_SIZE, device=device)) <= log_acc
    accepted.extend(z_batch[accept].cpu().unbind(0))
    attempted += BATCH_SIZE
    pbar.update(accept.sum().item())
pbar.close()

accepted = torch.stack(accepted[:n_needed])
samples_list = list(torch.chunk(accepted, args.n_trials, dim=0))

accept_rate = n_needed / attempted
print(f'RS-IR done. Accept rate: {accept_rate:.3f}  ({n_needed}/{attempted})', flush=True)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save({'samples': samples_list, 'accept_rate': accept_rate, 'prompt': PROMPT,
            'r_max': r_max}, args.output_path)
print(f'Saved to {args.output_path}')
