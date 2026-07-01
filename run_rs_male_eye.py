import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import torch.nn.functional as F
from tqdm import tqdm

from model_loader import load_models

BATCH_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--n_chains',    type=int, default=100)
parser.add_argument('--n_trials',    type=int, default=5)
parser.add_argument('--seed',        type=int, default=321)
parser.add_argument('--output_path', type=str, default='experiments/male_eye/results_rs.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, clf_eye, clf_male = load_models('eyeglasses', device)
clf_eye    = torch.compile(clf_eye)
clf_male   = torch.compile(clf_male)
stylegan.G = torch.compile(stylegan.G)
print('Models loaded.', flush=True)

torch.manual_seed(args.seed)

n_needed = args.n_chains * args.n_trials
accepted  = []
attempted = 0

# Accept prob = σ(clf_male) × σ(clf_eye) ≤ 1, so M=1 and no normalization needed.
pbar = tqdm(total=n_needed, desc='RS male+eye')
while len(accepted) < n_needed:
    z_batch = torch.randn(BATCH_SIZE, stylegan.latent_dim, device=device)
    with torch.no_grad():
        imgs = stylegan.G(z_batch, None)
        log_acc = (F.logsigmoid(clf_male(imgs).squeeze(-1))
                 + F.logsigmoid(clf_eye(imgs).squeeze(-1)))
    accept = torch.log(torch.rand(BATCH_SIZE, device=device)) <= log_acc
    accepted.extend(z_batch[accept].cpu().unbind(0))
    attempted += BATCH_SIZE
    pbar.update(accept.sum().item())
pbar.close()

accepted     = torch.stack(accepted[:n_needed])
samples_list = list(torch.chunk(accepted, args.n_trials, dim=0))
accept_rate  = n_needed / attempted
print(f'RS done. Accept rate: {accept_rate:.4f}  ({n_needed}/{attempted})', flush=True)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save({'samples': samples_list, 'accept_rate': accept_rate}, args.output_path)
print(f'Saved to {args.output_path}')
