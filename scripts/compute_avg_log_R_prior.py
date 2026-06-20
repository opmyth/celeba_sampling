import sys, os
sys.path.insert(0, os.path.abspath('/home/s2800722/dissertation/stylegan2-ada-pytorch'))

import torch
from tqdm import tqdm
from models import StyleGAN2Wrapper
import torch.nn.functional as F
import pickle

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cuda'
print(f'device being used: {device}')
with open('stylegan2_checkpoints/stylegan2_celeba.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

stylegan = StyleGAN2Wrapper(G).to(device)
stylegan.eval()


from models import classifier

smile_clf = classifier().to(device)
smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf_aug.pth', map_location=device, weights_only=False))
smile_clf.eval()

import torch
import numpy as np

n_trials = 5
n_chains = 1000
log_rewards = []

with torch.no_grad():
    for trial in tqdm(range(n_trials), desc='Trials'):
        z = torch.randn(n_chains, 512).to(device)
        imgs = stylegan(z)
        avg_log_r = F.logsigmoid(smile_clf(imgs)).mean().item()
        log_rewards.append(avg_log_r)

mean_lr = np.mean(log_rewards)
std_lr = np.std(log_rewards, ddof=1)
print(f'avg log reward across {n_trials} trials: {mean_lr:.4f} ± {std_lr:.4f}')
print(log_rewards)
