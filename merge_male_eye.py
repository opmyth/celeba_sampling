import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import torch
import numpy as np
from model_loader import load_models
from models import classifier
from utils import compute_w2, compute_diversity_cov, compute_avg_log_reward

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

stylegan, _, _ = load_models('smile', device)
stylegan.eval()

male_clf = classifier().to(device)
male_clf.load_state_dict(torch.load('clf_checkpoints/male_clf_aug.pth', weights_only=False, map_location=device))
male_clf.eval()

eye_clf = classifier().to(device)
eye_clf.load_state_dict(torch.load('clf_checkpoints/eyeglasses_clf_aug.pth', weights_only=False, map_location=device))
eye_clf.eval()

merged = torch.load('experiments/male_eye/results_merged.pt', weights_only=False)
d = merged['male_eye']

alr = {}
for s in ['RS', 'MALA', 'G_MH']:
    alr[s] = [compute_avg_log_reward(z.to(device), stylegan, [male_clf, eye_clf]) for z in d['samples'][s]]
    print(f'{s} ALR: {np.mean(alr[s]):.4f}±{np.std(alr[s], ddof=1):.4f}')

d['avg_log_reward'] = alr
torch.save(merged, 'experiments/male_eye/results_merged.pt')
print('Saved.')
