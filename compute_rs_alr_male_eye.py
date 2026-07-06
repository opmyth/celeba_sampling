import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import torch
import torch.nn.functional as F
import numpy as np

from model_loader import load_models
from models import classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('smile', device)
stylegan.eval()

male_clf = classifier().to(device)
male_clf.load_state_dict(torch.load('clf_checkpoints/male_clf_aug.pth', weights_only=False, map_location=device))
male_clf.eval()

eye_clf = classifier().to(device)
eye_clf.load_state_dict(torch.load('clf_checkpoints/eyeglasses_clf_aug.pth', weights_only=False, map_location=device))
eye_clf.eval()

r = torch.load('experiments/male_eye/results_merged.pt', weights_only=False, map_location='cpu')
d = r['male_eye']

rs_alr = []
for t, z in enumerate(d['samples']['RS']):
    with torch.no_grad():
        imgs = stylegan(z.to(device))
        lp = (F.logsigmoid(male_clf(imgs)) + F.logsigmoid(eye_clf(imgs))).squeeze().mean().item()
    rs_alr.append(lp)
    print(f'  trial {t+1}: ALR={lp:.4f}', flush=True)

print(f'\nRS ALR: {np.mean(rs_alr):.4f}±{np.std(rs_alr, ddof=1):.4f}', flush=True)

d['avg_log_reward']['RS'] = rs_alr
torch.save(r, 'experiments/male_eye/results_merged.pt')
print('Saved.', flush=True)
