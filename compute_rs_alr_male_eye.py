import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch, numpy as np

from model_loader import load_models
from models import classifier
from utils import compute_avg_log_reward

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

avg_log_reward = [compute_avg_log_reward(z.to(device), stylegan, [male_clf, eye_clf])
                  for z in d['samples']['RS']]
print(f'RS ALR: {np.mean(avg_log_reward):.4f}±{np.std(avg_log_reward, ddof=1):.4f}', flush=True)

d['avg_log_reward']['RS'] = avg_log_reward
torch.save(r, 'experiments/male_eye/results_merged.pt')
print('Saved.', flush=True)
