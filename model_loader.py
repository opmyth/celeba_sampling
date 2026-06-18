import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'stylegan2-ada-pytorch'))

import pickle
import torch
from models import classifier, StyleGAN2Wrapper

# Checkpoint filename exactly as downloaded from the StyleGAN2-ADA repo
_STYLEGAN_PKL = 'stylegan2_celeba.pkl'

def load_models(device):
    smile_clf = classifier().to(device)
    smile_clf.load_state_dict(torch.load(
        os.path.join(_HERE, 'clf_checkpoints', 'smile_clf_aug.pth'), weights_only=False))
    smile_clf.eval()

    male_clf = classifier().to(device)
    male_clf.load_state_dict(torch.load(
        os.path.join(_HERE, 'clf_checkpoints', 'male_clf_aug.pth'), weights_only=False))
    male_clf.eval()

    ckpt_path = os.path.join(_HERE, 'stylegan2_checkpoints', _STYLEGAN_PKL)
    with open(ckpt_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)

    stylegan = StyleGAN2Wrapper(G).to(device)
    stylegan.eval()
    stylegan.requires_grad_(False)
    smile_clf.requires_grad_(False)
    male_clf.requires_grad_(False)

    return stylegan, smile_clf, male_clf
