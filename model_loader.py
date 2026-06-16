import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

import pickle
import torch
from models import classifier, StyleGAN2Wrapper

def load_models(device):
    smile_clf = classifier().to(device)
    smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf_aug.pth', weights_only=False))
    smile_clf.eval()

    male_clf = classifier().to(device)
    male_clf.load_state_dict(torch.load('clf_checkpoints/male_clf_aug.pth', weights_only=False))
    male_clf.eval()

    with open('stylegan2_checkpoints/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)

    stylegan = StyleGAN2Wrapper(G).to(device)
    stylegan.eval()

    return stylegan, smile_clf, male_clf
