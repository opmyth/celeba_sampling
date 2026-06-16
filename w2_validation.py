import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

print("imports....")
from samplers import rejection_sampling
from utils import compute_w2, compute_w2
from models import classifier, StyleGAN2Wrapper

import pickle
import torch
torch.manual_seed(42)

print("device....")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

print("loading the classifier....")
smile_clf = classifier().to(device)
smile_clf.load_state_dict(torch.load('./clf_checkpoints/smile_clf.pth', map_location=device, weights_only=False))
smile_clf.eval()

print("loading stylegan checkpoint....")
with open('stylegan2_checkpoints/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
    
stylegan = StyleGAN2Wrapper(G).to(device)
stylegan.eval()

print("getting RS samples....")
# get rejection sampling samples
rs_1 = rejection_sampling(stylegan, smile_clf, 1000, device)
rs_2 = rejection_sampling(stylegan, smile_clf, 1000, device)

print("computing w2 (exact)....")
#compute w2 (exact)
rs_w2 = compute_w2(rs_1, rs_2)

w2_slices = {50: None, 100: None, 150: None, 200: None, 250: None, 300: None, 350: None, 400: None, 450: None, 500: None}

print("computing sliced_w2....")
for slice in w2_slices:
    w2_slices[slice] = compute_w2(rs_1, rs_2, slice)

print(f"The exact W2: {rs_w2}")

for k, v in w2_slices.items():
    print(f"# slices: {k} ==> sliced_w2: {v}")


import json

results = {
    'exact_w2': rs_w2,
    'sliced_w2': w2_slices
}

with open('w2_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
