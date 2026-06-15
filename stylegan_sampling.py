import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

import pickle
import torch
import matplotlib.pyplot as plt

from models import StyleGAN2Wrapper, classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

with open('stylegan2_checkpoints/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

model = StyleGAN2Wrapper(G).to(device)
model.eval()

smile_clf = classifier().to(device)
smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf.pth', map_location=device, weights_only=False))
smile_clf.eval()

n_samples = 10
torch.manual_seed(42)
z = torch.randn(n_samples, model.latent_dim, device=device)

with torch.no_grad():
    imgs = model(z)
    imgs_show = ((imgs + 1) / 2).clamp(0, 1)
    probs = torch.sigmoid(smile_clf(imgs)).squeeze().cpu().numpy()

print('Smile probabilities:', probs)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(n_samples):
    row, col = i // 5, i % 5
    axes[row, col].imshow(imgs_show[i].permute(1, 2, 0).cpu().numpy())
    axes[row, col].axis('off')
    color = 'green' if probs[i] > 0.7 else 'red' if probs[i] < 0.3 else 'orange'
    axes[row, col].set_title(f'{probs[i]:.2f}', color=color)

plt.suptitle('StyleGAN2 prior samples (no MCMC) + smile_clf scores')
plt.tight_layout()
plt.savefig('test_stylegan_prior_smile.png')
print('Saved figure to test_stylegan_prior_smile.png')
