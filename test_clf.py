import sys
sys.path.insert(0, '/home/s2800722/dissertation/stylegan2-ada-pytorch')

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models import classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

celeba_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.CelebA(root='./data', split='valid', target_type='attr', transform=celeba_transforms, download=True)

torch.manual_seed(0)
idx = torch.randint(0, len(dataset), (10,))

smile_clf = classifier().to(device)
smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf.pth', map_location=device, weights_only=False))
smile_clf.eval()

imgs, probs, labels = [], [], []
with torch.no_grad():
    for i in tqdm(idx, 'Image'):
        img, attrs = dataset[i]
        imgs.append(img)
        labels.append(attrs[31].item())  # 31 = Smiling
        logit = smile_clf(img.unsqueeze(0).to(device)).squeeze()
        probs.append(torch.sigmoid(logit).item())

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for i in range(10):
    row, col = i // 5, i % 5
    img_show = ((imgs[i] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    axes[row, col].imshow(img_show)
    axes[row, col].axis('off')
    color = 'green' if probs[i] > 0.7 else 'red' if probs[i] < 0.3 else 'orange'
    gt = 'smiling' if labels[i] == 1 else 'not smiling'
    axes[row, col].set_title(f'pred: {probs[i]:.2f}\ngt: {gt}', color=color)

plt.suptitle('Random CelebA samples + smile_clf scores (with ground truth)')
plt.tight_layout()
plt.savefig('test_celeba_real_smile.png')
print('Saved figure to test_celeba_real_smile.png')
