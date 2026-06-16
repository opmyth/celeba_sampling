import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from models import classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('attr', type=int)
parser.add_argument('attr_name', type=str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--subset_size', type=int, default=20000)
args = parser.parse_args()

def train_classifier(model, train_loader, attr, device, epochs=5, lr=1e-3):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in tqdm(range(epochs), desc='Training epochs'):
        print(f"epoch: {ep+1} running!")
        model.train()
        n_samples, correct, total_loss = 0, 0, 0.0
        for x, attrs in tqdm(train_loader, desc='Training steps'):
            x = x.to(device)
            y = attrs[:, attr].float().to(device)
            logits = model(x).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * x.size(0)
            pred = (logits > 0).long()
            correct += (pred == y.long()).sum().item()
            n_samples += x.size(0)
        print(f"epoch: {ep+1}/{epochs} | train_acc = {correct/n_samples:.3f} | loss: {total_loss/n_samples:.4f}")

celeba_train_transforms_aug = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

celeba_train_transforms_plain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

celeba_transforms = celeba_train_transforms_aug if args.augment else celeba_train_transforms_plain

print('Downloading the dataset')
celeba_dataset = datasets.CelebA(root='./data', split='train', target_type='attr', transform=celeba_transforms, download=True)

rng = np.random.default_rng(42)
indices = rng.choice(len(celeba_dataset), size=args.subset_size, replace=False)
celeba_dataset = Subset(celeba_dataset, indices)

train_loader = DataLoader(celeba_dataset, batch_size=256, shuffle=True)

clf = classifier().to(device)
train_classifier(clf, train_loader, args.attr, device)

suffix = '_aug' if args.augment else ''
torch.save(clf.state_dict(), f'./clf_checkpoints/{args.attr_name}_clf{suffix}.pth')
print(f"Saved to ./clf_checkpoints/{args.attr_name}_clf{suffix}.pth")
