import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

celeba_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_dataset = datasets.CelebA(root='./data', split='valid', target_type='attr', transform=celeba_transforms, download=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

def evaluate(model, loader, attr, device):
    model.to(device)
    model.eval()
    n_samples, correct, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for x, attrs in tqdm(loader, desc='Validation'):
            x = x.to(device)
            y = attrs[:, attr].float().to(device)
            logits = model(x).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = (logits > 0).long()
            correct += (pred == y.long()).sum().item()
            n_samples += x.size(0)

    print(f"val_acc = {correct/n_samples:.3f} | val_loss = {total_loss/n_samples:.4f}")

smile_clf = classifier()
smile_clf.load_state_dict(torch.load('clf_checkpoints/smile_clf.pth', weights_only=False))
print("Smile classifier:")
evaluate(smile_clf, val_loader, attr=31, device=device)

male_clf = classifier()
male_clf.load_state_dict(torch.load('clf_checkpoints/male_clf.pth', weights_only=False))
print("Male classifier:")
evaluate(male_clf, val_loader, attr=20, device=device)
