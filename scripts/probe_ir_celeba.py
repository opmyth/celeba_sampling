import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from types import ModuleType as _ModuleType
import sys as _sys
_sys.modules['ImageReward.ReFL'] = _ModuleType('ImageReward.ReFL')

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import ImageReward as RM

PROMPT = "a bald man"
N      = 10
BALD   = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

reward_model = RM.load("ImageReward-v1.0", device=str(device))
reward_model.eval()
print('ImageReward loaded.', flush=True)

BLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
BLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

dataset = datasets.CelebA(root='./data', split='all', target_type='attr',
                          transform=transforms.ToTensor(), download=False)

# grab first N images where bald=1
bald_imgs = []
for img, attrs in dataset:
    if attrs[BALD] == 1:
        bald_imgs.append(img)
    if len(bald_imgs) == N:
        break

imgs = torch.stack(bald_imgs).to(device)   # (N, 3, H, W) in [0, 1]

# tokenize prompt
text_input = reward_model.blip.tokenizer(
    PROMPT, padding='max_length', truncation=True, max_length=35, return_tensors='pt'
).to(device)
prompt_ids  = text_input.input_ids.expand(N, -1)
prompt_mask = text_input.attention_mask.expand(N, -1)

imgs_blip = F.interpolate(imgs, size=(224, 224), mode='bicubic', align_corners=False)
imgs_blip = (imgs_blip - BLIP_MEAN) / BLIP_STD

scores = reward_model.score_gard(prompt_ids, prompt_mask, imgs_blip).squeeze(-1)

print(f'\nPrompt : "{PROMPT}"')
print(f'N      : {N} real CelebA images (bald=1)')
print(f'Mean   : {scores.mean().item():.4f}')
print(f'Std    : {scores.std().item():.4f}')
print(f'Min    : {scores.min().item():.4f}')
print(f'Max    : {scores.max().item():.4f}')
print(f'Scores : {[round(s, 3) for s in scores.tolist()]}')
print(f'\nRandom prior baseline: mean=-1.85, range=[-2.25, -0.33]')
