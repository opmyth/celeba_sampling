import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
from model_loader import load_models

# stub functions removed from transformers.modeling_utils in newer versions
import torch, transformers.modeling_utils as _mu

def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    if chunk_size > 0:
        chunks = [t.split(chunk_size, dim=chunk_dim) for t in input_tensors]
        return torch.cat([forward_fn(*c) for c in zip(*chunks)], dim=chunk_dim)
    return forward_fn(*input_tensors)

def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

def _prune_linear_layer(layer, index, dim=0):
    W = layer.weight.index_select(dim, index).clone().detach()
    b = layer.bias.index_select(0, index).clone().detach() if layer.bias is not None else None
    new = torch.nn.Linear(W.size(1), W.size(0), bias=b is not None).to(layer.weight.device)
    new.weight, new.bias = torch.nn.Parameter(W), torch.nn.Parameter(b) if b is not None else None
    return new

for _name, _fn in [('apply_chunking_to_forward', _apply_chunking_to_forward),
                   ('find_pruneable_heads_and_indices', _find_pruneable_heads_and_indices),
                   ('prune_linear_layer', _prune_linear_layer)]:
    if not hasattr(_mu, _name):
        setattr(_mu, _name, _fn)

import ImageReward as RM

PROMPT = "a bald man"
N      = 10
SEED   = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
print('StyleGAN2 loaded.', flush=True)

reward_model = RM.load("ImageReward-v1.0", device=str(device))
reward_model.eval()
print('ImageReward loaded.', flush=True)

# BLIP preprocessing constants
BLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
BLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

# Tokenize prompt once
text_input = reward_model.blip.tokenizer(
    PROMPT, padding='max_length', truncation=True, max_length=35, return_tensors='pt'
).to(device)
prompt_ids   = text_input.input_ids.expand(N, -1)
prompt_mask  = text_input.attention_mask.expand(N, -1)

# Sample z and decode
torch.manual_seed(SEED)
z    = torch.randn(N, stylegan.latent_dim, device=device)
imgs = stylegan.G(z, None)                                     # (N, 3, 256, 256) in [-1, 1]

# Preprocess for BLIP — keep tensor, no PIL
imgs_blip = F.interpolate((imgs + 1) / 2, size=(224, 224), mode='bicubic', align_corners=False)
imgs_blip = (imgs_blip - BLIP_MEAN) / BLIP_STD

# Score — score_gard keeps the computation graph
scores = reward_model.score_gard(prompt_ids, prompt_mask, imgs_blip).squeeze(-1)

print(f'\nPrompt : "{PROMPT}"')
print(f'N      : {N} random z vectors (seed={SEED})')
print(f'Mean   : {scores.mean().item():.4f}')
print(f'Std    : {scores.std().item():.4f}')
print(f'Min    : {scores.min().item():.4f}')
print(f'Max    : {scores.max().item():.4f}')
print(f'Scores : {[round(s, 3) for s in scores.tolist()]}')

# Save grid of images with scores as titles
import matplotlib.pyplot as plt
import numpy as np

imgs_np = ((imgs.detach().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs_np[i])
    ax.axis('off')
    s = scores[i].item()
    ax.set_title(f'{s:.3f}', fontsize=10, color='green' if s > 0 else 'red')
fig.suptitle(f'ImageReward scores — "{PROMPT}"', fontsize=12)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'imagereward_probe.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'\nSaved grid to {os.path.abspath(out)}')
