import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import numpy as np

from model_loader import load_models
from models import classifier
from utils import (compute_w2, compute_diversity, compute_diversity_cov,
                   load_imagereward, tokenize_prompt, _preprocess_for_blip)

N_TRIALS     = 5
N_PER_TRIAL  = 1000
BATCH_SIZE   = 64
PROMPT_IR    = "a bald man"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# load stylegan once
stylegan, _, _ = load_models('smile', device)
stylegan.eval()
print('StyleGAN loaded.', flush=True)

# load all classifiers
clfs = {}
for attr in ['smile', 'bald', 'eyeglasses', 'male']:
    clf = classifier().to(device)
    clf.load_state_dict(torch.load(
        f'clf_checkpoints/{attr}_clf_aug.pth', weights_only=False, map_location=device))
    clf.eval()
    clfs[attr] = clf
print('Classifiers loaded.', flush=True)

# load ImageReward
reward_model = load_imagereward(device)
print('ImageReward loaded.', flush=True)

results = {}

# --- shared z-space metrics (W2 + diversity, same for all experiments) ---
print('\n=== [1/3] Prior z-space metrics (W2, Diversity) ===', flush=True)
w2_p, div_p, div_cov_p = [], [], []
z_trials = []
for t in range(N_TRIALS):
    z_all = torch.randn(N_PER_TRIAL * 2, 512)
    z_a, z_b = z_all[:N_PER_TRIAL], z_all[N_PER_TRIAL:]
    z_trials.append(z_a)
    w2_p.append(compute_w2(z_a, z_b))
    div_p.append(compute_diversity(z_a))
    div_cov_p.append(compute_diversity_cov(z_a))
    print(f'  trial {t+1}/{N_TRIALS}: W2={w2_p[-1]:.2f}, Div={div_p[-1]:.1f}', flush=True)

results['shared'] = {'w2': w2_p, 'div': div_p, 'div_cov': div_cov_p}

# --- AvgLogR per classifier ---
print('\n=== [2/3] Prior AvgLogR per classifier ===', flush=True)
for attr, clf in clfs.items():
    alr = []
    for t, z_a in enumerate(z_trials):
        logits = []
        for i in range(0, N_PER_TRIAL, BATCH_SIZE):
            z_b = z_a[i:i+BATCH_SIZE].to(device)
            with torch.no_grad():
                logits.append(clf(stylegan(z_b)).squeeze(-1).cpu())
        alr.append(F.logsigmoid(torch.cat(logits)).mean().item())
    results[attr] = {'alr': alr}
    print(f'  {attr}: {np.mean(alr):.4f}±{np.std(alr, ddof=1):.4f}', flush=True)

# male+eye: joint log reward
alr_me = []
for z_a in z_trials:
    logits = []
    for i in range(0, N_PER_TRIAL, BATCH_SIZE):
        z_b = z_a[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            imgs = stylegan(z_b)
            lp = (F.logsigmoid(clfs['male'](imgs)) +
                  F.logsigmoid(clfs['eyeglasses'](imgs))).squeeze(-1).cpu()
            logits.append(lp)
    alr_me.append(torch.cat(logits).mean().item())
results['male_eye'] = {'alr': alr_me}
print(f'  male_eye: {np.mean(alr_me):.4f}±{np.std(alr_me, ddof=1):.4f}', flush=True)

# --- bald_ir: log posterior = -½‖z‖² + IR_score ---
print('\n=== [3/3] Prior AvgLogR for bald_ir (ImageReward) ===', flush=True)
alr_ir, raw_ir = [], []
for t, z_a in enumerate(z_trials):
    log_prior_vals, ir_score_vals = [], []
    for i in range(0, N_PER_TRIAL, BATCH_SIZE):
        z_b = z_a[i:i+BATCH_SIZE].to(device)
        p_ids, p_mask = tokenize_prompt(reward_model, PROMPT_IR, device, z_b.shape[0])
        with torch.no_grad():
            imgs     = stylegan.G(z_b, None)
            imgs_blip = _preprocess_for_blip(imgs, device)
            ir_scores = reward_model.score_gard(p_ids, p_mask, imgs_blip).squeeze(-1).cpu()
        log_prior = -0.5 * (z_b.cpu() ** 2).sum(1)
        log_prior_vals.append(log_prior)
        ir_score_vals.append(ir_scores)
    log_post = (torch.cat(log_prior_vals) + torch.cat(ir_score_vals)).mean().item()
    raw      = torch.cat(ir_score_vals).mean().item()
    alr_ir.append(log_post)
    raw_ir.append(raw)
    print(f'  trial {t+1}: log_post={log_post:.3f}, raw_ir={raw:.3f}', flush=True)

results['bald_ir'] = {'alr': alr_ir, 'raw_ir': raw_ir}

os.makedirs('experiments', exist_ok=True)
out_path = 'experiments/prior_metrics.pt'
torch.save(results, out_path)
print(f'\nSaved to {out_path}', flush=True)
