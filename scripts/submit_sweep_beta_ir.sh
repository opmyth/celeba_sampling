#!/usr/bin/env bash
#SBATCH --job-name=sweep_beta_ir
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_beta_ir-%j.out
#SBATCH --error=logs/sweep_beta_ir-%j.err

JOB_START=$(date +%s)
echo "Job started: $(date)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
cd ~/celeba_sampling
mkdir -p logs

python - <<'EOF'
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
from model_loader import load_models
from utils import load_imagereward, tokenize_prompt, _preprocess_for_blip

PROMPT   = "a bald man"
DT       = 0.05
N_CHAINS = 20
N_WARMUP = 3
N_STEPS  = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

stylegan, _, _ = load_models('bald', device)
stylegan.G     = torch.compile(stylegan.G)
reward_model   = load_imagereward(device)
print('Models loaded.', flush=True)

prompt_ids, prompt_mask = tokenize_prompt(reward_model, PROMPT, device, N_CHAINS)
noise_scale = np.sqrt(2 * DT)

def ir_score(z):
    imgs      = stylegan.G(z, None)
    imgs_blip = _preprocess_for_blip(imgs, z.device)
    B = z.size(0)
    return reward_model.score_gard(
        prompt_ids[:B], prompt_mask[:B], imgs_blip
    ).squeeze(-1)

def grad_and_log_p(z, beta):
    z = z.detach().requires_grad_(True)
    log_p_list = []
    chunk = 8
    for start in range(0, z.size(0), chunk):
        zc = z[start:start+chunk]
        scores = ir_score(zc)
        lp = -0.5 * (zc**2).sum(1) + beta * scores
        lp.sum().backward()
        log_p_list.append(lp.detach())
    return z.grad.clone(), torch.cat(log_p_list)

def run_mala(beta):
    torch.manual_seed(42)
    z = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    z_grad, log_p_z = grad_and_log_p(z, beta)

    # warmup
    for _ in range(N_WARMUP):
        noise  = torch.randn_like(z)
        z_prop = z + DT * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_and_log_p(z_prop, beta)
        log_q_fwd = -((z_prop - (z + DT * z_grad))**2).sum(1) / (4*DT)
        log_q_bwd = -((z - (z_prop + DT * z_prop_grad))**2).sum(1) / (4*DT)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    # measured run
    n_accept    = 0
    ir_trace    = []
    for step in range(N_STEPS):
        noise  = torch.randn_like(z)
        z_prop = z + DT * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_and_log_p(z_prop, beta)
        log_q_fwd = -((z_prop - (z + DT * z_grad))**2).sum(1) / (4*DT)
        log_q_bwd = -((z - (z_prop + DT * z_prop_grad))**2).sum(1) / (4*DT)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)
        n_accept += accept.sum().item()
        # track raw IR score (remove prior term)
        with torch.no_grad():
            ir_trace.append(ir_score(z).mean().item())

    arr         = np.array(ir_trace)
    accept_rate = n_accept / (N_STEPS * N_CHAINS)
    early       = arr[:50].mean()
    late        = arr[50:].mean()
    slope       = np.polyfit(range(len(arr)), arr, 1)[0]
    return accept_rate, early, late, slope

print(f'\n{"="*65}', flush=True)
print(f'  MALA-IR beta sweep  dt={DT}  prompt: "{PROMPT}"', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps', flush=True)
print(f'{"="*65}', flush=True)
print(f'  {"beta":>6}  {"accept%":>8}  {"IR early":>10}  {"IR late":>10}  {"trend/step":>12}', flush=True)
print(f'  {"-"*54}', flush=True)

for beta in [10, 50, 100, 200]:
    torch.cuda.empty_cache()
    accept_rate, early, late, slope = run_mala(beta)
    flag = ' ← target' if 0.45 < accept_rate < 0.70 else ''
    print(f'  {beta:>6}  {accept_rate:>7.1%}  {early:>10.4f}  {late:>10.4f}  {slope:>+12.6f}{flag}', flush=True)

print(f'\nDone.', flush=True)
EOF

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
