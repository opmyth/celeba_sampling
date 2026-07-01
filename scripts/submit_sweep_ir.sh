#!/usr/bin/env bash
#SBATCH --job-name=sweep_ir
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_ir-%j.out
#SBATCH --error=logs/sweep_ir-%j.err

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
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
from model_loader import load_models
from utils import load_imagereward, tokenize_prompt, grad_and_log_posterior_ir

PROMPT   = "a bald man"
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

def run_mala(dt):
    noise_scale = np.sqrt(2 * dt)
    torch.manual_seed(42)
    z = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)

    # warmup
    z_grad, log_p_z = grad_and_log_posterior_ir(z, stylegan, reward_model, prompt_ids, prompt_mask)
    for _ in range(N_WARMUP):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_and_log_posterior_ir(z_prop, stylegan, reward_model, prompt_ids, prompt_mask)
        log_q_fwd = -((z_prop - (z + dt * z_grad))**2).sum(1) / (4*dt)
        log_q_bwd = -((z - (z_prop + dt * z_prop_grad))**2).sum(1) / (4*dt)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    # measured run
    n_accept   = 0
    log_p_trace = []
    for _ in range(N_STEPS):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_and_log_posterior_ir(z_prop, stylegan, reward_model, prompt_ids, prompt_mask)
        log_q_fwd = -((z_prop - (z + dt * z_grad))**2).sum(1) / (4*dt)
        log_q_bwd = -((z - (z_prop + dt * z_prop_grad))**2).sum(1) / (4*dt)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)
        n_accept   += accept.sum().item()
        log_p_trace.append(log_p_z.mean().item())

    accept_rate = n_accept / (N_STEPS * N_CHAINS)
    arr   = np.array(log_p_trace)
    early = arr[:20].mean()
    late  = arr[-20:].mean()
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    return accept_rate, early, late, slope

print(f'\n{"="*60}', flush=True)
print(f'  MALA-IR — target acceptance: 57%  prompt: "{PROMPT}"', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps', flush=True)
print(f'{"="*60}', flush=True)
print(f'  {"DT":>8}  {"accept%":>8}  {"log_p early":>12}  {"log_p late":>10}  {"trend/step":>10}', flush=True)
print(f'  {"-"*56}', flush=True)

for dt in [0.01, 0.05, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20]:
    torch.cuda.empty_cache()
    accept_rate, early, late, slope = run_mala(dt)
    flag = ' ← target' if 0.45 < accept_rate < 0.70 else ''
    print(f'  {dt:>8.3f}  {accept_rate:>7.1%}  {early:>12.4f}  {late:>10.4f}  {slope:>+10.4f}{flag}', flush=True)

print(f'\nG_MH: theoretical optimum sigma = 2.38/sqrt(512) = {2.38/math.sqrt(512):.4f}', flush=True)
print('Done.', flush=True)
EOF

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
