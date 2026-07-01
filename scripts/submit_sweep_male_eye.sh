#!/usr/bin/env bash
#SBATCH --job-name=sweep_male_eye
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=01:00:00
#SBATCH --output=logs/sweep_male_eye-%j.out
#SBATCH --error=logs/sweep_male_eye-%j.err

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
import sys, os, math
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, numpy as np
import torch.nn.functional as F
from model_loader import load_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# load_models('eyeglasses') returns (stylegan, clf_eye, clf_male)
stylegan, clf_eye, clf_male = load_models('eyeglasses', device)
clf_eye    = torch.compile(clf_eye)
clf_male   = torch.compile(clf_male)
stylegan.G = torch.compile(stylegan.G)
print('Models loaded.', flush=True)

N_CHAINS = 20
N_WARMUP = 5
N_STEPS  = 300

def grad_log_p(z):
    z = z.detach().requires_grad_(True)
    imgs    = stylegan.G(z, None)
    log_p   = (-0.5 * (z ** 2).sum(1)
               + F.logsigmoid(clf_male(imgs).squeeze(-1))
               + F.logsigmoid(clf_eye(imgs).squeeze(-1)))
    log_p.sum().backward()
    return z.grad.clone(), log_p.detach()

def run_mala(dt):
    noise_scale = np.sqrt(2 * dt)
    torch.manual_seed(42)
    z = torch.randn(N_CHAINS, stylegan.latent_dim, device=device)
    z_grad, log_p_z = grad_log_p(z)

    for _ in range(N_WARMUP):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_log_p(z_prop)
        log_q_fwd = -((z_prop - (z + dt * z_grad)) ** 2).sum(1) / (4 * dt)
        log_q_bwd = -((z - (z_prop + dt * z_prop_grad)) ** 2).sum(1) / (4 * dt)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    n_accept    = 0
    log_p_trace = []
    for _ in range(N_STEPS):
        noise  = torch.randn_like(z)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = grad_log_p(z_prop)
        log_q_fwd = -((z_prop - (z + dt * z_grad)) ** 2).sum(1) / (4 * dt)
        log_q_bwd = -((z - (z_prop + dt * z_prop_grad)) ** 2).sum(1) / (4 * dt)
        log_alpha = (log_p_prop + log_q_bwd - log_p_z - log_q_fwd).clamp(max=0)
        accept  = torch.log(torch.rand(N_CHAINS, device=device)) <= log_alpha
        mask    = accept.unsqueeze(1)
        z       = torch.where(mask, z_prop,      z)
        z_grad  = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)
        n_accept    += accept.sum().item()
        log_p_trace.append(log_p_z.mean().item())

    arr = np.array(log_p_trace)
    return (n_accept / (N_STEPS * N_CHAINS),
            arr[:50].mean(), arr[-50:].mean(),
            np.polyfit(range(len(arr)), arr, 1)[0])

print(f'\n{"="*65}', flush=True)
print(f'  MALA — Male + Eyeglasses combined posterior', flush=True)
print(f'  log p ∝ -½‖z‖² + log σ(clf_male) + log σ(clf_eye)', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps  target: 57%', flush=True)
print(f'{"="*65}', flush=True)
print(f'  {"DT":>8}  {"accept%":>8}  {"logp early":>11}  {"logp late":>10}  {"trend/step":>10}', flush=True)
print(f'  {"-"*54}', flush=True)

for dt in [0.01, 0.05, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2]:
    torch.cuda.empty_cache()
    accept_rate, early, late, slope = run_mala(dt)
    flag = ' ← target' if 0.45 < accept_rate < 0.70 else ''
    print(f'  {dt:>8.3f}  {accept_rate:>7.1%}  {early:>11.4f}  {late:>10.4f}  {slope:>+10.4f}{flag}', flush=True)

print(f'\nG_MH: theoretical sigma = 2.38/sqrt(512) = {2.38/math.sqrt(512):.4f}', flush=True)
print('Done.', flush=True)
EOF

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
