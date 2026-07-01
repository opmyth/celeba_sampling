#!/usr/bin/env bash
#SBATCH --job-name=sweep_male
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep_male-%j.out
#SBATCH --error=logs/sweep_male-%j.err

JOB_START=$(date +%s)
echo "Job started: $(date)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
cd ~/celeba_sampling

mkdir -p logs

python - <<'EOF'
import sys, os, math
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, numpy as np
from model_loader import load_models
from samplers import latent_MALA_celeba, latent_ULA_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models('male', device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_CHAINS  = 100
N_WARMUP  = 5
N_STEPS   = 300

def log_p_stats(trace):
    arr = np.array(trace)
    early = arr[:50].mean()
    late  = arr[-50:].mean()
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    return early, late, slope

# ── MALA sweep ────────────────────────────────────────────────────────────────
print(f'\n{"="*60}', flush=True)
print(f'  MALA — target acceptance: 57%', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps', flush=True)
print(f'{"="*60}', flush=True)
print(f'  {"DT":>8}  {"accept%":>8}  {"log_p early":>12}  {"log_p late":>10}  {"trend/step":>10}', flush=True)
print(f'  {"-"*56}', flush=True)

for dt in [0.01, 0.05, 0.07, 0.08, 0.09, 0.10, 0.15, 0.2]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    latent_MALA_celeba(model, clf, N_CHAINS, N_WARMUP, dt, device)
    _, accept_rate, log_p_trace = latent_MALA_celeba(
        model, clf, N_CHAINS, N_STEPS, dt, device, return_diagnostics=True)
    early, late, slope = log_p_stats(log_p_trace)
    flag = ' ← target' if 0.45 < accept_rate < 0.70 else ''
    print(f'  {dt:>8.3f}  {accept_rate:>7.1%}  {early:>12.2f}  {late:>10.2f}  {slope:>+10.4f}{flag}', flush=True)

# ── ULA sweep ─────────────────────────────────────────────────────────────────
print(f'\n{"="*60}', flush=True)
print(f'  ULA — check: log_p stabilises (slope ≈ 0)', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps', flush=True)
print(f'{"="*60}', flush=True)
print(f'  {"DT":>8}  {"log_p early":>12}  {"log_p late":>10}  {"trend/step":>10}  {"bias":>8}  {"stable?":>8}', flush=True)
print(f'  {"-"*64}', flush=True)

prev_stable_dt = None
for dt in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    latent_ULA_celeba(model, clf, N_CHAINS, N_WARMUP, dt, device)
    _, log_p_trace = latent_ULA_celeba(
        model, clf, N_CHAINS, N_STEPS, dt, device, return_diagnostics=True)
    early, late, slope = log_p_stats(log_p_trace)
    bias = late - early
    stable = abs(slope) < 0.01 and abs(bias) < 2.0
    if stable:
        prev_stable_dt = dt
    status = 'stable' if stable else 'BIASED'
    flag = ' ← largest stable' if stable and dt == prev_stable_dt else ''
    print(f'  {dt:>8.3f}  {early:>12.2f}  {late:>10.2f}  {slope:>+10.4f}  {bias:>+8.2f}  {status:>8}{flag}', flush=True)

print(f'\nG_MH: using theoretical optimum sigma = 2.38/sqrt(512) = {2.38/math.sqrt(512):.4f}', flush=True)
print(f'Done.', flush=True)
EOF

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
