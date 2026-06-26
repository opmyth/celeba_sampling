#!/usr/bin/env bash
#SBATCH --job-name=sweep_gmh
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=01:00:00
#SBATCH --output=logs/sweep_gmh-%j.out
#SBATCH --error=logs/sweep_gmh-%j.err

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

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
from samplers import latent_Gaussian_MH_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models('Bald', device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)

N_CHAINS = 100
N_WARMUP = 5
N_STEPS  = 300

def log_p_stats(trace):
    arr = np.array(trace)
    early = arr[:50].mean()
    late  = arr[-50:].mean()
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    return early, late, slope

print(f'\n{"="*60}', flush=True)
print(f'  G_MH — target acceptance: 23%', flush=True)
print(f'  Theory: sigma_opt = 2.38/sqrt(512) = {2.38/math.sqrt(512):.4f}', flush=True)
print(f'  {N_CHAINS} chains × {N_STEPS} steps', flush=True)
print(f'{"="*60}', flush=True)
print(f'  {"sigma":>8}  {"accept%":>8}  {"log_p early":>12}  {"log_p late":>10}', flush=True)
print(f'  {"-"*46}', flush=True)

for sigma in [0.05, 0.075, 0.105, 0.15, 0.2, 0.3, 0.5]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    latent_Gaussian_MH_celeba(model, clf, N_CHAINS, N_WARMUP, sigma, device)
    _, accept_rate, log_p_trace = latent_Gaussian_MH_celeba(
        model, clf, N_CHAINS, N_STEPS, sigma, device, return_diagnostics=True)
    early, late, _ = log_p_stats(log_p_trace)
    flag = ' ← target' if 0.15 < accept_rate < 0.32 else ''
    print(f'  {sigma:>8.3f}  {accept_rate:>7.1%}  {early:>12.2f}  {late:>10.2f}{flag}', flush=True)

print(f'\nDone.', flush=True)
EOF

echo "Finished: $(date -u)"
