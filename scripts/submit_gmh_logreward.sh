#!/usr/bin/env bash
#SBATCH --job-name=gmh_logreward
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=02:00:00
#SBATCH --output=logs/gmh_logreward-%j.out
#SBATCH --error=logs/gmh_logreward-%j.err

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
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from model_loader import load_models
from samplers import latent_Gaussian_MH_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)

# 100 chains × 3000 steps, burnin=1000, thin_k=200
# → 10 samples/chain → 1000 samples/trial, 5 trials
SIGMA    = 0.105
N_CHAINS = 100
N_TRIALS = 5
BURNIN   = 1000
THIN_K   = 200
N_KEEP   = 10
N_STEPS  = BURNIN + N_KEEP * THIN_K   # = 3000

print(f'\nG_MH sigma={SIGMA}: {N_CHAINS} chains × {N_STEPS} steps × {N_TRIALS} trials', flush=True)
print(f'burnin={BURNIN}, thin_k={THIN_K} → {N_KEEP} samples/chain → {N_CHAINS*N_KEEP} total/trial', flush=True)

avg_log_reward = []
for trial in range(N_TRIALS):
    torch.manual_seed(trial)
    t0 = time.perf_counter()
    chain_samples = latent_Gaussian_MH_celeba(
        model, clf, N_CHAINS, N_STEPS, SIGMA, device,
        burnin=BURNIN, thin_k=THIN_K)
    elapsed = time.perf_counter() - t0
    # chain_samples: list of N_KEEP tensors each (N_CHAINS, 512) → (1000, 512)
    z = torch.cat(chain_samples, dim=0)
    with torch.no_grad():
        lr = F.logsigmoid(clf(model(z))).mean().item()
    avg_log_reward.append(lr)
    print(f'  trial {trial+1}: {elapsed:.1f}s  log_reward={lr:.4f}', flush=True)

mean_lr = np.mean(avg_log_reward)
std_lr  = np.std(avg_log_reward, ddof=1)

PRIOR_MEAN, PRIOR_STD   = -1.27, 0.04
OLD_MH_MEAN, OLD_MH_STD = -1.23, 0.04
RS_MEAN, RS_STD          = -0.28, 0.01

print(f'\n{"="*60}', flush=True)
print(f'  G_MH log-reward  |  sigma=0.105 vs sigma=0.5 (broken)', flush=True)
print(f'{"="*60}', flush=True)
print(f'  NEW  G_MH (sigma=0.105):  {mean_lr:+.2f} ± {std_lr:.2f}', flush=True)
print(f'  OLD  G_MH (sigma=0.5):   {OLD_MH_MEAN:+.2f} ± {OLD_MH_STD:.2f}  (frozen ~ prior)', flush=True)
print(f'  Prior (z~N(0,I)):         {PRIOR_MEAN:+.2f} ± {PRIOR_STD:.2f}', flush=True)
print(f'  RS:                       {RS_MEAN:+.2f} ± {RS_STD:.2f}', flush=True)
print(f'', flush=True)
print(f'  delta vs prior:           {mean_lr - PRIOR_MEAN:+.3f}', flush=True)
print(f'  delta vs old G_MH:        {mean_lr - OLD_MH_MEAN:+.3f}', flush=True)
print(f'  delta vs RS:              {mean_lr - RS_MEAN:+.3f}', flush=True)
gap = (mean_lr - PRIOR_MEAN) / (RS_MEAN - PRIOR_MEAN)
print(f'  position prior->RS:       {gap:.0%}', flush=True)
print(f'{"="*60}', flush=True)
EOF

echo "Finished: $(date -u)"
