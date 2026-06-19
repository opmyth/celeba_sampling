#!/usr/bin/env bash
#SBATCH --job-name=gmh_logreward
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=03:00:00
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
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
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
torch.manual_seed(321)

# ── exact production setup ────────────────────────────────────────────────────
SIGMA      = 0.105
N_CHAINS   = 1000
N_TRIALS   = 5
N_STEPS    = 800
BATCH_SIZE = 64

# mirrors run_sampler_minibatched: runs total_chains in batches, keeps only the
# final sample per chain ([-1] from the sampler's step list)
def run_minibatched(total_chains):
    results = []
    n_batches = (total_chains + BATCH_SIZE - 1) // BATCH_SIZE
    for i, start in enumerate(range(0, total_chains, BATCH_SIZE)):
        size = min(BATCH_SIZE, total_chains - start)
        out = latent_Gaussian_MH_celeba(model, clf, size, N_STEPS, SIGMA, device)[-1]
        results.append(out)
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f'  batch {i+1}/{n_batches}', flush=True)
    return torch.cat(results, dim=0)

print(f'\nG_MH sigma={SIGMA}: {N_CHAINS} chains × {N_STEPS} steps × {N_TRIALS} trials', flush=True)
print(f'Batched at {BATCH_SIZE}: {(N_CHAINS * N_TRIALS + BATCH_SIZE - 1) // BATCH_SIZE} total batches', flush=True)

t0 = time.perf_counter()
all_samples = run_minibatched(N_CHAINS * N_TRIALS)
elapsed = time.perf_counter() - t0
print(f'Sampling done: {elapsed:.1f}s', flush=True)

# split into trials, exactly as run_sampler.py does
samples_list = list(torch.chunk(all_samples, N_TRIALS, dim=0))

# log reward: mirrors run_sampler.py line 95 exactly
# F.logsigmoid(smile_clf(stylegan(z))).mean()
avg_log_reward = []
for i, z in enumerate(samples_list):
    with torch.no_grad():
        lr = F.logsigmoid(clf(model(z))).mean().item()
    avg_log_reward.append(lr)
    print(f'  trial {i+1}: log_reward = {lr:.4f}', flush=True)

mean_lr = np.mean(avg_log_reward)
std_lr  = np.std(avg_log_reward, ddof=1)

PRIOR_MEAN, PRIOR_STD    = -1.27, 0.04
OLD_MH_MEAN, OLD_MH_STD  = -1.23, 0.04
RS_MEAN, RS_STD           = -0.28, 0.01

print(f'\n{"="*60}', flush=True)
print(f'  G_MH log-reward  |  sigma=0.105 vs sigma=0.5 (broken)', flush=True)
print(f'{"="*60}', flush=True)
print(f'  NEW  G_MH (σ=0.105):  {mean_lr:+.2f} ± {std_lr:.2f}', flush=True)
print(f'  OLD  G_MH (σ=0.5):   {OLD_MH_MEAN:+.2f} ± {OLD_MH_STD:.2f}  (frozen ≈ prior)', flush=True)
print(f'  Prior (z~N(0,I)):     {PRIOR_MEAN:+.2f} ± {PRIOR_STD:.2f}', flush=True)
print(f'  RS:                   {RS_MEAN:+.2f} ± {RS_STD:.2f}', flush=True)
print(f'', flush=True)
print(f'  Δ vs prior:           {mean_lr - PRIOR_MEAN:+.3f}', flush=True)
print(f'  Δ vs old G_MH:        {mean_lr - OLD_MH_MEAN:+.3f}', flush=True)
print(f'  Δ vs RS:              {mean_lr - RS_MEAN:+.3f}', flush=True)
gap = (mean_lr - PRIOR_MEAN) / (RS_MEAN - PRIOR_MEAN)
print(f'  Position prior→RS:    {gap:.0%}', flush=True)
print(f'{"="*60}', flush=True)
EOF

echo "Finished: $(date -u)"
