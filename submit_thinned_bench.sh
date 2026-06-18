#!/usr/bin/env bash
#SBATCH --job-name=thinned_bench
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=00:15:00
#SBATCH --output=logs/thinned_bench-%j.out
#SBATCH --error=logs/thinned_bench-%j.err

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
nvidia-smi

python - <<'EOF'
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
from model_loader import load_models
from samplers import latent_MALA_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

BURNIN  = 100
THIN_K  = 5
TARGET  = 50    # final sample count
DT      = 0.01  # production DT

n_steps_total = BURNIN + TARGET * THIN_K  # = 350
print(f"\nThinning params: burnin={BURNIN}, thin_k={THIN_K}, target={TARGET} → total_steps={n_steps_total}", flush=True)

# Throughput estimate: time 25 steps at batch=1 (no thinning)
N_WARMUP = 5
N_TIME   = 25
_ = latent_MALA_celeba(model, clf, 1, N_WARMUP, DT, device)
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = latent_MALA_celeba(model, clf, 1, N_TIME, DT, device)
torch.cuda.synchronize()
rate_s = time.perf_counter() - t0
per_step_ms = rate_s / N_TIME * 1000
print(f"Throughput (batch=1, no thinning): {per_step_ms:.1f} ms/step", flush=True)
print(f"Estimated test time: {rate_s / N_TIME * n_steps_total:.1f} s", flush=True)

# Full thinning correctness test
print(f"\nRunning thinning test ({n_steps_total} steps)...", flush=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
samples = latent_MALA_celeba(model, clf, 1, n_steps_total, DT, device,
                              burnin=BURNIN, thin_k=THIN_K)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

result = torch.cat(samples, dim=0)
print(f"Elapsed: {elapsed:.2f} s", flush=True)
print(f"Collected {len(samples)} sample tensors → stacked shape: {tuple(result.shape)}", flush=True)

assert len(samples) == TARGET, f"FAIL: expected {TARGET} samples, got {len(samples)}"
assert result.shape == (TARGET, model.latent_dim), \
    f"FAIL: expected ({TARGET}, {model.latent_dim}), got {tuple(result.shape)}"
print("All shape assertions passed.", flush=True)
EOF

echo "Finished: $(date -u)"
