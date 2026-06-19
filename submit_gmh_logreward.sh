#!/usr/bin/env bash
#SBATCH --job-name=gmh_logreward
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=00:20:00
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
from model_loader import load_models
from samplers import latent_Gaussian_MH_celeba
from utils import log_posterior_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

SIGMA        = 0.105
N_CHAINS     = 64
BURNIN       = 200   # discard first 200 steps (mixing)
THIN_K       = 5     # keep every 5th step after burnin
N_KEPT_STEPS = 50    # → 50 × 64 = 3200 samples
total_steps  = BURNIN + N_KEPT_STEPS * THIN_K  # = 450

print(f'\nG_MH sigma={SIGMA}: {N_CHAINS} chains, burnin={BURNIN}, thin_k={THIN_K}', flush=True)
print(f'Kept steps: {N_KEPT_STEPS}, total_steps: {total_steps}', flush=True)
print(f'Total samples collected: {N_CHAINS * N_KEPT_STEPS}', flush=True)

# warmup (compile + cache)
_ = latent_Gaussian_MH_celeba(model, clf, N_CHAINS, 5, SIGMA, device)

torch.cuda.synchronize()
t0 = time.perf_counter()
samples_list, accept_rate, _ = latent_Gaussian_MH_celeba(
    model, clf, N_CHAINS, total_steps, SIGMA, device,
    burnin=BURNIN, thin_k=THIN_K, return_diagnostics=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

print(f'Elapsed: {elapsed:.1f}s  |  accept_rate: {accept_rate:.1%}', flush=True)

# Stack all samples: (N_CHAINS * N_KEPT_STEPS, 512)
all_z = torch.cat(samples_list, dim=0)
print(f'Sample tensor shape: {tuple(all_z.shape)}', flush=True)

# log reward = log σ(clf(G(z))) = log_p - prior = log_p + 0.5||z||²
# log_posterior_celeba already handles chunking internally
log_p = log_posterior_celeba(all_z, model, clf)
prior = -0.5 * (all_z ** 2).sum(dim=1)
log_reward = log_p - prior  # = F.logsigmoid(clf(G(z)))

mean_lr = log_reward.mean().item()
std_lr  = log_reward.std().item()

PRIOR_MEAN, PRIOR_STD   = -1.27, 0.04
OLD_MH_MEAN, OLD_MH_STD = -1.23, 0.04
RS_MEAN, RS_STD          = -0.28, 0.01

print(f'\n{"="*60}', flush=True)
print(f'  G_MH log-reward: sigma=0.105 (corrected) vs sigma=0.5 (broken)', flush=True)
print(f'{"="*60}', flush=True)
print(f'  Samples:              {len(log_reward)}', flush=True)
print(f'  Accept rate:          {accept_rate:.1%}  (target ~23%)', flush=True)
print(f'', flush=True)
print(f'  NEW  G_MH (σ=0.105): {mean_lr:+.2f} ± {std_lr:.2f}', flush=True)
print(f'  OLD  G_MH (σ=0.5):   {OLD_MH_MEAN:+.2f} ± {OLD_MH_STD:.2f}  ← frozen (≈ prior)', flush=True)
print(f'  Prior (z~N(0,I)):     {PRIOR_MEAN:+.2f} ± {PRIOR_STD:.2f}', flush=True)
print(f'  RS:                   {RS_MEAN:+.2f} ± {RS_STD:.2f}', flush=True)
print(f'', flush=True)
print(f'  Δ vs old G_MH:        {mean_lr - OLD_MH_MEAN:+.2f}', flush=True)
print(f'  Δ vs prior:           {mean_lr - PRIOR_MEAN:+.2f}', flush=True)
print(f'  Δ vs RS:              {mean_lr - RS_MEAN:+.2f}', flush=True)

gap_to_prior = mean_lr - PRIOR_MEAN
gap_to_rs    = RS_MEAN - PRIOR_MEAN  # total gap from prior to RS

print(f'', flush=True)
if gap_to_prior > 0.15 * gap_to_rs:
    frac = gap_to_prior / gap_to_rs
    print(f'  VERDICT: Chains are mixing — {frac:.0%} of the way from prior to RS.', flush=True)
else:
    print(f'  VERDICT: Still near prior — chains may still be stuck.', flush=True)

print(f'{"="*60}', flush=True)
EOF

echo "Finished: $(date -u)"
