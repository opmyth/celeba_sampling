#!/usr/bin/env bash
#SBATCH --job-name=h200_bench
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=01:30:00
#SBATCH --output=logs/h200_bench-%j.out
#SBATCH --error=logs/h200_bench-%j.err

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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python - <<'EOF'
import sys, os, time
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from model_loader import load_models
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba

device = torch.device('cuda')

# Load two copies: raw (for batch sweep) and compiled (for sampler benchmarks)
model_raw, clf_raw, _ = load_models(device)
model_raw.eval()

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
# warm up compile at the batch size we actually use (64)
with torch.no_grad():
    _ = clf(model.G(torch.randn(64, 512, device=device), None))
print('Done.\n', flush=True)

N_WARMUP = 15
N_TIME   = 40

def measure(fn, nw=N_WARMUP, nt=N_TIME):
    for _ in range(nw): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(nt): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / nt * 1000  # ms

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1e6

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 66, flush=True)
print('  SECTION 1: Raw forward pass (no compile — avoids recompilation noise)', flush=True)
print('  G + clf, fp32.  Does larger batch improve throughput?', flush=True)
print('=' * 66, flush=True)
print(f'  {"batch":>6}  {"G (ms)":>8}  {"clf (ms)":>9}  {"total (ms)":>10}  {"imgs/s":>8}  {"peak MB":>8}', flush=True)
print(f'  {"-"*58}', flush=True)

for bs in [16, 32, 64, 128, 256]:
    try:
        z    = torch.randn(bs, 512, device=device)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            imgs = model_raw.G(z, None)
            g_ms = measure(lambda: model_raw.G(z, None))
            c_ms = measure(lambda: clf_raw(imgs))
            t_ms = measure(lambda: clf_raw(model_raw.G(z, None)))
        pmb  = peak_mb()
        tput = bs / (t_ms / 1000)
        print(f'  {bs:>6}  {g_ms:>8.1f}  {c_ms:>9.1f}  {t_ms:>10.1f}  {tput:>8.0f}  {pmb:>8.0f}', flush=True)
    except Exception as e:
        print(f'  {bs:>6}  ERROR: {e}', flush=True)
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 66}', flush=True)
print('  SECTION 2: Forward + gradient (no compile — MALA path)', flush=True)
print('  Largest safe chunk size before OOM or kernel error', flush=True)
print('=' * 66, flush=True)
print(f'  {"batch":>6}  {"fwd+bwd (ms)":>13}  {"imgs/s":>8}  {"peak MB":>8}', flush=True)
print(f'  {"-"*44}', flush=True)

for bs in [16, 32, 64, 128, 256]:
    try:
        torch.cuda.reset_peak_memory_stats()
        z = torch.randn(bs, 512, device=device, requires_grad=True)
        def fwd_bwd():
            if z.grad is not None: z.grad.zero_()
            imgs   = model_raw.G(z, None)
            logits = clf_raw(imgs).squeeze()
            (-0.5 * z.pow(2).sum(1) + F.logsigmoid(logits)).sum().backward()
        ms   = measure(fwd_bwd)
        pmb  = peak_mb()
        print(f'  {bs:>6}  {ms:>13.1f}  {bs/(ms/1000):>8.0f}  {pmb:>8.0f}', flush=True)
        del z; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  {bs:>6}  ERROR: {e}', flush=True)
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 66}', flush=True)
print('  SECTION 3: Mixed precision bf16 vs fp32 (no compile, forward only)', flush=True)
print('=' * 66, flush=True)
print(f'  {"batch":>6}  {"fp32 (ms)":>10}  {"bf16 (ms)":>10}  {"speedup":>8}', flush=True)
print(f'  {"-"*40}', flush=True)

for bs in [32, 64, 128]:
    try:
        z = torch.randn(bs, 512, device=device)
        with torch.no_grad():
            fp32_ms = measure(lambda: clf_raw(model_raw.G(z, None)))
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            bf16_ms = measure(lambda: clf_raw(model_raw.G(z, None)))
        print(f'  {bs:>6}  {fp32_ms:>10.1f}  {bf16_ms:>10.1f}  {fp32_ms/bf16_ms:>8.2f}x', flush=True)
    except Exception as e:
        print(f'  {bs:>6}  ERROR: {e}', flush=True)
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# Sections 4-6 use compiled model at fixed batch (no recompilation)
print(f'\n{"=" * 66}', flush=True)
print('  SECTION 4: Sampler throughput vs n_chains (compiled, max_batch_size=64)', flush=True)
print('  chain-steps/sec = n_chains / (ms_per_step / 1000)', flush=True)
print('=' * 66, flush=True)
print(f'  {"n_chains":>8}  {"ULA cps":>10}  {"MALA cps":>10}  {"G_MH cps":>10}', flush=True)
print(f'  {"-"*44}', flush=True)

N_STEPS = 20
for n_chains in [64, 128, 256, 512, 1000]:
    row = [f'  {n_chains:>8}']
    for name, fn, param in [
        ('ULA',  latent_ULA_celeba,         0.02),
        ('MALA', latent_MALA_celeba,        0.1),
        ('G_MH', latent_Gaussian_MH_celeba, 0.105),
    ]:
        try:
            torch.cuda.empty_cache()
            fn(model, clf, n_chains, 5, param, device)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(model, clf, n_chains, N_STEPS, param, device)
            torch.cuda.synchronize()
            ms  = (time.perf_counter() - t0) / N_STEPS * 1000
            cps = n_chains / (ms / 1000)
            row.append(f'{cps:>10.0f}')
        except Exception as e:
            row.append(f'{"ERR":>10}')
            torch.cuda.empty_cache()
    print('  '.join(row), flush=True)

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 66}', flush=True)
print('  SECTION 5: Thinning overhead (MALA, n_chains=64, compiled)', flush=True)
print('=' * 66, flush=True)
print(f'  {"config":>32}  {"ms/step":>8}', flush=True)
print(f'  {"-"*44}', flush=True)

torch.cuda.empty_cache()
latent_MALA_celeba(model, clf, 64, 5, 0.1, device)  # warmup

for label, kwargs in [
    ('no collection (takes [-1])',  {}),
    ('thin_k=1  (keep every step)', {'burnin': 0, 'thin_k': 1}),
    ('thin_k=10',                   {'burnin': 0, 'thin_k': 10}),
    ('thin_k=50',                   {'burnin': 0, 'thin_k': 50}),
]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent_MALA_celeba(model, clf, 64, N_STEPS, 0.1, device, **kwargs)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_STEPS * 1000
    print(f'  {label:>32}  {ms:>8.1f}', flush=True)

print(f'\nDone.', flush=True)
EOF

echo "Finished: $(date -u)"
