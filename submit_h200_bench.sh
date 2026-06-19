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
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from model_loader import load_models
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba

device = torch.device('cuda')

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.\n', flush=True)

N_WARMUP = 10
N_TIME   = 30

def measure(fn, n_warmup=N_WARMUP, n_time=N_TIME):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_time * 1000  # ms

def mem_gb():
    return torch.cuda.memory_allocated() / 1e9

def peak_mem_gb():
    return torch.cuda.max_memory_allocated() / 1e9

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 64, flush=True)
print('  SECTION 1: Raw forward pass  (G + clf, fp32)', flush=True)
print('  Vary batch size — tells us if max_batch_size=64 is leaving perf on table', flush=True)
print('=' * 64, flush=True)
print(f'  {"batch":>6}  {"G (ms)":>8}  {"clf (ms)":>8}  {"total (ms)":>10}  {"imgs/s":>8}  {"peak mem GB":>11}', flush=True)
print(f'  {"-"*60}', flush=True)

for bs in [32, 64, 128, 256, 512, 1024]:
    try:
        z = torch.randn(bs, 512, device=device)
        torch.cuda.reset_peak_memory_stats()

        g_ms = measure(lambda: model.G(z, None))
        with torch.no_grad():
            imgs = model.G(z, None)
        c_ms = measure(lambda: clf(imgs))
        t_ms = measure(lambda: clf(model.G(z, None)))
        pmem = peak_mem_gb()
        throughput = bs / (t_ms / 1000)
        print(f'  {bs:>6}  {g_ms:>8.1f}  {c_ms:>8.1f}  {t_ms:>10.1f}  {throughput:>8.0f}  {pmem:>11.2f}', flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f'  {bs:>6}  OOM', flush=True)
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 64}', flush=True)
print('  SECTION 2: Forward pass with gradients  (MALA path, fp32)', flush=True)
print('  Chunked backward — what is the largest chunk before OOM?', flush=True)
print('=' * 64, flush=True)
print(f'  {"batch":>6}  {"fwd+bwd (ms)":>12}  {"imgs/s":>8}  {"peak mem GB":>11}', flush=True)
print(f'  {"-"*44}', flush=True)

for bs in [32, 64, 128, 256, 512]:
    try:
        torch.cuda.reset_peak_memory_stats()
        z = torch.randn(bs, 512, device=device, requires_grad=True)

        def fwd_bwd():
            if z.grad is not None:
                z.grad.zero_()
            imgs = model.G(z, None)
            logits = clf(imgs).squeeze()
            log_p = (-0.5 * z.pow(2).sum(1) + F.logsigmoid(logits)).sum()
            log_p.backward()

        ms = measure(fwd_bwd)
        pmem = peak_mem_gb()
        print(f'  {bs:>6}  {ms:>12.1f}  {bs/(ms/1000):>8.0f}  {pmem:>11.2f}', flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f'  {bs:>6}  OOM', flush=True)
        torch.cuda.empty_cache()
    finally:
        del z
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 64}', flush=True)
print('  SECTION 3: Mixed precision bf16  (forward only)', flush=True)
print('=' * 64, flush=True)
print(f'  {"batch":>6}  {"fp32 (ms)":>10}  {"bf16 (ms)":>10}  {"speedup":>8}', flush=True)
print(f'  {"-"*40}', flush=True)

for bs in [64, 128, 256]:
    try:
        z = torch.randn(bs, 512, device=device)
        with torch.no_grad():
            fp32_ms = measure(lambda: clf(model.G(z, None)))
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            bf16_ms = measure(lambda: clf(model.G(z, None)))
        print(f'  {bs:>6}  {fp32_ms:>10.1f}  {bf16_ms:>10.1f}  {fp32_ms/bf16_ms:>8.2f}x', flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f'  {bs:>6}  OOM', flush=True)
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 64}', flush=True)
print('  SECTION 4: Sampler throughput vs max_batch_size', flush=True)
print('  n_chains=512, n_steps=20 — override max_batch_size and measure end-to-end', flush=True)
print('=' * 64, flush=True)
print(f'  {"max_bs":>6}  {"ULA ms/step":>12}  {"MALA ms/step":>13}  {"G_MH ms/step":>13}', flush=True)
print(f'  {"-"*50}', flush=True)

N_CHAINS = 512
N_STEPS  = 20

for mbs in [64, 128, 256]:
    model.max_batch_size = mbs
    row = [f'  {mbs:>6}']
    for name, fn, param in [
        ('ULA',  latent_ULA_celeba,         0.02),
        ('MALA', latent_MALA_celeba,        0.1),
        ('G_MH', latent_Gaussian_MH_celeba, 0.105),
    ]:
        try:
            torch.cuda.empty_cache()
            fn(model, clf, N_CHAINS, 5, param, device)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(model, clf, N_CHAINS, N_STEPS, param, device)
            torch.cuda.synchronize()
            ms_per_step = (time.perf_counter() - t0) / N_STEPS * 1000
            row.append(f'{ms_per_step:>12.1f}')
        except torch.cuda.OutOfMemoryError:
            row.append(f'{"OOM":>12}')
            torch.cuda.empty_cache()
    print('  '.join(row), flush=True)

model.max_batch_size = 64  # reset

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 64}', flush=True)
print('  SECTION 5: Sampler throughput vs n_chains  (max_batch_size=64)', flush=True)
print('  chain-steps/sec = n_chains / (ms_per_step / 1000)', flush=True)
print('=' * 64, flush=True)
print(f'  {"n_chains":>8}  {"ULA ch-stp/s":>13}  {"MALA ch-stp/s":>14}  {"G_MH ch-stp/s":>14}', flush=True)
print(f'  {"-"*56}', flush=True)

for n_chains in [64, 128, 256, 512, 1000]:
    row = [f'  {n_chains:>8}']
    for name, fn, param in [
        ('ULA',  latent_ULA_celeba,         0.02),
        ('MALA', latent_MALA_celeba,        0.1),
        ('G_MH', latent_Gaussian_MH_celeba, 0.105),
    ]:
        try:
            torch.cuda.empty_cache()
            fn(model, clf, n_chains, 5, param, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(model, clf, n_chains, N_STEPS, param, device)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / N_STEPS * 1000
            cps = n_chains / (ms / 1000)
            row.append(f'{cps:>13.0f}')
        except torch.cuda.OutOfMemoryError:
            row.append(f'{"OOM":>13}')
            torch.cuda.empty_cache()
    print('  '.join(row), flush=True)

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"=" * 64}', flush=True)
print('  SECTION 6: Thinning overhead  (MALA, n_chains=64)', flush=True)
print('=' * 64, flush=True)
print(f'  {"config":>30}  {"ms/step":>8}', flush=True)
print(f'  {"-"*42}', flush=True)

torch.cuda.empty_cache()
latent_MALA_celeba(model, clf, 64, 5, 0.1, device)

for label, kwargs in [
    ('no collection ([-1] only)',  {}),
    ('thin_k=1 (keep every step)', {'burnin': 0, 'thin_k': 1}),
    ('thin_k=10',                  {'burnin': 0, 'thin_k': 10}),
    ('thin_k=50',                  {'burnin': 0, 'thin_k': 50}),
]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = latent_MALA_celeba(model, clf, 64, N_STEPS, 0.1, device, **kwargs)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_STEPS * 1000
    print(f'  {label:>30}  {ms:>8.1f}', flush=True)

print(f'\nDone.', flush=True)
EOF

echo "Finished: $(date -u)"
