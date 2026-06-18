#!/bin/bash
# Sweeps max_batch_size (internal StyleGAN2 chunk size) to find optimal value.
# Holds total chains fixed at 256 so the number of chunks varies with chunk size.
cd "$(dirname "$0")"

python - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, time
from model_loader import load_models
from samplers import latent_MALA_celeba, latent_ULA_celeba, latent_Gaussian_MH_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_WARMUP    = 5
N_STEPS     = 20
DT          = 0.5
SIGMA       = 0.5
TOTAL_CHAINS = 256  # fixed; chunk_size divides this

CHUNK_SIZES = [16, 32, 48, 64, 96, 128]

SAMPLERS = {
    'MALA': (latent_MALA_celeba,        DT),
    'ULA':  (latent_ULA_celeba,         DT),
    'G_MH': (latent_Gaussian_MH_celeba, SIGMA),
}

for name, (fn, param) in SAMPLERS.items():
    print(f'\n{"="*50}', flush=True)
    print(f'  {name}  (total_chains={TOTAL_CHAINS})', flush=True)
    print(f'{"="*50}', flush=True)
    print(f'{"chunk":>8}  {"ms/step":>10}  {"ms/sample":>12}  {"projected":>12}', flush=True)
    print('-' * 48)

    for chunk_size in CHUNK_SIZES:
        try:
            torch.cuda.empty_cache()
            model.max_batch_size = chunk_size

            _ = fn(model, clf, TOTAL_CHAINS, N_WARMUP, param, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn(model, clf, TOTAL_CHAINS, N_STEPS, param, device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            per_step_ms = elapsed / N_STEPS * 1000
            per_sample_us = elapsed / N_STEPS / TOTAL_CHAINS * 1e6
            projected_h = (elapsed / N_STEPS) * 800 * (1000 * 5 / TOTAL_CHAINS) / 3600
            print(f'{chunk_size:>8}  {per_step_ms:>9.1f}ms  {per_sample_us:>10.1f}us  {projected_h:>10.2f}h', flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'{chunk_size:>8}  {"OOM":>10}  {"---":>12}  {"---":>12}', flush=True)

# restore default
model.max_batch_size = 64
EOF
