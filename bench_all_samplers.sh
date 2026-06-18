#!/bin/bash
# Benchmarks MALA, ULA, G_MH across batch sizes using the production code path.
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

N_WARMUP = 5
N_STEPS  = 15
DT       = 0.5
SIGMA    = 0.5

# G_MH can go larger since it has no backward pass
BATCH_SIZES = {
    'MALA':  [64, 128, 256, 512, 1000],
    'ULA':   [64, 128, 256, 512, 1000],
    'G_MH':  [64, 128, 256, 512, 1000, 2000, 5000],
}

SAMPLERS = {
    'MALA':  (latent_MALA_celeba,        DT),
    'ULA':   (latent_ULA_celeba,         DT),
    'G_MH':  (latent_Gaussian_MH_celeba, SIGMA),
}

for name, (fn, param) in SAMPLERS.items():
    print(f'\n{"="*50}', flush=True)
    print(f'  {name}', flush=True)
    print(f'{"="*50}', flush=True)
    print(f'{"batch":>8}  {"ms/step":>10}  {"projected":>12}', flush=True)
    print('-' * 36)

    for batch_size in BATCH_SIZES[name]:
        try:
            torch.cuda.empty_cache()
            # warmup (triggers torch.compile)
            _ = fn(model, clf, batch_size, N_WARMUP, param, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn(model, clf, batch_size, N_STEPS, param, device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            per_step_ms = elapsed / N_STEPS * 1000
            projected_h = (elapsed / N_STEPS) * 800 * (1000 * 5 / batch_size) / 3600
            print(f'{batch_size:>8}  {per_step_ms:>9.1f}ms  {projected_h:>10.2f}h', flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'{batch_size:>8}  {"OOM":>10}  {"---":>12}', flush=True)
EOF
