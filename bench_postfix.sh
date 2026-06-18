#!/usr/bin/env bash
# Post-fix benchmark at production parameters (DT=0.01, SIGMA=0.5, n_chains=64).
# 5 warmup + 25 timed steps per sampler; projects full production run time.
cd "$(dirname "$0")"
mkdir -p logs

python - <<'EOF'
import sys, os, math, time
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch
from model_loader import load_models
from samplers import latent_ULA_celeba, latent_MALA_celeba, latent_Gaussian_MH_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf     = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_WARMUP    = 5
N_STEPS     = 25
N_CHAINS    = 64       # one production minibatch
DT          = 0.01    # production step size
SIGMA       = 0.5     # production sigma
PROD_STEPS  = 800
PROD_CHAINS = 1000
PROD_TRIALS = 5
N_BATCHES   = math.ceil(PROD_CHAINS / N_CHAINS)  # ceil(1000/64) = 16

SAMPLERS = [
    ('ULA',  latent_ULA_celeba,         DT),
    ('MALA', latent_MALA_celeba,        DT),
    ('G_MH', latent_Gaussian_MH_celeba, SIGMA),
]

print(f'\nProjection: (ms/step) × {PROD_STEPS} steps × {N_BATCHES} batches × {PROD_TRIALS} trials', flush=True)
print(f'  = (ms/step) × {PROD_STEPS * N_BATCHES * PROD_TRIALS}', flush=True)

for name, fn, param in SAMPLERS:
    print(f'\n{"="*52}', flush=True)
    print(f'  {name}', flush=True)
    print(f'{"="*52}', flush=True)

    torch.cuda.empty_cache()

    _ = fn(model, clf, N_CHAINS, N_WARMUP, param, device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    _ = fn(model, clf, N_CHAINS, N_STEPS, param, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_step_ms = elapsed / N_STEPS * 1000
    projected_s = (elapsed / N_STEPS) * PROD_STEPS * N_BATCHES * PROD_TRIALS
    projected_h = projected_s / 3600

    print(f'  ms/step:         {per_step_ms:>8.1f} ms', flush=True)
    print(f'  Projected total: {projected_h:>8.2f} h  ({projected_s:.0f} s)', flush=True)
EOF
