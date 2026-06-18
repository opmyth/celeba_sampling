#!/bin/bash
# Benchmarks the actual production code path:
# - uses samplers.py (latent_MALA_celeba) directly
# - applies torch.compile the same way run_sampler.py does
# - confirms production matches bench numbers
cd "$(dirname "$0")"

python - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, time
from model_loader import load_models
from samplers import latent_MALA_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)

print('Compiling models...', flush=True)
clf = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)

torch.manual_seed(42)

N_CHAINS  = 64
N_WARMUP  = 5   # steps to trigger compilation + warm up GPU
N_STEPS   = 25  # timed steps
DT        = 0.5

print(f'Warming up ({N_WARMUP} steps)...', flush=True)
_ = latent_MALA_celeba(model, clf, N_CHAINS, N_WARMUP, DT, device)

torch.cuda.synchronize()
print(f'Timing {N_STEPS} steps...', flush=True)
t0 = time.perf_counter()
_ = latent_MALA_celeba(model, clf, N_CHAINS, N_STEPS, DT, device)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

per_step_ms = elapsed / N_STEPS * 1000
projected_h = (elapsed / N_STEPS) * 800 * (1000 * 5 / N_CHAINS) / 3600

print(f'\n[PRODUCTION] {N_CHAINS} chains, {N_STEPS} steps')
print(f'  Per step:  {per_step_ms:.1f}ms')
print(f'  Projected full run (n_chains=1000, n_trials=5, n_steps=800): {projected_h:.2f}h')
EOF
