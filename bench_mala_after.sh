#!/bin/bash
# Benchmarks MALA with the NEW logic (1 forward+backward per step, log_p_z cached).
# Self-contained: does not depend on the current state of samplers.py.
cd "$(dirname "$0")"

python - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, time, numpy as np
import torch.nn.functional as F
from model_loader import load_models

def grad_and_log_posterior(z, model, clf):
    z = z.detach().requires_grad_(True)
    imgs = model(z)
    logits = clf(imgs).squeeze()
    log_p = -0.5 * torch.sum(z**2, dim=1) + F.logsigmoid(logits)
    log_p.sum().backward()
    return z.grad.clone(), log_p.detach()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_CHAINS, N_WARMUP, N_STEPS, DT = 64, 10, 25, 0.5  # extra warmup to absorb compilation
z = torch.randn(N_CHAINS, model.latent_dim).to(device)
z_grad, log_p_z = grad_and_log_posterior(z, model, clf)

print(f'Warming up ({N_WARMUP} steps)...', flush=True)
for _ in range(N_WARMUP):
    z_prop = z + DT * z_grad + (2*DT)**0.5 * torch.randn_like(z)
    z_prop_grad, log_p_prop = grad_and_log_posterior(z_prop, model, clf)
    log_q_fwd = -torch.sum((z_prop - (z + DT*z_grad))**2, dim=1) / (4*DT)
    log_q_bwd = -torch.sum((z - (z_prop + DT*z_prop_grad))**2, dim=1) / (4*DT)
    log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)
    accept = torch.log(torch.rand(N_CHAINS).to(device)) <= log_alpha
    z = torch.where(accept.unsqueeze(1), z_prop, z)
    z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
    log_p_z = torch.where(accept, log_p_prop, log_p_z)

if device.type == 'cuda':
    torch.cuda.synchronize()
print(f'Timing {N_STEPS} steps...', flush=True)
t0 = time.perf_counter()
for _ in range(N_STEPS):
    z_prop = z + DT * z_grad + (2*DT)**0.5 * torch.randn_like(z)
    z_prop_grad, log_p_prop = grad_and_log_posterior(z_prop, model, clf)
    log_q_fwd = -torch.sum((z_prop - (z + DT*z_grad))**2, dim=1) / (4*DT)
    log_q_bwd = -torch.sum((z - (z_prop + DT*z_prop_grad))**2, dim=1) / (4*DT)
    log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)
    accept = torch.log(torch.rand(N_CHAINS).to(device)) <= log_alpha
    z = torch.where(accept.unsqueeze(1), z_prop, z)
    z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
    log_p_z = torch.where(accept, log_p_prop, log_p_z)
if device.type == 'cuda':
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

per_step_ms = elapsed / N_STEPS * 1000
projected_s = (elapsed / N_STEPS) * 800 * (1000 * 5 / N_CHAINS)
print(f'\n[AFTER FIX] {N_CHAINS} chains, {N_STEPS} steps')
print(f'  Total:        {elapsed:.2f}s')
print(f'  Per step:     {per_step_ms:.1f}ms')
print(f'  Projected full run (n_chains=1000, n_trials=5, n_steps=800, batch={N_CHAINS}): {projected_s/3600:.2f}h')
EOF
