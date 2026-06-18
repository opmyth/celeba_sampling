#!/bin/bash
# Benchmarks MALA with the NEW logic across multiple batch sizes.
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
    log_p_list = []
    chunk_size = model.max_batch_size  # 64; backward per chunk so activations freed immediately
    for start in range(0, z.size(0), chunk_size):
        z_chunk = z[start:start + chunk_size]
        imgs = model.G(z_chunk, None)
        logits = clf(imgs).squeeze()
        log_p_chunk = -0.5 * torch.sum(z_chunk**2, dim=1) + F.logsigmoid(logits)
        log_p_chunk.sum().backward()
        log_p_list.append(log_p_chunk.detach())
    return z.grad.clone(), torch.cat(log_p_list)

def run_bench(model, clf, batch_size, n_warmup, n_steps, dt, device):
    # max_batch_size stays at 64: StyleGAN2Wrapper chunks internally, autograd flows correctly
    z = torch.randn(batch_size, model.latent_dim).to(device)
    z_grad, log_p_z = grad_and_log_posterior(z, model, clf)

    for _ in range(n_warmup):
        z_prop = z + dt * z_grad + (2*dt)**0.5 * torch.randn_like(z)
        z_prop_grad, log_p_prop = grad_and_log_posterior(z_prop, model, clf)
        log_q_fwd = -torch.sum((z_prop - (z + dt*z_grad))**2, dim=1) / (4*dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt*z_prop_grad))**2, dim=1) / (4*dt)
        log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)
        accept = torch.log(torch.rand(batch_size).to(device)) <= log_alpha
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        z_prop = z + dt * z_grad + (2*dt)**0.5 * torch.randn_like(z)
        z_prop_grad, log_p_prop = grad_and_log_posterior(z_prop, model, clf)
        log_q_fwd = -torch.sum((z_prop - (z + dt*z_grad))**2, dim=1) / (4*dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt*z_prop_grad))**2, dim=1) / (4*dt)
        log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)
        accept = torch.log(torch.rand(batch_size).to(device)) <= log_alpha
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling models...', flush=True)
clf = torch.compile(clf)
model.G = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_WARMUP, N_STEPS, DT = 10, 25, 0.5

print(f'\n{"batch":>8}  {"ms/step":>10}  {"projected":>12}', flush=True)
print('-' * 36)

for batch_size in [64, 128, 256, 512, 1000]:
    try:
        torch.cuda.empty_cache()
        elapsed = run_bench(model, clf, batch_size, N_WARMUP, N_STEPS, DT, device)
        per_step_ms = elapsed / N_STEPS * 1000
        projected_h = (elapsed / N_STEPS) * 800 * (1000 * 5 / batch_size) / 3600
        print(f'{batch_size:>8}  {per_step_ms:>9.1f}ms  {projected_h:>10.2f}h', flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'{batch_size:>8}  {"OOM":>10}  {"---":>12}', flush=True)
EOF
