#!/bin/bash
# Analyses MCMC mixing properties: ACF, burn-in, ESS, thinning recommendations.
# Runs 32 chains × 300 steps per sampler and records log_p trace.
cd "$(dirname "$0")"

python - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
sys.path.insert(0, os.getcwd())

import torch, numpy as np, time
from model_loader import load_models
from utils import grad_and_log_posterior_celeba, log_posterior_celeba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

model, clf, _ = load_models(device)
print('Compiling...', flush=True)
clf       = torch.compile(clf)
model.G   = torch.compile(model.G)
print('Done.', flush=True)
torch.manual_seed(42)

N_CHAINS = 32
N_STEPS  = 300
DT       = 0.5
SIGMA    = 0.5
latent_dim = model.latent_dim

# ── samplers (inline, recording log_p trace) ──────────────────────────────────

def run_mala(n_chains, n_steps, dt):
    z = torch.randn(n_chains, latent_dim, device=device)
    z_grad, log_p = grad_and_log_posterior_celeba(z, model, clf)
    log_p_hist, accepts = [], []
    noise_scale = (2*dt) ** 0.5
    for _ in range(n_steps):
        log_p_hist.append(log_p.cpu().float().numpy())
        z_prop = z + dt * z_grad + noise_scale * torch.randn(n_chains, latent_dim, device=device)
        z_prop_grad, log_p_prop = grad_and_log_posterior_celeba(z_prop, model, clf)
        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad))**2,       dim=1) / (4*dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad))**2,  dim=1) / (4*dt)
        log_alpha  = torch.clamp(log_p_prop + log_q_bwd - log_p - log_q_fwd, max=0)
        accept     = torch.log(torch.rand(n_chains, device=device)) <= log_alpha
        accepts.append(accept.float().mean().item())
        z      = torch.where(accept.unsqueeze(1), z_prop,      z)
        z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
        log_p  = torch.where(accept, log_p_prop, log_p)
    return np.array(log_p_hist), np.mean(accepts)

def run_ula(n_chains, n_steps, dt):
    z = torch.randn(n_chains, latent_dim, device=device)
    z_grad, log_p = grad_and_log_posterior_celeba(z, model, clf)
    log_p_hist = []
    noise_scale = (2*dt) ** 0.5
    for _ in range(n_steps):
        log_p_hist.append(log_p.cpu().float().numpy())
        z      = z + dt * z_grad + noise_scale * torch.randn(n_chains, latent_dim, device=device)
        z_grad, log_p = grad_and_log_posterior_celeba(z, model, clf)
    return np.array(log_p_hist), 1.0

def run_gmh(n_chains, n_steps, sigma):
    z = torch.randn(n_chains, latent_dim, device=device)
    log_p = log_posterior_celeba(z, model, clf)
    log_p_hist, accepts = [], []
    for _ in range(n_steps):
        log_p_hist.append(log_p.cpu().float().numpy())
        z_prop    = z + sigma * torch.randn_like(z)
        log_p_prop = log_posterior_celeba(z_prop, model, clf)
        log_alpha  = torch.clamp(log_p_prop - log_p, max=0)
        accept     = torch.log(torch.rand(n_chains, device=device)) <= log_alpha
        accepts.append(accept.float().mean().item())
        z     = torch.where(accept.unsqueeze(1), z_prop, z)
        log_p = torch.where(accept, log_p_prop, log_p)
    return np.array(log_p_hist), np.mean(accepts)

# ── diagnostics ───────────────────────────────────────────────────────────────

def compute_acf(traces, max_lag):
    # traces: (n_steps, n_chains)
    x   = traces - traces.mean(0)
    var = x.var() + 1e-12
    return np.array([(x[:-lag] * x[lag:]).mean() / var for lag in range(1, max_lag+1)])

def compute_ess(n_steps, n_chains, acf):
    # initial positive sequence estimator
    rho_sum = 0.0
    for r in acf:
        if r <= 0:
            break
        rho_sum += r
    ess_per_chain = n_steps / max(1.0, 1 + 2*rho_sum)
    return ess_per_chain * n_chains

def find_burnin(traces, window=30):
    mean_trace = traces.mean(1)           # (n_steps,)
    final      = mean_trace[-window:].mean()
    std_final  = mean_trace[-window:].std() + 1e-6
    for i in range(len(mean_trace) - window):
        if abs(mean_trace[i:i+window].mean() - final) < 2 * std_final:
            return i
    return len(mean_trace) // 2

# ── per-step time from bench (batch=64, A6000) ───────────────────────────────
# MALA/ULA: ~300ms/step per chunk-of-64; G_MH: ~540ms/step per chunk-of-64
# For n_c chains: n_chunks = ceil(n_c/64)
MS_PER_CHUNK = {'MALA': 300, 'ULA': 300, 'G_MH': 540}

def projected_h(n_chains, total_steps, sampler):
    import math
    n_chunks    = math.ceil(n_chains / 64)
    ms_per_step = n_chunks * MS_PER_CHUNK[sampler]
    return ms_per_step * total_steps / 3_600_000

# ── report ────────────────────────────────────────────────────────────────────

def report(name, traces, accept_rate, elapsed, param):
    n_steps, n_chains = traces.shape
    max_lag = min(150, n_steps // 2)
    acf     = compute_acf(traces, max_lag)
    ess     = compute_ess(n_steps, n_chains, acf)
    burnin  = find_burnin(traces)
    eff_lag = next((i+1 for i, a in enumerate(acf) if a < 0.05), max_lag)
    neg_lag = next((i+1 for i, a in enumerate(acf) if a <= 0.0),  max_lag)

    print(f'\n{"="*60}', flush=True)
    print(f'  {name}  (param={param})', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Runtime:           {elapsed:.1f}s  ({elapsed/n_steps*1000:.0f} ms/step)', flush=True)
    if name != 'ULA':
        print(f'  Acceptance rate:   {accept_rate:.1%}', flush=True)
    print(f'  ESS total:         {ess:.0f}  ({ess/n_chains:.1f}/chain over {n_steps} steps)', flush=True)
    print(f'  ESS/step:          {ess/n_steps:.3f}', flush=True)
    print(f'  Burn-in estimate:  ~{burnin} steps', flush=True)
    print(f'  ACF < 0.05 at lag: {eff_lag}', flush=True)
    print(f'  ACF first neg:     {neg_lag}', flush=True)

    print(f'\n  ACF profile:', flush=True)
    for lag in [1, 2, 5, 10, 20, 50, 100, 150]:
        if lag <= max_lag:
            bar = '#' * max(0, int(acf[lag-1] * 30))
            print(f'    lag {lag:>3}: {acf[lag-1]:+.4f}  {bar}', flush=True)

    k = max(1, eff_lag)
    print(f'\n  --- Thinning strategy (target ~1000 samples, k={k}) ---', flush=True)
    print(f'  {"chains":>6}  {"burn-in":>8}  {"samp steps":>11}  {"samples":>8}  {"tot steps":>10}  {"proj time":>10}', flush=True)
    for n_c in [50, 100, 200]:
        samp_steps   = k * max(1, (1000 // n_c))   # enough thinned steps to get 1000/n_c samples
        total_steps  = burnin + samp_steps
        n_samples    = n_c * (samp_steps // k)
        ph           = projected_h(n_c, total_steps, name)
        print(f'  {n_c:>6}  {burnin:>8}  {samp_steps:>11}  {n_samples:>8}  {total_steps:>10}  {ph:>9.2f}h', flush=True)

    print(f'\n  Mean log_p trace (every 30 steps):', flush=True)
    mean_trace = traces.mean(1)
    for i in range(0, n_steps, 30):
        marker = ' <-- burn-in' if i == burnin else ''
        print(f'    step {i:>3}: {mean_trace[i]:+.2f}{marker}', flush=True)


# ── run all three ─────────────────────────────────────────────────────────────

for name, runner, param in [
    ('MALA', run_mala, DT),
    ('ULA',  run_ula,  DT),
    ('G_MH', run_gmh,  SIGMA),
]:
    print(f'\nRunning {name} ({N_CHAINS} chains × {N_STEPS} steps)...', flush=True)
    t0 = time.perf_counter()
    traces, accept_rate = runner(N_CHAINS, N_STEPS, param)
    elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()
    report(name, traces, accept_rate, elapsed, param)

EOF
