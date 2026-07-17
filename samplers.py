import torch
from tqdm import tqdm

from posteriors import Posterior

# NOTE: torch.randn_like/torch.rand_like don't accept a `generator=` kwarg,
# so every noise draw below uses the explicit-shape form instead.


def _draw_noise(shape, device, generator):
    return torch.randn(shape, device=device, generator=generator)


def latent_ULA_celeba(posterior: Posterior, n_chains, n_steps, dt, latent_dim, device,
                       generator=None, burnin=0, thin_k=1, z_init=None,
                       return_diagnostics=False):
    """Unadjusted Langevin. Returns (samples, log_p_kept, accept_rate, log_p_trace):
      samples, log_p_kept: lists of (n_chains, latent_dim)/(n_chains,) tensors, one per
        kept step (step >= burnin and (step - burnin) % thin_k == 0). log_p_kept lets
        callers derive avg reward (= log_p + 0.5||z||^2) without a second forward pass.
      accept_rate: None (ULA has no accept/reject step).
      log_p_trace: (n_steps, n_chains) tensor if return_diagnostics else None.
    """
    z = z_init.clone().to(device) if z_init is not None else \
        _draw_noise((n_chains, latent_dim), device, generator)
    samples, log_p_kept = [], []
    noise_scale = (2 * dt) ** 0.5
    log_p_trace = [] if return_diagnostics else None

    for step in tqdm(range(n_steps), desc="ULA"):
        z_grad, log_p_z = posterior.grad_and_log_p_fn(z)

        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())
            log_p_kept.append(log_p_z.detach().clone())
        if return_diagnostics:
            log_p_trace.append(log_p_z.detach().cpu().clone())

        noise = _draw_noise(z.shape, device, generator)
        z = z + dt * z_grad + noise_scale * noise

    if return_diagnostics:
        log_p_trace = torch.stack(log_p_trace)
    return samples, log_p_kept, None, log_p_trace


def latent_MALA_celeba(posterior: Posterior, n_chains, n_steps, dt, latent_dim, device,
                        generator=None, burnin=0, thin_k=1, z_init=None,
                        return_diagnostics=False):
    """Metropolis-adjusted Langevin. Returns (samples, log_p_kept, accept_rate, log_p_trace)
    — see latent_ULA_celeba for the shape of each; accept_rate is a float here."""
    z = z_init.clone().to(device) if z_init is not None else \
        _draw_noise((n_chains, latent_dim), device, generator)
    samples, log_p_kept = [], []
    noise_scale = (2 * dt) ** 0.5
    z_grad, log_p_z = posterior.grad_and_log_p_fn(z)
    accept_count = 0
    log_p_trace = [] if return_diagnostics else None

    for step in tqdm(range(n_steps), desc="MALA"):
        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())
            log_p_kept.append(log_p_z.detach().clone())
        if return_diagnostics:
            log_p_trace.append(log_p_z.detach().cpu().clone())

        noise = _draw_noise(z.shape, device, generator)
        z_prop = z + dt * z_grad + noise_scale * noise
        z_prop_grad, log_p_prop = posterior.grad_and_log_p_fn(z_prop)

        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad)) ** 2, dim=1) / (4 * dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad)) ** 2, dim=1) / (4 * dt)
        log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)

        u = torch.rand(n_chains, device=device, generator=generator)
        accept = torch.log(u) <= log_alpha
        accept_count += accept.float().mean().item()

        mask = accept.unsqueeze(1)
        z = torch.where(mask, z_prop, z)
        z_grad = torch.where(mask, z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    if return_diagnostics:
        log_p_trace = torch.stack(log_p_trace)
    return samples, log_p_kept, accept_count / n_steps, log_p_trace


def anneal_temps(n_temps):
    """Geometric temperature ladder sqrt(2)^n for n = n_temps..0 (largest T
    first, ends at 1). n_temps=6 -> 7 temps, ~8 down to 1."""
    return [2 ** (n / 2) for n in range(n_temps, -1, -1)]


def anneal_dt_schedule(dt_base, n_temps, mode='flat'):
    """Per-temperature annealing step sizes, aligned with anneal_temps
    (largest-T first). 'flat' -> None (caller uses dt_base at every temp);
    'inv_sqrt_t' -> dt_base / sqrt(T). Returns a list (or None for flat)
    suitable for the dt_anneal argument below."""
    if mode == 'flat':
        return None
    if mode == 'inv_sqrt_t':
        return [dt_base / (T ** 0.5) for T in anneal_temps(n_temps)]
    raise ValueError(f"anneal_dt_mode must be 'flat' or 'inv_sqrt_t', got {mode!r}")


def anneal_to_T1(posterior: Posterior, z, n_chains, latent_dim, device, generator,
                 n_temps, annealing_steps, dt, dt_anneal=None, return_trace=False):
    """Run the annealing phase in place on z: MALA targeting pi_T for T from
    sqrt(2)^n_temps down to 1, annealing_steps split evenly across temps. NO
    samples kept (this is burn-in). Returns (z_final, per_temp, trace) where
    per_temp is a list of (T, dt_T, accept_rate). Shared by
    latent_annealed_MALA_celeba and the accept-rate probe so both compute
    exactly the same thing.

    Likelihood-only tempering: log pi_T = -0.5||z||^2 + (1/T) log r(z),
    grad = -z + (1/T) grad log r(z) - derived arithmetically from the
    untempered Posterior. scale_drift variant (drift dt*grad, noise
    sqrt(2dt)*eta), accept ratio + proposal q both under pi_T."""
    temps = anneal_temps(n_temps)
    if dt_anneal is None:
        dt_anneal = [dt] * len(temps)
    if len(dt_anneal) != len(temps):
        raise ValueError(f'dt_anneal needs {len(temps)} entries (one per temperature), got {len(dt_anneal)}')
    steps_per_temp = annealing_steps // len(temps)

    def _tempered_grad_and_log_p(z, T):
        g, lp = posterior.grad_and_log_p_fn(z)
        if T == 1:
            return g, lp
        log_prior = -0.5 * (z ** 2).sum(1)
        reward = lp - log_prior          # log r(z)
        reward_grad = g + z              # grad log r(z)
        return -z + reward_grad / T, log_prior + reward / T

    per_temp = []
    trace = [] if return_trace else None
    for T, dt_T in zip(temps, dt_anneal):
        if steps_per_temp == 0:
            break
        noise_scale = (2 * dt_T) ** 0.5
        z_grad, log_p_z = _tempered_grad_and_log_p(z, T)
        accept_count = 0
        for _ in tqdm(range(steps_per_temp), desc=f'anneal T={T:.2f}'):
            noise = _draw_noise(z.shape, device, generator)
            z_prop = z + dt_T * z_grad + noise_scale * noise
            z_prop_grad, log_p_prop = _tempered_grad_and_log_p(z_prop, T)

            log_q_fwd = -torch.sum((z_prop - (z + dt_T * z_grad)) ** 2, dim=1) / (4 * dt_T)
            log_q_bwd = -torch.sum((z - (z_prop + dt_T * z_prop_grad)) ** 2, dim=1) / (4 * dt_T)
            log_alpha = torch.clamp(log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)

            u = torch.rand(n_chains, device=device, generator=generator)
            accept = torch.log(u) <= log_alpha
            accept_count += accept.float().mean().item()

            mask = accept.unsqueeze(1)
            z = torch.where(mask, z_prop, z)
            z_grad = torch.where(mask, z_prop_grad, z_grad)
            log_p_z = torch.where(accept, log_p_prop, log_p_z)
            if return_trace:
                trace.append(log_p_z.detach().cpu().clone())
        rate = accept_count / steps_per_temp
        per_temp.append((T, dt_T, rate))
        print(f'  anneal T={T:.3f} (dt={dt_T:.4g}): accept={rate:.1%}', flush=True)
    return z, per_temp, trace


def latent_annealed_MALA_celeba(posterior: Posterior, n_chains, n_steps, dt, latent_dim, device,
                                 generator=None, burnin=0, thin_k=1, z_init=None,
                                 return_diagnostics=False,
                                 n_temps=6, annealing_steps=700, dt_anneal=None):
    """Annealed MALA with LIKELIHOOD-ONLY tempering (supervisor-confirmed):

        log pi_T(z) = -0.5||z||^2 + (1/T) * log r(z)
        grad       = -z + (1/T) * grad log r(z)

    The prior term is never tempered - only the reward. Both tempered
    quantities are derived arithmetically from the untempered Posterior
    (log r = log_p + 0.5||z||^2, grad log r = grad + z), so this works for
    any experiment kind with no posteriors.py changes.

    Follows the 'scale_drift' variant of annealed-langvein-ULA-MALA.ipynb:
    drift dt * grad log pi_T, noise unchanged sqrt(2dt)*eta - i.e. each
    temperature phase is literally plain MALA targeting pi_T, with the
    acceptance ratio and proposal q both computed under pi_T consistently.

    Two phases:
      a. annealing: temps = sqrt(2)^n for n = n_temps..0 (n_temps+1
         temperatures, ~8 -> 1 at the default n_temps=6), annealing_steps
         split evenly across them. NO samples kept here - T>1 samples are
         from a tempered (wrong) distribution and never count as posterior
         samples. This phase is conceptually the (extra) burn-in.
      b. T=1 tail: n_steps of the plain latent_MALA_celeba (literally
         delegated to it, so tail behavior is identical to the non-annealed
         sampler by construction) with the usual burnin/thin_k applied to
         the tail ONLY - kept_per_chain stays comparable to non-annealed
         runs at the same n_steps/burnin/thin_k.

    dt_anneal: optional per-temperature step size list (len n_temps+1),
    largest-T first, letting dt shrink as T->1 while the target sharpens.
    None = use `dt` at every temperature. The tail always uses `dt`.

    Returns the standard (samples, log_p_kept, accept_rate, log_p_trace)
    contract: accept_rate is the T=1 tail's (per-temperature annealing
    accept rates are printed); log_p_trace, if requested, is the full
    annealing+tail concatenation ((annealing_steps_used + n_steps, n_chains))
    with annealing entries being log pi_T at that step's own temperature.
    """
    z = z_init.clone().to(device) if z_init is not None else \
        _draw_noise((n_chains, latent_dim), device, generator)

    # Phase a - annealing (no samples kept). Extracted to anneal_to_T1 so the
    # accept-rate probe computes exactly this; RNG draw order is unchanged from
    # the original inline loop, so the eyeglasses validation still reproduces.
    z, _per_temp, anneal_trace = anneal_to_T1(
        posterior, z, n_chains, latent_dim, device, generator,
        n_temps, annealing_steps, dt, dt_anneal, return_trace=return_diagnostics)

    # Phase b - T=1 tail: delegate to the plain sampler so tail behavior is
    # identical to the non-annealed pipeline by construction (same code, same
    # burnin/thin_k) - the annealed run only differs by its z_init.
    samples, log_p_kept, accept_rate, tail_trace = latent_MALA_celeba(
        posterior, n_chains, n_steps, dt, latent_dim, device,
        generator=generator, burnin=burnin, thin_k=thin_k, z_init=z,
        return_diagnostics=return_diagnostics)

    log_p_trace = None
    if return_diagnostics:
        log_p_trace = torch.cat([torch.stack(anneal_trace), tail_trace]) if anneal_trace else tail_trace
    return samples, log_p_kept, accept_rate, log_p_trace


def latent_Gaussian_MH_celeba(posterior: Posterior, n_chains, n_steps, sigma, latent_dim, device,
                               generator=None, burnin=0, thin_k=1, z_init=None,
                               return_diagnostics=False):
    """Random-walk Gaussian MH. Returns (samples, log_p_kept, accept_rate, log_p_trace)."""
    z = z_init.clone().to(device) if z_init is not None else \
        _draw_noise((n_chains, latent_dim), device, generator)
    log_p_z = posterior.log_p_fn(z)
    samples, log_p_kept = [], []
    accept_count = 0
    log_p_trace = [] if return_diagnostics else None

    for step in tqdm(range(n_steps), desc="G_MH"):
        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())
            log_p_kept.append(log_p_z.detach().clone())
        if return_diagnostics:
            log_p_trace.append(log_p_z.detach().cpu().clone())

        noise = _draw_noise(z.shape, device, generator)
        z_prop = z + sigma * noise
        log_p_prop = posterior.log_p_fn(z_prop)

        log_alpha = torch.clamp(log_p_prop - log_p_z, max=0)
        u = torch.rand(n_chains, device=device, generator=generator)
        accept = torch.log(u) <= log_alpha
        accept_count += accept.float().mean().item()

        mask = accept.unsqueeze(1)
        z = torch.where(mask, z_prop, z)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    if return_diagnostics:
        log_p_trace = torch.stack(log_p_trace)
    return samples, log_p_kept, accept_count / n_steps, log_p_trace


def rejection_sampling(posterior: Posterior, n_target, latent_dim, device,
                        generator=None, batch_size=64):
    """Accept z ~ N(0,I) with prob exp(log_accept_prob_fn(z)) <= 1. Returns
    (samples, accept_rate); reward/log_p for the accepted set is cheap to get
    afterward via posterior.reward_only_fn (one pass over the final set only,
    not per-proposal, so nothing is recomputed redundantly here)."""
    total_accepted, total_proposed = 0, 0
    samples = []
    pbar = tqdm(total=n_target, desc='Rejection Sampling')
    while total_accepted < n_target:
        z_prop = _draw_noise((batch_size, latent_dim), device, generator)
        log_accept = posterior.log_accept_prob_fn(z_prop)
        u = torch.rand(batch_size, device=device, generator=generator)
        accept = torch.log(u) <= log_accept
        current_accepted = z_prop[accept]

        if current_accepted.size(0) > 0:
            samples.append(current_accepted)
            total_accepted += current_accepted.size(0)
            pbar.update(current_accepted.size(0))
        total_proposed += batch_size
    pbar.close()

    accept_rate = total_accepted / total_proposed
    return torch.cat(samples, dim=0)[:n_target], accept_rate
