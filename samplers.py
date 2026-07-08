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
