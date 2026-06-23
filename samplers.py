import torch
import numpy as np
from tqdm import tqdm

from utils import log_posterior_celeba, grad_log_posterior_celeba, grad_and_log_posterior_celeba

def latent_ULA_celeba(model, clf, n_chains, n_steps, dt, device, burnin=0, thin_k=1, return_diagnostics=False, z_init=None):
    latent_dim = model.latent_dim
    z = z_init.clone().to(device) if z_init is not None else torch.randn(n_chains, latent_dim, device=device)
    samples = []
    noise_scale = np.sqrt(2*dt)
    log_p_trace = []

    for step in tqdm(range(n_steps), desc="ULA"):
        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())

        if return_diagnostics:
            grad, log_p = grad_and_log_posterior_celeba(z, model, clf)
            log_p_trace.append(log_p.mean().item())
        else:
            grad = grad_log_posterior_celeba(z, model, clf)

        z = z + dt * grad + noise_scale * torch.randn(n_chains, latent_dim, device=device)

    if return_diagnostics:
        return samples, log_p_trace
    return samples

def latent_MALA_celeba(model, clf, n_chains, n_steps, dt, device, burnin=0, thin_k=1, return_diagnostics=False, z_init=None):
    latent_dim = model.latent_dim
    z = z_init.clone().to(device) if z_init is not None else torch.randn(n_chains, latent_dim, device=device)
    samples = []
    noise_scale = np.sqrt(2*dt)
    z_grad, log_p_z = grad_and_log_posterior_celeba(z, model, clf)
    accept_count = 0
    log_p_trace = []

    for step in tqdm(range(n_steps), desc="MALA"):
        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())

        if return_diagnostics:
            log_p_trace.append(log_p_z.mean().item())

        z_prop = z + dt * z_grad + noise_scale * torch.randn(n_chains, latent_dim, device=device)
        z_prop_grad, log_p_prop = grad_and_log_posterior_celeba(z_prop, model, clf)

        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad))**2, dim=1) / (4*dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad))**2, dim=1) / (4*dt)

        log_alpha = torch.clamp(
            log_p_prop + log_q_bwd - log_p_z - log_q_fwd, max=0)

        accept = torch.log(torch.rand(n_chains, device=device)) <= log_alpha
        if return_diagnostics:
            accept_count += accept.float().mean().item()
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    if return_diagnostics:
        return samples, accept_count / n_steps, log_p_trace
    return samples

def latent_Gaussian_MH_celeba(model, clf, n_chains, n_steps, sigma, device, burnin=0, thin_k=1, return_diagnostics=False, z_init=None):
    latent_dim = model.latent_dim
    z = z_init.clone().to(device) if z_init is not None else torch.randn(n_chains, latent_dim, device=device)
    log_p_z = log_posterior_celeba(z, model, clf)
    samples = []
    accept_count = 0
    log_p_trace = []

    for step in tqdm(range(n_steps), desc="G_MH"):
        if step >= burnin and (step - burnin) % thin_k == 0:
            samples.append(z.detach().clone())

        if return_diagnostics:
            log_p_trace.append(log_p_z.mean().item())

        z_prop = z + sigma * torch.randn_like(z)
        log_p_prop = log_posterior_celeba(z_prop, model, clf)

        log_alpha = torch.clamp(log_p_prop - log_p_z, max=0)
        accept = torch.log(torch.rand(n_chains, device=device)) <= log_alpha
        if return_diagnostics:
            accept_count += accept.float().mean().item()
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)

    if return_diagnostics:
        return samples, accept_count / n_steps, log_p_trace
    return samples

def rejection_sampling(model, clf, n_chains, device):
    total_accepted, total_proposed = 0, 0
    samples = []
    pbar = tqdm(total=n_chains, desc='Rejection Sampling')
    batch_size = 64
    while total_accepted < n_chains:
        z_prop = torch.randn(batch_size, model.latent_dim).to(device)
        with torch.no_grad():
            imgs_norm = model(z_prop)
            probs = torch.sigmoid(clf(imgs_norm)).squeeze()

        accept = torch.rand(batch_size).to(device) <= probs
        current_accepted = z_prop[accept]

        if current_accepted.size(0) > 0:
            samples.append(current_accepted)
            total_accepted+=current_accepted.size(0)
            pbar.update(current_accepted.size(0))

        total_proposed+=z_prop.size(0)
        print(f"accepted: {total_accepted}/{n_chains}, proposed: {total_proposed}", flush=True)
    # print(f'accept_rate: {total_accepted/total_proposed:.2f}')
    pbar.close()
    return torch.cat(samples, dim=0)[:n_chains]
        
