import torch
import gc
import numpy as np
from tqdm import tqdm

from utils import log_posterior_celeba, grad_log_posterior_celeba

def latent_ULA_celeba(model, clf, n_chains, n_steps, dt, device):
    latent_dim = model.latent_dim
    z = z = torch.randn(n_chains, latent_dim).to(device)
    samples = []
    
    for step in tqdm(range(n_steps), desc="ULA"):
    # for step in range(n_steps):
        samples.append(z.detach().clone())

        z = z + dt * grad_log_posterior_celeba(z, model, clf) + np.sqrt(2*dt) * torch.randn(n_chains, latent_dim).to(device)
        
        if step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return samples

def latent_MALA_celeba(model, clf, n_chains, n_steps, dt, device):
    latent_dim = model.latent_dim
    z = torch.randn(n_chains, latent_dim).to(device)
    samples = []
    z_grad = grad_log_posterior_celeba(z, model, clf)
    
    for step in tqdm(range(n_steps), desc="MALA"):
    # for step in range(n_steps):
        samples.append(z.detach().clone())
        
        z_prop = z + dt * z_grad + np.sqrt(2*dt) * torch.randn(n_chains, latent_dim).to(device)
        z_prop_grad = grad_log_posterior_celeba(z_prop, model, clf)
        
        log_q_fwd = -torch.sum((z_prop - (z + dt * z_grad))**2, dim=1) / (4*dt)
        log_q_bwd = -torch.sum((z - (z_prop + dt * z_prop_grad))**2, dim=1) / (4*dt)
        
        log_alpha = torch.clamp(
            log_posterior_celeba(z_prop, model, clf) + log_q_bwd - 
            log_posterior_celeba(z, model, clf) - log_q_fwd, max=0)
        
        accept = torch.log(torch.rand(n_chains).to(device)) <= log_alpha
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        z_grad = torch.where(accept.unsqueeze(1), z_prop_grad, z_grad)
    
        if step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return samples

def latent_Gaussian_MH_celeba(model, clf, n_chains, n_steps, sigma, device):
    latent_dim = model.latent_dim
    z = torch.randn(n_chains, latent_dim).to(device)
    log_p_z = log_posterior_celeba(z, model, clf)
    samples = []
    
    for step in tqdm(range(n_steps), desc="G_MH"):
    # for step in range(n_steps):
        samples.append(z.detach().clone())
        
        z_prop = z + sigma * torch.randn_like(z)
        log_p_prop = log_posterior_celeba(z_prop, model, clf)
        
        log_alpha = torch.clamp(log_p_prop - log_p_z, max=0)
        accept = torch.log(torch.rand(n_chains).to(device)) <= log_alpha
        z = torch.where(accept.unsqueeze(1), z_prop, z)
        log_p_z = torch.where(accept, log_p_prop, log_p_z)
        
        if step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return samples[int(0.2*n_steps):]

def rejection_sampling(model, clf, n_chains, device):
    total_accepted, total_proposed = 0, 0
    samples = []
    pbar = tqdm(total=n_chains, desc='Rejection Sampling')
    batch_size = 512
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
        
