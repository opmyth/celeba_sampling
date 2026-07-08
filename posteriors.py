"""Posterior factories: build the callables samplers.py needs for a given
reward type (single/joint classifier, or ImageReward prompt), so the sampler
loops themselves never know whether they're sampling smile, male+eyeglasses,
or "a bald man".
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from utils import tokenize_prompt, _preprocess_for_blip


@dataclass
class Posterior:
    log_p_fn: Callable[[torch.Tensor], torch.Tensor]
    grad_and_log_p_fn: Callable[[torch.Tensor], tuple]
    log_accept_prob_fn: Callable[[torch.Tensor], torch.Tensor]
    reward_only_fn: Callable[[torch.Tensor], torch.Tensor]
    estimate_r_max: Optional[Callable] = None


def classifier_posterior(model, clfs: list) -> Posterior:
    """log p(z) = -0.5||z||^2 + sum_i logsigmoid(clf_i(G(z))). Covers both the
    single-classifier case (smile/eyeglasses/bald/male) and the joint case
    (male_eye = [male_clf, eye_clf]) with the same code path."""
    chunk = model.max_batch_size

    def _reward_chunk(z_chunk):
        imgs = model.G(z_chunk, None)
        return sum(F.logsigmoid(clf(imgs).squeeze(-1)) for clf in clfs)

    def log_p_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk):
                z_chunk = z[start:start + chunk]
                out.append(-0.5 * (z_chunk ** 2).sum(1) + _reward_chunk(z_chunk))
        return torch.cat(out)

    def grad_and_log_p_fn(z):
        z = z.detach().requires_grad_(True)
        out = []
        for start in range(0, z.size(0), chunk):
            z_chunk = z[start:start + chunk]
            log_p_chunk = -0.5 * (z_chunk ** 2).sum(1) + _reward_chunk(z_chunk)
            log_p_chunk.sum().backward()
            out.append(log_p_chunk.detach())
        return z.grad.clone(), torch.cat(out)

    def reward_only_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk):
                out.append(_reward_chunk(z[start:start + chunk]))
        return torch.cat(out)

    def log_accept_prob_fn(z):
        # Prior N(0,I) is the RS proposal, so the prior term cancels exactly
        # and accept prob = prod_i sigmoid(clf_i(G(z))) <= 1 always (no bound needed).
        return reward_only_fn(z)

    return Posterior(log_p_fn, grad_and_log_p_fn, log_accept_prob_fn, reward_only_fn)


def imagereward_posterior(model, reward_model, prompt, device, chunk_size=32, beta=1.0) -> Posterior:
    """log p(z) = -0.5||z||^2 + beta * IR(G(z), prompt). IR is a Bradley-Terry
    reward used directly as log-energy (no sigmoid squashing), so RS needs an
    estimated upper bound r_max before it can compute an acceptance prob.
    beta tempers the reward term (used by sweep_hyperparams.py's beta sweep);
    it's folded into the differentiable expression itself (not applied to the
    gradient afterward) so autograd produces the correct tempered gradient.
    reward_only_fn always returns the untempered (beta=1) raw score."""
    prompt_ids, prompt_mask = tokenize_prompt(reward_model, prompt, device, chunk_size)
    r_max_holder = {}

    def _scores_chunk(z_chunk):
        B = z_chunk.size(0)
        imgs = model.G(z_chunk, None)
        imgs_blip = _preprocess_for_blip(imgs, z_chunk.device)
        return reward_model.score_gard(prompt_ids[:B], prompt_mask[:B], imgs_blip).squeeze(-1)

    def log_p_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk_size):
                z_chunk = z[start:start + chunk_size]
                out.append(-0.5 * (z_chunk ** 2).sum(1) + beta * _scores_chunk(z_chunk))
        return torch.cat(out)

    def grad_and_log_p_fn(z):
        z = z.detach().requires_grad_(True)
        out = []
        for start in range(0, z.size(0), chunk_size):
            z_chunk = z[start:start + chunk_size]
            log_p_chunk = -0.5 * (z_chunk ** 2).sum(1) + beta * _scores_chunk(z_chunk)
            log_p_chunk.sum().backward()
            out.append(log_p_chunk.detach())
        return z.grad.clone(), torch.cat(out)

    def reward_only_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk_size):
                out.append(_scores_chunk(z[start:start + chunk_size]))
        return torch.cat(out)

    def estimate_r_max(n_samples, generator):
        scan = []
        for _ in range(n_samples // chunk_size):
            z_scan = torch.randn(chunk_size, model.latent_dim, device=device, generator=generator)
            scan.append(reward_only_fn(z_scan).cpu())
        r_max = torch.cat(scan).max().item() + 0.5  # +0.5 headroom
        r_max_holder['value'] = r_max
        return r_max

    def log_accept_prob_fn(z):
        if 'value' not in r_max_holder:
            raise RuntimeError('call posterior.estimate_r_max(n_samples, generator) first')
        return (reward_only_fn(z) - r_max_holder['value']).clamp(max=0)

    return Posterior(log_p_fn, grad_and_log_p_fn, log_accept_prob_fn, reward_only_fn,
                      estimate_r_max=estimate_r_max)
