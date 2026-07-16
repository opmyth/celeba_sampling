"""Posterior factories: build the callables samplers.py needs for a given
reward type (single/joint classifier, or ImageReward prompt), so the sampler
loops themselves never know whether they're sampling smile, male+eyeglasses,
or "a bald man".
"""
from dataclasses import dataclass
from typing import Callable
import os

import torch
import torch.nn.functional as F

from utils import tokenize_prompt, _preprocess_for_blip


@dataclass
class Posterior:
    log_p_fn: Callable[[torch.Tensor], torch.Tensor]
    grad_and_log_p_fn: Callable[[torch.Tensor], tuple]
    log_accept_prob_fn: Callable[[torch.Tensor], torch.Tensor]
    reward_only_fn: Callable[[torch.Tensor], torch.Tensor]


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


def estimate_r_max(model, reward_model, prompt, device, n_samples, generator, chunk_size=32):
    """Scan n_samples prior draws z~N(0,I), score each with the RAW (unclipped)
    ImageReward Bradley-Terry model, return the max observed score. This is M:
    the bound imagereward_posterior uses to define the reward everywhere (RS,
    ULA, MALA, G_MH) as r_tilde(z) = min(IR(z), M), making r_tilde/M <= 1 hold
    *exactly* by construction instead of approximately for whatever a finite
    scan happened to see. No headroom is added on top of the observed max -
    headroom would defeat the point, since M isn't estimating some other
    unknown quantity anymore, it *is* the bound (r_tilde is defined in terms
    of it). Run with as large n_samples as compute allows: a bigger scan can
    only raise M (flattening less of the reward landscape via clipping),
    never silently break correctness the way an underestimated bound could
    under the old pre-clipping approach.
    """
    prompt_ids, prompt_mask = tokenize_prompt(reward_model, prompt, device, chunk_size)
    scan = []
    for start in range(0, n_samples, chunk_size):
        size = min(chunk_size, n_samples - start)
        z_chunk = torch.randn(size, model.latent_dim, device=device, generator=generator)
        with torch.no_grad():
            imgs = model.G(z_chunk, None)
            imgs_blip = _preprocess_for_blip(imgs, device)
            scores = reward_model.score_gard(prompt_ids[:size], prompt_mask[:size], imgs_blip).squeeze(-1)
        scan.append(scores.cpu())
    return torch.cat(scan).max().item()


def load_r_max(prompt, path='experiments/bald_ir/r_max.pt'):
    """Loads the precomputed clipping bound M for `prompt` (see estimate_r_max
    above / estimate_r_max.py). Every imagereward_posterior caller needs this
    before constructing the posterior - M has to be known upfront now, since
    the reward itself is defined in terms of it, not just RS's acceptance
    probability."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found - run `python estimate_r_max.py` first to '
            f'precompute the ImageReward clipping bound M for every bald_ir prompt.')
    r_max_by_prompt = torch.load(path, weights_only=False)
    if prompt not in r_max_by_prompt:
        raise KeyError(
            f'No precomputed r_max for prompt {prompt!r} in {path} - rerun '
            f"`python estimate_r_max.py` (it covers every prompt in "
            f"config.EXPERIMENTS['bald_ir'].prompts).")
    return r_max_by_prompt[prompt]


def imagereward_posterior(model, reward_model, prompt, device, r_max, chunk_size=32, beta=1.0) -> Posterior:
    """log p(z) = -0.5||z||^2 + beta * r_tilde(G(z), prompt), where
    r_tilde(z) = min(IR(z), r_max) clips the raw ImageReward Bradley-Terry
    score at a precomputed bound M (= r_max, see estimate_r_max/load_r_max
    above). Clipping - not just estimating a bound for RS after the fact -
    means r_tilde/M <= 1 holds exactly for every z, not approximately for
    whatever a finite scan happened to see. It also means ULA/MALA/G_MH see
    the identical clipped reward, not a different one than RS: torch.clamp's
    autograd gives exactly zero gradient wherever IR(z) > r_max (flat there,
    same ceiling as RS's envelope), so all four samplers target the same
    r_tilde-based posterior.
    beta tempers the (already-clipped) reward term (used by
    sweep_hyperparams.py's beta sweep); folded into the differentiable
    expression itself (not applied to the gradient afterward) so autograd
    produces the correct tempered gradient. reward_only_fn always returns the
    untempered (beta=1) clipped score."""
    prompt_ids, prompt_mask = tokenize_prompt(reward_model, prompt, device, chunk_size)

    def _reward_chunk(z_chunk):
        B = z_chunk.size(0)
        imgs = model.G(z_chunk, None)
        imgs_blip = _preprocess_for_blip(imgs, z_chunk.device)
        raw = reward_model.score_gard(prompt_ids[:B], prompt_mask[:B], imgs_blip).squeeze(-1)
        return torch.clamp(raw, max=r_max)

    def log_p_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk_size):
                z_chunk = z[start:start + chunk_size]
                out.append(-0.5 * (z_chunk ** 2).sum(1) + beta * _reward_chunk(z_chunk))
        return torch.cat(out)

    def grad_and_log_p_fn(z):
        z = z.detach().requires_grad_(True)
        out = []
        for start in range(0, z.size(0), chunk_size):
            z_chunk = z[start:start + chunk_size]
            log_p_chunk = -0.5 * (z_chunk ** 2).sum(1) + beta * _reward_chunk(z_chunk)
            log_p_chunk.sum().backward()
            out.append(log_p_chunk.detach())
        return z.grad.clone(), torch.cat(out)

    def reward_only_fn(z):
        out = []
        with torch.no_grad():
            for start in range(0, z.size(0), chunk_size):
                out.append(_reward_chunk(z[start:start + chunk_size]))
        return torch.cat(out)

    def log_accept_prob_fn(z):
        # r_tilde(z) <= r_max by construction (torch.clamp above), so this is
        # <= 0 exactly - the .clamp(max=0) here only guards float rounding,
        # it isn't covering for an unverified bound like the old headroom did.
        return (reward_only_fn(z) - r_max).clamp(max=0)

    return Posterior(log_p_fn, grad_and_log_p_fn, log_accept_prob_fn, reward_only_fn)
