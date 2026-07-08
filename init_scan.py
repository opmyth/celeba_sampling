import torch


def get_init_z(kind, reward_only_fn, n_chains, latent_dim, device, generator,
                n_candidates=10000, batch_size=512):
    """Chain initialisation: 'random' draws directly; 'cold'/'warm' scan
    n_candidates ~N(0,I) draws and keep the lowest/highest-scoring n_chains
    by reward_only_fn (any monotonic reward proxy - raw prob, logsigmoid sum,
    or raw IR score all give the same ranking as their un-transformed prob).
    """
    if kind == 'random':
        return torch.randn(n_chains, latent_dim, device=device, generator=generator)

    if kind not in ('cold', 'warm'):
        raise ValueError(f"kind must be 'random', 'cold', or 'warm', got {kind!r}")

    scores, zs = [], []
    for start in range(0, n_candidates, batch_size):
        size = min(batch_size, n_candidates - start)
        z_cand = torch.randn(size, latent_dim, device=device, generator=generator)
        score = reward_only_fn(z_cand)
        scores.append(score.detach().cpu())
        zs.append(z_cand.cpu())
    all_scores = torch.cat(scores)
    all_z = torch.cat(zs)
    idx = torch.argsort(all_scores)
    selected = idx[:n_chains] if kind == 'cold' else idx[-n_chains:]
    print(f'  {kind} init score range: [{all_scores[selected].min():.4f}, '
          f'{all_scores[selected].max():.4f}]', flush=True)
    return all_z[selected].to(device)
