import numpy as np

# ----------------------------------------------------------
# Same functions as your experiment
# ----------------------------------------------------------

def make_gmm():
    means = np.array([
        [-2.0, -2.0],
        [-2.0,  2.0],
        [ 2.0, -2.0],
        [ 2.0,  2.0],
    ])
    weights = np.ones(4) / 4
    sigma = 1.8
    return means, weights, sigma


def log_prob_and_grad(z, means, weights, sigma):
    diff = z[:, None, :] - means[None, :, :]
    sqdist = np.sum(diff**2, axis=-1)

    log_comp = -0.5 * sqdist / sigma**2 + np.log(weights)[None, :]

    m = np.max(log_comp, axis=1, keepdims=True)
    exp_shift = np.exp(log_comp - m)
    denom = np.sum(exp_shift, axis=1, keepdims=True)

    log_pi = m[:, 0] + np.log(denom[:, 0])
    resp = exp_shift / denom

    grad = -np.sum(resp[:, :, None] * diff, axis=1) / sigma**2

    return log_pi, grad


def mala_step(z, log_pi_z, grad_z, dt, means, weights, sigma, eps, u):

    z_prop = z + dt * grad_z + np.sqrt(2 * dt) * eps

    log_pi_prop, grad_prop = log_prob_and_grad(
        z_prop, means, weights, sigma
    )

    fwd = -np.sum((z_prop - z - dt * grad_z) ** 2, axis=1) / (4 * dt)
    bwd = -np.sum((z - z_prop - dt * grad_prop) ** 2, axis=1) / (4 * dt)

    log_alpha = np.minimum(
        0.0,
        (log_pi_prop + bwd) - (log_pi_z + fwd)
    )

    accept = np.log(u) <= log_alpha

    z_new = np.where(accept[:, None], z_prop, z)
    log_pi_new = np.where(accept, log_pi_prop, log_pi_z)
    grad_new = np.where(accept[:, None], grad_prop, grad_z)

    return z_new, log_pi_new, grad_new, accept


# ----------------------------------------------------------
# Sweep
# ----------------------------------------------------------

means, weights, sigma = make_gmm()

rng = np.random.default_rng(0)

z = np.array([[0.0, 0.0]])
log_pi, grad = log_prob_and_grad(z, means, weights, sigma)

n_steps = 5000

dt_values = [1, 2, 4, 6, 8, 10, 15, 20, 30]

for dt in dt_values:
    zz = z.copy()
    lp = log_pi.copy()
    gr = grad.copy()

    accepts = []
    noise_rng = np.random.default_rng(1)
    uniform_rng = np.random.default_rng(2)

    for _ in range(n_steps):
        eps = noise_rng.standard_normal((1, 2))
        u = uniform_rng.uniform(size=1)

        zz, lp, gr, acc = mala_step(
            zz, lp, gr, dt,
            means, weights, sigma,
            eps, u
        )

        accepts.append(acc[0])

    print(f"dt={dt:>4}: acceptance={np.mean(accepts):.3f}")