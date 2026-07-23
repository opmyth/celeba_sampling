"""
Visual demonstration of MALA coupling on a 2D four-component GMM.

The same two initial states are used in both experiments:

1. Shared randomness:
   Both chains use the same Langevin noise and accept/reject uniform draw.

2. Independent randomness:
   Each chain uses its own Langevin noise and uniform draw.

Outputs:
- GMM density contours with both chain trajectories.
- Distance between the chains over time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = "gmm_2d_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Four-component 2D GMM
# ---------------------------------------------------------

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
    """
    Evaluate the unnormalised log-density and gradient of the GMM.

    z has shape (n, 2).
    """
    diff = z[:, None, :] - means[None, :, :]
    squared_distance = np.sum(diff**2, axis=-1)

    log_components = (
        -0.5 * squared_distance / sigma**2
        + np.log(weights)[None, :]
    )

    maximum = np.max(log_components, axis=1, keepdims=True)
    shifted = np.exp(log_components - maximum)
    denominator = np.sum(shifted, axis=1, keepdims=True)

    log_pi = maximum[:, 0] + np.log(denominator[:, 0])
    responsibilities = shifted / denominator

    grad = (
        -np.sum(responsibilities[:, :, None] * diff, axis=1)
        / sigma**2
    )

    return log_pi, grad


# ---------------------------------------------------------
# MALA
# ---------------------------------------------------------

def mala_step(
    z,
    log_pi_z,
    grad_z,
    dt,
    means,
    weights,
    sigma,
    epsilon,
    uniform,
):
    z_proposed = (
        z
        + dt * grad_z
        + np.sqrt(2.0 * dt) * epsilon
    )

    log_pi_proposed, grad_proposed = log_prob_and_grad(
        z_proposed,
        means,
        weights,
        sigma,
    )

    log_q_forward = (
        -np.sum(
            (z_proposed - z - dt * grad_z) ** 2,
            axis=1,
        )
        / (4.0 * dt)
    )

    log_q_reverse = (
        -np.sum(
            (z - z_proposed - dt * grad_proposed) ** 2,
            axis=1,
        )
        / (4.0 * dt)
    )

    log_alpha = np.minimum(
        0.0,
        log_pi_proposed
        + log_q_reverse
        - log_pi_z
        - log_q_forward,
    )

    accepted = np.log(uniform) <= log_alpha

    z_new = np.where(accepted[:, None], z_proposed, z)
    log_pi_new = np.where(
        accepted,
        log_pi_proposed,
        log_pi_z,
    )
    grad_new = np.where(
        accepted[:, None],
        grad_proposed,
        grad_z,
    )

    return z_new, log_pi_new, grad_new, accepted


# ---------------------------------------------------------
# Coupled-chain experiment
# ---------------------------------------------------------

def run_pair(
    z1_initial,
    z2_initial,
    means,
    weights,
    sigma,
    dt,
    n_steps,
    seed,
    shared_randomness,
):
    z1 = z1_initial.copy()
    z2 = z2_initial.copy()

    log_pi1, grad1 = log_prob_and_grad(
        z1,
        means,
        weights,
        sigma,
    )
    log_pi2, grad2 = log_prob_and_grad(
        z2,
        means,
        weights,
        sigma,
    )

    noise_rng1 = np.random.default_rng(seed + 100)
    noise_rng2 = np.random.default_rng(seed + 101)

    uniform_rng1 = np.random.default_rng(seed + 200)
    uniform_rng2 = np.random.default_rng(seed + 201)

    path1 = np.zeros((n_steps + 1, 2))
    path2 = np.zeros((n_steps + 1, 2))

    distances = np.zeros(n_steps + 1)
    mismatches = np.zeros(n_steps, dtype=bool)

    accepts1 = np.zeros(n_steps, dtype=bool)
    accepts2 = np.zeros(n_steps, dtype=bool)

    path1[0] = z1[0]
    path2[0] = z2[0]
    distances[0] = np.linalg.norm(z1 - z2)

    for t in range(n_steps):
        if shared_randomness:
            epsilon = noise_rng1.standard_normal((1, 2))
            uniform = uniform_rng1.uniform(size=1)

            epsilon1 = epsilon
            epsilon2 = epsilon
            uniform1 = uniform
            uniform2 = uniform

        else:
            epsilon1 = noise_rng1.standard_normal((1, 2))
            epsilon2 = noise_rng2.standard_normal((1, 2))

            uniform1 = uniform_rng1.uniform(size=1)
            uniform2 = uniform_rng2.uniform(size=1)

        z1, log_pi1, grad1, accepted1 = mala_step(
            z1,
            log_pi1,
            grad1,
            dt,
            means,
            weights,
            sigma,
            epsilon1,
            uniform1,
        )

        z2, log_pi2, grad2, accepted2 = mala_step(
            z2,
            log_pi2,
            grad2,
            dt,
            means,
            weights,
            sigma,
            epsilon2,
            uniform2,
        )

        path1[t + 1] = z1[0]
        path2[t + 1] = z2[0]

        distances[t + 1] = np.linalg.norm(z1 - z2)

        accepts1[t] = accepted1[0]
        accepts2[t] = accepted2[0]
        mismatches[t] = accepted1[0] != accepted2[0]

    return {
        "path1": path1,
        "path2": path2,
        "distances": distances,
        "mismatches": mismatches,
        "accepts1": accepts1,
        "accepts2": accepts2,
    }


# ---------------------------------------------------------
# Density grid
# ---------------------------------------------------------

def evaluate_density_grid(means, weights, sigma):
    x_values = np.linspace(-7, 7, 300)
    y_values = np.linspace(-7, 7, 300)

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    points = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
    ])

    log_density, _ = log_prob_and_grad(
        points,
        means,
        weights,
        sigma,
    )

    density = np.exp(
        log_density.reshape(x_grid.shape)
        - np.max(log_density)
    )

    return x_grid, y_grid, density


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_trajectory(
    ax,
    result,
    x_grid,
    y_grid,
    density,
    means,
    title,
):
    ax.contourf(
        x_grid,
        y_grid,
        density,
        levels=30,
        alpha=0.7,
    )

    ax.contour(
        x_grid,
        y_grid,
        density,
        levels=12,
        linewidths=0.5,
        alpha=0.6,
    )

    path1 = result["path1"]
    path2 = result["path2"]

    ax.plot(
        path1[:, 0],
        path1[:, 1],
        linewidth=1.1,
        label="Chain 1",
    )

    ax.plot(
        path2[:, 0],
        path2[:, 1],
        linewidth=1.1,
        label="Chain 2",
    )

    # Initial points
    ax.scatter(
        path1[0, 0],
        path1[0, 1],
        s=100,
        marker="o",
        edgecolors="black",
        label="Chain 1 start",
    )

    ax.scatter(
        path2[0, 0],
        path2[0, 1],
        s=100,
        marker="s",
        edgecolors="black",
        label="Chain 2 start",
    )

    # Final points
    ax.scatter(
        path1[-1, 0],
        path1[-1, 1],
        s=130,
        marker="*",
        edgecolors="black",
        label="Chain 1 end",
    )

    ax.scatter(
        path2[-1, 0],
        path2[-1, 1],
        s=130,
        marker="X",
        edgecolors="black",
        label="Chain 2 end",
    )

    # GMM component means
    ax.scatter(
        means[:, 0],
        means[:, 1],
        marker="+",
        s=130,
        linewidths=2,
        label="Component means",
    )

    ax.set_title(title)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)


def print_summary(name, result):
    print(f"\n{name}")
    print(
        f"  initial distance: "
        f"{result['distances'][0]:.3f}"
    )
    print(
        f"  minimum distance: "
        f"{result['distances'].min():.6f}"
    )
    print(
        f"  final distance: "
        f"{result['distances'][-1]:.3f}"
    )
    print(
        f"  chain 1 acceptance rate: "
        f"{result['accepts1'].mean():.3f}"
    )
    print(
        f"  chain 2 acceptance rate: "
        f"{result['accepts2'].mean():.3f}"
    )
    print(
        f"  accept/reject mismatches: "
        f"{result['mismatches'].sum()}"
    )


def main():
    means, weights, sigma = make_gmm()

    dt = 0.1
    n_steps = 1000
    seed = 7

    # Start near opposite components.
    z1_initial = np.array([[-3.5, -3.0]])
    z2_initial = np.array([[ 3.5,  3.0]])

    shared_result = run_pair(
        z1_initial,
        z2_initial,
        means,
        weights,
        sigma,
        dt,
        n_steps,
        seed,
        shared_randomness=True,
    )

    independent_result = run_pair(
        z1_initial,
        z2_initial,
        means,
        weights,
        sigma,
        dt,
        n_steps,
        seed,
        shared_randomness=False,
    )

    print_summary(
        "SHARED NOISE AND UNIFORMS",
        shared_result,
    )
    print_summary(
        "INDEPENDENT NOISE AND UNIFORMS",
        independent_result,
    )

    x_grid, y_grid, density = evaluate_density_grid(
        means,
        weights,
        sigma,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    plot_trajectory(
        axes[0, 0],
        shared_result,
        x_grid,
        y_grid,
        density,
        means,
        "Shared noise and accept/reject draws",
    )

    plot_trajectory(
        axes[0, 1],
        independent_result,
        x_grid,
        y_grid,
        density,
        means,
        "Independent noise and accept/reject draws",
    )

    axes[1, 0].plot(
        shared_result["distances"],
        linewidth=1.2,
    )
    axes[1, 0].set_title(
        "Distance between chains: shared randomness"
    )
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel(r"$\|z_t^{(1)}-z_t^{(2)}\|_2$")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        independent_result["distances"],
        linewidth=1.2,
    )
    axes[1, 1].set_title(
        "Distance between chains: independent randomness"
    )
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel(r"$\|z_t^{(1)}-z_t^{(2)}\|_2$")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        "Effect of shared randomness on two MALA chains",
        fontsize=15,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "mala_gmm_2d_shared_vs_independent.png",
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()