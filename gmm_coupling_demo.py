"""
Two MALA chains on a 2D four-component GMM: shared Langevin noise +
accept/reject draws vs. independent ones.

Outputs (same experiment, same initial states, in both figures):
- distance_comparison.png  distance between the chains over time
- trajectories.png         the chains themselves, over the GMM contours

Sampler and target maths are imported unchanged from temp.py; this file only
picks the parameters and draws the plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from temp import run_pair, evaluate_density_grid

# ~57% MALA acceptance, four visually separated components.
MEANS = np.array([[-2.5, -2.5], [-2.5, 2.5], [2.5, -2.5], [2.5, 2.5]])
WEIGHTS = np.ones(4) / 4
SIGMA = 1.0
DT = 1.2
N_STEPS = 1000
SEED = 7

Z1_INITIAL = np.array([[-3.5, -3.0]])
Z2_INITIAL = np.array([[3.5, 3.0]])

OUTPUT_PATH = "gmm_2d_results/distance_comparison.png"
TRAJECTORY_PATH = "gmm_2d_results/trajectories.png"
N_SHOW = 300          # trajectory steps drawn (coalescence happens at ~284)
AXIS_LIMIT = 7


common = dict(means=MEANS, weights=WEIGHTS, sigma=SIGMA, dt=DT,
              n_steps=N_STEPS, seed=SEED)

shared = run_pair(Z1_INITIAL, Z2_INITIAL, shared_randomness=True, **common)
independent = run_pair(Z1_INITIAL, Z2_INITIAL, shared_randomness=False,
                       **common)

print(f"acceptance rate: {shared['accepts1'].mean():.2f}")
print(f"final distance:  shared {shared['distances'][-1]:.2e}, "
      f"independent {independent['distances'][-1]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

axes[0].plot(shared["distances"])
axes[0].set_title("Shared noise and accept/reject draws")
axes[0].set_ylabel("distance between chains")

axes[1].plot(independent["distances"])
axes[1].set_title("Independent noise and accept/reject draws")

for ax in axes:
    ax.set_xlabel("MALA step")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150)

print(f"saved {OUTPUT_PATH}")


# Trajectories over the target density - same two runs as above.
x_grid, y_grid, density = evaluate_density_grid(MEANS, WEIGHTS, SIGMA)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for ax, result, title in [
    (axes[0], shared, "Shared noise and accept/reject draws"),
    (axes[1], independent, "Independent noise and accept/reject draws"),
]:
    ax.contour(x_grid, y_grid, density, levels=8, colors="gray",
               linewidths=0.6, alpha=0.6)
    ax.plot(MEANS[:, 0], MEANS[:, 1], "+", color="black", markersize=9,
            label="component means")

    path1 = result["path1"][:N_SHOW + 1]
    path2 = result["path2"][:N_SHOW + 1]

    # Tie-lines join the two chains at the SAME step: they shrink to nothing
    # under shared randomness and stay long under independent randomness.
    for t in range(0, N_SHOW + 1, 25):
        ax.plot([path1[t, 0], path2[t, 0]], [path1[t, 1], path2[t, 1]],
                color="0.6", lw=0.6, alpha=0.7, zorder=1)

    ax.plot(path1[:, 0], path1[:, 1], lw=0.8, alpha=0.5, label="chain 1")
    ax.plot(path2[:, 0], path2[:, 1], lw=0.8, alpha=0.5, label="chain 2")
    ax.plot(path1[0, 0], path1[0, 1], "o", color="C0", markersize=10,
            markeredgecolor="black", label="chain 1 start")
    ax.plot(path2[0, 0], path2[0, 1], "s", color="C1", markersize=10,
            markeredgecolor="black", label="chain 2 start")
    ax.plot(path1[-1, 0], path1[-1, 1], "*", color="C0", markersize=17,
            markeredgecolor="black", zorder=3,
            label=f"chain 1 at step {N_SHOW}")
    ax.plot(path2[-1, 0], path2[-1, 1], "X", color="C1", markersize=11,
            markeredgecolor="black", zorder=3,
            label=f"chain 2 at step {N_SHOW}")

    gap = result["distances"][N_SHOW]
    ax.set_title(f"{title}\ndistance at step {N_SHOW}: {gap:.2f}")
    ax.set_xlabel("$z_1$")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")

axes[0].set_ylabel("$z_2$")
axes[0].legend(loc="lower right", fontsize=8)   # empty corner in both panels

fig.suptitle(f"First {N_SHOW} MALA steps")
fig.tight_layout()
fig.savefig(TRAJECTORY_PATH, dpi=150)

print(f"saved {TRAJECTORY_PATH}")
