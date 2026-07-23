"""
Coupled MALA on a "rippled Gaussian" target.

    log pi(z)      = -0.5||z||^2 + lambda * sum_k cos(omega * z_k)
    grad log pi(z) = -z - lambda * omega * sin(omega * z_k)     (per coordinate)

Motivation: a smooth GMM target reproduces shared-noise coalescence but NOT
the repeated escape/re-convergence cycling seen on the real StyleGAN2
posterior. The ripple term adds local structure at a controllable length
scale (2*pi/omega) and depth (lambda), so two very close points can sit on
different parts of a ripple and keep meaningfully different gradients - and
therefore different acceptance probabilities - even at tiny separation.

The target is a proper density for every (lambda, omega): cos is bounded, so
pi(z) <= exp(-0.5||z||^2 + lambda*d), which is integrable. The Gaussian term
always dominates as ||z|| -> infinity.

The MALA/coupling machinery mirrors temp.py exactly (same proposal, same
Metropolis-Hastings ratio, same RNG stream layout); the only change is that
the target is passed in as a callable so it works in any dimension.
`self_test()` checks this re-implementation reproduces temp.py's run_pair
bit-for-bit on the GMM target.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = "ripple_results"

DIM = 30
N_STEPS = 20000
SEED = 11


# ---------------------------------------------------------
# Rippled Gaussian target
# ---------------------------------------------------------

def make_ripple_target(lam, omega):
    """Return a log_prob_and_grad(z) closure for the rippled Gaussian."""

    def log_prob_and_grad(z):
        """z has shape (n, d). Returns (log_pi, grad) - unnormalised."""
        log_pi = (
            -0.5 * np.sum(z**2, axis=1)
            + lam * np.sum(np.cos(omega * z), axis=1)
        )

        grad = -z - lam * omega * np.sin(omega * z)

        return log_pi, grad

    return log_prob_and_grad


def ripple_curvature_ratio(lam, omega):
    """
    lambda*omega^2, the ripple's peak curvature relative to the Gaussian's.

    Per coordinate the log-density curvature is
        f''(x) = -1 - lambda*omega^2*cos(omega*x),
    which turns positive (locally convex log-density, i.e. genuine local
    structure rather than a gentle wobble) wherever cos(omega*x) < 0 once
    lambda*omega^2 > 1. Below 1 the ripple only perturbs a globally
    log-concave target.
    """
    return lam * omega**2


def make_noisy_target(base_target, tau, seed):
    """
    Wrap a target so every evaluation carries fresh independent noise.

    This mimics the real pipeline: StyleGAN2Wrapper calls `self.G(z, None)`
    without overriding `noise_mode`, so the synthesis network's default
    `noise_mode='random'` injects fresh per-layer noise on every forward
    pass. Evaluating the SAME latent twice therefore gives different rewards
    and different gradients - the log-density is a stochastic function of z.

    Crucially the two coupled chains do NOT share this noise: sharing the
    Langevin noise and the accept/reject uniform says nothing about the
    generator's internal randomness.
    """
    rng = np.random.default_rng(seed)

    def target(z):
        log_pi, grad = base_target(z)
        return (log_pi + tau * rng.standard_normal(log_pi.shape),
                grad + tau * rng.standard_normal(grad.shape))

    return target


def check_gradient(lam, omega, dim=8, n_points=5, seed=0, eps=1e-6):
    """Verify the analytic gradient against central finite differences."""
    target = make_ripple_target(lam, omega)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_points, dim)) * 1.5

    _, analytic = target(z)

    numerical = np.zeros_like(z)
    for k in range(dim):
        step = np.zeros_like(z)
        step[:, k] = eps
        log_plus, _ = target(z + step)
        log_minus, _ = target(z - step)
        numerical[:, k] = (log_plus - log_minus) / (2 * eps)

    max_abs = np.max(np.abs(analytic - numerical))
    scale = np.maximum(np.abs(analytic), 1.0)
    max_rel = np.max(np.abs(analytic - numerical) / scale)

    return max_abs, max_rel


# ---------------------------------------------------------
# MALA + coupling (structure identical to temp.py)
# ---------------------------------------------------------

def mala_step(z, log_pi_z, grad_z, dt, log_prob_and_grad, epsilon, uniform):
    z_proposed = (
        z
        + dt * grad_z
        + np.sqrt(2.0 * dt) * epsilon
    )

    log_pi_proposed, grad_proposed = log_prob_and_grad(z_proposed)

    log_q_forward = (
        -np.sum((z_proposed - z - dt * grad_z) ** 2, axis=1)
        / (4.0 * dt)
    )

    log_q_reverse = (
        -np.sum((z - z_proposed - dt * grad_proposed) ** 2, axis=1)
        / (4.0 * dt)
    )

    log_alpha = np.minimum(
        0.0,
        log_pi_proposed + log_q_reverse - log_pi_z - log_q_forward,
    )

    accepted = np.log(uniform) <= log_alpha

    z_new = np.where(accepted[:, None], z_proposed, z)
    log_pi_new = np.where(accepted, log_pi_proposed, log_pi_z)
    grad_new = np.where(accepted[:, None], grad_proposed, grad_z)

    return z_new, log_pi_new, grad_new, accepted, log_alpha


def run_pair(z1_initial, z2_initial, log_prob_and_grad, dt, n_steps, seed,
             shared_randomness, store_paths=False):
    """
    Two MALA chains on the same target, sharing or not sharing randomness.

    Returns per-step distance, accept flags, accept/reject mismatch flags,
    and the per-step gap in acceptance probability (|alpha_1 - alpha_2|),
    which is what drives mismatches when the chains are close.
    """
    z1 = z1_initial.copy()
    z2 = z2_initial.copy()

    log_pi1, grad1 = log_prob_and_grad(z1)
    log_pi2, grad2 = log_prob_and_grad(z2)

    noise_rng1 = np.random.default_rng(seed + 100)
    noise_rng2 = np.random.default_rng(seed + 101)

    uniform_rng1 = np.random.default_rng(seed + 200)
    uniform_rng2 = np.random.default_rng(seed + 201)

    dim = z1.shape[1]

    distances = np.zeros(n_steps + 1)
    mismatches = np.zeros(n_steps, dtype=bool)
    accepts1 = np.zeros(n_steps, dtype=bool)
    accepts2 = np.zeros(n_steps, dtype=bool)
    alpha_gap = np.zeros(n_steps)
    norms = np.zeros(n_steps + 1)

    distances[0] = np.linalg.norm(z1 - z2)
    norms[0] = np.linalg.norm(z1)

    paths = ([z1[0].copy()], [z2[0].copy()]) if store_paths else None

    for t in range(n_steps):
        if shared_randomness:
            epsilon = noise_rng1.standard_normal((1, dim))
            uniform = uniform_rng1.uniform(size=1)

            epsilon1 = epsilon
            epsilon2 = epsilon
            uniform1 = uniform
            uniform2 = uniform

        else:
            epsilon1 = noise_rng1.standard_normal((1, dim))
            epsilon2 = noise_rng2.standard_normal((1, dim))

            uniform1 = uniform_rng1.uniform(size=1)
            uniform2 = uniform_rng2.uniform(size=1)

        z1, log_pi1, grad1, accepted1, log_alpha1 = mala_step(
            z1, log_pi1, grad1, dt, log_prob_and_grad, epsilon1, uniform1,
        )

        z2, log_pi2, grad2, accepted2, log_alpha2 = mala_step(
            z2, log_pi2, grad2, dt, log_prob_and_grad, epsilon2, uniform2,
        )

        distances[t + 1] = np.linalg.norm(z1 - z2)
        norms[t + 1] = np.linalg.norm(z1)

        accepts1[t] = accepted1[0]
        accepts2[t] = accepted2[0]
        mismatches[t] = accepted1[0] != accepted2[0]
        alpha_gap[t] = abs(np.exp(log_alpha1[0]) - np.exp(log_alpha2[0]))

        if store_paths:
            paths[0].append(z1[0].copy())
            paths[1].append(z2[0].copy())

    result = {
        "distances": distances,
        "mismatches": mismatches,
        "accepts1": accepts1,
        "accepts2": accepts2,
        "alpha_gap": alpha_gap,
        "norms": norms,
    }

    if store_paths:
        result["path1"] = np.array(paths[0])
        result["path2"] = np.array(paths[1])

    return result


# ---------------------------------------------------------

def self_test():
    """
    1. Analytic gradient vs. finite differences.
    2. This run_pair vs. temp.py's, on temp.py's own GMM target.
    """
    print("gradient check (analytic vs. central differences)")
    for lam, omega in [(0.5, 2.0), (2.0, 6.0), (5.0, 12.0)]:
        max_abs, max_rel = check_gradient(lam, omega)
        status = "OK" if max_rel < 1e-6 else "FAIL"
        print(f"  lambda={lam:>4}, omega={omega:>5}: "
              f"max abs err {max_abs:.2e}, max rel err {max_rel:.2e}  {status}")

    import temp

    means, weights, sigma = temp.make_gmm()

    def gmm_target(z):
        return temp.log_prob_and_grad(z, means, weights, sigma)

    z1 = np.array([[-3.5, -3.0]])
    z2 = np.array([[3.5, 3.0]])

    print("\ncoupling machinery check (vs. temp.run_pair, GMM target)")
    for shared in (True, False):
        mine = run_pair(z1, z2, gmm_target, dt=0.1, n_steps=300, seed=7,
                        shared_randomness=shared)
        theirs = temp.run_pair(z1, z2, means, weights, sigma, dt=0.1,
                               n_steps=300, seed=7, shared_randomness=shared)

        same = all(
            np.array_equal(mine[key], theirs[key])
            for key in ("distances", "mismatches", "accepts1", "accepts2")
        )
        print(f"  shared_randomness={shared!s:>5}: identical -> {same}")


# ---------------------------------------------------------
# dt tuning and cycle detection
# ---------------------------------------------------------

def tune_dt(target, dim=DIM, target_accept=0.575, lo=1e-5, hi=3.0,
            iters=18, n_steps=1500, seed=1):
    """Bisect dt (acceptance is monotone decreasing in dt) to hit ~57%."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(1, dim))

    def acceptance(dt, steps):
        r = run_pair(z, z + 0.0, target, dt, steps, seed,
                     shared_randomness=True)
        return r["accepts1"].mean()

    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if acceptance(mid, n_steps) > target_accept:
            lo = mid
        else:
            hi = mid

    dt = np.sqrt(lo * hi)
    return dt, acceptance(dt, 4000)


def escape_cycles(distances, hi_mult=100.0, lo_mult=10.0):
    """
    Count collapse -> escape -> collapse cycles, relative to the run's own
    floor (the equilibrium separation), since that floor is set by the
    evaluation-noise scale rather than being zero.

    Returns (n_cycles, floor). The initial state is taken from the first
    sample so the opening descent is not miscounted as an escape.
    """
    positive = distances[distances > 0]
    if positive.size == 0:
        return 0, 0.0

    floor = np.percentile(positive, 10)
    high, low = floor * hi_mult, floor * lo_mult

    cycles = 0
    state = "high" if distances[0] > high else "low"

    for value in distances:
        if state == "low" and value > high:
            cycles += 1
            state = "high"
        elif state == "high" and value < low:
            state = "low"

    return cycles, floor


def escape_mismatch_alignment(result, hi_mult=100.0, lo_mult=10.0,
                              horizon=50, near=(1, 2)):
    """
    Do escapes coincide with accept/reject mismatches, as on the real data?

    Mirrors analyze_escape_mismatch.py's definitions (escape start = last
    collapsed step before the first crossing above the escape threshold;
    a mismatch counts if it lands within [start-1, start+2]), but with
    thresholds relative to this run's own floor instead of the real data's
    absolute COLLAPSE=2 / ESCAPE=10, which are scaled for 512 dimensions.
    """
    d = result["distances"]
    mismatch = result["mismatches"]          # index t aligns with d[t] -> d[t+1]

    _, floor = escape_cycles(d, hi_mult, lo_mult)
    if floor == 0.0:
        return 0, 0

    collapsed = d < floor * lo_mult
    big = d > floor * hi_mult

    starts = []
    for t in range(1, len(d)):
        if big[t] and not big[t - 1]:
            lo = max(0, t - horizon)
            idx = np.flatnonzero(collapsed[lo:t])
            if idx.size:
                starts.append(lo + idx[-1])
    starts = sorted(set(starts))

    hits = 0
    for s in starts:
        lo = max(0, s - near[0])
        hi = min(len(mismatch), s + near[1] + 1)
        if mismatch[lo:hi].any():
            hits += 1

    return len(starts), hits


def summarise(name, result):
    d = result["distances"]
    cycles, floor = escape_cycles(d)
    n_escapes, n_aligned = escape_mismatch_alignment(result)
    return (f"  {name:<34} floor {floor:>9.1e}  median {np.median(d):>9.1e}  "
            f"max {d.max():>7.2f}  exact0 {int((d == 0).sum()):>6}  "
            f"cycles {cycles:>4}  mismatches {result['mismatches'].sum():>6}  "
            f"escapes w/ mismatch {n_aligned:>4}/{n_escapes:<4}")


# ---------------------------------------------------------
# Experiments
# ---------------------------------------------------------

def _trace_panel(ax, result, title):
    d = result["distances"]
    ax.semilogy(np.maximum(d, 1e-17), lw=0.6)

    mismatch_steps = np.flatnonzero(result["mismatches"])
    if mismatch_steps.size:
        ax.plot(mismatch_steps, np.full(mismatch_steps.size, 2e-17),
                "|", color="red", ms=5, alpha=0.35)

    n_escapes, n_aligned = escape_mismatch_alignment(result)

    ax.set_ylim(3e-18, 100)
    ax.set_title(
        f"{title}\n{len(mismatch_steps)} mismatches (red ticks) · "
        f"{n_aligned}/{n_escapes} escapes with a mismatch at onset",
        fontsize=9)
    ax.set_xlabel("MALA step")
    ax.set_ylabel("distance")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    self_test()

    smooth = make_ripple_target(0.0, 0.0)          # pure Gaussian
    rippled = make_ripple_target(2.0, 6.0)         # lambda*omega^2 = 72

    print(f"\ndt tuning (target ~57.5% acceptance, dim={DIM})")
    targets = {}
    for name, base in [("smooth", smooth), ("rippled", rippled)]:
        dt, acc = tune_dt(base)
        targets[name] = (base, dt)
        print(f"  {name:<8} dt = {dt:.3e}  acceptance {acc:.3f}")

    rng = np.random.default_rng(SEED)
    z_far1 = rng.normal(size=(1, DIM))
    z_far2 = rng.normal(size=(1, DIM))
    z_close1 = rng.normal(size=(1, DIM))
    z_close2 = z_close1 + 1e-3 * rng.normal(size=(1, DIM))

    print(f"\nfar-start separation   {np.linalg.norm(z_far1 - z_far2):.3f}")
    print(f"close-start separation {np.linalg.norm(z_close1 - z_close2):.3e}")

    # --- deterministic targets: far and close start, shared and independent
    print(f"\nDETERMINISTIC TARGETS (dim={DIM}, {N_STEPS} steps)")
    for name, (base, dt) in targets.items():
        print(f"\n{name} target, dt={dt:.3e}")
        for label, a, b in [("far start", z_far1, z_far2),
                            ("close start", z_close1, z_close2)]:
            for shared in (True, False):
                tag = "shared" if shared else "independent"
                r = run_pair(a, b, base, dt, N_STEPS, SEED,
                             shared_randomness=shared)
                print(summarise(f"{label}, {tag}", r))

    # --- stochastic evaluation, close start, shared randomness
    # Outcomes on the rippled target turned out to be start-dependent, so
    # repeat over several close starts rather than reporting one.
    print(f"\nSTOCHASTIC EVALUATION (close start, shared randomness)")
    starts = []
    start_rng = np.random.default_rng(SEED)
    for _ in range(3):
        a = start_rng.normal(size=(1, DIM))
        starts.append((a, a + 1e-3 * start_rng.normal(size=(1, DIM))))

    panels = []
    for name, (base, dt) in targets.items():
        for tau in (0.0, 1e-3):
            for i, (a, b) in enumerate(starts):
                target = (base if tau == 0.0
                          else make_noisy_target(base, tau, SEED + 5))
                r = run_pair(a, b, target, dt, N_STEPS, SEED,
                             shared_randomness=True)
                print(summarise(f"{name}, tau={tau:g}, start {i}", r))
                # start 1 is plotted: on the rippled target it is one of the
                # starts that does collapse at tau=0, so the panels compare
                # like with like.
                if i == 1:
                    panels.append((f"{name} target, tau={tau:g}", r))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (title, result) in zip(axes.ravel(), panels):
        _trace_panel(ax, result, title)
    fig.suptitle("Coupled MALA, close start, shared noise + shared uniforms")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distance_traces.png")
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
