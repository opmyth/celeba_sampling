"""Single source of truth for per-experiment hyperparameters. Replaces the
per-attribute dt tables that used to be duplicated independently in
run_all.sh, run_trajectory_init.py, and each *_ir.py/*_male_eye.py script."""
from dataclasses import dataclass, field
from typing import List, Optional

_SIGMA_OPT = 2.38 / (512 ** 0.5)  # Roberts-Gelman-Gilks optimal RWMH scale in d=512


@dataclass
class ExperimentConfig:
    name: str
    kind: str                              # 'classifier' | 'imagereward'
    clf_names: Optional[List[str]] = None  # e.g. ['smile'] or ['male', 'eyeglasses']
    prompt: Optional[str] = None           # default prompt, imagereward only
    prompts: Optional[List[str]] = None    # alternative prompts to try, imagereward only

    samplers: List[str] = field(default_factory=lambda: ['ULA', 'MALA', 'G_MH'])
    dt_mala: float = 0.1
    dt_ula: Optional[float] = 0.01
    sigma_gmh: float = _SIGMA_OPT

    n_chains: int = 100     # parallel walkers for ULA/MALA/G_MH
    n_trials: int = 5
    n_steps: int = 3000
    burnin: int = 1000
    thin_k: int = 200

    # RS target sample count per trial. Named separately from n_chains above
    # because the two have historically meant different things (RS = total
    # samples wanted; MCMC n_chains = parallel walkers before thinning) -
    # see EXPERIMENTS.md / plan notes: today RS undersamples relative to the
    # thinned MCMC count (100 vs kept_per_chain*n_chains=1000) for every
    # experiment except male_eye, which was already fixed to 1000.
    rs_target: int = 100

    @property
    def kept_per_chain(self) -> int:
        return (self.n_steps - self.burnin) // self.thin_k


EXPERIMENTS = {
    'smile': ExperimentConfig(
        name='smile', kind='classifier', clf_names=['smile'],
        dt_mala=0.1, dt_ula=0.01, sigma_gmh=0.105,
    ),
    'eyeglasses': ExperimentConfig(
        name='eyeglasses', kind='classifier', clf_names=['eyeglasses'],
        dt_mala=0.05, dt_ula=0.03, sigma_gmh=0.105,
    ),
    'bald': ExperimentConfig(
        name='bald', kind='classifier', clf_names=['bald'],
        dt_mala=0.1, dt_ula=0.01, sigma_gmh=0.105,
    ),
    'male': ExperimentConfig(
        name='male', kind='classifier', clf_names=['male'],
        dt_mala=0.1, dt_ula=0.03, sigma_gmh=0.1052,
    ),
    # dt_mala/dt_ula confirmed via sweep_hyperparams.py: dt_mala=0.1 gave 65.5%
    # MALA accept (target band, matches male's own dt_mala); dt_ula=0.005 is the
    # largest value with a clearly non-negative log_p trend (0.01 was already
    # borderline negative).
    'notmale': ExperimentConfig(
        name='notmale', kind='classifier', clf_names=['not_male'],
        dt_mala=0.1, dt_ula=0.005,
        rs_target=1000,
    ),
    'male_eye': ExperimentConfig(
        name='male_eye', kind='classifier', clf_names=['male', 'eyeglasses'],
        samplers=['MALA', 'G_MH'],   # no ULA run for this experiment historically
        dt_mala=0.05, dt_ula=None, sigma_gmh=_SIGMA_OPT,
        rs_target=1000,
    ),
    'bald_ir': ExperimentConfig(
        name='bald_ir', kind='imagereward',
        prompt='a bald man',
        prompts=['a bald man', 'a bald person', 'a person with a shaved head'],
        dt_mala=0.05, dt_ula=0.01, sigma_gmh=_SIGMA_OPT,
        # 2026-07-16: was n_steps=1000, burnin=200, thin_k=80 - same
        # kept_per_chain=10 but only a third of the classifier experiments'
        # step budget, so chains had 3x less time to converge/mix with no
        # documented justification. Now matches the classifier schedule
        # exactly; the only remaining difference vs. classifiers is runtime
        # (ImageReward/BLIP is ~3-4x slower per step).
        # rs_target also bumped 100 -> 1000 in the same pass (the known
        # RS-undersampling inconsistency, already fixed for male_eye and
        # never present in the _hat/notmale experiments). ~1h RS stage at
        # the observed ~5% accept rate, vs ~6min at 100.
        rs_target=1000,
    ),
    # dt_mala/dt_ula confirmed via sweep_hyperparams.py (accept rate in
    # target range + flat/positive log_p trend for MALA; largest dt_ula
    # with a non-negative trend before it turns unstable for ULA).
    'wearing_hat': ExperimentConfig(
        name='wearing_hat', kind='classifier', clf_names=['WearingHat'],
        dt_mala=0.05, dt_ula=0.01, sigma_gmh=0.105,
        rs_target=1000,
    ),
    'male_hat': ExperimentConfig(
        name='male_hat', kind='classifier', clf_names=['male', 'WearingHat'],
        dt_mala=0.05, dt_ula=0.01, sigma_gmh=_SIGMA_OPT,
        rs_target=1000,
    ),
    'notmale_hat': ExperimentConfig(
        name='notmale_hat', kind='classifier', clf_names=['not_male', 'WearingHat'],
        dt_mala=0.05, dt_ula=0.01, sigma_gmh=_SIGMA_OPT,
        rs_target=1000,
    ),
}
