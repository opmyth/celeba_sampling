#!/usr/bin/env bash
# Hyperparameter sweep for any experiment in config.py (replaces
# submit_sweep.sh/_male.sh/_ir.sh/_beta_ir.sh - 4 files, each a heredoc).
#
# Usage: sbatch scripts/submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta> <comma-separated values>
#        sbatch scripts/submit_sweep.sh <experiment> both <dt_mala values> <dt_ula values>
#   both: runs dt_mala then dt_ula sequentially, one job/log for both tables
# Example: sbatch scripts/submit_sweep.sh eyeglasses dt_mala 0.05,0.07,0.08,0.09,0.10
#          sbatch scripts/submit_sweep.sh wearing_hat both 0.01,0.05,0.1 0.005,0.01,0.03
#
#SBATCH --job-name=sweep
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

USAGE="Usage: sbatch submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta|both> <values> [ula_values if both]"
EXPR=${1:?$USAGE}
SWEEP=${2:?$USAGE}
VALUES=${3:?$USAGE}

# N_CHAINS: optional env var override (default is sweep_hyperparams.py's own
# default of 100). Lower it (e.g. 32, below StyleGAN2Wrapper's chunk size of
# 64) to fit on a smaller MIG slice - still plenty of chains for a step-size
# diagnostic sweep, just not full publication-quality sample counts.
CHAIN_ARGS=()
if [ -n "${N_CHAINS:-}" ]; then CHAIN_ARGS=(--n_chains "$N_CHAINS"); fi

if [ "$SWEEP" = "both" ]; then
    ULA_VALUES=${4:?$USAGE}
    python sweep_hyperparams.py --experiment "$EXPR" --sweep dt_mala --values "$VALUES" "${CHAIN_ARGS[@]}"
    python sweep_hyperparams.py --experiment "$EXPR" --sweep dt_ula --values "$ULA_VALUES" "${CHAIN_ARGS[@]}"
else
    python sweep_hyperparams.py --experiment "$EXPR" --sweep "$SWEEP" --values "$VALUES" "${CHAIN_ARGS[@]}"
fi
