#!/usr/bin/env bash
# Hyperparameter sweep for any experiment in config.py (replaces
# submit_sweep.sh/_male.sh/_ir.sh/_beta_ir.sh - 4 files, each a heredoc).
#
# Usage: sbatch scripts/submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta> <comma-separated values>
# Example: sbatch scripts/submit_sweep.sh eyeglasses dt_mala 0.05,0.07,0.08,0.09,0.10
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

EXPR=${1:?Usage: sbatch submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta> <values>}
SWEEP=${2:?Usage: sbatch submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta> <values>}
VALUES=${3:?Usage: sbatch submit_sweep.sh <experiment> <dt_mala|dt_ula|sigma_gmh|beta> <values>}

python sweep_hyperparams.py --experiment "$EXPR" --sweep "$SWEEP" --values "$VALUES"
