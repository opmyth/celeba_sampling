#!/usr/bin/env bash
# Precompute the ImageReward clipping bound M (= r_max) for every bald_ir
# prompt (see estimate_r_max.py). No MCMC, just a large forward-pass scan -
# runs fine on the fallback (non-landonia11/saxa) nodes.
#
# Usage: sbatch scripts/submit_estimate_r_max.sh [--n_samples 50000]
#
#SBATCH --job-name=estimate_r_max
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --exclude=landonia11,saxa,opencast
#SBATCH --time=02:00:00
#SBATCH --output=logs/estimate_r_max-%j.out
#SBATCH --error=logs/estimate_r_max-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

python estimate_r_max.py "$@"
