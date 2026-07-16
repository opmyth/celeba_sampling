#!/usr/bin/env bash
# Annealed-MALA validation gate (see validate_annealed.py).
#
# Usage: sbatch scripts/submit_validate_annealed.sh --probe        # timing probe (~15 min)
#        sbatch --time=05:00:00 scripts/submit_validate_annealed.sh  # full 5-trial validation
#
# All args pass straight through to validate_annealed.py.
#
#SBATCH --job-name=validate_annealed
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_annealed-%j.out
#SBATCH --error=logs/validate_annealed-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

python validate_annealed.py "$@"
