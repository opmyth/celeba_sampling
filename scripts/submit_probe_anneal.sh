#!/usr/bin/env bash
# Accept-rate probe for annealed MALA (probe_anneal.py) - all 5 target
# posteriors (bald_ir counts twice, 2 prompts) in one job, flat vs dt/sqrt(T).
# ~1-1.5h (bald_ir/ImageReward dominates). TF32 on to match the annealed runs.
#
# Usage: sbatch scripts/submit_probe_anneal.sh            # all targets
#        sbatch scripts/submit_probe_anneal.sh --experiment bald   # one
#
#SBATCH --job-name=probe_anneal
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_anneal-%j.out
#SBATCH --error=logs/probe_anneal-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
export TF32=1

python probe_anneal.py "$@"
