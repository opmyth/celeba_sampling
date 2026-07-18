#!/usr/bin/env bash
# CPU-only inject job: add one sampler's stage results into an experiment's
# existing merged results_stylegan.pt (inject_sampler.py) - for experiments
# whose per-stage files are gone (bald). Chain after the sampler stage via
#   sbatch --dependency=afterok:<stage_jid> scripts/submit_inject.sh <exp> <sampler>
# PROMPT env passes through for imagereward experiments.
#
#SBATCH --job-name=inject_sampler
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --exclude=opencast,damnii12
#SBATCH --time=00:20:00
#SBATCH --output=logs/inject-%j.out
#SBATCH --error=logs/inject-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_inject.sh <experiment> <sampler>}
SAMPLER=${2:?Usage: sbatch submit_inject.sh <experiment> <sampler>}

PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi

python inject_sampler.py --experiment "$EXPR" --sampler "$SAMPLER" "${PROMPT_ARGS[@]}"
