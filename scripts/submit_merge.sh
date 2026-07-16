#!/usr/bin/env bash
# CPU-only merge job: combines the per-stage results_*.pt into
# results_stylegan.pt. Meant to be chained after per-stage sampler jobs via
#   sbatch --dependency=afterok:<mala_jid>:<gmh_jid> scripts/submit_merge.sh <experiment>
# PROMPT env passes through for imagereward experiments (prompt-nested dirs).
#
#SBATCH --job-name=merge_results
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_merge.sh <experiment>}

PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi

python merge_results.py --experiment "$EXPR" "${PROMPT_ARGS[@]}"
