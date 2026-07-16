#!/usr/bin/env bash
# Run ONE sampler stage of the pipeline as its own job - the stages are
# independent given results_rs.pt, so running them as separate jobs turns a
# ~20h serial pipeline into ~9h wall-clock (and schedules far better than
# one long job). Paths resolve into the prompt-nested experiment dir
# automatically via run_sampler.py's defaults.
#
# Usage: sbatch scripts/submit_sampler_stage.sh <experiment> <ULA|MALA|G_MH>
#   PROMPT="..." (imagereward experiments) and TF32=1 pass through as env vars.
#
#SBATCH --job-name=sampler_stage
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=12:00:00
#SBATCH --output=logs/sampler_stage-%j.out
#SBATCH --error=logs/sampler_stage-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_sampler_stage.sh <experiment> <ULA|MALA|G_MH>}
SAMPLER=${2:?Usage: sbatch submit_sampler_stage.sh <experiment> <ULA|MALA|G_MH>}

PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi

python run_sampler.py --experiment "$EXPR" --sampler "$SAMPLER" "${PROMPT_ARGS[@]}"
