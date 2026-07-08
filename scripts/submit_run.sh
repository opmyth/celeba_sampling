#!/usr/bin/env bash
# Full pipeline for any experiment in config.py (replaces submit_run_male.sh,
# submit_ir_pipeline.sh, submit_male_eye_pipeline.sh).
#
# Usage: sbatch scripts/submit_run.sh <experiment> [init]
#
# Default GPU target below is landonia11/A6000 (the reliable general-purpose
# choice per cluster.md). For bald_ir, prefer saxa's H200 slice when free by
# overriding on the command line, e.g.:
#   sbatch --nodelist=saxa --gres=gpu:h200_3g.71gb:1 --time=04:00:00 \
#       scripts/submit_run.sh bald_ir
#
#SBATCH --job-name=celeba_run
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=08:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_run.sh <experiment> [init]}
INIT=${2:-random}

bash run_all.sh "$EXPR" "$INIT"
