#!/usr/bin/env bash
# One-off recovery for the 2026-07-16 bald_ir pipeline runs (jobs 3553176/
# 3553178) that hit their 16h time limit mid-MALA: real per-step cost on the
# h200_3g slice measured 0.52 it/s -> full pipeline needs ~20h, not the 12-14h
# estimated. RS/Prior/ULA completed and saved per-stage into the prompt-nested
# dirs; this job runs only the missing MALA -> G_MH -> merge against those
# existing files (run_sampler/merge_results defaults resolve into
# experiments/bald_ir/prompt_<slug>/ automatically when --prompt is given).
#
# Usage: sbatch scripts/submit_finish_bald_ir.sh                      # default prompt
#        PROMPT="a person with a shaved head" sbatch scripts/submit_finish_bald_ir.sh
#
#SBATCH --job-name=finish_bald_ir
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=15:00:00
#SBATCH --output=logs/finish_bald_ir-%j.out
#SBATCH --error=logs/finish_bald_ir-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi

echo "=== [1/3] MALA ==="
python run_sampler.py --experiment bald_ir --sampler MALA "${PROMPT_ARGS[@]}"

echo "=== [2/3] G_MH ==="
python run_sampler.py --experiment bald_ir --sampler G_MH "${PROMPT_ARGS[@]}"

echo "=== [3/3] merge ==="
python merge_results.py --experiment bald_ir "${PROMPT_ARGS[@]}"

echo "Done - results_stylegan.pt merged from existing RS/Prior/ULA + new MALA/G_MH."
