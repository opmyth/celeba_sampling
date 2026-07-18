#!/usr/bin/env bash
# Trajectory diagnostics for any experiment in config.py (replaces
# submit_trajectory.sh/_male.sh/_male_eye.sh/_ir.sh/_ir_init.sh - 5 files).
#
# Usage: sbatch scripts/submit_trajectory.sh <experiment> <stepsize|init|all> [same|indep|both]
#   both (init/all mode): runs same then indep, each with its own 3 plots
#   all: runs init (both noise) then stepsize, in one job
#
#SBATCH --job-name=trajectory
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=08:00:00
#SBATCH --output=logs/trajectory-%j.out
#SBATCH --error=logs/trajectory-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_trajectory.sh <experiment> <stepsize|init|all> [same|indep|both]}
MODE=${2:?Usage: sbatch submit_trajectory.sh <experiment> <stepsize|init|all> [same|indep|both]}
NOISE=${3:-both}

# PROMPT (imagereward experiments only) / N_STEPS / SAMPLER: optional env
# var overrides. SAMPLER=ANNEALED_MALA runs annealed trajectory (output nests
# under trajectory/annealed/, beside the plain-MALA trajectory).
PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi
STEP_ARGS=()
if [ -n "${N_STEPS:-}" ]; then STEP_ARGS=(--n_steps "$N_STEPS"); fi
SAMPLER_ARGS=()
if [ -n "${SAMPLER:-}" ]; then SAMPLER_ARGS=(--sampler "$SAMPLER"); fi

run_init() {
    if [ "$NOISE" = "both" ]; then
        NOISES=(same indep)
    else
        NOISES=("$NOISE")
    fi
    for N in "${NOISES[@]}"; do
        python run_trajectory.py --experiment "$EXPR" --mode init --noise "$N" "${PROMPT_ARGS[@]}" "${STEP_ARGS[@]}" "${SAMPLER_ARGS[@]}"
        python plot_trajectory.py --experiment "$EXPR" --plot init --noise "$N" "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
        # across-chain mean convergence trend only (single-chain0 traces dropped
        # 2026-07-17 - the 25-chain mean supersedes the noisy single walker)
        python plot_trajectory.py --experiment "$EXPR" --plot jump_distance --mode init --noise "$N" --chain mean "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
        python plot_trajectory.py --experiment "$EXPR" --plot log_reward --mode init --noise "$N" --chain mean "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
    done
}

run_stepsize() {
    python run_trajectory.py --experiment "$EXPR" --mode stepsize "${PROMPT_ARGS[@]}" "${STEP_ARGS[@]}" "${SAMPLER_ARGS[@]}"
    python plot_trajectory.py --experiment "$EXPR" --plot stepsize "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
    # across-chain mean convergence trend only (single-chain0 traces dropped
    # 2026-07-17 - the 25-chain mean supersedes the noisy single walker)
    python plot_trajectory.py --experiment "$EXPR" --plot jump_distance --mode stepsize --chain mean "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
    python plot_trajectory.py --experiment "$EXPR" --plot log_reward --mode stepsize --chain mean "${PROMPT_ARGS[@]}" "${SAMPLER_ARGS[@]}"
}

case "$MODE" in
    init) run_init ;;
    stepsize) run_stepsize ;;
    all) run_init; run_stepsize ;;
    *) echo "MODE must be stepsize, init, or all" >&2; exit 1 ;;
esac
