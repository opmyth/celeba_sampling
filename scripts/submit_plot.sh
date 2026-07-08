#!/usr/bin/env bash
# Re-render trajectory plots for any experiment without re-running MCMC
# (replaces submit_plot_ir.sh).
#
# Usage: sbatch scripts/submit_plot.sh <experiment> <stepsize|init|jump_distance|log_reward> [mode] [noise]
#   mode is required for jump_distance/log_reward (stepsize|init)
#
#SBATCH --job-name=plot
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=00:30:00
#SBATCH --output=logs/plot-%j.out
#SBATCH --error=logs/plot-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"

EXPR=${1:?Usage: sbatch submit_plot.sh <experiment> <stepsize|init|jump_distance|log_reward> [mode] [noise]}
PLOT=${2:?Usage: sbatch submit_plot.sh <experiment> <stepsize|init|jump_distance|log_reward> [mode] [noise]}
MODE=${3:-}
NOISE=${4:-both}

ARGS=(--experiment "$EXPR" --plot "$PLOT" --noise "$NOISE")
if [ -n "$MODE" ]; then ARGS+=(--mode "$MODE"); fi

python plot_trajectory.py "${ARGS[@]}"
