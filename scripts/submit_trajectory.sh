#!/usr/bin/env bash
# Trajectory diagnostics for any experiment in config.py (replaces
# submit_trajectory.sh/_male.sh/_male_eye.sh/_ir.sh/_ir_init.sh - 5 files).
#
# Usage: sbatch scripts/submit_trajectory.sh <experiment> <stepsize|init> [same|indep]
#
#SBATCH --job-name=trajectory
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=04:00:00
#SBATCH --output=logs/trajectory-%j.out
#SBATCH --error=logs/trajectory-%j.err

source "$(dirname "$0")/env.sh"

EXPR=${1:?Usage: sbatch submit_trajectory.sh <experiment> <stepsize|init> [same|indep]}
MODE=${2:?Usage: sbatch submit_trajectory.sh <experiment> <stepsize|init> [same|indep]}
NOISE=${3:-same}

if [ "$MODE" = "init" ]; then
    python run_trajectory.py --experiment "$EXPR" --mode init --noise "$NOISE"
    python plot_trajectory.py --experiment "$EXPR" --plot init --noise "$NOISE"
    python plot_trajectory.py --experiment "$EXPR" --plot jump_distance --mode init --noise "$NOISE"
    python plot_trajectory.py --experiment "$EXPR" --plot log_reward --mode init --noise "$NOISE"
else
    python run_trajectory.py --experiment "$EXPR" --mode stepsize
    python plot_trajectory.py --experiment "$EXPR" --plot stepsize
    python plot_trajectory.py --experiment "$EXPR" --plot jump_distance --mode stepsize
    python plot_trajectory.py --experiment "$EXPR" --plot log_reward --mode stepsize
fi
