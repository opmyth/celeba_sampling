#!/usr/bin/env bash
#SBATCH --job-name=prior_metrics
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --exclude=landonia11,saxa,opencast
#SBATCH --time=02:00:00
#SBATCH --output=logs/prior_metrics-%j.out
#SBATCH --error=logs/prior_metrics-%j.err

source "$(dirname "$0")/env.sh"

python run_prior_metrics.py
