#!/usr/bin/env bash
# Accept-rate probe for annealed MALA (probe_anneal.py) - all 5 target
# posteriors (bald_ir counts twice, 2 prompts) in one job, flat vs dt/sqrt(T).
# ~50-60 min on A6000 (bald_ir/ImageReward dominates). TF32 on to match the
# annealed runs. Targets landonia11/A6000 (reliable per cluster.md); override
# to saxa's h200_3g.71gb slice on the command line when it's free for ~1.5x.
#
# Usage: sbatch scripts/submit_probe_anneal.sh            # all targets
#        sbatch scripts/submit_probe_anneal.sh --experiment bald   # one
#
#SBATCH --job-name=probe_anneal
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_anneal-%j.out
#SBATCH --error=logs/probe_anneal-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
export TF32=1

python probe_anneal.py "$@"
