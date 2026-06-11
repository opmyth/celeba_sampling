#!/usr/bin/env bash
#SBATCH --job-name=dissertation_sampling
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/sampling-%j.out
#SBATCH --error=logs/sampling-%j.err

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate

cd ~/dissertation

nvidia-smi

python run_sampling.py \
    --n_chains 1000 \
    --n_steps 2 \
    --n_trials 1 \
    --dt 0.01 \
    --sigma 0.5 \
    --output_path results_benchmark.pt

echo "Finished: $(date -u)"


