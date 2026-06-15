#!/usr/bin/env bash
#SBATCH --job-name=dissertation_sampling
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/testing_stylegan-%j.out
#SBATCH --error=logs/testing_stylegan-%j.err
#SBATCH --exclude=opencast

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

module load cuda/12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate

cd ~/dissertation

nvidia-smi

python stylegan_sampling.py

echo "Finished: $(date -u)"


