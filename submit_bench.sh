#!/bin/bash
#SBATCH --job-name=mala_bench
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.log
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/s2800722/celeba_sampling

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
. /home/htang2/toolchain-20251006/toolchain.rc

source /home/s2800722/venv/bin/activate

# Use /tmp so CUDA plugin builds don't hang on NFS FileBaton locks
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

bash bench_mala_after.sh 2>&1
