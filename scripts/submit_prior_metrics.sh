#!/usr/bin/env bash
#SBATCH --job-name=prior_metrics
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --exclude=landonia11,saxa,opencast
#SBATCH --time=02:00:00
#SBATCH --output=logs/prior_metrics-%j.out
#SBATCH --error=logs/prior_metrics-%j.err

echo "Job started: $(date)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
cd ~/celeba_sampling
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python run_prior_metrics.py

echo "Job finished: $(date)"
