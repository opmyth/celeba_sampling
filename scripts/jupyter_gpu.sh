#!/usr/bin/env bash
#SBATCH --job-name=jupyter
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/jupyter-%j.out
#SBATCH --error=logs/jupyter-%j.err

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
cd ~/celeba_sampling

echo "Node: $(hostname)"
echo "Port: 8888"
echo "Started: $(date -u)"

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
