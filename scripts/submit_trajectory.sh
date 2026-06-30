#!/usr/bin/env bash
#SBATCH --job-name=trajectory
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=08:00:00
#SBATCH --output=logs/trajectory-%j.out
#SBATCH --error=logs/trajectory-%j.err

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
cd ~/celeba_sampling
mkdir -p logs

echo "=== [1/8] smile — same noise ==="
python run_trajectory_init.py --attribute smile --noise same

echo "=== [2/8] smile — indep noise ==="
python run_trajectory_init.py --attribute smile --noise indep

echo "=== [3/8] eyeglasses — same noise ==="
python run_trajectory_init.py --attribute eyeglasses --noise same

echo "=== [4/8] eyeglasses — indep noise ==="
python run_trajectory_init.py --attribute eyeglasses --noise indep

echo "=== [5/8] bald — same noise ==="
python run_trajectory_init.py --attribute bald --noise same

echo "=== [6/8] bald — indep noise ==="
python run_trajectory_init.py --attribute bald --noise indep

echo "=== [7/8] step size sweep (MALA, eyeglasses) ==="
python run_trajectory_stepsize.py --attribute eyeglasses

echo "=== [8/8] step size sweep (MALA, bald) ==="
python run_trajectory_stepsize.py --attribute bald

echo "All trajectory runs done."
