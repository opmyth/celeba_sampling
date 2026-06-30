#!/usr/bin/env bash
#SBATCH --job-name=trajectory
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=04:00:00
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

echo "=== [1/7] smile — same noise ==="
python run_trajectory_init.py --attribute smile --noise same

echo "=== [2/7] smile — indep noise ==="
python run_trajectory_init.py --attribute smile --noise indep

echo "=== [3/7] eyeglasses — same noise ==="
python run_trajectory_init.py --attribute eyeglasses --noise same

echo "=== [4/7] eyeglasses — indep noise ==="
python run_trajectory_init.py --attribute eyeglasses --noise indep

echo "=== [5/7] bald — same noise ==="
python run_trajectory_init.py --attribute bald --noise same

echo "=== [6/7] bald — indep noise ==="
python run_trajectory_init.py --attribute bald --noise indep

echo "=== [7/7] step size sweep (MALA, eyeglasses) ==="
python run_trajectory_stepsize.py

echo "All trajectory runs done."
