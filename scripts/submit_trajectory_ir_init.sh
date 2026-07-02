#!/usr/bin/env bash
#SBATCH --job-name=traj_ir_init
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --exclude=opencast
#SBATCH --time=03:00:00
#SBATCH --output=logs/traj_ir_init-%j.out
#SBATCH --error=logs/traj_ir_init-%j.err

JOB_START=$(date +%s)
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

echo "=== [1/2] Init — same noise ==="
python run_trajectory_init_ir.py --noise same

echo "=== [2/2] Init — indep noise ==="
python run_trajectory_init_ir.py --noise indep

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
