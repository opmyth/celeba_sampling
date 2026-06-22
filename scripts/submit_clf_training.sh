#!/usr/bin/env bash
#SBATCH --job-name=clf_bald
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/bald-clf-%j.out
#SBATCH --error=logs/bald-clf-%j.err

set -x
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
cd ~/celeba_sampling

mkdir -p logs

nvidia-smi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
echo "About to run python..."
python scripts/train_classifier.py 4 Bald --augment
echo "Python done."

echo "Finished: $(date -u)"
