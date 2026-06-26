#!/usr/bin/env bash
#SBATCH --job-name=celeba_smile
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/run_smile-%j.out
#SBATCH --error=logs/run_smile-%j.err

INIT=${INIT:-random}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Init: ${INIT}"
echo "Started: $(date -u)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
cd ~/celeba_sampling

mkdir -p logs

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

bash run_all.sh smile "$INIT"

echo "Finished: $(date -u)"
