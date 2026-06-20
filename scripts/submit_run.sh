#!/usr/bin/env bash
#SBATCH --job-name=celeba_full
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/full_run-%j.out
#SBATCH --error=logs/full_run-%j.err
#SBATCH --exclude=saxa,opencast,damnii[07-12],landonia01,landonia02,landonia03,landonia05,landonia08,landonia23,landonia25

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

nvidia-smi

bash run_all.sh

echo "Finished: $(date -u)"
