#!/usr/bin/env bash
#SBATCH --job-name=train_clf
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_clf-%j.out
#SBATCH --error=logs/train_clf-%j.err

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~/celeba_sampling:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/home/s2800722/.torch_extensions
cd ~/celeba_sampling
mkdir -p logs clf_checkpoints

JOB_START=$(date +%s)
echo "Job started: $(date)"

echo "=== [1/1] Training Young classifier (attr=39) ==="
`python scripts/train_classifier.py 39 young --augment`

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
echo "Checkpoint saved to clf_checkpoints/young_clf_aug.pth"
