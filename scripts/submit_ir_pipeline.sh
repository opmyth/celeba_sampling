#!/usr/bin/env bash
#SBATCH --job-name=ir_pipeline
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=08:00:00
#SBATCH --output=logs/ir_pipeline-%j.out
#SBATCH --error=logs/ir_pipeline-%j.err

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
mkdir -p logs experiments/bald_ir

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "=== [1/3] Rejection Sampling (IR) ==="
python run_rs_ir.py \
    --n_chains 100 \
    --n_trials 5 \
    --output_path experiments/bald_ir/results_rs_ir.pt

echo "=== [2/3] MALA (IR) ==="
python run_mala_ir.py \
    --n_trials 5 \
    --rs_path   experiments/bald_ir/results_rs_ir.pt \
    --output_path experiments/bald_ir/results_mala_ir.pt

echo "=== [3/3] Gaussian MH (IR) ==="
python run_gmh_ir.py \
    --n_trials 5 \
    --rs_path   experiments/bald_ir/results_rs_ir.pt \
    --output_path experiments/bald_ir/results_gmh_ir.pt

echo "=== Merging ==="
python merge_results_ir.py \
    --rs_path   experiments/bald_ir/results_rs_ir.pt \
    --mala_path experiments/bald_ir/results_mala_ir.pt \
    --gmh_path  experiments/bald_ir/results_gmh_ir.pt \
    --output_path experiments/bald_ir/results_merged_ir.pt

JOB_END=$(date +%s)
echo "Job finished: $(date) — Total: $(( (JOB_END - JOB_START) / 60 )) min"
