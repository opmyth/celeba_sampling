#!/usr/bin/env bash
#SBATCH --job-name=cuda_test
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=saxa
#SBATCH --time=00:05:00
#SBATCH --output=logs/cuda_test-%j.out
#SBATCH --error=logs/cuda_test-%j.err

echo "=== Test 1: no module, venv only ==="
source /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"

echo "=== Test 2: with module load ==="
module load cuda/12.8.0
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/cuda-12.8.0/lib64:${LD_LIBRARY_PATH:-}
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"

echo "=== Test 3: cuda/13.2.1 ==="
module load cuda/13.2.1
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/cuda-13.2.1/lib64:${LD_LIBRARY_PATH:-}
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
