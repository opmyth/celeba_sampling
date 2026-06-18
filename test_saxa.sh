#!/usr/bin/env bash
#SBATCH --job-name=test_saxa
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_saxa-%j.out
#SBATCH --error=logs/test_saxa-%j.err

echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

nvidia-smi

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
x = torch.randn(1000, 1000, device='cuda')
print('Tensor op OK:', (x @ x).shape)
"

echo "Finished: $(date -u)"
