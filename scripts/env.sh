#!/usr/bin/env bash
# Shared cluster environment setup, sourced by every submit script:
#   source "$(dirname "$0")/env.sh"
#
# TORCH_EXTENSIONS_DIR is pinned to the compute node's local /tmp (job-scoped)
# to avoid the NFS FileBaton hang documented in scripts/cluster.md. Previously
# this was set per-script and 16 of 17 submit scripts used the NFS home dir
# instead (~/.torch_extensions) - the exact thing cluster.md already
# documents as a fixed issue. Centralizing it here means there's one place to
# get it right instead of seventeen.
set -eo pipefail

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

echo "Job started: $(date -u)  node=${SLURMD_NODENAME:-local}  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
