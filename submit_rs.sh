#!/usr/bin/env bash
#SBATCH --job-name=dissertation_sampling
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/rs-testing_w2-%j.out
#SBATCH --error=logs/rs-testing_w2-%j.err
#SBATCH --exclude=saxa,opencast,damnii[07-12],landonia[01-03,05,08,23,25]

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Started: $(date -u)"

module load cuda/12.8.0
export CUDA_HOME=/opt/cuda-12.8.0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
. /home/htang2/toolchain-20251006/toolchain.rc
. /home/s2800722/venv/bin/activate
export LD_LIBRARY_PATH=/home/s2800722/venv/lib/python3.12/site-packages/torch/lib:/home/s2800722/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

cd ~/dissertation

nvidia-smi

python run_rs.py --n_chains 300 --n_trials 5 --output_path results_rs_test.pt

echo "Finished: $(date -u)"


