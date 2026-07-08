#!/usr/bin/env bash
#SBATCH --job-name=train_clf
#SBATCH -p Teaching
#SBATCH --account=general-teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodelist=landonia11
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_clf-%j.out
#SBATCH --error=logs/train_clf-%j.err

source "$SLURM_SUBMIT_DIR/scripts/env.sh"
mkdir -p clf_checkpoints

echo "=== [1/1] Training Young classifier (attr=39) ==="
python scripts/train_classifier.py 39 young --augment

echo "Checkpoint saved to clf_checkpoints/young_clf_aug.pth"
