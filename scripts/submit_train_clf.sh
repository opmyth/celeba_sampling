#!/usr/bin/env bash
# Usage: sbatch scripts/submit_train_clf.sh <attr_idx> <attr_name>
# e.g.:  sbatch scripts/submit_train_clf.sh 35 WearingHat
#
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

ATTR_IDX=${1:?Usage: sbatch submit_train_clf.sh <attr_idx> <attr_name>}
ATTR_NAME=${2:?Usage: sbatch submit_train_clf.sh <attr_idx> <attr_name>}

echo "=== Training ${ATTR_NAME} classifier (attr=${ATTR_IDX}) ==="
python scripts/train_classifier.py "$ATTR_IDX" "$ATTR_NAME" --augment

echo "Checkpoint saved to clf_checkpoints/${ATTR_NAME}_clf_aug.pth"
