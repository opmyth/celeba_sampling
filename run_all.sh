#!/usr/bin/env bash
# Full end-to-end pipeline: RS → ULA → MALA → G_MH → merge
# Usage: bash run_all.sh <experiment_name> [init]
#   init: random (default), cold, warm
set -eo pipefail

EXPR=${1:?Usage: bash run_all.sh <experiment_name> [init]}
INIT=${2:-random}
if [ "$INIT" = "random" ]; then
    EXPR_DIR="experiments/${EXPR}"
else
    EXPR_DIR="experiments/${EXPR}_${INIT}"
fi

N_CHAINS=100
N_TRIALS=5
N_STEPS=3000
DT_MALA=0.1
DT_ULA=0.01
SIGMA=0.105
BATCH_SIZE=64
BURNIN=1000
THIN_K=200

mkdir -p logs "${EXPR_DIR}"

cat > "${EXPR_DIR}/config.txt" <<CFG
experiment:  ${EXPR}
init:        ${INIT}
timestamp:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')
n_chains:    ${N_CHAINS}
n_trials:    ${N_TRIALS}
n_steps:     ${N_STEPS}
dt_ula:      ${DT_ULA}
dt_mala:     ${DT_MALA}
sigma:       ${SIGMA}
batch_size:  ${BATCH_SIZE}
burnin:      ${BURNIN}
thin_k:      ${THIN_K}
clf:         clf_checkpoints/${EXPR}_clf_aug.pth
CFG

echo "=== [1/5] Rejection Sampling ==="
python run_rs.py \
    --clf_name "$EXPR" \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --output_path "${EXPR_DIR}/results_rs.pt" \
    2>&1 | tee logs/rs.log

echo "=== [2/5] ULA ==="
python run_sampler.py \
    --clf_name "$EXPR" \
    --sampler ULA \
    --init "$INIT" \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --dt "$DT_ULA" \
    --batch_size "$BATCH_SIZE" \
    --burnin "$BURNIN" \
    --thin_k "$THIN_K" \
    --rs_path "${EXPR_DIR}/results_rs.pt" \
    --output_path "${EXPR_DIR}/results_ula.pt" \
    2>&1 | tee logs/ula.log

echo "=== [3/5] MALA ==="
python run_sampler.py \
    --clf_name "$EXPR" \
    --sampler MALA \
    --init "$INIT" \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --dt "$DT_MALA" \
    --batch_size "$BATCH_SIZE" \
    --burnin "$BURNIN" \
    --thin_k "$THIN_K" \
    --rs_path "${EXPR_DIR}/results_rs.pt" \
    --output_path "${EXPR_DIR}/results_mala.pt" \
    2>&1 | tee logs/mala.log

echo "=== [4/5] G_MH ==="
python run_sampler.py \
    --clf_name "$EXPR" \
    --sampler G_MH \
    --init "$INIT" \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --sigma "$SIGMA" \
    --batch_size "$BATCH_SIZE" \
    --burnin "$BURNIN" \
    --thin_k "$THIN_K" \
    --rs_path "${EXPR_DIR}/results_rs.pt" \
    --output_path "${EXPR_DIR}/results_gmh.pt" \
    2>&1 | tee logs/gmh.log

echo "=== [5/5] Merging results ==="
python merge_results.py \
    --rs_path   "${EXPR_DIR}/results_rs.pt" \
    --ula_path  "${EXPR_DIR}/results_ula.pt" \
    --mala_path "${EXPR_DIR}/results_mala.pt" \
    --gmh_path  "${EXPR_DIR}/results_gmh.pt" \
    --output_path "${EXPR_DIR}/results_stylegan.pt" \
    2>&1 | tee logs/merge.log

echo ""
echo "All done. Final results in ${EXPR_DIR}/results_stylegan.pt"
