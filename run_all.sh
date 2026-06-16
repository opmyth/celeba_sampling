#!/usr/bin/env bash
# Full end-to-end pipeline: RS → ULA → MALA → G_MH → merge
# Usage: bash run_all.sh
set -e

N_CHAINS=10
N_TRIALS=1
N_STEPS=800
DT=0.01
SIGMA=0.5
BATCH_SIZE=64

mkdir -p logs

echo "=== [1/5] Rejection Sampling ==="
python run_rs.py \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --output_path results_rs.pt \
    2>&1 | tee logs/rs.log

echo "=== [2/5] ULA ==="
python run_sampler.py \
    --sampler ULA \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --dt "$DT" \
    --batch_size "$BATCH_SIZE" \
    --rs_path results_rs.pt \
    --output_path results_ula.pt \
    2>&1 | tee logs/ula.log

echo "=== [3/5] MALA ==="
python run_sampler.py \
    --sampler MALA \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --dt "$DT" \
    --batch_size "$BATCH_SIZE" \
    --rs_path results_rs.pt \
    --output_path results_mala.pt \
    2>&1 | tee logs/mala.log

echo "=== [4/5] G_MH ==="
python run_sampler.py \
    --sampler G_MH \
    --n_chains "$N_CHAINS" \
    --n_trials "$N_TRIALS" \
    --n_steps "$N_STEPS" \
    --sigma "$SIGMA" \
    --batch_size "$BATCH_SIZE" \
    --rs_path results_rs.pt \
    --output_path results_gmh.pt \
    2>&1 | tee logs/gmh.log

echo "=== [5/5] Merging results ==="
python merge_results.py \
    --rs_path   results_rs.pt \
    --ula_path  results_ula.pt \
    --mala_path results_mala.pt \
    --gmh_path  results_gmh.pt \
    --output_path results_stylegan.pt \
    2>&1 | tee logs/merge.log

echo ""
echo "All done. Final results in results_stylegan.pt"
