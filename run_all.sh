#!/usr/bin/env bash
# Full end-to-end pipeline: RS -> Prior -> {ULA,MALA,G_MH per config.py} -> merge.
# Works for every experiment in config.py (including male_eye and bald_ir,
# which previously had no orchestrator of their own).
#
# Usage: bash run_all.sh <experiment> [init]
#   init: random (default), cold, warm
#
# Optional overrides (env vars): DT_MALA, DT_ULA, SIGMA, PROMPT, N_STEPS, BURNIN, THIN_K
set -eo pipefail

EXPR=${1:?Usage: bash run_all.sh <experiment> [init]}
INIT=${2:-random}
if [ "$INIT" = "random" ]; then
    EXPR_DIR="experiments/${EXPR}"
else
    EXPR_DIR="experiments/${EXPR}_${INIT}"
fi

mkdir -p logs "${EXPR_DIR}"

SAMPLERS=$(python -c "from config import EXPERIMENTS; print(' '.join(EXPERIMENTS['${EXPR}'].samplers))")

PROMPT_ARGS=()
if [ -n "${PROMPT:-}" ]; then PROMPT_ARGS=(--prompt "$PROMPT"); fi
STEP_ARGS=()
if [ -n "${N_STEPS:-}" ]; then STEP_ARGS+=(--n_steps "$N_STEPS"); fi
if [ -n "${BURNIN:-}" ]; then STEP_ARGS+=(--burnin "$BURNIN"); fi
if [ -n "${THIN_K:-}" ]; then STEP_ARGS+=(--thin_k "$THIN_K"); fi

cat > "${EXPR_DIR}/config.txt" <<CFG
experiment:  ${EXPR}
init:        ${INIT}
timestamp:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')
samplers:    ${SAMPLERS}
dt_mala:     ${DT_MALA:-<config.py default>}
dt_ula:      ${DT_ULA:-<config.py default>}
sigma:       ${SIGMA:-<config.py default>}
prompt:      ${PROMPT:-<config.py default>}
n_steps:     ${N_STEPS:-<config.py default>}
burnin:      ${BURNIN:-<config.py default>}
thin_k:      ${THIN_K:-<config.py default>}
CFG

echo "=== [1] Rejection Sampling ==="
python run_rs.py --experiment "$EXPR" "${PROMPT_ARGS[@]}" \
    --output_path "${EXPR_DIR}/results_rs.pt" \
    2>&1 | tee logs/rs.log

echo "=== [2] Prior ==="
python run_prior.py --experiment "$EXPR" "${PROMPT_ARGS[@]}" \
    --rs_path "${EXPR_DIR}/results_rs.pt" \
    --output_path "${EXPR_DIR}/results_prior.pt" \
    2>&1 | tee logs/prior.log

STEP=3
for SAMPLER in $SAMPLERS; do
    echo "=== [${STEP}] ${SAMPLER} ==="
    SAMPLER_ARGS=("${PROMPT_ARGS[@]}" "${STEP_ARGS[@]}")
    if [ "$SAMPLER" = "ULA" ] && [ -n "${DT_ULA:-}" ]; then SAMPLER_ARGS+=(--dt "$DT_ULA"); fi
    if [ "$SAMPLER" = "MALA" ] && [ -n "${DT_MALA:-}" ]; then SAMPLER_ARGS+=(--dt "$DT_MALA"); fi
    if [ "$SAMPLER" = "G_MH" ] && [ -n "${SIGMA:-}" ]; then SAMPLER_ARGS+=(--sigma "$SIGMA"); fi

    LOWER=$(echo "$SAMPLER" | tr '[:upper:]' '[:lower:]')
    python run_sampler.py \
        --experiment "$EXPR" --sampler "$SAMPLER" --init "$INIT" \
        "${SAMPLER_ARGS[@]}" \
        --rs_path "${EXPR_DIR}/results_rs.pt" \
        --output_path "${EXPR_DIR}/results_${LOWER}.pt" \
        2>&1 | tee "logs/${LOWER}.log"
    STEP=$((STEP + 1))
done

echo "=== [${STEP}] Merging results ==="
python merge_results.py --experiment "$EXPR" --expr_dir "$EXPR_DIR" \
    --output_path "${EXPR_DIR}/results_stylegan.pt" \
    2>&1 | tee logs/merge.log

echo ""
echo "All done. Final results in ${EXPR_DIR}/results_stylegan.pt"
