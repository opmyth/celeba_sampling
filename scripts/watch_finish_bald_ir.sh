#!/usr/bin/env bash
# Login-node watcher for the 2026-07-16 timeout recovery: the running bald_ir
# pipeline job cannot finish inside its 16h limit (measured 0.52 it/s ->
# ~20h needed), but its ULA stage saves results_ula.pt on completion. The
# moment that file lands, everything the doomed job would still compute is
# better done as parallel per-stage jobs - so this script:
#   1. waits for results_ula.pt in the prompt's experiment dir
#   2. waits 2 more minutes (file-write settle + z_final save)
#   3. scancels the pipeline job (its remaining MALA work is unsalvageable -
#      MALA saves nothing until all 5 trials finish, which the limit prevents)
#   4. submits MALA + G_MH as parallel stage jobs (TF32=1) and a dependent
#      CPU merge job
#
# Run on the LOGIN node (not sbatch - it's a sleep loop, negligible CPU):
#   nohup bash scripts/watch_finish_bald_ir.sh 3553176 > logs/watch_3553176.log 2>&1 &
#   nohup bash scripts/watch_finish_bald_ir.sh 3553178 "a person with a shaved head" > logs/watch_3553178.log 2>&1 &
set -eo pipefail

JOB=${1:?Usage: watch_finish_bald_ir.sh <pipeline_jobid> [prompt]}
PROMPT_IN=${2:-}

cd "$(dirname "$0")/.."   # repo root (fine here: NOT run under sbatch, $0 is real)

PROMPT_FOR_DIR=${PROMPT_IN:-$(python -c "from config import EXPERIMENTS; print(EXPERIMENTS['bald_ir'].prompt)")}
SLUG=$(echo "$PROMPT_FOR_DIR" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
DIR="experiments/bald_ir/prompt_${SLUG}"
ULA_FILE="$DIR/results_ula.pt"

echo "$(date '+%F %T') watching for $ULA_FILE (pipeline job $JOB, prompt: '$PROMPT_FOR_DIR')"

until [ -f "$ULA_FILE" ]; do
    # bail out if the pipeline job died before ever saving ULA (e.g. crash) -
    # in that case a human needs to look, auto-submitting would be wrong
    if ! squeue -h -j "$JOB" 2>/dev/null | grep -q .; then
        echo "$(date '+%F %T') job $JOB left the queue but $ULA_FILE never appeared - NOT submitting recovery, investigate."
        exit 1
    fi
    sleep 60
done

echo "$(date '+%F %T') $ULA_FILE exists - settling 120s before cancel"
sleep 120

echo "$(date '+%F %T') cancelling pipeline job $JOB"
scancel "$JOB" || true

PROMPT_ENV=()
if [ -n "$PROMPT_IN" ]; then export PROMPT="$PROMPT_IN"; fi
export TF32=1

J_MALA=$(sbatch --parsable scripts/submit_sampler_stage.sh bald_ir MALA)
J_GMH=$(sbatch --parsable scripts/submit_sampler_stage.sh bald_ir G_MH)
J_MERGE=$(sbatch --parsable --dependency=afterok:${J_MALA}:${J_GMH} scripts/submit_merge.sh bald_ir)

echo "$(date '+%F %T') submitted: MALA=$J_MALA G_MH=$J_GMH merge=$J_MERGE (TF32=1, prompt: '$PROMPT_FOR_DIR')"
echo "done."
