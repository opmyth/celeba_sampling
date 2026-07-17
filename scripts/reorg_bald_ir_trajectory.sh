#!/usr/bin/env bash
# One-off (2026-07-17): move bald_ir trajectory output from the old scattered
# layout, where prompt_<slug>/ sat at three different depths --
#   trajectory/prompt_<slug>/                (stepsize)
#   trajectory/<noise>/prompt_<slug>/        (init)
# while pipeline results lived at experiments/bald_ir/prompt_<slug>/ --
# into the clean prompt-first layout matching run_trajectory.py/plot_trajectory.py
# after the refactor:
#   experiments/bald_ir/prompt_<slug>/trajectory/{stepsize files, same_noise/, indep_noise/}
#
# Idempotent: no-op if the old trajectory/ dir is already gone. Run from repo
# root (locally AND on the cluster - the cluster copy also has the .pt trace
# files, which this moves along with the PNGs since it globs everything).
set -eo pipefail
cd "$(dirname "$0")/.."

OLD="experiments/bald_ir/trajectory"
if [ ! -d "$OLD" ]; then
    echo "Already reorganized (no $OLD) - nothing to do."
    exit 0
fi

for P in prompt_a_bald_man prompt_a_person_with_a_shaved_head; do
    mkdir -p "experiments/bald_ir/$P/trajectory"
    if [ -d "$OLD/$P" ]; then
        mv "$OLD/$P"/* "experiments/bald_ir/$P/trajectory/" 2>/dev/null || true
        rmdir "$OLD/$P" 2>/dev/null || true
    fi
    for N in same_noise indep_noise; do
        if [ -d "$OLD/$N/$P" ]; then
            mkdir -p "experiments/bald_ir/$P/trajectory/$N"
            mv "$OLD/$N/$P"/* "experiments/bald_ir/$P/trajectory/$N/" 2>/dev/null || true
            rmdir "$OLD/$N/$P" 2>/dev/null || true
        fi
    done
done
rmdir "$OLD/same_noise" "$OLD/indep_noise" "$OLD" 2>/dev/null || true
echo "Done. bald_ir trajectory now nests under each prompt_<slug>/."
