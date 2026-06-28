#!/bin/bash
# 2D info-content sweep: data source (baseline/phrasebank, template, ollama)
#                       x distribution  (gaussian, flat, sir)
#                       x seed          (42, 43, 44)
#
# Reuses pre-existing runs where available:
#   gaussian baseline seed42 -> gauss_inject_r10_seed42
#   sir      baseline seed42 -> sir_inject_cal2x_v2_seed42
#   sir      template seed42 -> case_compare_template_seed42
#   sir      ollama   seed42 -> case_compare_ollama_seed42
#
# Appends a one-line result summary to results/unknown_disease/case_compare_sweep_notes.log
# after every run so progress can be reviewed without waiting for the whole sweep to finish.
set -uo pipefail

SEEDS=(42 43 44)
RESULTS_DIR="results/unknown_disease"
NOTES="$RESULTS_DIR/case_compare_sweep_notes.log"

source .venv/bin/activate

log_note() {
    local run_name="$1"
    local summary="$RESULTS_DIR/$run_name/summary.json"
    if [ -f "$summary" ]; then
        python3 -c "
import json, sys
d = json.load(open('$summary'))
sc = d.get('silhouette_curve', [])
first = sc[0]['silhouette'] if sc else float('nan')
last = sc[-1]['silhouette'] if sc else float('nan')
print(f\"$(date -Iseconds) $run_name acc={d.get('final_diag_acc')} wall={d.get('wall_seconds'):.0f}s sil_first={first:.3f} sil_last={last:.3f}\")
" >> "$NOTES"
    else
        echo "$(date -Iseconds) $run_name FAILED (no summary.json)" >> "$NOTES"
    fi
}

run_variant() {
    local schedule="$1"   # gaussian | flat | sir
    local datatype="$2"   # baseline | template | ollama
    local seed="$3"
    local run_name="sweep_${schedule}_${datatype}_seed${seed}"

    if [ -d "$RESULTS_DIR/$run_name" ] && [ -f "$RESULTS_DIR/$run_name/summary.json" ]; then
        echo ">>> SKIP (already done) $run_name"
        return
    fi

    local extra=()
    case "$datatype" in
        baseline) ;;
        template) extra=(--case-summary) ;;
        ollama)   extra=(--case-summary --ollama-summary) ;;
    esac

    local mode=()
    case "$schedule" in
        sir) mode=(--sir --sir-n-agents 150 --sir-days-per-round 2) ;;
        *)   mode=(--schedule "$schedule") ;;
    esac

    echo ">>> RUN $run_name"
    if python3 -u run_unknown_disease.py \
        "${mode[@]}" \
        --n-rounds 20 --n-silos 3 --injection-round 10 \
        --results-dir "$RESULTS_DIR" --seed "$seed" --run-name "$run_name" \
        "${extra[@]}"; then
        log_note "$run_name"
    else
        echo "$(date -Iseconds) $run_name CRASHED" >> "$NOTES"
    fi
}

# fold pre-existing runs into the matrix under canonical names (symlink, skip re-run)
mkdir -p "$RESULTS_DIR"
for pair in \
    "gaussian:baseline:42:gauss_inject_r10_seed42" \
    "sir:baseline:42:sir_inject_cal2x_v2_seed42" \
    "sir:template:42:case_compare_template_seed42" \
    "sir:ollama:42:case_compare_ollama_seed42"
do
    IFS=: read -r sched dtype seed src <<< "$pair"
    canon="sweep_${sched}_${dtype}_seed${seed}"
    if [ ! -e "$RESULTS_DIR/$canon" ] && [ -d "$RESULTS_DIR/$src" ]; then
        ln -s "$src" "$RESULTS_DIR/$canon"
        log_note "$canon"
    fi
done

for SCHEDULE in gaussian flat sir; do
    for DATATYPE in baseline template ollama; do
        for SEED in "${SEEDS[@]}"; do
            run_variant "$SCHEDULE" "$DATATYPE" "$SEED"
        done
    done
done

echo "$(date -Iseconds) SWEEP DONE" >> "$NOTES"
echo "DONE"
