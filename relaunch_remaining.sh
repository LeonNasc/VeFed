#!/bin/bash
# Relaunch seeds 43, 44 and all prototype runs (seed 42 SIR already complete).

set -e
PYTHON=".venv/bin/python3"
DEVICE="${1:-cuda}"

echo "[$(date)] Starting seed 43..."
$PYTHON -u run_10silo.py \
    --device "$DEVICE" --seed 43 --no-sir-ref \
    --out-dir results/scalability_10silo/seed_43 \
    2>&1 | tee run_10silo_seed43.log
echo "[$(date)] Seed 43 complete."

echo "[$(date)] Starting seed 44..."
$PYTHON -u run_10silo.py \
    --device "$DEVICE" --seed 44 --no-sir-ref \
    --out-dir results/scalability_10silo/seed_44 \
    2>&1 | tee run_10silo_seed44.log
echo "[$(date)] Seed 44 complete."

echo "[$(date)] Running 10-silo prototype bank (seeds 42, 43, 44)..."
for SEED in 42 43 44; do
    echo "[$(date)]   proto seed=$SEED ..."
    $PYTHON -u run_prototype.py \
        --device "$DEVICE" \
        --n-silos 10 \
        --seed "$SEED" \
        --run-name "proto_10silo_seed${SEED}" \
        --results-dir "results/prototype/10silo" \
        2>&1 | tee "run_proto_10silo_seed${SEED}.log"
    echo "[$(date)]   proto seed=$SEED done."
done

echo "[$(date)] All done."
