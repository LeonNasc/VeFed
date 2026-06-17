#!/bin/bash
# Queue seeds 43 and 44 after current seed=42 run completes,
# then run 10-silo prototype bank (n_silos=10) for all 3 seeds.
# Each seed writes to its own output dir to avoid cache collisions.

set -e

DEVICE="${1:-cuda}"
PYTHON=".venv/bin/python3"

# ── Wait for seed 42 ──────────────────────────────────────────────────────────
echo "[$(date)] Waiting for seed 42 to complete..."
while pgrep -f "run_10silo.py.*--seed 42" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Seed 42 complete."

# ── Seed 43 ───────────────────────────────────────────────────────────────────
echo "[$(date)] Starting seed 43..."
$PYTHON -u run_10silo.py \
    --device "$DEVICE" \
    --seed 43 \
    --no-sir-ref \
    --out-dir results/scalability_10silo/seed_43 \
    2>&1 | tee run_10silo_seed43.log
echo "[$(date)] Seed 43 complete."

# ── Seed 44 ───────────────────────────────────────────────────────────────────
echo "[$(date)] Starting seed 44..."
$PYTHON -u run_10silo.py \
    --device "$DEVICE" \
    --seed 44 \
    --no-sir-ref \
    --out-dir results/scalability_10silo/seed_44 \
    2>&1 | tee run_10silo_seed44.log
echo "[$(date)] Seed 44 complete."

# ── 10-silo prototype bank — 3 seeds ─────────────────────────────────────────
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

echo "[$(date)] All replicas complete. Run: python3 gen_10silo_report.py"
