#!/bin/bash
# Run 10-silo comprehensive report: 3 replicas of all experiments
# Seeds: 42, 43, 44
# Each run includes: IID, Non-IID, Unknown disease with prototype classification eval
# Runs sequentially; each takes ~90 min

set -e

SEEDS=(42 43 44)
DEVICE="${1:-cuda}"
LOG_DIR="."

echo "=========================================="
echo "10-silo 3-replica comprehensive report"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Starting replica seed=$SEED ($(date '+%H:%M:%S'))"
    python3 -u run_10silo.py \
        --device "$DEVICE" \
        --seed "$SEED" \
        --no-sir-ref \
        2>&1 | tee "run_10silo_seed${SEED}.log"
    echo "<<< Completed seed=$SEED ($(date '+%H:%M:%S'))"
done

echo ""
echo "=========================================="
echo "All 3 replicas complete — aggregating results"
echo "=========================================="
python3 -c "
import json
from pathlib import Path

OUT = Path('results/scalability_10silo')
all_seeds = {}
for seed in [42, 43, 44]:
    summary_file = OUT / 'summary.json'
    if summary_file.exists():
        all_seeds[f'seed_{seed}'] = json.loads(summary_file.read_text())

# Save aggregated summary for report generation
(OUT / 'summary_3replicas.json').write_text(json.dumps(all_seeds, indent=2))
print(f'Aggregated results: {OUT}/summary_3replicas.json')
"

echo ""
echo "Results location: results/scalability_10silo/"
echo "Ready for report generation and meeting tomorrow"
