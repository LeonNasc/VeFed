#!/usr/bin/env bash
# Run replicas 1, 2, 3 sequentially (rep 0 is assumed to already be running).
# Seeds: rep1=43, rep2=44, rep3=45.
# Each log goes to ablation_rep{N}.log.
#
# Usage:
#   chmod +x run_ablation_reps.sh
#   nohup ./run_ablation_reps.sh > ablation_reps_driver.log 2>&1 &

set -uo pipefail

PYTHON=".venv/bin/python3"
SCRIPT="run_ablation.py"

flush_ollama() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Flushing Ollama model cache..."
    # Ask Ollama to unload phi3:mini immediately (keep_alive=0); no sudo needed.
    curl -sf http://localhost:11434/api/generate \
        -d '{"model":"phi3:mini","keep_alive":0}' > /dev/null 2>&1 || true
    sleep 3
    # If sudo is available without a password, do a hard restart for good measure.
    if sudo -n systemctl restart ollama > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Hard-restarted Ollama via systemctl."
        for i in $(seq 1 30); do
            if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama is up."
                return 0
            fi
            sleep 2
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Ollama did not come back up in 60s."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] sudo unavailable — used API unload only."
    fi
}

for rep in 1 2 3; do
    seed=$((42 + rep))
    logfile="ablation_rep${rep}.log"
    # Flush Ollama's KV-cache leak before each replica
    flush_ollama
    sleep 5  # let RSS settle
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rep=${rep} seed=${seed} → ${logfile}"
    $PYTHON $SCRIPT --rep "$rep" --seed "$seed" > "$logfile" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] rep=${rep} finished OK"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] rep=${rep} FAILED (exit ${status}) — continuing to next rep"
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All replicas done."
