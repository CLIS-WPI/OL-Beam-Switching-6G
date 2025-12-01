#!/bin/bash
echo "=========================================="
echo "OPTIMIZED EXPERIMENTS - H100 MAX PERFORMANCE"
echo "BATCH_SIZE=4096, HIDDEN_SIZE=768, AMP=ON"
echo "=========================================="

monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
        sleep 60
    done
}

# Start GPU monitoring in background
monitor_gpu &
MONITOR_PID=$!

echo "[1/4] Hero Result (Balanced, 5 runs)..."
python3 main.py --mode balanced --runs 5 > results_proposed_final.txt 2>&1
echo "✅ Hero complete: $(tail -1 results_proposed_final.txt)"

echo "[2/4] Vanilla Baseline (5 runs)..."
python3 main.py --mode vanilla --runs 5 > results_vanilla_final.txt 2>&1
echo "✅ Vanilla complete"

echo "[3/4] Ablation Study (5 runs)..."
python3 main.py --mode balanced --ablation --runs 5 > results_ablation_final.txt 2>&1
echo "✅ Ablation complete"

echo "[4/4] Latency Measurement (1 run)..."
python3 main.py --mode balanced --measure_latency --runs 1 > results_latency_final.txt 2>&1
echo "✅ Latency complete"

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null

echo "=========================================="
echo "ALL 4 EXPERIMENTS COMPLETE!"
date
echo "=========================================="
