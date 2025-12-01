#!/bin/bash
echo "Waiting for vanilla to complete..."
while ps aux | grep -q "[p]ython3 main.py.*vanilla"; do sleep 60; done
echo "✅ Vanilla complete!"

echo "[3/4] Starting Ablation Study..."
python3 main.py --mode balanced --ablation --runs 5 > results_ablation_final.txt 2>&1
echo "✅ Ablation complete!"

echo "[4/4] Starting Latency Measurement..."
python3 main.py --mode balanced --measure_latency --runs 1 > results_latency_final.txt 2>&1
echo "✅ Latency complete!"

echo "=========================================="
echo "ALL 4 EXPERIMENTS COMPLETE!"
echo "=========================================="
