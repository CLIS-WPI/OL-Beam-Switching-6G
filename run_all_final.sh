#!/bin/bash
echo "=========================================="
echo "FINAL EXPERIMENTS - GPU Optimized"
echo "=========================================="

echo "[1/4] Hero Result (Balanced Mode, 5 runs)..."
python3 main.py --mode balanced --runs 5 > results_proposed_final.txt 2>&1
echo "✅ Hero complete"

echo "[2/4] Vanilla Baseline (5 runs)..."
python3 main.py --mode vanilla --runs 5 > results_vanilla_final.txt 2>&1
echo "✅ Vanilla complete"

echo "[3/4] Ablation Study (5 runs)..."
python3 main.py --mode balanced --ablation --runs 5 > results_ablation_final.txt 2>&1
echo "✅ Ablation complete"

echo "[4/4] Latency Measurement (1 run)..."
python3 main.py --mode balanced --measure_latency --runs 1 > results_latency_final.txt 2>&1
echo "✅ Latency complete"

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
