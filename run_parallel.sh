#!/bin/bash
echo "=========================================="
echo "PARALLEL EXPERIMENTS - BOTH GPUs"
echo "=========================================="

echo "Phase 1: Running Hero on GPU0 and Vanilla on GPU1..."
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode balanced --runs 5 > results_proposed_final.txt 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode vanilla --runs 5 > results_vanilla_final.txt 2>&1 &
PID2=$!

echo "Waiting for Phase 1 to complete... (PIDs: $PID1, $PID2)"
wait $PID1 $PID2
echo "✅ Phase 1 complete!"

echo "Phase 2: Running Ablation on GPU0 and Latency on GPU1..."
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode balanced --ablation --runs 5 > results_ablation_final.txt 2>&1 &
PID3=$!
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode balanced --measure_latency --runs 1 > results_latency_final.txt 2>&1 &
PID4=$!

echo "Waiting for Phase 2 to complete... (PIDs: $PID3, $PID4)"
wait $PID3 $PID4
echo "✅ Phase 2 complete!"

echo "=========================================="
echo "ALL 4 EXPERIMENTS COMPLETE!"
date
echo "=========================================="
