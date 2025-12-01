#!/bin/bash
echo "=========================================="
echo "HERO RUN MONITOR - H100-Stable Config"
echo "Target: Stability Score < 0.80"
echo "=========================================="
echo ""

while true; do
    # Check if process is running
    if ! ps aux | grep -q "[p]ython3 main.py.*balanced"; then
        if grep -q "EXPERIMENT COMPLETE" results_proposed_final.txt 2>/dev/null; then
            echo ""
            echo "🎉 HERO RUN COMPLETE!"
            echo ""
            echo "=== FINAL RESULTS ==="
            tail -50 results_proposed_final.txt | grep -E "(Stability Score|DRL|MAB|Average SNR)" | head -10
            break
        else
            echo "⚠️ Process stopped but not complete. Checking for errors..."
            tail -20 results_proposed_final.txt | grep -E "Error|Exception|Traceback" || echo "No errors found"
            break
        fi
    fi
    
    # Get current progress
    last_line=$(tail -1 results_proposed_final.txt 2>/dev/null)
    
    # Extract training progress
    if echo "$last_line" | grep -q "Training"; then
        percent_str=$(echo "$last_line" | grep -oE "[0-9.]+%" | head -1)
        percent_int=$(echo "$percent_str" | sed 's/%//' | cut -d. -f1)
        run_num=$(grep -oE "Run [0-9]/5" results_proposed_final.txt 2>/dev/null | tail -1 || echo "Run ?/5")
        epsilon=$(echo "$last_line" | grep -oE "ε=[0-9.]+" | sed 's/ε=//' || echo "?")
        
        # Calculate progress bar (use integer division)
        filled=$((percent_int / 2))
        if [ $filled -gt 50 ]; then filled=50; fi
        empty=$((50 - filled))
        bar=$(printf "%${filled}s" | tr ' ' '█')
        empty_bar=$(printf "%${empty}s" | tr ' ' '-')
        
        echo -ne "\r🔄 $run_num | Training: [$bar$empty_bar] ${percent_str} | ε=$epsilon"
    elif echo "$last_line" | grep -q "Evaluation"; then
        percent_str=$(echo "$last_line" | grep -oE "[0-9.]+%" | head -1)
        percent_int=$(echo "$percent_str" | sed 's/%//' | cut -d. -f1)
        filled=$((percent_int / 2))
        if [ $filled -gt 50 ]; then filled=50; fi
        empty=$((50 - filled))
        bar=$(printf "%${filled}s" | tr ' ' '█')
        empty_bar=$(printf "%${empty}s" | tr ' ' '-')
        echo -ne "\r🔄 Evaluation: [$bar$empty_bar] ${percent_str}"
    elif echo "$last_line" | grep -q "Completed"; then
        echo ""
        echo "✅ Run completed! Starting next run..."
    else
        echo -ne "\r⏳ Initializing..."
    fi
    
    # GPU status
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | tr -d ' %' | cut -d, -f1)
    if [ ! -z "$gpu_util" ]; then
        echo -n " | GPU: ${gpu_util}%"
    fi
    
    sleep 5
done
echo ""
