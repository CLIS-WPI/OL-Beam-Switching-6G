#!/bin/bash
echo "=========================================="
echo "مانیتورینگ 3 آزمایش باقی‌مانده"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    date
    echo "=========================================="
    echo ""
    
    # GPU Status
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -2
    echo ""
    
    # Running processes
    running=$(ps aux | grep "[p]ython3 main.py" | grep -v grep | wc -l)
    echo "آزمایش‌های در حال اجرا: $running"
    echo ""
    
    # Progress for each experiment
    echo "پیشرفت:"
    for f in results_vanilla_final.txt results_ablation_final.txt results_latency_final.txt; do
        name=$(basename "$f" .txt | sed 's/results_//' | sed 's/_final//')
        if grep -q "EXPERIMENT COMPLETE" "$f" 2>/dev/null; then
            echo "  ✅ $name: تکمیل شد"
        else
            progress=$(tail -3 "$f" 2>/dev/null | grep -oE "(Run [0-9]/[0-9]|Training.*%|Evaluation.*%|Completed)" | tail -1)
            if [ -z "$progress" ]; then
                progress=$(tail -2 "$f" 2>/dev/null | grep -oE "ε=[0-9.]+" | tail -1 || echo "در حال راه‌اندازی...")
            fi
            echo "  🔄 $name: $progress"
        fi
    done
    
    # Count completed
    complete=$(grep -l "EXPERIMENT COMPLETE" results_vanilla_final.txt results_ablation_final.txt results_latency_final.txt 2>/dev/null | wc -l)
    echo ""
    echo "تکمیل شده: $complete/3"
    
    # Check if all complete
    if [ "$complete" -eq 3 ]; then
        echo ""
        echo "🎉 همه آزمایش‌ها تکمیل شدند!"
        echo ""
        echo "=== نتایج نهایی ==="
        for f in results_vanilla_final.txt results_ablation_final.txt results_latency_final.txt; do
            name=$(basename "$f" .txt | sed 's/results_//' | sed 's/_final//')
            echo ""
            echo "--- $name ---"
            tail -20 "$f" | grep -E "(Stability|Average SNR|DRL|MAB)" | head -5
        done
        break
    fi
    
    # Check if processes stopped unexpectedly
    if [ "$running" -eq 0 ] && [ "$complete" -lt 3 ]; then
        echo ""
        echo "⚠️ هیچ فرآیندی در حال اجرا نیست اما آزمایش‌ها تکمیل نشده‌اند!"
        echo "بررسی خطاها..."
        for f in results_vanilla_final.txt results_ablation_final.txt results_latency_final.txt; do
            if grep -q "Error\|Exception\|Traceback" "$f" 2>/dev/null; then
                echo "  ❌ $f: خطا دارد"
            fi
        done
        break
    fi
    
    sleep 30
done
