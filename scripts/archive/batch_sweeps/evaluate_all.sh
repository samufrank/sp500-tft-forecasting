#!/bin/bash
# Evaluate all experiments across all phase directories
# Works with subdirectory structure: experiments/{phase}/{exp_name}/

echo "========================================================================"
echo "EVALUATING ALL EXPERIMENTS"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

TOTAL=0
EVALUATED=0
SKIPPED=0

# Iterate through phase directories (00_*, 01_*, etc.)
for phase_dir in experiments/0*; do
    if [ ! -d "$phase_dir" ]; then
        continue
    fi
    
    phase_name=$(basename "$phase_dir")
    echo "Processing phase: $phase_name"
    echo "----------------------------------------"
    
    # Iterate through experiments in this phase
    for exp_dir in "$phase_dir"*/; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        full_path="${phase_name}/${exp_name}"
        
        TOTAL=$((TOTAL + 1))
        
        # Skip if no checkpoints
        if [ ! -d "$exp_dir/checkpoints" ]; then
            echo "  [SKIP] $full_path - no checkpoints"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        # Skip if no non-last checkpoints
        ckpt_count=$(ls "$exp_dir/checkpoints"/*.ckpt 2>/dev/null | grep -v "last.ckpt" | wc -l)
        if [ "$ckpt_count" -eq 0 ]; then
            echo "  [SKIP] $full_path - no valid checkpoints"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        # Check if already evaluated (optional - remove to force re-evaluation)
        if [ -f "$exp_dir/evaluation/evaluation_metrics.json" ]; then
            echo "  [SKIP] $full_path - already evaluated"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        echo "  [EVAL] $full_path"
        
        # Run evaluation
        python train/evaluate_tft.py --experiment-name "$full_path" \
            > "logs/eval_${phase_name}_${exp_name}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  [DONE] $full_path"
            EVALUATED=$((EVALUATED + 1))
        else
            echo "  [FAIL] $full_path - check logs/eval_${phase_name}_${exp_name}.log"
        fi
        
        echo ""
    done
    
    echo ""
done

echo "========================================================================"
echo "EVALUATION COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Summary:"
echo "  Total experiments found: $TOTAL"
echo "  Evaluated: $EVALUATED"
echo "  Skipped: $SKIPPED"
echo ""
echo "Next:"
echo "  - check logs/ for failures"
echo "  - run: python scripts/summarize_experiments.py"
echo ""
