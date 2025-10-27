#!/bin/bash

for phase_dir in experiments/*/; do
    # Skip if not a directory or doesn't start with digit (phase folders are 00_, 01_, etc)
    if [ ! -d "$phase_dir" ] || [[ ! "$(basename $phase_dir)" =~ ^[0-9] ]]; then
        continue
    fi
    
    for exp_dir in "$phase_dir"*/; do
        exp_name=$(basename "$exp_dir")
        phase_name=$(basename "$phase_dir")
        full_path="$phase_name/$exp_name"
        
        # Skip if no checkpoints directory or no .ckpt files
        if [ ! -d "$exp_dir/checkpoints" ] || [ -z "$(ls -A $exp_dir/checkpoints/*.ckpt 2>/dev/null | grep -v last.ckpt)" ]; then
            echo "Skipping $full_path - no checkpoints found"
            continue
        fi
        
        echo "Processing: $full_path"
        
        # Run evaluation with --experiment-name flag
        python train/evaluate_tft.py --experiment-name "$full_path" || echo "WARNING: Evaluation failed for $full_path"
        
        echo "---"
    done
done

echo "All evaluations complete!"
