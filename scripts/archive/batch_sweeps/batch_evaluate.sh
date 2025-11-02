#!/bin/bash
PHASE="03_distribution_loss"
EXPERIMENTS=("balanced_005" "balanced_01" "emphasize_mean" "emphasize_std" "mean_01" "mean_02" "std_01" "std_02")

for exp in "${EXPERIMENTS[@]}"; do
    echo "=== Evaluating $exp ==="
    python train/evaluate_tft.py --experiment-name "$PHASE/$exp"
    echo ""
done
