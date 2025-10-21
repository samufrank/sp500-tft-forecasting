#!/bin/bash

# Capacity analysis sweep: Test if collapse is due to parameter/sample ratio
# 
# Strategy:
#   1. Test h16 (works) and h24 (collapses) with different data sizes
#   2. Test h24 with more regularization
#   3. Document the capacity threshold
#
# Run with: bash scripts/sweep_capacity_analysis.sh

echo "Starting capacity analysis sweep at $(date)"
echo ""
echo "Goal: Determine if collapse is due to parameter/sample ratio"
echo ""

# ============================================================================
# BASELINE: Confirm working vs collapsed with monitoring
# ============================================================================
echo "Phase 1: Baseline with full monitoring..."

python train/train_with_monitoring.py \
    --experiment-name capacity_baseline_h16 \
    --hidden-size 16 \
    --learning-rate 0.0005 \
    --dropout 0.15 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

python train/train_with_monitoring.py \
    --experiment-name capacity_baseline_h24 \
    --hidden-size 24 \
    --learning-rate 0.0005 \
    --dropout 0.15 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

# ============================================================================
# TEST: Can h24 work with more regularization?
# ============================================================================
echo ""
echo "Phase 2: Testing h24 with aggressive regularization..."

# Higher dropout
python train/train_with_monitoring.py \
    --experiment-name capacity_h24_drop0.3 \
    --hidden-size 24 \
    --learning-rate 0.0005 \
    --dropout 0.3 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

python train/train_with_monitoring.py \
    --experiment-name capacity_h24_drop0.4 \
    --hidden-size 24 \
    --learning-rate 0.0005 \
    --dropout 0.4 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

# Lower learning rate
python train/train_with_monitoring.py \
    --experiment-name capacity_h24_lr0.0002 \
    --hidden-size 24 \
    --learning-rate 0.0002 \
    --dropout 0.15 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

# Combined: more dropout + lower lr
python train/train_with_monitoring.py \
    --experiment-name capacity_h24_conservative \
    --hidden-size 24 \
    --learning-rate 0.0002 \
    --dropout 0.3 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

# ============================================================================
# TEST: Edge cases h20, h22 to find exact threshold
# ============================================================================
echo ""
echo "Phase 3: Testing intermediate sizes to find collapse threshold..."

python train/train_with_monitoring.py \
    --experiment-name capacity_h20 \
    --hidden-size 20 \
    --learning-rate 0.0005 \
    --dropout 0.15 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

python train/train_with_monitoring.py \
    --experiment-name capacity_h22 \
    --hidden-size 22 \
    --learning-rate 0.0005 \
    --dropout 0.15 \
    --max-encoder-length 20 \
    --max-epochs 50 \
    --monitor-every-n-epochs 2

# ============================================================================
# TEST: Multiple random seeds to check consistency
# ============================================================================
echo ""
echo "Phase 4: Testing multiple seeds for h16, h18, h20, h24..."

for seed in 42 123 456; do
    for hs in 16 18 20 24; do
        python train/train_with_monitoring.py \
            --experiment-name capacity_h${hs}_seed${seed} \
            --hidden-size ${hs} \
            --learning-rate 0.0005 \
            --dropout 0.15 \
            --max-encoder-length 20 \
            --max-epochs 50 \
            --seed ${seed} \
            --monitor-every-n-epochs 5
    done
done

echo ""
echo "Capacity analysis sweep completed at $(date)"
echo ""
echo "Total experiments: ~20"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/analyze_capacity_sweep.py"
echo "  2. Look for patterns in collapse_monitoring/ directories"
echo "  3. Plot prediction_std over epochs for each hidden size"

