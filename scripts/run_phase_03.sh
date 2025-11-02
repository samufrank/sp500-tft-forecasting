#!/bin/bash
# Phase 03: Distribution Loss Grid
# Tests anti-collapse and anti-drift penalties on vintage alignment
#
# 1. Trains 9 distribution loss experiments on vintage alignment
# 2. Evaluates each after training
# 3. Runs in parallel with configurable job limit

echo "========================================================================"
echo "PHASE 03: DISTRIBUTION LOSS GRID"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
MAX_PARALLEL=9
LOG_DIR="logs/phase_03"
SPLITS_DIR="data/splits"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Parallel execution: $MAX_PARALLEL jobs"
echo "Total experiments: 9"
echo "Base config: h=16, dropout=0.15, lr=0.0005 (best Phase 00)"
echo ""

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 30
    done
}

# Function to train and evaluate
train_and_eval() {
    local exp_name=$1
    local mean_weight=$2
    local std_weight=$3
    
    echo "[$exp_name] Training (mean=$mean_weight, std=$std_weight)..."
    
    python train/train_tft.py \
        --experiment-name "03_distribution_loss/$exp_name" \
        --splits-dir "$SPLITS_DIR" \
        --alignment vintage \
        --frequency daily \
        --feature-set core_proposal \
        --hidden-size 16 \
        --dropout 0.15 \
        --learning-rate 0.0005 \
        --max-epochs 100 \
        --early-stop-patience 15 \
        --batch-size 64 \
        --max-encoder-length 20 \
        --attention-heads 2 \
        --hidden-continuous-size 8 \
        --gradient-clip 0.1 \
        --dist-loss-mean-weight $mean_weight \
        --dist-loss-std-weight $std_weight \
        --no-staleness \
        --overwrite \
        > "$LOG_DIR/train_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Training complete, evaluating..."
    
    python train/evaluate_tft.py \
        --experiment-name "03_distribution_loss/$exp_name" \
        > "$LOG_DIR/eval_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Complete"
}

echo "========================================================================"
echo "STARTING EXPERIMENTS"
echo "========================================================================"
echo ""

# ============================================================================
# CONTROL
# ============================================================================

echo "--- Control (no penalties) ---"
wait_for_slot
train_and_eval "control" 0.0 0.0 &
sleep 2

# ============================================================================
# ANTI-COLLAPSE ONLY (std penalty)
# ============================================================================

echo "--- Anti-Collapse Only ---"
wait_for_slot
train_and_eval "std_01" 0.0 0.1 &
sleep 2

wait_for_slot
train_and_eval "std_02" 0.0 0.2 &
sleep 2

# ============================================================================
# ANTI-DRIFT ONLY (mean penalty)
# ============================================================================

echo "--- Anti-Drift Only ---"
wait_for_slot
train_and_eval "mean_01" 0.1 0.0 &
sleep 2

wait_for_slot
train_and_eval "mean_02" 0.2 0.0 &
sleep 2

# ============================================================================
# BALANCED (mean = std)
# ============================================================================

echo "--- Balanced Penalties ---"
wait_for_slot
train_and_eval "balanced_005" 0.05 0.05 &
sleep 2

wait_for_slot
train_and_eval "balanced_01" 0.1 0.1 &
sleep 2

# ============================================================================
# UNBALANCED (emphasize one)
# ============================================================================

echo "--- Unbalanced Penalties ---"
wait_for_slot
train_and_eval "emphasize_std" 0.1 0.2 &
sleep 2

wait_for_slot
train_and_eval "emphasize_mean" 0.2 0.1 &
sleep 2

# ============================================================================
# WAIT FOR COMPLETION
# ============================================================================

echo ""
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "========================================================================"
echo "PHASE 03 COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Summary:"
echo "  Control: 1"
echo "  Anti-collapse only: 2"
echo "  Anti-drift only: 2"
echo "  Balanced: 2"
echo "  Unbalanced: 2"
echo "  Total: 9"
echo ""
echo "Results saved to: experiments/03_distribution_loss/"
echo "Logs saved to: $LOG_DIR/"
echo ""
