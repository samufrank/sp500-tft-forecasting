#!/bin/bash
# Phase 02: Vintage Validation Experiments
# Tests baseline and staleness configs on vintage alignment
#
# 1. Trains 6 experiments (3 baseline + 3 staleness) on vintage alignment
# 2. Evaluates each after training
# 3. Runs in parallel with configurable job limit

echo "========================================================================"
echo "PHASE 02: VINTAGE VALIDATION"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
MAX_PARALLEL=6
LOG_DIR="logs/phase_02"
SPLITS_DIR="data/splits"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Parallel execution: $MAX_PARALLEL jobs"
echo "Total experiments: 6"
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
    local alignment=$2
    local no_staleness=$3
    local hidden_size=$4
    local dropout=$5
    local learning_rate=$6
    
    echo "[$exp_name] Training..."
    
    local cmd="python train/train_tft.py \
        --experiment-name \"02_vintage_baseline/$exp_name\" \
        --splits-dir \"$SPLITS_DIR\" \
        --alignment $alignment \
        --frequency daily \
        --feature-set core_proposal \
        --hidden-size $hidden_size \
        --dropout $dropout \
        --learning-rate $learning_rate \
        --max-epochs 100 \
        --early-stop-patience 10 \
        --batch-size 64 \
        --max-encoder-length 20 \
        --attention-heads 2 \
        --hidden-continuous-size 16 \
        --gradient-clip 0.1 \
        --overwrite"
    
    if [ "$no_staleness" = "true" ]; then
        cmd="$cmd --no-staleness"
    fi
    
    # Train
    eval $cmd > "$LOG_DIR/train_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Training complete, evaluating..."
    
    # Evaluate
    python train/evaluate_tft.py \
        --experiment-name "02_vintage_baseline/$exp_name" \
        > "$LOG_DIR/eval_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Complete"
}

echo "========================================================================"
echo "STARTING EXPERIMENTS"
echo "========================================================================"
echo ""

# ============================================================================
# BASELINE EXPERIMENTS (no staleness)
# ============================================================================

echo "--- Baseline Experiments (no staleness) ---"
echo ""

# Best Phase 00 config
wait_for_slot
train_and_eval "baseline_vintage_h16_drop0.15_lr5" "vintage" "true" 16 0.15 0.0005 &
sleep 2

# High dropout
wait_for_slot
train_and_eval "baseline_vintage_h16_drop0.25_lr5" "vintage" "true" 16 0.25 0.0005 &
sleep 2

# Low dropout
wait_for_slot
train_and_eval "baseline_vintage_h16_drop0.10_lr5" "vintage" "true" 16 0.1 0.0005 &
sleep 2

# ============================================================================
# STALENESS EXPERIMENTS (with staleness features)
# ============================================================================

echo "--- Staleness Experiments (with staleness features) ---"
echo ""

# h=16 with staleness
wait_for_slot
train_and_eval "staleness_vintage_h16_drop0.25_lr5" "vintage" "false" 16 0.25 0.0005 &
sleep 2

# h=14 partial collapse variants
wait_for_slot
train_and_eval "staleness_vintage_h14_drop0.15_lr5" "vintage" "false" 14 0.15 0.0005 &
sleep 2

wait_for_slot
train_and_eval "staleness_vintage_h14_drop0.20_lr5" "vintage" "false" 14 0.2 0.0005 &
sleep 2

# ============================================================================
# WAIT FOR COMPLETION
# ============================================================================

echo ""
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "========================================================================"
echo "PHASE 02 COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Summary:"
echo "  Baseline experiments: 3"
echo "  Staleness experiments: 3"
echo "  Total: 6"
echo ""
echo "Results saved to: experiments/02_vintage_baseline/"
echo "Logs saved to: $LOG_DIR/"
echo ""
