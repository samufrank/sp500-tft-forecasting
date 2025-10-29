#!/bin/bash
# Master Retraining Script - All Phases
# Retrains all 121 experiments with fixed validation bug and collapse monitoring
# 
# Usage: bash retrain_all_experiments.sh
#
# This script:
# 1. Reads experiment_configs.csv to get exact hyperparameters
# 2. Retrains all experiments in original order
# 3. Evaluates each experiment after training
# 4. Preserves exact naming and directory structure

#set -e  # Exit on error

echo "========================================================================"
echo "MASTER RETRAINING SCRIPT - ALL PHASES"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
CONFIG_FILE="experiments/experiment_configs.csv"
SPLITS_DIR="data/splits/fixed"
MAX_PARALLEL=2  

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: $CONFIG_FILE not found"
    echo "Run the Python script to generate it first"
    exit 1
fi

# Function to wait for available training slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 30
    done
}

# Function to train a single experiment
train_experiment() {
    local phase=$1
    local exp_name=$2
    local frequency=$3
    local hidden_size=$4
    local dropout=$5
    local attention_heads=$6
    local hidden_continuous=$7
    local learning_rate=$8
    local batch_size=$9
    local max_epochs=${10}
    local encoder_length=${11}
    local gradient_clip=${12}
    local early_stop_patience=${13}
    local has_staleness=${14}
    
    local full_exp_name="${phase}/${exp_name}"
    
    echo "Training: $full_exp_name"
    
    # Build command with all parameters
    local cmd="python train/train_tft.py \
        --experiment-name \"$full_exp_name\" \
        --splits-dir \"$SPLITS_DIR\" \
        --frequency \"$frequency\" \
        --hidden-size $hidden_size \
        --dropout $dropout \
        --attention-heads $attention_heads \
        --hidden-continuous-size $hidden_continuous \
        --learning-rate $learning_rate \
        --batch-size $batch_size \
        --max-epochs $max_epochs \
        --max-encoder-length $encoder_length \
        --gradient-clip $gradient_clip \
        --early-stop-patience $early_stop_patience \
        --overwrite"
    
    # Add --no-staleness flag if needed
    if [ "$has_staleness" = "False" ]; then
        cmd="$cmd --no-staleness"
    fi
    
    # Execute training
    eval $cmd > "logs/retrain_${phase}_${exp_name}.log" 2>&1
    
    echo "  ✓ Training complete: $full_exp_name"
}

# Function to evaluate a single experiment
evaluate_experiment() {
    local phase=$1
    local exp_name=$2
    local full_exp_name="${phase}/${exp_name}"
    
    echo "Evaluating: $full_exp_name"
    
    python train/evaluate_tft.py \
        --experiment-name "$full_exp_name" \
        > "logs/eval_${phase}_${exp_name}.log" 2>&1
    
    echo "  ✓ Evaluation complete: $full_exp_name"
}

# Create logs directory
mkdir -p logs

# ============================================================================
# PARSE CSV AND RETRAIN ALL EXPERIMENTS
# ============================================================================

echo "Reading configurations from: $CONFIG_FILE"
echo ""

# Count total experiments
TOTAL_EXPS=$(tail -n +2 "$CONFIG_FILE" | wc -l)
COUNTER=0

echo "Total experiments to retrain: $TOTAL_EXPS"
echo ""
echo "========================================================================"
echo "STARTING RETRAINING"
echo "========================================================================"
echo ""

# Read CSV and process each experiment
tail -n +2 "$CONFIG_FILE" | while IFS=',' read -r phase experiment frequency hidden_size dropout attention_heads hidden_continuous learning_rate batch_size max_epochs encoder_length gradient_clip early_stop_patience has_staleness; do
    
    COUNTER=$((COUNTER + 1))
    
    echo "[$COUNTER/$TOTAL_EXPS] Processing: ${phase}/${experiment}"
    
    # Wait for available slot if running in parallel
    # wait_for_slot
    
    # Train experiment (sequential for now - uncomment wait_for_slot and add & to parallelize)
    train_experiment "$phase" "$experiment" "$frequency" "$hidden_size" "$dropout" \
        "$attention_heads" "$hidden_continuous" "$learning_rate" "$batch_size" \
        "$max_epochs" "$encoder_length" "$gradient_clip" "$early_stop_patience" \
        "$has_staleness"
    
    # Evaluate immediately after training
    evaluate_experiment "$phase" "$experiment"
    
    echo ""
done

# Wait for any remaining background jobs
wait

echo ""
echo "========================================================================"
echo "RETRAINING COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Summary:"
echo "  Total experiments: $TOTAL_EXPS"
echo "  Phase 0 (baseline exploration): 57 experiments"
echo "  Phase 1 (staleness features): 64 experiments"
echo ""
