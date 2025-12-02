#!/bin/bash
# Phase 06c: Refined Weekly Sweep
# Fine-tunes around best configs from 06a
#
# Best from 06a:
#   h16_enc8_d025_bs32 - 58.1% dir_acc
#   h16_enc12_d015_bs16 - 57.9% dir_acc
#
# This sweep explores:
#   - Higher hidden size (24) - 16 beat 8, maybe more helps
#   - Shorter encoder (4, 6) - weekly regimes change fast
#   - Learning rate (never swept in 06a)
#   - Total: 2 × 3 × 2 = 12 experiments

echo "========================================================================"
echo "PHASE 06c: REFINED WEEKLY SWEEP"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
MAX_PARALLEL=4
LOG_DIR="logs/phase_06c"
SPLITS_DIR="data/splits"
PHASE_NAME="06c_weekly_refined"

mkdir -p "$LOG_DIR"

echo "Parallel execution: $MAX_PARALLEL jobs"
echo "Total experiments: 12"
echo "Fixed params: dropout=0.25, batch_size=32, frequency=weekly"
echo ""

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
    done
}

# Function to train and evaluate all checkpoints
train_and_eval() {
    local exp_name=$1
    local hidden_size=$2
    local encoder_len=$3
    local learning_rate=$4
    
    local full_exp_name="$PHASE_NAME/$exp_name"
    
    echo "[$exp_name] Training..."
    
    python train/train_tft.py \
        --experiment-name "$full_exp_name" \
        --splits-dir "$SPLITS_DIR" \
        --alignment vintage \
        --frequency weekly \
        --feature-set core_proposal \
        --hidden-size $hidden_size \
        --max-encoder-length $encoder_len \
        --dropout 0.25 \
        --batch-size 32 \
        --learning-rate $learning_rate \
        --max-epochs 100 \
        --early-stop-patience 15 \
        --attention-heads 2 \
        --hidden-continuous-size $hidden_size \
        --gradient-clip 0.1 \
        --overwrite \
        > "$LOG_DIR/train_${exp_name}.log" 2>&1
    
    local train_status=$?
    
    if [ $train_status -ne 0 ]; then
        echo "[$exp_name] Training FAILED (exit code: $train_status)"
        return 1
    fi
    
    echo "[$exp_name] Training complete, evaluating all checkpoints..."
    
    python train/evaluate_checkpoints.py \
        "experiments/$full_exp_name" \
        > "$LOG_DIR/eval_${exp_name}.log" 2>&1
    
    local eval_status=$?
    
    if [ $eval_status -ne 0 ]; then
        echo "[$exp_name] Evaluation FAILED (exit code: $eval_status)"
        return 1
    fi
    
    echo "[$exp_name] Complete"
    return 0
}

echo "========================================================================"
echo "STARTING SWEEP"
echo "========================================================================"
echo ""

# Grid definition
HIDDEN_SIZES=(16 24)
ENCODER_LENGTHS=(4 6 8)
LEARNING_RATES=(0.0005 0.0001)

# Calculate total
total=$((${#HIDDEN_SIZES[@]} * ${#ENCODER_LENGTHS[@]} * ${#LEARNING_RATES[@]}))

counter=1
for h in "${HIDDEN_SIZES[@]}"; do
    for enc in "${ENCODER_LENGTHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            # Format: h16_enc8_lr5e4
            lr_fmt=$(echo $lr | sed 's/0.000//' | sed 's/0.00//')
            if [ "$lr" == "0.0005" ]; then
                lr_fmt="5e4"
            elif [ "$lr" == "0.0001" ]; then
                lr_fmt="1e4"
            elif [ "$lr" == "0.001" ]; then
                lr_fmt="1e3"
            fi
            exp_name=$(printf "h%d_enc%d_lr%s" $h $enc $lr_fmt)
            
            echo "[$counter/$total] Queuing: $exp_name (h=$h, enc=$enc, lr=$lr)"
            
            wait_for_slot
            train_and_eval "$exp_name" $h $enc $lr &
            sleep 2
            
            counter=$((counter + 1))
        done
    done
done

echo ""
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "========================================================================"
echo "PHASE 06c COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""

# Aggregate results
echo "Aggregating results..."
python scripts/aggregate_checkpoints.py "experiments/$PHASE_NAME" \
    --output "experiments/$PHASE_NAME/best_per_experiment.csv" \
    > "$LOG_DIR/aggregate.log" 2>&1

echo ""
echo "Results saved to: experiments/$PHASE_NAME/"
echo "Logs saved to: $LOG_DIR/"
echo ""
