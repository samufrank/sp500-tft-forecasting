#!/bin/bash
# Phase 1 Exhaustive Staleness Sweep
# Tests staleness features across wide hyperparameter space

echo "========================================================================"
echo "PHASE 1 EXHAUSTIVE STALENESS SWEEP"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
SPLITS_DIR="data/splits/fixed"
MAX_EPOCHS=50
ENCODER_LENGTH=20
MAX_PARALLEL=4  # Number of parallel training runs

# Create logs directory
mkdir -p logs/sweep_staleness

# Hyperparameter grid
HIDDEN_SIZES=(12 14 16 18 20 22)
DROPOUTS=(0.15 0.20 0.25 0.30 0.35)
LEARNING_RATES=(0.0003 0.0005)

# Count total experiments
TOTAL_DAILY=$((${#HIDDEN_SIZES[@]} * ${#DROPOUTS[@]} * ${#LEARNING_RATES[@]}))
TOTAL_EXPERIMENTS=$((TOTAL_DAILY + 1))  # +1 for monthly

echo "Experiment configuration:"
echo "  Hidden sizes: ${HIDDEN_SIZES[*]}"
echo "  Dropouts: ${DROPOUTS[*]}"
echo "  Learning rates: ${LEARNING_RATES[*]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Total daily experiments: $TOTAL_DAILY"
echo "  Total with monthly: $TOTAL_EXPERIMENTS"
echo ""

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
    done
}

# Counter
COUNTER=0

# ============================================================================
# DAILY DATA SWEEP
# ============================================================================
echo "========================================================================"
echo "DAILY DATA SWEEP (${TOTAL_DAILY} experiments)"
echo "========================================================================"
echo ""

for h in "${HIDDEN_SIZES[@]}"; do
    for drop in "${DROPOUTS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            COUNTER=$((COUNTER + 1))
            
            # Format learning rate for experiment name (remove decimal)
            lr_name=$(echo $lr | sed 's/0\.//g' | sed 's/^0*//g')
            
            EXP_NAME="sweep_stale_h${h}_drop${drop}_lr${lr_name}"
            
            echo "[$COUNTER/$TOTAL_DAILY] Starting: $EXP_NAME"
            
            # Wait for available slot
            wait_for_slot
            
            # Launch training in background
            python train/train_tft.py \
                --experiment-name $EXP_NAME \
                --splits-dir $SPLITS_DIR \
                --hidden-size $h \
                --dropout $drop \
                --learning-rate $lr \
                --max-epochs $MAX_EPOCHS \
                --max-encoder-length $ENCODER_LENGTH \
                > logs/sweep_staleness/${EXP_NAME}.log 2>&1 &
            
            # Small delay to avoid race conditions
            sleep 2
        done
    done
done

# Wait for all daily experiments to complete
echo ""
echo "Waiting for all daily experiments to complete..."
wait

echo ""
echo "Daily sweep completed at: $(date)"
echo ""

# ============================================================================
# MONTHLY DATA EXPERIMENT
# ============================================================================
echo "========================================================================"
echo "MONTHLY DATA EXPERIMENT"
echo "========================================================================"

EXP_NAME="sweep_stale_monthly_h16_drop0.25"

echo "Starting monthly experiment: $EXP_NAME"

python train/train_tft.py \
    --experiment-name $EXP_NAME \
    --splits-dir data/splits/fixed \
    --frequency monthly \
    --hidden-size 16 \
    --dropout 0.25 \
    --learning-rate 0.0005 \
    --max-epochs 50 \
    --max-encoder-length 12 \
    > logs/sweep_staleness/${EXP_NAME}.log 2>&1

echo "Monthly experiment completed at: $(date)"
echo ""

# After all training completes, before analysis section:

echo "========================================================================"
echo "EVALUATING ALL MODELS"
echo "========================================================================"

for exp_path in experiments/sweep_stale_*; do
    exp_name=$(basename $exp_path)
    
    # Skip if already evaluated
    if [ -f "$exp_path/evaluation/evaluation_metrics.json" ]; then
        echo "Skipping $exp_name (already evaluated)"
        continue
    fi
    
    echo "Evaluating: $exp_name"
    
    # Wait for available slot
    wait_for_slot
    
    python train/evaluate_tft.py \
        --experiment-name $exp_name \
        > logs/sweep_staleness/${exp_name}_eval.log 2>&1 &
    
    sleep 2
done

# Wait for all evaluations
echo "Waiting for evaluations to complete..."
wait

echo "All evaluations completed at: $(date)"
echo ""

# ============================================================================
# ANALYSIS
# ============================================================================
echo "========================================================================"
echo "ANALYZING RESULTS"
echo "========================================================================"

python << 'EOF'
import os
import json
import pandas as pd
from pathlib import Path

results = []

# Scan all sweep_stale_* experiments
exp_dir = Path('experiments')
for exp_path in sorted(exp_dir.glob('sweep_stale_*')):
    exp_name = exp_path.name
    
    # Load config
    config_path = exp_path / 'config.json'
    if not config_path.exists():
        print(f"Skipping {exp_name} - no config")
        continue
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load final metrics
    metrics_path = exp_path / 'final_metrics.json'
    if not metrics_path.exists():
        print(f"Skipping {exp_name} - no metrics")
        continue
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    result = {
        'experiment': exp_name,
        'frequency': config.get('frequency', 'daily'),
        'hidden_size': config['architecture']['hidden_size'],
        'dropout': config['architecture']['dropout'],
        'learning_rate': config['training']['learning_rate'],
        'val_loss': metrics['best_val_loss'],
        'epochs': metrics['total_epochs'],
        'early_stopped': metrics.get('early_stopped', False),
    }
    
    # Check for evaluation results
    eval_path = exp_path / 'evaluation' / 'evaluation_metrics.json'
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        result['sharpe'] = eval_data['financial_metrics']['sharpe_ratio']
        result['dir_acc'] = eval_data['financial_metrics']['directional_accuracy']
        result['mse'] = eval_data['statistical_metrics']['mse']
        result['r2'] = eval_data['statistical_metrics']['r2']
        result['evaluated'] = True
    else:
        result['evaluated'] = False
    
    results.append(result)

if not results:
    print("No results found")
    exit()

df = pd.DataFrame(results)

# Save full results
df.to_csv('sweep_staleness_results.csv', index=False)
print(f"\nSaved results to: sweep_staleness_results.csv")

# Summary statistics
print("\n" + "="*80)
print("SWEEP SUMMARY")
print("="*80)
print(f"Total experiments: {len(df)}")
print(f"Completed: {len(df)}")
print(f"Evaluated: {df['evaluated'].sum() if 'evaluated' in df.columns else 0}")

# Best by validation loss
print("\n" + "="*80)
print("TOP 5 BY VALIDATION LOSS")
print("="*80)
top5 = df.nsmallest(5, 'val_loss')[['experiment', 'hidden_size', 'dropout', 'learning_rate', 'val_loss', 'epochs']]
print(top5.to_string(index=False))

# By frequency
if 'monthly' in df['frequency'].values:
    print("\n" + "="*80)
    print("MONTHLY EXPERIMENT")
    print("="*80)
    monthly = df[df['frequency'] == 'monthly']
    print(monthly[['experiment', 'val_loss', 'epochs']].to_string(index=False))

# By hidden size
print("\n" + "="*80)
print("RESULTS BY HIDDEN SIZE")
print("="*80)
by_hidden = df.groupby('hidden_size')['val_loss'].agg(['mean', 'min', 'count'])
print(by_hidden)

# By dropout
print("\n" + "="*80)
print("RESULTS BY DROPOUT")
print("="*80)
by_dropout = df.groupby('dropout')['val_loss'].agg(['mean', 'min', 'count'])
print(by_dropout)

# By learning rate
print("\n" + "="*80)
print("RESULTS BY LEARNING RATE")
print("="*80)
by_lr = df.groupby('learning_rate')['val_loss'].agg(['mean', 'min', 'count'])
print(by_lr)

EOF

echo ""
echo "========================================================================"
echo "EXHAUSTIVE SWEEP COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved to:"
echo "  sweep_staleness_results.csv"
echo "  logs/sweep_staleness/*.log"
echo ""
echo "Next step: Evaluate promising models with:"
echo "  python train/evaluate_tft.py --experiment-name <exp_name>"
echo ""

