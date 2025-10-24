#!/bin/bash
# Phase 1 Parallel Training Suite - Staleness Feature Validation
# Runs baseline and staleness experiments in parallel

echo "========================================================================"
echo "PHASE 1 PARALLEL TRAINING SUITE - STALENESS FEATURE VALIDATION"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
SPLITS_DIR="data/splits/fixed"
HIDDEN_SIZE=16
DROPOUT=0.25
LR=0.0005
MAX_EPOCHS=50
ENCODER_LENGTH=20

echo "Configuration:"
echo "  Splits: $SPLITS_DIR"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Dropout: $DROPOUT"
echo "  Learning rate: $LR"
echo "  Max epochs: $MAX_EPOCHS"
echo ""

# Create logs directory
mkdir -p logs

# ============================================================================
# PARALLEL TRAINING: Baseline + Staleness
# ============================================================================
echo "========================================================================"
echo "Starting parallel training (baseline + staleness)"
echo "========================================================================"

# Launch baseline training
python train/train_tft.py \
    --experiment-name baseline_fixed_h16_drop0.25 \
    --splits-dir $SPLITS_DIR \
    --hidden-size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --learning-rate $LR \
    --max-epochs $MAX_EPOCHS \
    --max-encoder-length $ENCODER_LENGTH \
    --no-staleness \
    > logs/baseline_training.log 2>&1 &

BASELINE_PID=$!

# Launch staleness training
python train/train_tft.py \
    --experiment-name staleness_fixed_h16_drop0.25 \
    --splits-dir $SPLITS_DIR \
    --hidden-size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --learning-rate $LR \
    --max-epochs $MAX_EPOCHS \
    --max-encoder-length $ENCODER_LENGTH \
    > logs/staleness_training.log 2>&1 &

STALENESS_PID=$!

echo "Baseline training PID: $BASELINE_PID"
echo "Staleness training PID: $STALENESS_PID"

# Wait for both trainings to complete
echo "Waiting for training to complete..."
wait $BASELINE_PID || true
echo "Baseline training completed at: $(date)"
wait $STALENESS_PID || true
echo "Staleness training completed at: $(date)"
echo ""

# ============================================================================
# EVALUATION: Run on test set
# ============================================================================
echo "========================================================================"
echo "Evaluating models on test set"
echo "========================================================================"

echo "Evaluating baseline..."
python train/evaluate_tft.py \
    --experiment-name baseline_fixed_h16_drop0.25 \
	--test-split data/splits/fixed/core_proposal_daily_test.csv \
    > logs/baseline_evaluation.log 2>&1

echo "Evaluating staleness model..."
python train/evaluate_tft.py \
    --experiment-name staleness_fixed_h16_drop0.25 \
	--test-split data/splits/fixed/core_proposal_daily_test.csv \
    > logs/staleness_evaluation.log 2>&1

echo "Evaluation complete at: $(date)"
echo ""

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
echo "========================================================================"
echo "RESULTS SUMMARY"
echo "========================================================================"

python << 'EOF' | tee logs/results_summary.log
import json

def load_metrics(exp_name):
    try:
        with open(f'experiments/{exp_name}/evaluation/evaluation_metrics.json') as f:
            return json.load(f)
    except:
        return None

baseline = load_metrics('baseline_fixed_h16_drop0.25')
staleness = load_metrics('staleness_fixed_h16_drop0.25')

print("\nBaseline (no staleness):")
if baseline:
    print(f"  Sharpe Ratio: {baseline['financial_metrics']['sharpe_ratio']:.4f}")
    print(f"  Dir Accuracy: {baseline['financial_metrics']['directional_accuracy']:.4f}")
    print(f"  MSE: {baseline['statistical_metrics']['mse']:.4f}")
    print(f"  R²: {baseline['statistical_metrics']['r2']:.4f}")
else:
    print("  Evaluation not found")

print("\nWith Staleness Features:")
if staleness:
    print(f"  Sharpe Ratio: {staleness['financial_metrics']['sharpe_ratio']:.4f}")
    print(f"  Dir Accuracy: {staleness['financial_metrics']['directional_accuracy']:.4f}")
    print(f"  MSE: {staleness['statistical_metrics']['mse']:.4f}")
    print(f"  R²: {staleness['statistical_metrics']['r2']:.4f}")
else:
    print("  Evaluation not found")

if baseline and staleness:
    sharpe_delta = staleness['financial_metrics']['sharpe_ratio'] - baseline['financial_metrics']['sharpe_ratio']
    acc_delta = staleness['financial_metrics']['directional_accuracy'] - baseline['financial_metrics']['directional_accuracy']
    
    print("\nImprovement:")
    print(f"  Sharpe: {sharpe_delta:+.4f} ({sharpe_delta/baseline['financial_metrics']['sharpe_ratio']*100:+.1f}%)")
    print(f"  Dir Acc: {acc_delta:+.4f} ({acc_delta/baseline['financial_metrics']['directional_accuracy']*100:+.1f}%)")
EOF

echo ""
echo "========================================================================"
echo "PHASE 1 PARALLEL TRAINING SUITE COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved to:"
echo "  experiments/baseline_fixed_h16_drop0.25/"
echo "  experiments/staleness_fixed_h16_drop0.25/"
echo "  logs/results_summary.log"
echo ""
