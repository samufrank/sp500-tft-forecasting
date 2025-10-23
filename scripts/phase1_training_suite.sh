#!/bin/bash
# Phase 1 Training Suite - Staleness Feature Validation
# Run comprehensive experiments comparing baseline vs staleness features

echo "========================================================================"
echo "PHASE 1 TRAINING SUITE - STALENESS FEATURE VALIDATION"
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

mkdir -p experiments/01_staleness_features

echo "Configuration:"
echo "  Splits: $SPLITS_DIR"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Dropout: $DROPOUT"
echo "  Learning rate: $LR"
echo "  Max epochs: $MAX_EPOCHS"
echo ""

# ============================================================================
# EXPERIMENT 1: Baseline (no staleness) - 50 epochs
# ============================================================================
echo "========================================================================"
echo "EXPERIMENT 1/4: Baseline (no staleness features)"
echo "========================================================================"
python train/train_tft.py \
    --experiment-name baseline_fixed_h16_drop0.25 \
    --splits-dir $SPLITS_DIR \
    --hidden-size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --learning-rate $LR \
    --max-epochs $MAX_EPOCHS \
    --max-encoder-length $ENCODER_LENGTH \
    --no-staleness

echo ""
echo "Baseline training complete at: $(date)"
echo ""

# ============================================================================
# EXPERIMENT 2: With staleness features - 50 epochs
# ============================================================================
echo "========================================================================"
echo "EXPERIMENT 2/4: With staleness features"
echo "========================================================================"
python train/train_tft.py \
    --experiment-name staleness_fixed_h16_drop0.25 \
    --splits-dir $SPLITS_DIR \
    --hidden-size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --learning-rate $LR \
    --max-epochs $MAX_EPOCHS \
    --max-encoder-length $ENCODER_LENGTH

echo ""
echo "Staleness training complete at: $(date)"
echo ""

# ============================================================================
# EXPERIMENT 3: Evaluate baseline
# ============================================================================
echo "========================================================================"
echo "EXPERIMENT 3/4: Evaluating baseline on test set"
echo "========================================================================"
python train/evaluate_tft.py \
    --experiment-name baseline_fixed_h16_drop0.25

echo ""
echo "Baseline evaluation complete at: $(date)"
echo ""

# ============================================================================
# EXPERIMENT 4: Evaluate staleness model
# ============================================================================
echo "========================================================================"
echo "EXPERIMENT 4/4: Evaluating staleness model on test set"
echo "========================================================================"
python train/evaluate_tft.py \
    --experiment-name staleness_fixed_h16_drop0.25

echo ""
echo "Staleness evaluation complete at: $(date)"
echo ""

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
echo "========================================================================"
echo "RESULTS SUMMARY"
echo "========================================================================"

# Extract key metrics using Python
python << 'EOF'
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
echo "PHASE 1 TRAINING SUITE COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved to:"
echo "  experiments/baseline_fixed_h16_drop0.25/"
echo "  experiments/staleness_fixed_h16_drop0.25/"
echo ""
