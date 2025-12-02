#!/bin/bash
# Phase 06b: Rolling Window Evaluation
# Tests model robustness across different market regimes (2016-2024)
#
# Strategy:
# - 10-year rolling training window
# - 1-year validation, 1-year test
# - 9 folds: test years 2016, 2017, ..., 2024
# - Run for both daily and weekly frequencies
# - Use best hyperparameters from fixed-split experiments

echo "========================================================================"
echo "PHASE 06b: ROLLING WINDOW EVALUATION"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

LOG_DIR="logs/phase_06b"
mkdir -p "$LOG_DIR"

# ============================================================================
# DAILY ROLLING EVALUATION
# ============================================================================

echo "----------------------------------------------------------------------"
echo "DAILY ROLLING EVALUATION"
echo "----------------------------------------------------------------------"
echo ""

# Daily baseline (h16, enc20, drop0.10 - best from Phase 02b)
echo "[1/3] Running daily baseline..."
python train/rolling_evaluation.py \
    --experiment-prefix 06b_rolling/daily_h16_baseline \
    --frequency daily \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --hidden-size 16 \
    --max-encoder-length 20 \
    --dropout 0.10 \
    --batch-size 64 \
    --max-epochs 100 \
    --early-stop-patience 10 \
    > "$LOG_DIR/daily_h16_baseline.log" 2>&1

echo "[1/3] Daily baseline complete. Status: $?"

# ============================================================================
# WEEKLY ROLLING EVALUATION
# ============================================================================

echo ""
echo "----------------------------------------------------------------------"
echo "WEEKLY ROLLING EVALUATION"
echo "----------------------------------------------------------------------"
echo ""

# Weekly config 1: Best from 06a sweep (h16_enc8_d025_bs32)
echo "[2/3] Running weekly h16_enc8_d025_bs32 (best from sweep)..."
python train/rolling_evaluation.py \
    --experiment-prefix 06b_rolling/weekly_h16_enc8_d025_bs32 \
    --frequency weekly \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --hidden-size 16 \
    --max-encoder-length 8 \
    --dropout 0.25 \
    --batch-size 32 \
    --max-epochs 100 \
    --early-stop-patience 15 \
    > "$LOG_DIR/weekly_h16_enc8_d025_bs32.log" 2>&1

echo "[2/3] Weekly h16_enc8_d025_bs32 complete. Status: $?"

# Weekly config 2: Second best from 06a sweep (h16_enc12_d015_bs16)
echo "[3/3] Running weekly h16_enc12_d015_bs16 (second best from sweep)..."
python train/rolling_evaluation.py \
    --experiment-prefix 06b_rolling/weekly_h16_enc12_d015_bs16 \
    --frequency weekly \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --hidden-size 16 \
    --max-encoder-length 12 \
    --dropout 0.15 \
    --batch-size 16 \
    --max-epochs 100 \
    --early-stop-patience 15 \
    > "$LOG_DIR/weekly_h16_enc12_d015_bs16.log" 2>&1

echo "[3/3] Weekly h16_enc12_d015_bs16 complete. Status: $?"

# ============================================================================
# ANALYSIS
# ============================================================================

echo ""
echo "========================================================================"
echo "ANALYZING RESULTS"
echo "========================================================================"
echo ""

# Analyze all rolling experiments
python scripts/analyze_rolling.py \
    experiments/06b_rolling/daily_h16_baseline \
    experiments/06b_rolling/weekly_h16_enc8_d025_bs32 \
    experiments/06b_rolling/weekly_h16_enc12_d015_bs16 \
    --compare experiments/02b_vintage_sweep/baseline_h16_drop0.10 \
    --output experiments/06b_rolling/rolling_comparison.csv \
    --plot-dir experiments/06b_rolling \
    > "$LOG_DIR/analysis.log" 2>&1

echo ""
echo "========================================================================"
echo "PHASE 06b COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  experiments/06b_rolling/daily_h16_baseline/"
echo "  experiments/06b_rolling/weekly_h16_enc8_d025_bs32/"
echo "  experiments/06b_rolling/weekly_h16_enc12_d015_bs16/"
echo "  experiments/06b_rolling/rolling_comparison.csv"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "To view summary:"
echo "  cat experiments/06b_rolling/rolling_comparison.csv"
echo ""
