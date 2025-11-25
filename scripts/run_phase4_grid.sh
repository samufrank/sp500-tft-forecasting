#!/bin/bash
# Phase 4: Complete Loss Modification Grid at lr=0.0005
# Reruns all Phase 4 experiments at correct learning rate for fair comparison with Phase 02b
#
# Total: 63 experiments (100 epochs each)
# - h16d015: 47 experiments (35 redos + 12 magnitude tests)
# - h16d010: 8 experiments (new cross-config validation)
# - h16d025: 8 experiments (new cross-config validation)

echo "========================================================================"
echo "PHASE 4: COMPLETE LOSS MODIFICATION GRID (lr=0.0005)"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

# Configuration
MAX_PARALLEL=10
LOG_DIR="logs/phase_04_lr0.0005"
SPLITS_DIR="data/splits"
EXPERIMENT_BASE="experiments/04_custom_losses"

mkdir -p "$LOG_DIR"

echo "Parallel execution: $MAX_PARALLEL jobs"
echo "Total experiments: 63 (all at 100 epochs, lr=0.0005)"
echo "Fixed params: h=16, encoder_len=20, batch=64"
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
    local hidden_size=$2
    local dropout=$3
    local dirw=${4:-0.0}
    local colw=${5:-0.0}
    local colt=${6:-0.005}
    local extreme_w=${7:-1.0}
    local extreme_p=${8:-95}
    local mag_alpha=${9:-0.0}
    
    echo "[$exp_name] Training..."
    
    # Build command
    CMD="python train/train_tft.py \
        --experiment-name $exp_name \
        --splits-dir $SPLITS_DIR \
        --alignment vintage \
        --frequency daily \
        --feature-set core_proposal \
        --hidden-size $hidden_size \
        --dropout $dropout \
        --learning-rate 0.0005 \
        --max-epochs 100 \
        --early-stop-patience 10 \
        --batch-size 64 \
        --max-encoder-length 20 \
        --attention-heads 2 \
        --hidden-continuous-size 16 \
        --gradient-clip 0.1 \
        --overwrite"
    
    # Add loss modification parameters if non-default
    if (( $(echo "$dirw > 0" | bc -l) )); then
        CMD="$CMD --directional-weight $dirw"
    fi
    
    if (( $(echo "$colw > 0" | bc -l) )); then
        CMD="$CMD --dist-loss-std-weight $colw --collapse-threshold $colt"
    fi
    
    if (( $(echo "$extreme_w > 1.0" | bc -l) )); then
        CMD="$CMD --extreme-move-weight $extreme_w --extreme-move-percentile $extreme_p"
    fi
    
    if (( $(echo "$mag_alpha > 0" | bc -l) )); then
        CMD="$CMD --magnitude-weight-alpha $mag_alpha"
    fi
    
    # Run training
    eval $CMD > "$LOG_DIR/train_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Training complete, evaluating..."
    
    # Run evaluation
    python train/evaluate_tft.py \
        --experiment-name "$exp_name" \
        > "$LOG_DIR/eval_${exp_name}.log" 2>&1
    
    echo "[$exp_name] Complete"
}

echo "========================================================================"
echo "STARTING PHASE 4 GRID (63 experiments)"
echo "========================================================================"
echo ""

counter=1

# ========================================================================
# PART 1: h16d015 - Baseline
# ========================================================================
echo "Part 1/8: h16d015 baseline (1 experiment)"
echo "------------------------------------------------------------------------"

wait_for_slot
echo "[$counter/63] h16d015_baseline"
train_and_eval "h16d015_baseline" 16 0.15 &
counter=$((counter + 1))
sleep 2

# ========================================================================
# PART 2: h16d015 - Directional only sweep
# ========================================================================
echo ""
echo "Part 2/8: h16d015 directional only (9 experiments)"
echo "------------------------------------------------------------------------"

DIRW_VALUES=(0.1 0.2 0.5 1.0 2.0 3.0 5.0 10.0 100.0)
for dirw in "${DIRW_VALUES[@]}"; do
    wait_for_slot
    exp_name=$(printf "h16d015_dirw%.1f" $dirw)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 $dirw &
    counter=$((counter + 1))
    sleep 2
done

# ========================================================================
# PART 3: h16d015 - Directional + collapse combined
# ========================================================================
echo ""
echo "Part 3/8: h16d015 directional + collapse (15 experiments)"
echo "------------------------------------------------------------------------"

# Standard combined (dirw + colw=0.1, thresh=0.035)
DIRW_COMBINED=(0.1 0.2 1.0 2.0 3.0 5.0 10.0)
for dirw in "${DIRW_COMBINED[@]}"; do
    wait_for_slot
    exp_name=$(printf "h16d015_dirw%.1f_colw0.1t0.035" $dirw)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 $dirw 0.1 0.035 &
    counter=$((counter + 1))
    sleep 2
done

# Refined combined (dirw=2.0 with different collapse settings)
COLLAPSE_CONFIGS=(
    "0.5 0.005"
    "0.5 0.01"
    "1.0 0.005"
    "1.0 0.01"
)
for config in "${COLLAPSE_CONFIGS[@]}"; do
    read colw colt <<< "$config"
    wait_for_slot
    exp_name=$(printf "h16d015_dirw2.0_colw%.1ft%.3f" $colw $colt)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 2.0 $colw $colt &
    counter=$((counter + 1))
    sleep 2
done

# Refined combined (dirw=3.0 with different collapse settings)
for config in "${COLLAPSE_CONFIGS[@]}"; do
    read colw colt <<< "$config"
    wait_for_slot
    exp_name=$(printf "h16d015_dirw3.0_colw%.1ft%.3f" $colw $colt)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 3.0 $colw $colt &
    counter=$((counter + 1))
    sleep 2
done

# ========================================================================
# PART 4: h16d015 - Variance only (failed experiments)
# ========================================================================
echo ""
echo "Part 4/8: h16d015 variance only (10 experiments - expect failure)"
echo "------------------------------------------------------------------------"

VARIANCE_CONFIGS=(
    "0.1 0.005"
    "0.1 0.02"
    "0.1 0.035"
    "0.1 0.05"
    "0.1 0.075"
    "0.1 1.0"
    "0.5 0.005"
    "0.5 0.01"
    "1.0 0.005"
    "1.0 0.01"
)
for config in "${VARIANCE_CONFIGS[@]}"; do
    read colw colt <<< "$config"
    wait_for_slot
    exp_name=$(printf "h16d015_varcw%.1ft%.3f" $colw $colt)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 0.0 $colw $colt &
    counter=$((counter + 1))
    sleep 2
done

# ========================================================================
# PART 5: h16d015 - Extreme weighting variations (12 experiments)
# ========================================================================
echo ""
echo "Part 5/8: h16d015 extreme/magnitude weighting (12 experiments)"
echo "------------------------------------------------------------------------"

# Linear magnitude (alpha)
ALPHA_VALUES=(0.1 0.5 1.0)
for alpha in "${ALPHA_VALUES[@]}"; do
    wait_for_slot
    exp_name=$(printf "h16d015_alpha%.1f" $alpha)
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.15 2.0 0.0 0.005 1.0 95 $alpha &
    counter=$((counter + 1))
    sleep 2
done

# Extreme weighting variations
EXTREME_CONFIGS=(
    "0.0 2.0 90"    # dirw2.0_extremew2.0p90
    "0.0 2.0 95"    # dirw2.0_extremew2.0p95
    "0.0 3.0 90"    # dirw2.0_extremew3.0p90
    "0.0 3.0 95"    # dirw2.0_extremew3.0p95
    "0.0 5.0 75"    # dirw2.0_extremew5.0p75
    "0.0 5.0 90"    # dirw2.0_extremew5.0p90
    "0.0 5.0 95"    # dirw2.0_extremew5.0p95
    "0.0 10.0 90"   # dirw2.0_extremew10.0p90
    "1.0 5.0 90"    # dirw1.0_extremew5.0p90
)
for config in "${EXTREME_CONFIGS[@]}"; do
    read dirw extreme_w extreme_p <<< "$config"
    wait_for_slot
    if (( $(echo "$dirw > 0" | bc -l) )); then
        exp_name=$(printf "h16d015_dirw%.1f_extremew%.1fp%d" $dirw $extreme_w $extreme_p)
    else
        exp_name=$(printf "h16d015_dirw2.0_extremew%.1fp%d" $extreme_w $extreme_p)
    fi
    echo "[$counter/63] $exp_name"
    # Note: all extreme experiments have dirw=2.0 unless specified
    if (( $(echo "$dirw == 0" | bc -l) )); then
        train_and_eval "$exp_name" 16 0.15 2.0 0.0 0.005 $extreme_w $extreme_p &
    else
        train_and_eval "$exp_name" 16 0.15 $dirw 0.0 0.005 $extreme_w $extreme_p &
    fi
    counter=$((counter + 1))
    sleep 2
done

# ========================================================================
# PART 6: h16d010 - Full sweep on stable baseline
# ========================================================================
echo ""
echo "Part 6/8: h16d010 full sweep (8 experiments)"
echo "------------------------------------------------------------------------"

# Define h16d010 configs
H16D010_CONFIGS=(
    "baseline 0.0 0.0 0.005 1.0 95"
    "dirw1.0 1.0 0.0 0.005 1.0 95"
    "dirw2.0 2.0 0.0 0.005 1.0 95"
    "dirw1.0_colw0.1t0.035 1.0 0.1 0.035 1.0 95"
    "dirw2.0_colw0.1t0.035 2.0 0.1 0.035 1.0 95"
    "extremew5.0p90 0.0 0.0 0.005 5.0 90"
    "dirw1.0_extremew5.0p90 1.0 0.0 0.005 5.0 90"
    "dirw2.0_extremew5.0p90 2.0 0.0 0.005 5.0 90"
)

for config in "${H16D010_CONFIGS[@]}"; do
    read name dirw colw colt extreme_w extreme_p <<< "$config"
    wait_for_slot
    exp_name="h16d010_$name"
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.10 $dirw $colw $colt $extreme_w $extreme_p &
    counter=$((counter + 1))
    sleep 2
done

# ========================================================================
# PART 7: h16d025 - Full sweep on marginal baseline
# ========================================================================
echo ""
echo "Part 7/8: h16d025 full sweep (8 experiments)"
echo "------------------------------------------------------------------------"

# Same configs as h16d010 but with dropout=0.25
for config in "${H16D010_CONFIGS[@]}"; do
    read name dirw colw colt extreme_w extreme_p <<< "$config"
    wait_for_slot
    exp_name="h16d025_$name"
    echo "[$counter/63] $exp_name"
    train_and_eval "$exp_name" 16 0.25 $dirw $colw $colt $extreme_w $extreme_p &
    counter=$((counter + 1))
    sleep 2
done

echo ""
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "========================================================================"
echo "PHASE 4 COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""

#echo "Summarizing results..."
#python scripts/analyze_penalty_sweep.py > "$LOG_DIR/phase4_summary.log" 2>&1

echo ""
echo "Results saved to: $EXPERIMENT_BASE/"
echo "Logs saved to: $LOG_DIR/"
echo "Summary: $LOG_DIR/phase4_summary.log"
echo ""
echo "Next steps:"
echo "  1. Review summary statistics"
echo "  2. Compare against Phase 02b baselines"
echo "  3. Identify best loss modifications for staleness testing"
echo ""