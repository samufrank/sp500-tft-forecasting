#!/bin/bash

# Targeted sweep around hidden_16 (the working configuration)
# Run with: bash scripts/sweep2_targeted.sh

echo "Starting targeted sweep at $(date)"

# ============================================================================
# PHASE 1: Test smaller hidden sizes (including 8 for fun)
# ============================================================================
echo "Phase 1: Testing smaller hidden sizes..."

for hs in 8 10 12 14 16 18 20; do
    python train/train_tft.py --experiment-name sweep2_hidden_${hs} \
        --hidden-size ${hs} \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-encoder-length 20 \
        --max-epochs 50
done

# ============================================================================
# PHASE 2: Learning rate variations with hidden_16
# ============================================================================
echo "Phase 2: Testing learning rates with hidden_16..."

for lr in 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008; do
    python train/train_tft.py --experiment-name sweep2_h16_lr_${lr} \
        --hidden-size 16 \
        --learning-rate ${lr} \
        --dropout 0.15 \
        --max-encoder-length 20 \
        --max-epochs 50
done

# ============================================================================
# PHASE 3: Encoder length with hidden_16
# ============================================================================
echo "Phase 3: Testing encoder lengths with hidden_16..."

for enc in 15 20 30 40 50 60; do
    python train/train_tft.py --experiment-name sweep2_h16_enc_${enc} \
        --hidden-size 16 \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-encoder-length ${enc} \
        --max-epochs 50
done

# ============================================================================
# PHASE 4: Dropout with hidden_16
# ============================================================================
echo "Phase 4: Testing dropout with hidden_16..."

for drop in 0.05 0.1 0.15 0.2 0.25 0.3; do
    python train/train_tft.py --experiment-name sweep2_h16_drop_${drop} \
        --hidden-size 16 \
        --learning-rate 0.0005 \
        --dropout ${drop} \
        --max-encoder-length 20 \
        --max-epochs 50
done

# ============================================================================
# PHASE 5: Best combo guesses based on patterns
# ============================================================================
echo "Phase 5: Testing promising combinations..."

# Small model + longer context
python train/train_tft.py --experiment-name sweep2_combo1 \
    --hidden-size 12 \
    --learning-rate 0.0004 \
    --dropout 0.2 \
    --max-encoder-length 40 \
    --max-epochs 75

# Very small model + moderate settings
python train/train_tft.py --experiment-name sweep2_combo2 \
    --hidden-size 10 \
    --learning-rate 0.0005 \
    --dropout 0.15 \
    --max-encoder-length 30 \
    --max-epochs 75

# Slightly larger than 16 + conservative
python train/train_tft.py --experiment-name sweep2_combo3 \
    --hidden-size 18 \
    --learning-rate 0.0003 \
    --dropout 0.25 \
    --max-encoder-length 30 \
    --max-epochs 75

echo "Targeted sweep completed at $(date)"
echo ""
echo "Total experiments: ~30"
echo "Run analysis with: python scripts/analyze_sweep.py"

