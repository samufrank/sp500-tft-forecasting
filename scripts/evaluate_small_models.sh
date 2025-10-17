#!/bin/bash

# Evaluate all small models (hidden ≤ 20) from both sweeps
# These are most likely to make varied predictions

echo "Evaluating small models from sweeps..."
echo "Started at $(date)"

# From original sweep
python train/evaluate_tft.py --experiment-name sweep_hidden_16

# From sweep2 - all hidden sizes 8-20
python train/evaluate_tft.py --experiment-name sweep2_hidden_8
python train/evaluate_tft.py --experiment-name sweep2_hidden_10
python train/evaluate_tft.py --experiment-name sweep2_hidden_12
python train/evaluate_tft.py --experiment-name sweep2_hidden_14
python train/evaluate_tft.py --experiment-name sweep2_hidden_16
python train/evaluate_tft.py --experiment-name sweep2_hidden_18
python train/evaluate_tft.py --experiment-name sweep2_hidden_20

# Combo experiments with small models
python train/evaluate_tft.py --experiment-name sweep2_combo1  # h12
python train/evaluate_tft.py --experiment-name sweep2_combo2  # h10
python train/evaluate_tft.py --experiment-name sweep2_combo3  # h18

# Some hidden_16 variations
python train/evaluate_tft.py --experiment-name sweep2_h16_enc_40
python train/evaluate_tft.py --experiment-name sweep2_h16_enc_50
python train/evaluate_tft.py --experiment-name sweep2_h16_drop_0.2
python train/evaluate_tft.py --experiment-name sweep2_h16_drop_0.25

echo "Completed at $(date)"
echo ""
echo "Now analyze with: python scripts/compare_evaluations.py"

