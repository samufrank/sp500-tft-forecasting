#!/bin/bash

# Find and evaluate the best model from sweep
# Run after sweep completes

echo "Analyzing sweep results..."
python scripts/analyze_sweep.py

echo ""
echo "Enter the best experiment name to evaluate (e.g., sweep_lr_0.0005):"
read exp_name

echo "Evaluating $exp_name..."
python train/evaluate_tft.py --experiment-name $exp_name
