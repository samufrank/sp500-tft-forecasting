#!/bin/bash

# Hyperparameter sweep for TFT baseline optimization
# Run overnight on PC with: bash scripts/sweep_hyperparameters.sh

echo "Starting hyperparameter sweep at $(date)"

# Baseline (repeat for verification)
python train/train_tft.py --experiment-name sweep_baseline \
    --learning-rate 0.001 --hidden-size 32 --max-encoder-length 20 --max-epochs 50

# Learning rate sweep (most important)
for lr in 0.0003 0.0005 0.0007 0.001 0.002; do
    python train/train_tft.py --experiment-name sweep_lr_${lr} \
        --learning-rate ${lr} --hidden-size 32 --max-encoder-length 20 --max-epochs 50
done

# Model size sweep
for hs in 16 24 32 48 64; do
    python train/train_tft.py --experiment-name sweep_hidden_${hs} \
        --learning-rate 0.0005 --hidden-size ${hs} --max-encoder-length 20 --max-epochs 50
done

# Context window sweep
for enc in 10 20 40 60; do
    python train/train_tft.py --experiment-name sweep_encoder_${enc} \
        --learning-rate 0.0005 --hidden-size 32 --max-encoder-length ${enc} --max-epochs 50
done

# Dropout sweep
for drop in 0.1 0.15 0.2 0.3; do
    python train/train_tft.py --experiment-name sweep_dropout_${drop} \
        --learning-rate 0.0005 --hidden-size 32 --max-encoder-length 20 \
        --dropout ${drop} --max-epochs 50
done

# Best combo guesses (based on typical patterns)
python train/train_tft.py --experiment-name sweep_combo1 \
    --learning-rate 0.0005 --hidden-size 48 --max-encoder-length 60 \
    --dropout 0.2 --max-epochs 100

python train/train_tft.py --experiment-name sweep_combo2 \
    --learning-rate 0.0003 --hidden-size 64 --max-encoder-length 40 \
    --dropout 0.15 --max-epochs 100

python train/train_tft.py --experiment-name sweep_combo3 \
    --learning-rate 0.0007 --hidden-size 32 --max-encoder-length 60 \
    --dropout 0.25 --max-epochs 75

echo "Sweep completed at $(date)"
