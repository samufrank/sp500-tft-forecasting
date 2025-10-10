#!/bin/bash

export PYTHONUNBUFFERED=1

# base directories
mkdir -p logs experiments

echo "=== Starting experiments at $(date) ==="

# create splits once
echo "Creating daily splits at $(date)"
python scripts/create_splits.py --feature-set core_proposal --frequency daily 2>&1 | tee logs/create_splits_daily.log

# Baseline
echo "Starting exp001_baseline at $(date)"
mkdir -p experiments/exp001_baseline
time python train/train_tft.py --experiment-name exp001_baseline --overwrite 2>&1 | tee experiments/exp001_baseline/terminal.log
echo "Completed exp001_baseline at $(date)"
echo ""

# Ablation: model size
echo "Starting exp002_small at $(date)"
mkdir -p experiments/exp002_small
time python train/train_tft.py --experiment-name exp002_small --hidden-size 16 --overwrite 2>&1 | tee experiments/exp002_small/terminal.log
echo "Completed exp002_small at $(date)"
echo ""

echo "Starting exp003_large at $(date)"
mkdir -p experiments/exp003_large
time python train/train_tft.py --experiment-name exp003_large --hidden-size 64 --overwrite 2>&1 | tee experiments/exp003_large/terminal.log
echo "Completed exp003_large at $(date)"
echo ""

# Ablation: attention heads
echo "Starting exp004_attn1 at $(date)"
mkdir -p experiments/exp004_attn1
time python train/train_tft.py --experiment-name exp004_attn1 --attention-heads 1 --overwrite 2>&1 | tee experiments/exp004_attn1/terminal.log
echo "Completed exp004_attn1 at $(date)"
echo ""

echo "Starting exp005_attn4 at $(date)"
mkdir -p experiments/exp005_attn4
time python train/train_tft.py --experiment-name exp005_attn4 --attention-heads 4 --overwrite 2>&1 | tee experiments/exp005_attn4/terminal.log
echo "Completed exp005_attn4 at $(date)"
echo ""

# Monthly comparison
echo "Creating monthly splits at $(date)"
python scripts/create_splits.py --feature-set core_proposal --frequency monthly 2>&1 | tee logs/create_splits_monthly.log

echo "Starting exp006_monthly at $(date)"
mkdir -p experiments/exp006_monthly
time python train/train_tft.py --experiment-name exp006_monthly \
	--overwrite \
    --frequency monthly \
    --max-encoder-length 12 2>&1 | tee experiments/exp006_monthly/terminal.log
echo "Completed exp006_monthly at $(date)"
echo ""

echo "All experiments complete!"