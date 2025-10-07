# TFT Training Pipeline (Early Development)

Quick reference for running TFT experiments on S&P 500 data.

## Setup
```bash
pip install torch pytorch-forecasting pytorch-lightning pandas numpy
```

## Usage

### 1. Create data splits (run once)
```bash
python create_splits.py --feature-set core_proposal --frequency daily
```

### 2. Train TFT
```bash
python train_tft.py --experiment-name tft_baseline
```

Results go to `experiments/tft_baseline/` with checkpoints, logs, and config.

### 3. Quick test (small model, fast)
```bash
python train_tft.py --experiment-name test --hidden-size 16 --max-epochs 5
```

## Common Options

**Training:**
- `--hidden-size 32` - model size (16/32/64)
- `--attention-heads 2` - attention heads (1/2/4)
- `--max-epochs 50` - training epochs
- `--learning-rate 0.001` - learning rate

**Data:**
- `--frequency daily` - daily or monthly
- `--feature-set core_proposal` - which features to use

**Safety:**
- `--overwrite` - allow overwriting existing experiment
- `--seed 42` - random seed (default: 42)

See all options: `python train_tft.py --help`

## Files

- `create_splits.py` - make train/val/test splits
- `train_tft.py` - train TFT models
- `feature_configs.py` - feature set definitions
- `data_utils.py` - data loading utilities

## Current Status

⚠️ **Early development** - data processing still being validated

- Data collection: ✅
- Split creation: ✅
- TFT training pipeline: ✅
- Evaluation metrics: TODO
- ARIMAX/LSTM baselines: TODO
