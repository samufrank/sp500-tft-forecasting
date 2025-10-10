# S&P 500 TFT Forecasting

Temporal Fusion Transformer (TFT) experiments for S&P 500 return prediction using macroeconomic features.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (use exact versions for compatibility)
pip install pytorch-lightning==1.9.5 pytorch-forecasting==0.10.3
pip install torch pandas numpy matplotlib scipy

# Install project as package
pip install -e .
```

**Critical:** Must use `pytorch-lightning==1.9.5` and `pytorch-forecasting==0.10.3` for compatibility.

## Quick Start

```bash
# 1. Create data splits (run once per feature set)
python scripts/create_splits.py --feature-set core_proposal --frequency daily

# 2. Train TFT model
python train/train_tft.py --experiment-name my_experiment

# 3. Evaluate on test set (uses best checkpoint automatically)
python train/evaluate_tft.py --experiment-name my_experiment

# 4. List available experiments and checkpoints
python scripts/list_checkpoints.py
python scripts/list_checkpoints.py my_experiment
```

## Project Structure

```
sp500-tft-forecasting/
├── src/
│   ├── feature_configs.py    # Feature set definitions
│   └── data_utils.py          # Data loading utilities
├── scripts/
│   ├── create_splits.py       # Create train/val/test splits
│   └── list_checkpoints.py    # View experiments and checkpoints
├── train/
│   ├── train_tft.py          # Train TFT models
│   └── evaluate_tft.py       # Evaluate trained models
├── data/
│   ├── financial_dataset_daily.csv
│   └── splits/               # Generated train/val/test CSVs
└── experiments/              # Auto-generated outputs
    └── {experiment_name}/
        ├── config.json
        ├── final_metrics.json
        ├── checkpoints/
        ├── logs/
        └── evaluation/
```

## Usage Details

### Creating Data Splits

```bash
# Daily frequency (default)
python scripts/create_splits.py --feature-set core_proposal --frequency daily

# Monthly frequency
python scripts/create_splits.py --feature-set core_proposal --frequency monthly
```

**Feature sets available:**
- `core_proposal` - 5 core features (VIX, Treasury yields, inflation)
- TODO: `core_plus_credit` - Core + credit spreads
- TODO: `macro_heavy` - Emphasis on macro fundamentals
- TODO: `market_only` - Pure market features, no macro
- TODO: `kitchen_sink` - All available features

Splits are saved to `data/splits/` as CSVs with 70% train / 15% val / 15% test (temporal, not random).

### Training Models

```bash
# Basic training with defaults
python train/train_tft.py --experiment-name baseline

# Custom architecture
python train/train_tft.py --experiment-name large_model \
    --hidden-size 64 \
    --attention-heads 4 \
    --max-epochs 100

# Small test run
python train/train_tft.py --experiment-name quick_test \
    --hidden-size 16 \
    --max-epochs 5

# Different feature set
python train/train_tft.py --experiment-name macro_exp \
    --feature-set macro_heavy
```

**Key arguments:**
- `--experiment-name` (required) - Name for outputs
- `--feature-set` - Which feature config to use (default: core_proposal)
- `--frequency` - daily or monthly (default: daily)
- `--hidden-size` - Model capacity: 16/32/64 (default: 32)
- `--attention-heads` - Number of attention heads: 1/2/4 (default: 2)
- `--max-encoder-length` - Lookback window (default: 20)
- `--max-epochs` - Training epochs (default: 50)
- `--learning-rate` - Learning rate (default: 0.001)
- `--batch-size` - Batch size (default: 64)
- `--seed` - Random seed for reproducibility (default: 42)
- `--overwrite` - Allow overwriting existing experiment

See all options: `python train/train_tft.py --help`

**Outputs:**
- `experiments/{name}/config.json` - All hyperparameters
- `experiments/{name}/final_metrics.json` - Best validation loss, epochs
- `experiments/{name}/checkpoints/` - Model checkpoints (best 3 + last)
- `experiments/{name}/logs/` - TensorBoard and CSV logs

### Evaluating Models

```bash
# Automatic (uses best checkpoint from training)
python train/evaluate_tft.py --experiment-name baseline

# Manual checkpoint selection
python train/evaluate_tft.py --experiment-name baseline \
    --checkpoint experiments/baseline/checkpoints/tft-epoch=05-val_loss=0.1234.ckpt

# Custom test split
python train/evaluate_tft.py --experiment-name baseline \
    --test-split data/splits/custom_test.csv
```

**Outputs** (saved to `experiments/{name}/evaluation/`):
- `evaluation_metrics.json` - All metrics in JSON format
- `predictions.csv` - Actual vs predicted values
- `diagnostic_plots.png` - Statistical diagnostics (4 plots)
- `performance_plots.png` - Financial performance (2 plots)

**Metrics computed:**
- Statistical: MSE, RMSE, MAE, R²
- Financial: Directional accuracy, Sharpe ratio, total return, max drawdown
- Diagnostics: Residual analysis, normality tests, Q-Q plots

### Managing Experiments

```bash
# List all experiments
python scripts/list_checkpoints.py

# Show checkpoints for specific experiment
python scripts/list_checkpoints.py baseline

# Example output:
# CHECKPOINTS FOR: baseline
# Best model (val_loss=0.119101):
#   experiments/baseline/checkpoints/tft-epoch=00-val_loss=0.1191.ckpt
# All checkpoints (4 total):
#   tft-epoch=00-val_loss=0.1191.ckpt (2.4 MB) ← BEST
#   tft-epoch=01-val_loss=0.1552.ckpt (2.4 MB)
#   last.ckpt (2.4 MB)
```

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds (`--seed 42`)
- Deterministic CUDA operations
- Temporal (not random) data splits
- Complete config logging
- Version-locked dependencies

To reproduce an experiment:
```bash
# Check config
cat experiments/{name}/config.json

# Recreate splits with same settings
python scripts/create_splits.py --feature-set {from_config} --frequency {from_config}

# Retrain with same hyperparameters
python train/train_tft.py --experiment-name {new_name} {args_from_config}
```

## Current Status

**Completed:**
- Data collection and preprocessing
- Intial feature engineering with proper temporal alignment
- Data split creation with versioning
- TFT training pipeline with full logging
- Evaluation pipeline with comprehensive metrics
- Checkpoint management utilities

**In Progress:**
- Baseline experiments across feature sets
- Architectural modifications

**Baseline Results (exp004, core features):**
- Test period: 2020-07-22 to 2025-10-06 (1,280 days)
- Directional accuracy: 53.67%
- Sharpe ratio: 0.83
- Total return: 92.16%
- Note: Model collapsed to predicting unconditional mean (~0.35% daily)

## Notes

- **Financial forecasting is hard:** R² near zero is expected, not a bug
- **52-54% directional accuracy is good** - random walk baseline is 50%
- **Lead with Sharpe ratio** when presenting results, not accuracy
- Test period (2020-2025) includes COVID crash and recovery - strong bull market
- Early epoch convergence suggests need for hyperparameter tuning

## References

- Lim et al. (2021) - Temporal Fusion Transformers
- Gu, Kelly & Xiu (2020) - Empirical Asset Pricing via Machine Learning
- Welch & Goyal (2008) - Equity Premium Prediction

## License

Academic project for Deep Learning (Fall 2025)
