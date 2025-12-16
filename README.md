# S&P 500 Forecasting with Temporal Fusion Transformers

Systematic characterization of Temporal Fusion Transformer (TFT) performance on financial time series forecasting across 450+ experiments spanning 11 experimental phases. This project provides comprehensive documentation of TFT behavior on financial data, discovering fundamental limitations in standard TFT architecture and identifying configuration strategies that achieve modest improvements.

**Key Finding**: TFT exhibits gradient collapse in financial forecasting - the encoder successfully learns market regime structure, but the output layer fails to translate these representations into stable predictions. Weekly resampling achieves 59.1±8.3% directional accuracy across 9 years of rolling evaluation, representing the best performance observed across all configurations, though models primarily match base rate expectations rather than demonstrating strong directional forecasting skill.

**Research Contributions**:
- Systematic TFT characterization on financial data (11 experimental phases, 450+ models)
- Discovery of gradient collapse failure mode specific to financial forecasting (83% output layer collapse, 279% encoder gradient increase)
- Comprehensive negative results documenting what doesn't work (staleness features, regime-conditional output, multi-horizon forecasting, feature engineering)
- Identification of weekly resampling as most robust configuration
- Extensive attention mechanism analysis demonstrating post-attention failure
- Proper rolling evaluation methodology exposing fixed-split overfitting

---

## Key Results

### Fixed-Split Performance (Test: 2020-2025)

| Model | Dir Acc | Baseline | Excess | Sharpe |
|-------|---------|----------|--------|--------|
| ARIMAX(3,0,3) | 54.0% | 53.6% | +0.4% | 1.56 |
| LSTM | 53.3% | 53.6% | -0.3% | 0.39 |
| TFT (daily, single-step) | 55.2% | 53.6% | +1.6% | 1.21 |
| TFT (daily, 10-step)¹ | 56.1% | 53.6% | +2.1% | 2.08 |
| TFT (weekly) | 58.1% | 56.0% | +2.1% | 1.05 |
| TFT + Regime Attn (weekly)² | 59.4% | 56.0% | +3.4% | 1.22 |

¹ Multi-horizon forecasting: predictions averaged over 10-step window  
² Regime attention gains observed in fixed-split do not generalize to rolling evaluation

**Baseline definitions**: Frequency-matched naive prediction rate (fraction of positive returns in test set). Daily baseline: 53.6%, Weekly baseline: 56.0%.

**Checkpoint selection**: Models selected by validation prediction diversity (`val_pred_std`), not validation loss. Validation loss anti-correlated with directional accuracy (r=-0.46).

### Rolling Evaluation (9 folds, 2016-2024)

Rolling evaluation across nine years reveals that fixed-split gains do not generalize. Models achieve approximately zero excess accuracy when accounting for fold-specific base rates.

| Model | Dir Acc (mean ± std) | Base Rate | Excess | Sharpe |
|-------|---------------------|-----------|--------|--------|
| Weekly baseline | 59.1 ± 8.3% | 60.2% | -1.1pp | 1.52 ± 1.33 |
| Daily baseline | 53.3 ± 5.2% | 54.5% | -1.2pp | 0.95 ± 1.31 |
| Daily multi-horizon (h=10, 7q) | 50.9 ± 5.5% | 54.5% | -3.6pp | 0.94 ± 1.04 |
| Daily multi-horizon (h=10, 3q)³ | 53.8 ± 5.2% | 54.5% | -0.7pp | 0.97 ± 1.10 |

³ Reduced quantiles (3q vs 7q) mitigate output layer capacity constraints

**Base rate context**: Rolling base rates represent the average fraction of positive returns across 9 test folds (2016-2024), accounting for varying market conditions. Daily ranges from 43.8% (2022 bear) to 59.6% (2019 bull). Weekly ranges from 45.3% (2022 bear) to 67.2% (2024 bull).

**Key insight**: Models achieve near-zero excess accuracy by predicting predominantly positive returns, matching performance to market drift rather than learning directional signals. Weekly achieves higher raw accuracy (59.1%) primarily because weekly data exhibits higher positive bias (60.2% vs 54.5%).

**Universal limitation**: All models fail during 2022 Federal Reserve tightening period (~40% accuracy) regardless of architecture, checkpoint selection, or feature set.

---

## Core Discoveries

### 1. Gradient Collapse in Financial TFT

Standard TFT architecture exhibits systematic failure when applied to financial forecasting. Within the first five epochs, output layer gradients collapse by 83% while encoder gradients continue increasing by 279% throughout training. The encoder successfully learns meaningful regime representations - it can detect market shifts and adapt its internal state - but the output layer settles into a low-gradient state and cannot translate these insights into stable, directionally accurate predictions.

This failure manifests as validation loss becoming anti-correlated with directional accuracy (r=-0.46). Models that optimize quantile loss well often produce the worst directional predictions, learning to exploit market drift rather than genuine temporal patterns. The gradient flow analysis shows output gradients dropping sharply in early epochs then plateauing, while encoder gradients continue rising throughout training.

The implication is clear: TFT's attention mechanism works correctly. It detects regime shifts and adapts temporal focus appropriately. The breakdown occurs in post-attention processing, where the model fails to generate diverse, directionally accurate predictions despite having learned meaningful representations.

### 2. Weekly Resampling as Best Configuration

Weekly resampling achieves the highest directional accuracy observed across all experiments: 59.1±8.3% in rolling evaluation versus 53.3±5.2% for daily models. However, this performance must be understood in context.

Rolling evaluation reveals that weekly models achieve approximately 1pp below their base rate (59.1% vs 60.2% positive rate across folds), while daily models similarly operate 1pp below their base rate (53.3% vs 54.5%). The higher raw accuracy of weekly models primarily reflects the higher positive bias in weekly-resampled data rather than superior directional forecasting skill.

The key contribution of weekly resampling is stability across different configurations. While both frequencies achieve near-zero excess accuracy, weekly provides:
- Consistent collapse patterns across folds (healthy % variance: ±1.0% vs ±13.9% for daily)
- Higher Sharpe ratios (1.52 vs 0.95)
- Better performance during volatile periods (2020 COVID recovery: 60% vs 56.7% accuracy)

Important limitation: gradient collapse still occurs with weekly frequency. Output layer gradients exhibit similar collapse patterns to daily models; the aggregation primarily reduces prediction noise rather than solving the underlying gradient pathology.

### 3. Comprehensive Negative Results

Several modifications caused systematic failures or degraded performance.

Staleness features (Phase 1, 64 experiments) showed a 94% collapse rate when using continuous staleness counters like `days_since_CPI_update`. Phase 11 staleness-aware attention experiments revealed that attention entropy drops from 2.48 to 1.37 with these features, over-constraining the mechanism. The root cause is a monotonic recency gradient in forward-filled data - any mechanism that "discounts stale data" mathematically becomes "attend to recent timesteps," which the model exploits for collapse. Sparse binary flags (`CPI_is_fresh`) avoid collapse but provide no predictive value. [Phase 1 Details](experiments/01_staleness_features_fixed/) | [Phase 11 Details](experiments/11_staleness_attention/)

Regime-conditional output layers using mixture-of-experts (Phase 5a, 48 experiments) showed that while the router successfully learns regime signals (VIX correlation r=0.83-0.87), all configurations exhibit expert collapse to static per-regime biases. The experts output nearly constant values regardless of encoder input, suggesting that output-layer modifications alone are insufficient without addressing the gradient pathology. [Details](experiments/05a_regime_output/)

Cumulative return targets (Phase 10) produced smoother, lower-noise signals but showed negative excess accuracy. Models learned to predict drift rather than genuine patterns, with single-period returns remaining the optimal target. The implementation tested single-step forecasting with cumret_10 targets; true multi-horizon cumulative (h=1→cumret_1, h=2→cumret_2, etc.) might perform differently with multi-task learning benefits but requires custom Dataset implementation.

Multi-horizon forecasting (Phase 10, 53 experiments) achieved +2.1% excess accuracy in fixed-split evaluation but failed in rolling validation. Daily h=10 with 7q degraded to -3.6pp excess in rolling evaluation, while 3q configuration (reducing output layer capacity) improved to -0.7pp excess but still showed no genuine skill. Longer horizons showed monotonic performance degradation (h5: -2.3pp, h10: -3.6pp, h30: -7.2pp), contradicting the hypothesis that prediction horizon alignment with macro feature update frequency would improve performance. The fixed-split gains resulted from overfitting to the 2020-2025 bull market test period. [Details](experiments/06b_rolling/)

Feature engineering (Phase 12, 50 experiments) demonstrated no impact on performance. Three alternative feature sets (macro_heavy emphasizing monthly macro indicators, market_only using only daily market features, core_dynamics adding technical regime indicators) all achieved 54.2-54.5% directional accuracy in rolling evaluation, matching the 54.49% base rate. Fixed-split gains observed for macro_heavy (55.4% vs 55.7% for core_proposal) disappeared in rolling validation. The core_dynamics feature set additionally suffered from a 10-year training data handicap and 50% collapse rate. Hyperparameter variations (dropout 0.10 vs 0.15, hidden size 14/16/18) produced <0.5pp accuracy differences, confirming feature composition dominates hyperparameter tuning - but all feature sets perform equivalently. [Details](experiments/12_feature_ablation/)

### 4. Architectural Modifications with Mixed Results

Regime-aware attention (Phase 7) showed promising but inconclusive results. Fixed-split evaluation demonstrated +1.5% directional accuracy improvement (59.4% vs 57.9%) with improved Sharpe ratio (1.22 vs 1.05). The learned gates exhibit interpretable behavior, amplifying attention in high-volatility regimes (0.57) and dampening in low-volatility regimes (0.46). However, rolling evaluation across 9 folds showed no improvement over baseline (59.1% for both). Notably, the modification induces dramatic Variable Selection Network adaptation, with VIX weight jumping to 0.74 during the 2022 bear market. While gains didn't generalize robustly, the mechanism caused no performance degradation. [Details](experiments/07_regime_attention/)

Custom loss functions (Phase 4, 63 experiments) produced configuration-dependent effects. Directional diversity penalties rescued weak configurations (+0.33 Sharpe) but hurt already-optimized configurations (-0.27 Sharpe), limiting practical utility. [Details](experiments/04_custom_losses/)

---

## Setup

```bash
# Clone repository
git clone https://github.com/samufrank/sp500-tft-forecasting.git
cd sp500-tft-forecasting

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Critical**: Project requires `pytorch-lightning==1.9.5` and `pytorch-forecasting==0.10.3` due to breaking API changes in Lightning 2.x. These versions are locked in `requirements.txt`.

---

## Repository Structure

```
sp500-tft-forecasting/
├── src/                    # Core modules
│   ├── data_utils.py       # Data loading, preprocessing
│   ├── feature_configs.py  # Feature set definitions and metadata
│   ├── custom_losses.py    # Loss modifications (directional, distribution-aware)
│   ├── regime_attention.py # Regime-conditional attention gating
│   ├── regime_output.py    # Mixture-of-experts output layer
│   ├── staleness_attention.py  # Staleness-aware attention penalty
│   └── classification_tft.py   # Multi-task classification head
│
├── train/                  # Training and evaluation
│   ├── train_tft.py        # Main training script
│   ├── evaluate_tft.py     # Single model evaluation
│   ├── evaluate_checkpoints.py  # Checkpoint comparison
│   ├── rolling_evaluation.py    # Walk-forward validation
│   └── callbacks.py        # Collapse monitoring, checkpointing
│
├── scripts/                # Data collection and analysis
│   ├── collect_data.py     # Fetch data from FRED/yfinance
│   ├── create_splits.py    # Generate train/val/test splits
│   ├── analyze_experiments.py   # Rank and filter experiments
│   ├── summarize_experiments.py # Generate CSV summaries
│   ├── batch_analyze_attention.py  # Extract attention patterns
│   ├── summarize_attention_patterns.py # Attention analysis across experiments
│   ├── compare_phase_attention.py  # A/B test attention between phases
│   ├── analyze_vsn_weights.py      # Variable Selection Network analysis
│   └── [See scripts/README.md for complete reference]
│
├── experiments/            # All experimental results (Phases 0-11)
│   ├── 00_baseline_exploration/  # Hyperparameter characterization (93 experiments)
│   ├── 01_staleness_features_fixed/  # Staleness feature experiments (64 experiments)
│   ├── 02b_vintage_sweep/  # Vintage release alignment (30 experiments)
│   ├── 03_distribution_loss/  # Distribution-aware loss (9 experiments, checkpoint failure)
│   ├── 04_custom_losses/   # Loss function modifications (63 experiments)
│   ├── 05a_regime_output/  # Regime-conditional MoE (48 experiments)
│   ├── 06a_weekly_sweep/   # Weekly frequency experiments (24 experiments)
│   ├── 06b_rolling/        # Rolling window evaluation (27 experiments, 9 folds)
│   ├── 07_regime_attention/  # Regime-aware attention (7 + ablations)
│   ├── 10_quantile_horizon_sweep/  # Quantile and horizon grid (53 experiments)
│   └── 11_staleness_attention/     # Staleness attention penalty
│
├── results/                # Cross-experiment analysis
│   ├── attention_analysis/ # Temporal attention pattern comparisons
│   │   ├── phase00_vs_phase01/  # Baseline vs staleness features
│   │   ├── phase00_vs_phase02b/ # Fixed vs vintage alignment
│   │   └── [Other phase comparisons with READMEs]
│   ├── vsn_analysis/       # Variable Selection Network analysis
│   │   └── phase02b_vs_phase04/ # Feature selection across phases
│   ├── regime_analysis/    # VIX regime statistics
│   └── regime_ablation/    # MoE ablation study results
│
├── data/                   # Datasets and splits
│   ├── financial_dataset_daily_vintage.csv  # Daily data (1990-2025)
│   ├── financial_dataset_weekly_vintage.csv # Weekly resampled
│   ├── financial_dataset_monthly_vintage.csv # Monthly resampled
│   └── splits/             # Train/val/test splits (fixed and vintage alignment)
│       ├── vintage/        # Vintage ALFRED release dates (recommended)
│       └── fixed/          # Fixed month-end alignment
│
└── docs/                   # Development documentation (internal)
```

**Key Documentation**:
- Each `experiments/XX_*/` directory contains a phase-specific README with detailed findings
- `results/attention_analysis/` contains cross-phase attention comparisons with READMEs
- `results/vsn_analysis/` documents Variable Selection Network feature importance
- `scripts/README.md` provides complete analysis workflow reference

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Note: Uses pytorch-lightning==1.9.5 due to pytorch-forecasting compatibility
```

### Reproduce Key Results

```bash
# Generate experiment summary CSV first
python scripts/summarize_experiments.py --phase 06a_weekly_sweep

# Then analyze results
python scripts/analyze_experiments.py --phases 06a_weekly_sweep

# Evaluate specific model
python train/evaluate_tft.py experiments/06a_weekly_sweep/baseline_h16_enc12_d015_bs16

# Compare checkpoint selection strategies
python train/evaluate_checkpoints.py experiments/06a_weekly_sweep/baseline_h16_enc12_d015_bs16
```

### Train Your Own Model

**Weekly TFT** (recommended configuration):
```bash
python train/train_tft.py \
    --experiment-name my_weekly_baseline \
    --frequency weekly \
    --hidden-size 16 \
    --dropout 0.15 \
    --batch-size 16 \
    --max-encoder-length 12 \
    --feature-set core_proposal \
    --alignment vintage
```

**Daily Multi-Horizon**:
```bash
python train/train_tft.py \
    --experiment-name my_multihorizon \
    --frequency daily \
    --max-prediction-length 10 \
    --hidden-size 16 \
    --dropout 0.25 \
    --feature-set core_proposal \
    --alignment vintage
```

**With Custom Loss** (directional diversity penalty):
```bash
python train/train_tft.py \
    --experiment-name my_custom_loss \
    --frequency weekly \
    --hidden-size 16 \
    --dropout 0.25 \
    --directional-weight 0.1 \
    --directional-threshold 0.85
```

**Common Training Arguments**:
- `--experiment-name` (required) - Unique identifier for experiment outputs
- `--feature-set` - Feature configuration: `core_proposal` (default), `macro_heavy`, `market_only`, etc.
- `--frequency` - Data frequency: `daily` (default), `weekly`, `monthly`
- `--alignment` - Release date alignment: `vintage` (ALFRED, recommended) or `fixed` (month-end)
- `--hidden-size` - Model capacity (default: 16; larger models tend to collapse)
- `--attention-heads` - Number of attention heads (default: 2)
- `--dropout` - Dropout rate (default: 0.1; 0.25 recommended for financial data)
- `--max-encoder-length` - Lookback window (default: 20 for daily, 12 for weekly)
- `--max-prediction-length` - Forecast horizon: 1=single-step, >1=multi-horizon (default: 1)
- `--max-epochs` - Training epochs (default: 100)
- `--seed` - Random seed for reproducibility (default: 42)
- `--overwrite` - Allow overwriting existing experiment directory

**Training Outputs** (saved to `experiments/{name}/`):
- `config.json` - Complete hyperparameter configuration
- `final_metrics.json` - Best validation loss and training metadata  
- `checkpoints/` - Model checkpoints (best val_loss, best pred_std, best unique_preds, last)
- `collapse_monitor/` - Training dynamics (prediction diversity, gradient flow, attention entropy)
- `evaluation/` - Test set results (if evaluated)
- `attention_analysis_year/` - Attention patterns (if analyzed)

See `python train/train_tft.py --help` for complete options.

<details>
<summary>Advanced: Rolling evaluation, regime attention, classification head</summary>

**Rolling Window Evaluation**:
```bash
python train/rolling_evaluation.py \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --experiment-prefix my_rolling_test \
    --frequency weekly \
    --feature-set core_proposal \
    --hidden-size 16 \
    --dropout 0.25
```

**Regime-Aware Attention**:
```bash
python train/train_tft.py \
    --experiment-name my_regime_attention \
    --frequency daily \
    --hidden-size 16 \
    --regime-attention \
    --regime-attention-vix-threshold 25.0 \
    --regime-attention-grad-scale 100.0
```

**Multi-Task Classification Head**:
```bash
python train/train_tft.py \
    --experiment-name my_classification \
    --frequency daily \
    --classification \
    --classification-mode regime_volatility \
    --classification-weight 1.0 \
    --regression-weight 1.0
```

</details>

---

## Data Collection & Features

The repository includes pre-collected datasets (1990-2025). To update or extend:

```bash
# Collect data with vintage ALFRED alignment (slower, more accurate)
python scripts/collect_data.py \
    --start-date 2005-01-01 \
    --end-date 2025-10-31 \
    --use-vintage \
    --output-dir data/

# Create train/val/test splits (date-based)
python scripts/create_splits.py \
    --feature-set core_proposal \
    --frequency daily \
    --data-version vintage \
    --train-end 2019-12-31 \
    --val-end 2020-12-31 \
    --output-dir data/splits

# Or use percentage-based splits (70/15/15)
python scripts/create_splits.py \
    --feature-set core_proposal \
    --frequency daily \
    --data-version vintage \
    --train-pct 0.70 \
    --val-pct 0.15 \
    --output-dir data/splits
```

**Data Sources**:
- **Macroeconomic**: FRED API (CPI, unemployment, GDP, etc.)
- **Market**: yfinance (S&P 500, VIX, Treasury yields)
- **Vintage Alignment**: ALFRED database for accurate release dates

### Feature Sets

**Core Proposal** (default, recommended):
1. **Lagged S&P 500 Returns** - Momentum and mean reversion signal
2. **VIX** - Market volatility (daily updates)
3. **10-Year Treasury Yield** - Risk-free rate proxy (daily updates)
4. **Term Spread (10Y-2Y)** - Yield curve slope, recession indicator (daily updates)
5. **CPI Year-over-Year** - Inflation rate (monthly updates, ~14 day release lag)

Other available feature sets (see `src/feature_configs.py`):
- `macro_heavy` - Extended macro indicators (unemployment, GDP, housing)
- `market_only` - VIX, yields, term spread only
- `core_plus_credit` - Core + credit spreads
- `core_dynamics` - Core + momentum/volatility indicators
- `kitchen_sink` - All available features (experimental, may require debugging)

### Staleness Features

For low-frequency variables (CPI, unemployment), staleness indicators are automatically generated:

- `days_since_{feature}_update` - Continuous staleness counter (0, 1, 2, ..., 30)
- `{feature}_is_fresh` - Binary indicator (1 if updated today, 0 otherwise)

**Warning**: Continuous staleness features cause 94% collapse rate (Phase 1 finding). Sparse binary flags avoid collapse but provide no predictive value. Use `--staleness-mode fresh_only` if needed, or omit `--staleness` flag entirely (recommended).

**Note**: Staleness feature generation currently only fully tested with CPI in core_proposal. Use with other feature sets may require code modifications.

### Data Characteristics

- **Daily**: 8,913 observations (1990-2025)
- **Weekly**: Resampled from daily (last observation per week)
- **Monthly**: Resampled from daily (last observation per month)
- **Mixed-frequency alignment**: Low-frequency data forward-filled between releases
- **Vintage alignment**: Macroeconomic data aligned to actual FRED release dates (eliminates look-ahead bias)
- **Train/val/test split**: 70% / 15% / 15% (temporal ordering preserved, no shuffle)

---

## Experimental Journey

### Foundation (Phases 0-2)

Phase 0 characterized optimal hyperparameters through 93 baseline experiments, discovering that hidden size 16 with dropout 0.25 produces the tightest convergence (val_loss 0.392-0.403 across configurations). The experiments also revealed collapse sensitivity to model capacity - larger models consistently fail in financial forecasting contexts. [README](experiments/00_baseline_exploration/)

Phase 2 tested ALFRED vintage release dates versus fixed month-end alignment across 30 experiments. Vintage alignment improved model quality significantly, with 50-62% healthy predictions versus 21-48% for fixed alignment and reduced unidirectional behavior. This came with a modest (~2pp) directional accuracy tradeoff but validated that attention mechanisms adapt appropriately to data characteristics. [README](experiments/02b_vintage_sweep/)

### Failed Interventions (Phases 1, 5)

Phase 1 systematically tested staleness features across 64 experiments. Continuous staleness counters caused a 94% collapse rate, with attention entropy dropping from 2.48 to 1.37 as the mechanism over-constrained itself. Sparse binary flags avoided collapse but provided no signal. The fundamental issue: encoding staleness through input features creates an exploitable monotonic recency gradient. [README](experiments/01_staleness_features_fixed/)

Phase 5a explored regime-conditional output layers through 48 experiments testing learned/VIX routing, 2/3 regimes, and linear/MLP experts. While routers learned regime signals perfectly (r=0.83-0.87), all configurations exhibited expert collapse to static per-regime biases. The experts output nearly constant values regardless of encoder state, demonstrating that output-layer modifications alone cannot address the gradient pathology. [README](experiments/05a_regime_output/)

### Best Configuration Identified (Phases 6, 10)

Phase 6a identified weekly frequency as the best-performing configuration through 24 experiments. The optimal configuration (h16, encoder_length=12, dropout=0.15, batch_size=16) achieved 58.1% directional accuracy on fixed-split and 59.1±8.3% across 9 rolling folds, the highest performance observed across all experiments. Checkpoint selection analysis revealed that validation prediction diversity (`val_pred_std`) produces better test performance than validation loss. [README](experiments/06a_weekly_sweep/)

Phase 6b's rolling evaluation (108 experiments total across 3 baseline configs + Phase 10 multi-horizon + Phase 10 parameter budget testing, 9 folds spanning 2016-2024) revealed that models achieve approximately zero excess accuracy when accounting for fold-specific base rates. Daily models average -1.2pp below their 54.5% base rate; weekly models average -1.1pp below their 60.2% base rate. All models universally fail during the 2022 bear market (~40% accuracy) regardless of architecture or training regime. Multi-horizon forecasting showed monotonic degradation with increasing horizon length, contradicting fixed-split gains. [README](experiments/06b_rolling/)

Phase 10 tested quantile and horizon combinations across 53 experiments on fixed-split data. Multi-horizon forecasting (h=10) achieved +2.1% excess accuracy, but Phase 6b rolling validation revealed these gains don't generalize (-3.6pp excess for 7q, -0.7pp for 3q). Weekly remained optimal at single-step (h=1). Testing 7 quantiles proved marginally best but not critical. Cumulative return targets failed completely despite producing smoother signals. [README](experiments/10_quantile_horizon_sweep/)

### Feature Ablation (Phase 12)

Phase 12 tested alternative feature sets across 50 experiments to validate whether feature engineering could improve performance. Three alternatives (macro_heavy emphasizing monthly macroeconomic indicators, market_only using only daily market features, core_dynamics adding technical regime indicators) were tested across horizons h=1/3/5/10/20 on both fixed-split and rolling evaluation.

Fixed-split results showed macro_heavy competitive with core_proposal (55.4% vs 55.7% at optimal horizons), but rolling validation revealed all feature sets achieve identical performance: 54.2-54.5% directional accuracy, matching the 54.49% rolling base rate. Hyperparameter variations (dropout, hidden size, encoder length) produced <0.5pp differences. The core_dynamics feature set suffered from a 10-year training data handicap and 50% collapse rate, providing no compensating benefit.

Key finding: feature composition doesn't impact out-of-sample performance. All tested feature sets achieve approximately zero excess accuracy in rolling evaluation. Recommendation: use core_proposal as baseline - it's simpler, better-documented, and equivalent in performance. [README](experiments/12_feature_ablation/)

### Architectural Modifications (Phases 4, 7, 11)

Phase 4 explored custom loss functions across 63 experiments. Directional diversity penalties showed configuration-dependent effects - rescuing weak configurations (+0.33 Sharpe) while hurting already-optimized ones (-0.27 Sharpe). This limits practical utility, as the benefit depends on knowing in advance which configurations need rescue. [README](experiments/04_custom_losses/)

Phase 7 implemented regime-aware attention through per-head gating conditioned on VIX regimes (4 learned parameters total). Fixed-split evaluation showed +1.5% directional accuracy improvement and better Sharpe ratios. The gates learned interpretable behavior, amplifying attention in high-volatility and dampening in low-volatility regimes. However, rolling evaluation revealed these gains don't generalize across 9 folds. The most interesting finding was dramatic VSN adaptation - VIX weight jumping to 0.74 during the 2022 bear market as the model learned "when scared, watch VIX almost exclusively." While promising, the modification caused no performance degradation but also no robust improvement. [README](experiments/07_regime_attention/)

Phase 11 tested staleness-aware attention penalties but found the same fundamental issues as Phase 1 staleness features. [README](experiments/11_staleness_attention/)

---

## Analysis Workflows

### Post-Training Analysis

```bash
# Generate CSV summary for a phase
python scripts/summarize_experiments.py --phase 06a_weekly_sweep

# Rank and filter experiments
python scripts/analyze_experiments.py \
    --phases 06a_weekly_sweep \
    --sort-by dir_acc sharpe_ratio \
    --no-collapse \
    --top 10

# Comprehensive snapshot with attention/regime data
python scripts/aggregate_experiments.py \
    --experiments-dir experiments/06a_weekly_sweep
```

### Attention Analysis

```bash
# Extract attention patterns for entire phase
python scripts/batch_analyze_attention.py --phase 06a_weekly_sweep

# Aggregate and detect regime shifts
python scripts/summarize_attention_patterns.py experiments/06a_weekly_sweep/

# Compare two phases (A/B test)
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/06a_weekly_sweep/ \
    --output results/attention_comparison/
```

**Findings**:
- Attention mechanisms detect regime shifts (23/30 experiments show 2022→2023 transition)
- Weak correlation between attention patterns and model quality (|r|<0.20)
- Phase 0 vs Phase 1 comparison: [README](results/attention_analysis/phase00_vs_phase01/)
- Fixed vs Vintage alignment: [README](results/attention_analysis/phase00_vs_phase02b/)
- VSN feature selection: [README](results/vsn_analysis/phase02b_vs_phase04/)

### Checkpoint Comparison

```bash
# Evaluate all checkpoints for an experiment
python train/evaluate_checkpoints.py experiments/06a_weekly_sweep/baseline_h16_enc12

# Aggregate checkpoint strategies across phase
python scripts/aggregate_checkpoints.py experiments/06a_weekly_sweep
```

**Finding**: Checkpoint selection using `val_pred_std` (prediction diversity) produces better test performance than `val_loss` (anti-correlated with directional accuracy).

See `scripts/README.md` for complete analysis workflow reference.

---

## Known Issues & Limitations

### Gradient Collapse

**Symptom**: Models achieve low validation loss while producing constant or near-constant predictions.

**5-Mode Quality Classification**: Models are evaluated with temporal quality classification:

- **HEALTHY**: Predictions vary appropriately with strong directional accuracy (>52%) and positive correlation with actuals
- **DEGRADED**: Predictions vary but show poor quality (directional accuracy <48% or negative correlation with actuals)
- **UNIDIRECTIONAL**: Predictions vary but show extreme directional bias (>98% same sign, exploiting market drift)
- **WEAK_COLLAPSE**: Reduced prediction variation (2/3 structural collapse methods detect issues)
- **STRONG_COLLAPSE**: Near-constant predictions (3/3 structural collapse methods detect issues)

**Detection Methods**:
- **Structural**: Variance threshold, range check, consecutive-similarity analysis
- **Quality**: Rolling 60-day correlation and directional accuracy
- **Unidirectional**: Directional bias >98% threshold

This framework addresses the multi-dimensional nature of collapse in financial forecasting, where models can pass variance checks while still being fundamentally broken (anti-correlated predictions, unidirectional bias).

Monitoring is handled by a custom callback in `train/collapse_monitor.py` that tracks gradient flow, attention entropy, prediction diversity, and VSN activity during training.

Mitigation strategies include temporal aggregation (weekly frequency or multi-horizon forecasting), small model capacity (h16-18; larger models collapse universally), and checkpoint selection based on prediction diversity (`val_pred_std`) rather than validation loss.

### Universal 2022 Bear Market failure

All models (~40% accuracy) fail during 2022 Federal Reserve tightening period, regardless of:
- Architecture modifications
- Checkpoint selection strategy  
- Training data regime composition

**Hypothesis**: Fundamental regime mismatch - models cannot generalize to unprecedented monetary policy shock.

### Quantile Loss Anti-Correlation

Validation loss (quantile loss) weakly anti-correlated with directional accuracy (r=-0.46). Models optimizing quantile loss may learn to predict small positive values consistently (exploiting market drift) rather than genuine temporal patterns.

**Implication**: Financial forecasting requires domain-specific validation metrics; standard ML evaluation is insufficient.

**Financial forecasting context**: R² near zero is expected and not indicative of model failure. Daily S&P 500 returns have an extremely low signal-to-noise ratio (~0.03:1). Directional accuracy of 52-54% represents strong performance - random walk baseline is ~50%, and top quantitative hedge funds operate at 52-55%. Sharpe ratio and cumulative returns are the primary indicators of model quality.

---

## Technical Details

### Model Architecture

- **Base**: Temporal Fusion Transformer (pytorch-forecasting v0.10.3)
- **Key Components**:
  - Variable Selection Networks (VSN) for feature importance
  - Multi-head temporal attention (encoder/decoder)
  - Gated Residual Networks (GRN) for feature processing
  - Quantile regression for uncertainty estimation

### Evaluation Metrics

Models are evaluated across multiple dimensions:

**Statistical Metrics**:
- MSE, RMSE, MAE - Standard regression metrics
- R² - Coefficient of determination (near-zero expected for daily returns due to high noise)

**Classification Metrics**:
- Directional accuracy - Fraction of correct up/down predictions
- AUC-ROC - Area under receiver operating characteristic curve
- Precision/Recall - Positive class performance
- F1 Score - Harmonic mean of precision and recall

**Financial Metrics**:
- Sharpe ratio - Risk-adjusted returns (primary financial metric)
- Total return - Cumulative performance
- Max drawdown - Largest peak-to-trough decline
- Alpha - Excess return vs. buy-and-hold baseline

**Quality Metrics**:
- 5-mode temporal quality classification (HEALTHY/DEGRADED/UNIDIRECTIONAL/WEAK_COLLAPSE/STRONG_COLLAPSE)
- Prediction diversity (standard deviation, range, unique values)
- Temporal consistency (rolling correlations, autocorrelation)

### Dependencies

Key constraint: `pytorch-forecasting==0.10.3` requires `pytorch-lightning==1.9.5`, limiting access to modern PyTorch 2.x features.

---

## Future Work

The comprehensive experimental exploration documented here identifies several promising research directions.

**Near-term directions** (highest confidence):

Regime classification reformulates the task from regression to classification. The classification head diagnostic (Phase 10/11) demonstrated that the encoder achieves 100% VIX regime detection accuracy while directional prediction remains at base rate. Binary classification (bull/bear) or multi-class classification (bull/bear/volatile/sideways) could test whether the encoder's regime detection capability translates to classification performance above random baselines. This requires minimal architectural changes and uses existing infrastructure.

**Medium-term exploration**:

Multi-task learning with constituent returns represents unexplored territory with high potential. Instead of predicting only S&P 500 returns, the model would simultaneously predict individual constituent returns alongside the index return. With 51 prediction targets (50 constituents + index) versus the current single target, this provides 51× richer gradient signal and may prevent collapse through more diverse optimization landscape. Data for the top 50 constituents covering 2005-2025 has been collected. Implementation requires modifying the output layer and evaluation framework but uses the same TFT architecture.

Alternative asset classes could validate whether findings generalize beyond S&P 500. Testing on individual stocks (higher volatility), sector ETFs, VIX (volatility forecasting), or other indices would determine whether the observed limitations are specific to index-level equity prediction or represent fundamental TFT characteristics on financial data.

**Long-term research**:

Alternative information sources may provide directional signal absent in current features. News sentiment scores, options market data beyond VIX term structure, or cross-asset correlation signals represent fundamentally different information types than the market-level macro indicators tested across Phases 0-12. This would require new data collection infrastructure.

Regime-conditional models train separate models for each regime rather than using a unified architecture. Since models detect regimes perfectly but cannot predict direction overall, training dedicated bull-market and bear-market models (or evaluating a single model's performance separately by regime) could test whether predictive skill exists conditionally even when unconditional prediction fails.

The comprehensive negative results documented here provide valuable guidance for future work, eliminating approaches that don't generalize and identifying the most promising unexplored directions.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{frank2024sp500tft,
  author = {Frank, Sam},
  title = {S\&P 500 Forecasting with Temporal Fusion Transformers: Gradient Collapse and Temporal Aggregation Solutions},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/samufrank/sp500-tft-forecasting}
}
```

---

## Related Work

- **Lim et al. (2021)**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **pytorch-forecasting**: https://pytorch-forecasting.readthedocs.io/

---

## License

MIT License - See LICENSE file for details.

---