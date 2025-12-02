# Phase 06b: Rolling Window Evaluation (Nov 29, 2024)

Rolling window evaluation across 9 years (2016-2024) to test model robustness across different market regimes. Unlike fixed train/val/test splits that only evaluate on recent data, rolling evaluation reveals how models perform during volatility spikes, crashes, bear markets, and recoveries.

## Why rolling evaluation

Fixed splits (70/15/15) test on a single regime—typically recent bull market conditions. This can mask:
- Overfitting to recent market dynamics
- Failure during regime changes
- Sensitivity to specific market conditions

Rolling evaluation provides:
- Mean ± std across regimes (not just point estimates)
- Explicit testing on 2018 vol spike, 2020 COVID crash, 2022 bear market
- More rigorous robustness assessment
- Standard practice in financial ML literature

## Rolling window configuration

- Train: 10 years
- Validation: 1 year
- Test: 1 year
- Step: 1 year
- Folds: 9 (test years 2016-2024)

Example fold structure:
- Fold 2020: Train 2009-2018, Val 2019, Test 2020
- Fold 2022: Train 2011-2020, Val 2021, Test 2022

## Models evaluated

| Config | Frequency | Hidden | Encoder | Dropout | Batch |
|--------|-----------|--------|---------|---------|-------|
| daily_h16_baseline | daily | 16 | 20 | 0.10 | 64 |
| weekly_h16_enc8_d025_bs32 | weekly | 16 | 8 | 0.25 | 32 |
| weekly_h16_enc12_d015_bs16 | weekly | 16 | 12 | 0.15 | 16 |

Daily config from Phase 02b best baseline. Weekly configs from Phase 06a top performers.

## Summary results

| Config | Dir Acc | Sharpe | Healthy % | Total Return |
|--------|---------|--------|-----------|--------------|
| Daily | 53.3% ± 5.2% | 0.95 ± 1.31 | 17.6% ± 13.9% | 11.0% ± 12.7% |
| Weekly enc8 | 58.5% ± 7.6% | 1.22 ± 1.16 | 11.6% ± 0.8% | 10.5% ± 12.6% |
| Weekly enc12 | 59.1% ± 8.3% | 1.52 ± 1.33 | 13.0% ± 1.0% | 15.1% ± 20.9% |

## Key findings

**Weekly outperforms daily on accuracy:**
- Weekly enc12: 59.1% vs Daily: 53.3% (+5.8pp)
- Weekly enc8: 58.5% vs Daily: 53.3% (+5.2pp)

**Excess accuracy over baseline (critical metric):**
- Daily: 53.3% accuracy vs 53.9% positive rate = **-0.6% excess** (not learning)
- Weekly: 59.1% accuracy vs 56.8% positive rate = **+2.3% excess** (genuine signal)

Daily models essentially predict "always positive" and get credit for market drift. Weekly models extract actual predictive signal beyond naive baseline.

**Weekly healthy % is more stable:**
- Daily: 17.6% ± 13.9% (huge variance, 54.8% outlier in 2024)
- Weekly enc8: 11.6% ± 0.8% (consistent)
- Weekly enc12: 13.0% ± 1.0% (consistent)

**Sharpe ratio:**
- Weekly enc12 best: 1.52 ± 1.33
- Weekly enc8: 1.22 ± 1.16
- Daily: 0.95 ± 1.31

## Performance by regime

### Daily baseline
| Year | Regime | Dir Acc | Sharpe | Return |
|------|--------|---------|--------|--------|
| 2016 | Post-election rally | 52.2% | 1.36 | 15.1% |
| 2017 | Low vol bull | 58.3% | 2.67 | 17.5% |
| 2018 | Vol spike | 48.0% | -1.42 | -1.1% |
| 2019 | Recovery | 59.1% | 1.83 | 20.9% |
| 2020 | COVID crash/recovery | 56.7% | 0.52 | 11.7% |
| 2021 | Meme stocks/inflation | 56.5% | 2.14 | 27.1% |
| 2022 | Bear market | **43.7%** | **-0.61** | **-15.1%** |
| 2023 | AI rally | 54.6% | 1.37 | 16.3% |
| 2024 | Continued rally | 50.4% | 0.71 | 6.8% |

### Weekly enc12 (best)
| Year | Regime | Dir Acc | Sharpe | Return |
|------|--------|---------|--------|--------|
| 2016 | Post-election rally | 53.7% | 1.19 | 9.2% |
| 2017 | Low vol bull | 67.5% | 3.23 | 14.1% |
| 2018 | Vol spike | 57.5% | -0.25 | -4.0% |
| 2019 | Recovery | 62.5% | 1.87 | 15.7% |
| 2020 | COVID crash/recovery | 60.0% | 2.62 | **60.7%** |
| 2021 | Meme stocks/inflation | 65.9% | 2.49 | 21.8% |
| 2022 | Bear market | **40.0%** | **-0.82** | **-15.5%** |
| 2023 | AI rally | 62.5% | 1.95 | 20.1% |
| 2024 | Continued rally | 62.5% | 1.36 | 14.1% |

## Regime analysis

**2022 bear market failure:**
All models fail dramatically in 2022:
- Daily: 43.7% dir_acc, -0.61 Sharpe
- Weekly enc8: 40.9% dir_acc, -0.55 Sharpe
- Weekly enc12: 40.0% dir_acc, -0.82 Sharpe

This is expected—models trained on predominantly bullish data struggle when regime shifts to sustained bearish conditions. The failure is honest and reveals fundamental limitations.

Checkpoint selection does not fix this: evaluating all checkpoint metrics (val_loss, val_dir_acc, val_sharpe, val_pred_std) on the 2022 fold produces identical results (40% dir_acc, 0 negative predictions). The model predicts "always positive" regardless of checkpoint—regime mismatch cannot be fixed by checkpoint selection.

**2020 COVID recovery:**
Weekly models handle 2020 better:
- Weekly enc12: 60.0% dir_acc, 2.62 Sharpe, 60.7% return
- Daily: 56.7% dir_acc, 0.52 Sharpe, 11.7% return

Weekly captures the V-shaped recovery; daily gets whipsawed by daily volatility.

**Stable regimes (2017, 2021, 2023):**
Both frequencies perform well in trending markets, but weekly consistently outperforms by 5-10pp on directional accuracy.

## Conclusions

1. **Weekly frequency is superior for this task**: Higher accuracy, better Sharpe, more consistent behavior across regimes.

2. **Daily models don't learn signal**: -0.6% excess accuracy means daily predictions are essentially noise around market drift.

3. **Weekly models extract genuine signal**: +2.3% excess accuracy indicates actual predictive power beyond naive baseline.

4. **All models fail in bear markets**: 2022 exposes fundamental limitations. Models cannot predict regime reversals.

5. **Rolling evaluation should be primary presentation**: Reports honest mean ± std, reveals regime-specific failures, standard in financial ML.

## Recommended reporting for paper etc.

Use rolling evaluation results as primary metrics:
- Weekly enc12: 59.1% ± 8.3% dir_acc, 1.52 ± 1.33 Sharpe
- Compare to daily: 53.3% ± 5.2% dir_acc, 0.95 ± 1.31 Sharpe
- Emphasize excess over baseline: +2.3% (weekly) vs -0.6% (daily)
- Acknowledge 2022 failure explicitly (scientific honesty)

## Data

Rolling splits generated per fold in `data/splits/rolling/fold_XXXX/`:
- Daily: ~2,500 train samples, ~250 val, ~250 test per fold
- Weekly: ~520 train samples, ~52 val, ~52 test per fold

## Experiments structure

Total: 27 experiments (3 configs × 9 folds)

Directory structure:
```
experiments/06b_rolling/
├── daily_h16_baseline/
│   ├── fold_2016/
│   ├── fold_2017/
│   ├── ...
│   ├── fold_2024/
│   └── rolling_results_full.csv
├── weekly_h16_enc8_d025_bs32/
│   └── ...
└── weekly_h16_enc12_d015_bs16/
    └── ...
```

Each fold contains full experiment output (checkpoints, evaluation, logs).

## Reproducing results

Run rolling evaluation for a new config:
```bash
python train/rolling_evaluation.py \
    --experiment-prefix 06b_rolling/my_config \
    --frequency weekly \
    --mode rolling \
    --train-years 10 \
    --hidden-size 16 \
    --max-encoder-length 8 \
    --dropout 0.25 \
    --batch-size 32
```

Analyze results:
```bash
python scripts/analyze_rolling.py \
    experiments/06b_rolling/daily_h16_baseline \
    experiments/06b_rolling/weekly_h16_enc8_d025_bs32 \
    experiments/06b_rolling/weekly_h16_enc12_d015_bs16 \
    --output experiments/06b_rolling/comparison.csv
```

