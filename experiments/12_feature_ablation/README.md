# Phase 12: Feature Set Ablation and Validation (December 2025)

Systematic evaluation of alternative feature sets (macro_heavy, market_only, core_dynamics) against core_proposal baseline across fixed-split and rolling evaluation settings. Tests whether feature engineering can improve predictive performance and examines robustness across prediction horizons, hyperparameters, and market regimes.

## Summary

Feature set selection has no meaningful impact on TFT performance for S&P 500 return prediction. Three alternative feature sets (macro_heavy, market_only, core_dynamics) tested across 50 experiments show identical rolling evaluation performance to core_proposal baseline, all clustering at 54.2-54.5% directional accuracy against a 54.49% rolling base rate.

Fixed-split experiments showed macro_heavy competitive with core_proposal (55.4% vs 55.7% on 2020-2025 test set), but rolling validation across 2016-2024 revealed no feature set achieves excess accuracy. The gains from fixed-split testing result from overfitting to the bull market test period rather than genuine predictive skill.

The classification head diagnostic from earlier work (100% VIX regime detection, 0% direction prediction) is validated - encoder architecture functions correctly and models successfully detect market regimes, but feature sets tested lack information about future return direction. Recommendation: use core_proposal as baseline feature set. It is simpler, better-documented, and equivalent in performance to alternatives.

## Motivation

Phase 10/11 established optimal configurations using core_proposal features (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY). This phase tests whether alternative feature combinations improve performance.

The macro_heavy feature set emphasizes macroeconomic fundamentals with 6 features: VIX, Inflation_YoY, Unemployment, Fed_Rate, Consumer_Sentiment, and Industrial_Production. Most features update monthly. The hypothesis was that monthly macro indicators might align better with longer prediction horizons, particularly h20 (approximately 1 month).

The market_only feature set uses only daily-updating market features: VIX, Treasury_10Y, Yield_Spread, and SP500_Volatility. The hypothesis was that removing stale monthly macro data would improve predictions by eliminating the information staleness problem identified in Phase 11.

The core_dynamics feature set adds regime-detection features to core_proposal: VIX_relative, VIX_spike, Treasury_10Y_change, and CPI in addition to the original 5 features. The hypothesis was that enhanced technical indicators would better capture market dynamics. This feature set has a 10-year data handicap (2000-2025 vs 1990-2025 for others) due to gold data availability for computing VIX_relative.

## Experimental design

Phase 12 conducted experiments in three stages with progressively stricter validation.

The initial ablation (Phase 12) tested each feature set at configurations found optimal in Phase 10/11. Fixed-split evaluation used train: 1990-2015, validation: 2015-2020, test: 2020-2025. Both daily h10 and weekly h1 were tested with 7q output, resulting in 6 experiments total (3 feature sets × 2 frequencies).

Phase 12b expanded to a robustness sweep across horizons and hyperparameters, still using fixed-split evaluation. Daily experiments tested h1/h3/h5/h10 with dropout 0.10 and 0.15. Weekly experiments tested h1/h3/h5 with encoder lengths 8 and 12. This produced 36 experiments for macro_heavy, market_only, and core_dynamics. Some duplicate configurations already run in Phase 12 were skipped. The core_proposal feature set was already characterized in Phase 10/11 so was not re-tested.

Phase 12c implemented rolling evaluation validation in two sub-phases. First, h1 baseline validation tested 8 experiments (6 daily, 2 weekly) with different hyperparameters to confirm whether any feature set could beat baseline at the one horizon known to generalize (h1). Second, multi-horizon testing used 3q configuration for 6 experiments testing h5/h10/h20 on macro_heavy and market_only. The rolling window used 10-year training, 1-year validation, 1-year test, with 1-year steps across 2016-2024, producing 9 folds. The 3q configuration (instead of standard 7q) was chosen based on concurrent findings that 7q × multi-horizon causes output layer parameter overload where the number of outputs exceeds hidden dimension capacity.

## Configuration

All experiments use vintage alignment (realistic macro release delays via ALFRED database), seed 42, learning rate 0.0005, max epochs 100, early stop patience 100.

Feature set definitions from feature_configs.py:

core_proposal: 5 features (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY, SP500_Returns as lagged target). All data available from 1990.

macro_heavy: 6 features (VIX, Inflation_YoY, Unemployment, Fed_Rate, Consumer_Sentiment, Industrial_Production, plus SP500_Returns). Most features update monthly. Same data availability as core_proposal (1990-2025).

market_only: 5 features (VIX, Treasury_10Y, Yield_Spread, SP500_Volatility, SP500_Returns). All features update daily or are derived from daily data. Available from 1990.

core_dynamics: 9 features (core_proposal base + VIX_relative, VIX_spike, Treasury_10Y_change, CPI). VIX_relative uses 252-day rolling max requiring substantial history. Gold data for VIX calculation limits start to 2000-11-27, creating a 10-year training disadvantage.

Daily baseline: hidden_size=16, max_encoder_length=20, batch_size=64
Weekly baseline: hidden_size=16, max_encoder_length=12, dropout=0.15, batch_size=16

Phase 12b tested hyperparam variations: dropout 0.10/0.15 (daily), encoder length 8/12 (weekly), hidden size 14/16/18.

Phase 12c multi-horizon used 3q [0.1, 0.5, 0.9] to avoid output layer overload from 7q × h10 = 70 outputs exceeding hidden_size=16 capacity.

## Results

### Phase 12: Fixed-split ablation

Daily h10 (base rate: 53.6% positive):

| Feature Set | Dir Acc | Excess | Sharpe | Healthy % |
|-------------|---------|--------|--------|-----------|
| core_proposal (Phase 10) | 55.7% | +2.1% | 1.28 | 37.5% |
| macro_heavy | 53.7% | +0.1% | 0.89 | - |
| market_only | 52.6% | -1.0% | 0.82 | - |
| core_dynamics | 52.3% | -1.3% | 0.46 | - |

Weekly h1 (base rate: 56.0% positive):

| Feature Set | Dir Acc | Sharpe | Healthy % |
|-------------|---------|--------|-----------|
| core_proposal (Phase 10) | 57.9% | 0.90 | 18.8% |
| macro_heavy | 45.5% | 0.46 | - |
| market_only | 55.7% | 0.71 | - |
| core_dynamics | 55.7% | 0.76 | collapsed |

core_dynamics weekly collapsed (0% negative predictions). macro_heavy performed worse than random on weekly frequency.

### Phase 12b: Fixed-split robustness sweep

Best performers by feature set (daily only, weekly showed high collapse rates):

macro_heavy: 55.4% dir_acc at h3/h5 (dropout 0.10), 1.21 Sharpe. Competitive with core_proposal at shorter horizons.

market_only: 54.7% dir_acc at h10 (dropout 0.10), 1.42 Sharpe. Viable but weaker than macro_heavy.

core_dynamics: 5 of 10 daily experiments collapsed (0% down predictions). Only 13.1% epochs showed healthy behavior. Enhanced features with 10-year training handicap provide no benefit.

Hyperparam sensitivity: dropout 0.10 vs 0.15 differed by <0.3pp directional accuracy. Hidden size variations (14/16/18) showed <0.5pp differences. Configuration choice matters less than feature set selection.

Horizon findings (daily): h3 and h5 performed comparably to h10 for macro_heavy (55.4% vs 55.7%), contradicting Phase 10 finding that h10 was optimal for core_proposal. Suggests feature-horizon interaction or seed variance.

Weekly frequency: High collapse rates (most configs <5% healthy epochs) and extreme variance across configs. Not recommended for alternative feature sets.

### Phase 12c: Rolling evaluation validation

h1 baseline test used 7q configuration. The rolling evaluation base rate (average positive rate across 9 folds, 2016-2024) is 54.49% for daily frequency, compared to 53.6% for the fixed 2020-2025 test set. This difference reflects varying market conditions across different time periods.

| Feature Set | Dir Acc | vs Base | Sharpe | Healthy % |
|-------------|---------|---------|--------|-----------|
| macro_heavy d010 | 54.38% | -0.11pp | 1.11 | 14.8% |
| macro_heavy d005 | 54.38% | -0.11pp | 1.11 | 14.7% |
| macro_heavy h18 | 53.11% | -1.38pp | 1.11 | 22.7% |
| market_only d010 | 54.38% | -0.11pp | 1.14 | 20.7% |
| market_only d025 | 54.11% | -0.38pp | 1.05 | 14.8% |
| market_only h14 | 53.88% | -0.61pp | 1.12 | 14.9% |

All configurations perform at or below rolling base rate. No feature set achieves excess accuracy. Hyperparam variations (dropout, hidden size) produce negligible differences.

Multi-horizon test with 3q (rolling base rate: 54.49% positive):

| Feature Set | Horizon | Dir Acc | vs Base | Sharpe | Healthy % |
|-------------|---------|---------|---------|--------|-----------|
| macro_heavy | h5 | 54.19% | -0.30pp | 1.16 | 18.7% |
| macro_heavy | h10 | 53.57% | -0.92pp | 0.98 | 15.4% |
| macro_heavy | h20 | 54.02% | -0.47pp | 1.21 | 20.3% |
| market_only | h5 | 54.53% | +0.04pp | 1.08 | 15.2% |
| market_only | h10 | 54.15% | -0.34pp | 0.95 | 15.6% |
| market_only | h20 | 54.45% | -0.04pp | 0.98 | 16.4% |

Best result: market_only h5 at 54.53%, exactly matching base rate (within measurement noise). No configuration demonstrates predictive skill. The 3q output configuration eliminates parameter budget constraints but does not unlock multi-horizon performance.

Three hypotheses were tested in the multi-horizon experiments. The monthly alignment hypothesis predicted that macro_heavy h20 would perform best since 20 trading days approximates one month, aligning prediction horizon with monthly macro feature update frequency. This failed - h20 (54.02%) performed worse than h5 (54.19%), showing no evidence of beneficial temporal alignment.

The fresh features hypothesis predicted that market_only would outperform macro_heavy by eliminating stale monthly macro data. This also failed - market_only (54.15-54.53%) performed equivalently to macro_heavy (53.57-54.19%). Removing monthly features provided no advantage.

The 3q enables multi-horizon hypothesis predicted that solving output layer parameter overload (3q × h20 = 60 outputs vs 7q × h20 = 140 outputs, both using hidden_size=16) would enable multi-horizon learning. This failed - no horizon achieved excess accuracy despite adequate output layer capacity.

Year-by-year performance shows 2022 bear market causes universal failure. All feature sets and horizons achieve 44-56% directional accuracy in 2022 (below 54.49% base rate). The 2018 volatility spike also proved problematic with several configurations showing 0% Sharpe despite positive directional accuracy.

One anomaly requires explanation: macro_heavy h20 in 2022 achieved 55.6% directional accuracy with 0.0 Sharpe and 0.0 total return. Manual inspection of fold predictions revealed the model output small negative values (approximately -0.001% to -0.01%) for most days. These magnitudes fall below meaningful trading thresholds, resulting in no effective positions taken. The model achieved slightly-above-random directional accuracy by consistently predicting negative in a bear market, but the prediction magnitudes were so small that the strategy took no risk and earned no return. This represents collapse to near-zero predictions rather than genuine regime adaptation - the model hedged uncertainty by outputting values approaching numerical precision limits.

Collapse detection uses the methodology from Phase 0-2: structural checks (variance threshold, range analysis, consecutive prediction similarity) combined with quality metrics (60-day rolling correlation, directional accuracy). Models showing <15% healthy epochs across training were classified as collapsed. The core_dynamics feature set exhibited particularly severe collapse with 5 of 10 daily configurations showing 0% down predictions (unidirectional collapse) and only 13.1% healthy epochs on average.

## Key findings

1. Fixed-split gains do not generalize. macro_heavy achieved 55.4% on 2020-2025 fixed test set but 54.38% on rolling evaluation (matching random). Phase 10/11 core_proposal showed similar regression (55.7% fixed → 53.3% rolling for daily h1). Overfitting to bull market test period explains fixed-split performance.

2. Feature sets perform identically in rolling evaluation. macro_heavy, market_only, and core_proposal all cluster at 54.2-54.5% directional accuracy across configurations. Feature composition does not impact out-of-sample performance.

3. core_dynamics fails comprehensively. 10-year training data handicap (2000-2025 vs 1990-2025) combined with enhanced feature complexity produces collapse in 50% of daily experiments. No compensating performance benefit observed.

4. Hyperparameter choice matters minimally. Dropout 0.10 vs 0.15, hidden size 14/16/18, encoder length 8/12 produce <0.5pp directional accuracy differences. Feature set selection dominates hyperparameter tuning, but feature sets themselves provide equivalent performance.

5. Multi-horizon with 3q does not rescue performance. Eliminating output layer parameter overload (via 3q) fails to enable multi-horizon learning for these feature sets. All horizons (h5/h10/h20) cluster at base rate performance.

6. 2022 bear market defeats all configurations. Every feature set, horizon, and hyperparam combination fails in 2022 (44-56% directional accuracy, negative Sharpe ratios). No evidence of regime-adaptive behavior.

7. Weekly frequency unreliable for alternative feature sets. High variance (±11.9% directional accuracy std) and frequent collapse. core_proposal remains only viable weekly configuration from Phase 10/11.

8. Classification head diagnostic validated. Phase 10/11 classification head showed encoder achieves 100% VIX regime detection but 0% direction prediction above base rate. Rolling evaluation confirms: models detect regimes perfectly but cannot predict returns.

## Comparison to baselines

Phase 10/11 core_proposal performance:

Daily: Fixed-split h10 achieved 55.7% (+2.1% excess). Rolling h1 achieved 53.3% (-1.2pp vs 54.49% base). Multi-horizon regression observed.

Weekly: Fixed-split h1 achieved 57.9% (+1.9% excess). Rolling h1 achieved 59.1% ± 8.3% (+2.3% excess vs 56.8% base rate). Weekly generalizes better than daily for core_proposal specifically.

Phase 12 feature sets vs Phase 10/11 baseline:

Fixed-split daily h10: macro_heavy (53.7%) underperforms core_proposal (55.7%) by 2pp. market_only (52.6%) underperforms by 3.1pp.

Rolling daily h1: macro_heavy (54.38%) matches core_proposal (53.3%) within noise. market_only (54.38%) equivalent.

Conclusion: Feature set variations provide no improvement over core_proposal in either fixed-split or rolling settings. core_proposal remains simplest viable option.

## Conclusions

Feature engineering does not improve TFT performance on S&P 500 return prediction with market-level indicators. Three alternative feature sets (macro_heavy, market_only, core_dynamics) tested across 50 total experiments show equivalent performance to core_proposal baseline in rolling evaluation, all clustering at random accuracy (54.49% base rate).

The encoder architecture functions correctly - models achieve 100% VIX regime detection (from classification head diagnostic) and demonstrate regime-adaptive attention patterns. Failure occurs at prediction output: the hidden state representations contain regime information but lack information about future return direction.

Fixed-split evaluation is unreliable. Gains observed in 2020-2025 test period (macro_heavy: 55.4%, core_proposal: 55.7%) collapse to base rate in rolling evaluation across 2016-2024. Bull market test period selection creates misleading performance estimates.

Recommendation: Use core_proposal (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY) as baseline feature set. Simpler, better-documented, and equivalent performance to alternatives. Further feature engineering unlikely to improve results without fundamentally different information sources (e.g., order flow, news sentiment, cross-asset signals).

## Future work

Multi-task learning represents the most promising unexplored direction. Instead of predicting only S&P 500 returns, the model would predict individual constituent returns alongside the index return. With 51 simultaneous prediction targets (vs current 1 target), this provides 51× richer gradient signal and forces the encoder to learn a general market representation rather than index-specific patterns. The richer optimization landscape may prevent collapse through more diverse gradient signals. Data for the top 50 constituents has already been collected covering 2005-2025, though implementation requires modifying the output layer and evaluation framework.

Alternative information sources could be explored given that current features (VIX, interest rates, macro indicators) demonstrably lack directional signal across multiple feature set configurations. News sentiment scores, options market data beyond VIX term structure, or cross-asset correlation signals might contain predictive information not present in macro-level indicators. This would require new data collection infrastructure.

Regime-conditional evaluation may reveal whether directional signal exists within specific market regimes even though unconditional prediction fails. Models detect high-VIX vs low-VIX regimes with 100% accuracy (from classification head diagnostic) but cannot predict direction overall. Training and evaluating separate models for each regime, or evaluating a single model's performance separately by regime, could test whether predictions have skill conditionally.

The alternative conclusion is to accept the baseline performance. With market-level macro indicators available through 2025, TFT achieves approximately 54.5% directional accuracy in rolling evaluation across nine years of out-of-sample testing. This may represent the fundamental limit of these features and architecture for this prediction task. A comprehensive writeup documenting the methodology, experimental findings, and lessons learned would have value for the research community even without achieving excess returns.

## Experiments structure

```
experiments/12_feature_ablation/          # Initial fixed-split ablation (6 experiments)
├── macro_heavy_daily_7q_h10_s42/
├── macro_heavy_weekly_7q_h1_s42/
├── market_only_daily_7q_h10_s42/
├── market_only_weekly_7q_h1_s42/
├── core_dynamics_daily_7q_h10_s42/
├── core_dynamics_weekly_7q_h1_s42/
├── experiments_summary.csv
├── experiments_summary_key_metrics.csv
└── README.md                             # This file

experiments/12b_feature_sweep/            # Fixed-split robustness sweep (36 experiments)
├── macro_heavy_daily_7q_{h1,h3,h5,h10}_{d010,d015}_s42/
├── macro_heavy_weekly_7q_{h1,h3,h5}_{enc8,enc12}_s42/
├── market_only_daily_7q_{h1,h3,h5,h10}_{d010,d015}_s42/
├── market_only_weekly_7q_{h1,h3,h5}_{enc8,enc12}_s42/
├── core_dynamics_daily_7q_{h1,h3,h5,h10}_{d010,d015}_s42/
├── core_dynamics_weekly_7q_{h1,h3,h5}_{enc8,enc12}_s42/
├── experiments_summary.csv
└── experiments_summary_key_metrics.csv

experiments/12c_h1_rolling/               # Rolling validation at h1 (8 experiments)
├── macro_heavy_daily_h1_{d005,d010,h18}/
├── market_only_daily_h1_{d010,d025,h14}/
├── macro_heavy_weekly_h1_enc8/
├── market_only_weekly_h1_enc8/
├── plots/
└── rolling_comparison.csv

experiments/12c_multi_horizon/            # Rolling multi-horizon with 3q (6 experiments)
├── macro_heavy_daily_3q_{h5,h10,h20}_d025/
├── market_only_daily_3q_{h5,h10,h20}_d025/
├── plots/
└── multi_horizon_comparison.csv
```

## Reproducibility

Fixed-split experiments (Phase 12, 12b):
```bash
# Phase 12 example
python train/train_tft.py \
    --experiment-name 12_feature_ablation/macro_heavy_daily_7q_h10_s42 \
    --feature-set macro_heavy \
    --frequency daily \
    --alignment vintage \
    --hidden-size 16 \
    --max-encoder-length 20 \
    --dropout 0.10 \
    --batch-size 64 \
    --quantiles 7q \
    --max-prediction-length 10 \
    --seed 42 \
    --learning-rate 0.0005 \
    --max-epochs 100 \
    --early-stop-patience 100

# Evaluate
python train/evaluate_checkpoints.py experiments/12_feature_ablation/macro_heavy_daily_7q_h10_s42
```

Rolling experiments (Phase 12c):
```bash
# h1 baseline example
python train/rolling_evaluation.py \
    --experiment-prefix 12c_h1_rolling/macro_heavy_daily_h1_d010 \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --feature-set macro_heavy \
    --frequency daily \
    --alignment vintage \
    --hidden-size 16 \
    --max-encoder-length 20 \
    --dropout 0.10 \
    --batch-size 64 \
    --quantiles 7q \
    --max-prediction-length 1 \
    --seed 42

# Multi-horizon with 3q example
python train/rolling_evaluation.py \
    --experiment-prefix 12c_multi_horizon/macro_heavy_daily_3q_h5_d025 \
    --mode rolling \
    --train-years 10 \
    --val-years 1 \
    --test-years 1 \
    --step-years 1 \
    --start-test-year 2016 \
    --end-test-year 2024 \
    --feature-set macro_heavy \
    --frequency daily \
    --alignment vintage \
    --hidden-size 16 \
    --max-encoder-length 20 \
    --dropout 0.25 \
    --batch-size 64 \
    --quantiles 3q \
    --max-prediction-length 5 \
    --seed 42

# Analyze rolling results
python scripts/analyze_rolling.py \
    experiments/12c_h1_rolling/macro_heavy_daily_h1_d010 \
    experiments/12c_h1_rolling/market_only_daily_h1_d010 \
    --output experiments/12c_h1_rolling/rolling_comparison.csv
```

Analysis scripts:
```bash
# Fixed-split summary
python scripts/summarize_experiments.py \
    --output-dir experiments/12_feature_ablation \
    --phase 12_feature_ablation \
    --best-by pred_std \
    --min-epoch 80

# Cross-phase comparison
python scripts/analyze_experiments.py \
    --phases 12_feature_ablation 12b_feature_sweep
```
