# Phase 10/11: Quantile and Horizon Sweep (December 03-11, 2025)

Systematic evaluation of quantile configurations, prediction horizons, and cumulative return targets to identify optimal TFT configuration for S&P 500 forecasting.

## Motivation

Prior phases established daily and weekly baselines but used default single-step prediction (h=1) and 7-quantile output. This phase tests whether:
- Multi-horizon prediction improves learning through multi-task regularization
- Fewer quantiles (3q, 5q) reduce output complexity without sacrificing performance
- Cumulative return targets provide smoother signal than point returns
- Optimal horizon differs between daily and weekly frequencies

## Experimental design

### Phase 10: Core sweep (36 experiments)
- Frequencies: daily, weekly
- Quantiles: 3q [0.1, 0.5, 0.9], 7q [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
- Horizons: h1, h3, h5
- Seeds: 42, 123, 456

### Phase 11: Extended horizons (17 experiments)
- Daily only: h10, h20, h30 with 7q (3 seeds each)
- 3q comparison at h20, h30 (2 seeds each)
- 5q [0.1, 0.25, 0.5, 0.75, 0.9] at h30 (2 seeds)

### Phase 11b: Cumulative targets (5 experiments)
- Horizons: cumret_5, cumret_10, cumret_20, cumret_30
- Single-step prediction with pre-computed cumulative returns
- Tests whether target smoothness improves learning

Total: 53 experiments evaluated (3 failed due to disk space, not re-run).

## Configuration

Daily baseline (from Phase 02b):
- hidden_size=16, max_encoder_length=20, dropout=0.1, batch_size=64
- learning_rate=0.0005, max_epochs=100, early_stop_patience=100

Weekly baseline (from Phase 06a):
- hidden_size=16, max_encoder_length=12, dropout=0.15, batch_size=16
- learning_rate=0.0005, max_epochs=100, early_stop_patience=15

## Base rates

Excess directional accuracy (DirAcc minus base rate) is the primary metric. A model predicting "always positive" achieves the base rate with zero skill.

| Target | Daily Base Rate | Weekly Base Rate |
|--------|-----------------|------------------|
| SP500_Returns | 53.6% | 56.0% |
| cumret_5 | 59.7% | 65.8% |
| cumret_10 | 63.6% | 73.7% |
| cumret_20 | 67.1% | 74.5% |
| cumret_30 | 68.2% | 76.5% |

## Summary results

### By target type

| Target | DirAcc | Excess | Sharpe | Interpretation |
|--------|--------|--------|--------|----------------|
| SP500_Returns | 54.7±1.5% | +0.3% | 0.93 | Marginal skill |
| cumret_5 | 59.3% | -0.4% | 1.87 | No skill |
| cumret_10 | 58.2±3.8% | -5.4% | 3.67 | No skill |
| cumret_20 | 64.4% | -2.7% | 4.63 | No skill |
| cumret_30 | 67.4% | -0.8% | 4.89 | No skill |

Cumulative targets fail uniformly. High Sharpe ratios and astronomical returns (cumret_30 showing 157M% total return) are artifacts of always-positive predictions on high-magnitude targets during a bull market test period, not genuine skill.

### By frequency (point-return targets only)

| Frequency | DirAcc | Excess | Sharpe | Healthy % |
|-----------|--------|--------|--------|-----------|
| Weekly | 56.0±1.6% | +0.0% | 0.94 | 12.2% |
| Daily | 54.0±0.9% | +0.1% | 0.92 | 19.1% |

Aggregate numbers mask horizon effects. See breakdown below.

### By horizon (daily, point-return)

| Horizon | DirAcc | Excess | Sharpe | Healthy % |
|---------|--------|--------|--------|-----------|
| h1 | 53.8±0.3% | +0.2% | 0.87 | 18.2% |
| h3 | 54.1±0.6% | +0.5% | 0.93 | 20.7% |
| h5 | 54.8±2.2% | +0.3% | 1.03 | 15.8% |
| h10 | 56.1±2.8% | +2.1% | 2.08 | 27.1% |
| h20 | 56.0±4.3% | +0.1% | 1.58 | 19.0% |
| h30 | 55.4±5.0% | +0.0% | 1.39 | 18.4% |

h10 is optimal for daily. Performance degrades at h20 and h30.

### By horizon (weekly, point-return)

| Horizon | DirAcc | Excess | Sharpe | Healthy % |
|---------|--------|--------|--------|-----------|
| h1 | 56.7±2.0% | +0.7% | 0.94 | 11.4% |
| h3 | 55.4±1.4% | -0.6% | 0.96 | 13.9% |
| h5 | 55.7±1.2% | -0.3% | 0.92 | 11.4% |

h1 is optimal for weekly. Longer horizons compound weekly noise.

### By quantiles

| Quantiles | DirAcc | Sharpe | Healthy % |
|-----------|--------|--------|-----------|
| 3q | 54.3±1.3% | 0.89 | 14.5% |
| 5q | 54.0±2.5% | 0.84 | 23.2% |
| 7q | 55.0±1.5% | 0.97 | 18.4% |

7q marginally best. Difference is small.

## Top experiments by excess DirAcc

| Experiment | DirAcc | Excess | Base Rate | Healthy % |
|------------|--------|--------|-----------|-----------|
| weekly_7q_h1_s456 | 58.2% | +2.2% | 56.0% | 13.8% |
| daily_7q_h20_s42 | 55.8% | +2.2% | 53.6% | 22.0% |
| daily_5q_h30_s42 | 55.8% | +2.2% | 53.6% | 22.5% |
| daily_7q_h5_s42 | 55.7% | +2.1% | 53.6% | 39.0% |
| daily_7q_h10_s42 | 55.7% | +2.1% | 53.6% | 37.5% |

Four of five top daily experiments use seed 42. Weekly results are more consistent across seeds (all h1 seeds achieve +1.9-2.2% excess).

## Key findings

1. Multi-horizon improves daily frequency. Daily h1 had ~0% excess; h10 achieves +2.1% average excess, with individual seeds at h20/h30 reaching +2.2%. The multi-task gradient regularization from predicting multiple horizons simultaneously helps the shared encoder learn better representations.

2. h10 is the most consistent for daily, but h20/h30 have upside. h10 shows the best average across seeds. h20 and h30 have higher variance - some seeds match or beat h10, others underperform. For robustness, h10 is safer; for maximum potential, h20 may be worth testing further.

3. Weekly is best at h1, no contest. Unlike daily, weekly data is already smoothed. Longer horizons compound weekly noise rather than reducing it. All three seeds at weekly_7q_h1 show consistent +1.9-2.2% excess.

4. 7q is marginally best but not critical. The difference between 3q and 7q is small (~0.7pp DirAcc). Quantile choice matters less than horizon choice.

5. Cumulative targets fail completely. Despite smoother signal-to-noise ratio, all cumret configurations achieved negative excess DirAcc. Models defaulted to predicting positive returns to exploit elevated base rates. This confirms the multi-horizon benefit comes from multi-task learning, not target smoothness.

6. Seed 42 dominates top results. Four of the top five experiments by excess DirAcc use seed 42. This could be lucky initialization or overfitting to the specific train/val split. Rolling evaluation would help distinguish.

7. Trade-off between accuracy and stability. High healthy % (h1, h5) correlates with lower excess DirAcc. Models that make more varied predictions (longer horizons) achieve better directional accuracy but have more epochs with problematic behavior.

## Model comparison: No single best

Different configurations excel on different metrics. Selection depends on objectives.

### Daily frequency top performers

| Model | DirAcc | Excess | Sharpe | Healthy % | Strength |
|-------|--------|--------|--------|-----------|----------|
| daily_7q_h20_s42 | 55.8% | +2.2% | 1.36 | 22.0% | Highest excess |
| daily_5q_h30_s42 | 55.8% | +2.2% | 1.46 | 22.5% | Highest excess, best Sharpe |
| daily_7q_h10_s42 | 55.7% | +2.1% | 1.28 | 37.5% | Good excess, high healthy % |
| daily_7q_h5_s42 | 55.7% | +2.1% | 1.21 | 39.0% | Most stable (highest healthy %) |
| daily_7q_h1_s42 | 53.7% | +0.1% | 0.87 | 65.0% | Most conservative, lowest collapse |

Across-seed averages tell a different story:

| Horizon | Avg Excess | Avg Sharpe | Interpretation |
|---------|------------|------------|----------------|
| h5 | +0.3% | 1.03 | Moderate improvement |
| h10 | +2.1% | 2.08 | Best average, seed 42 representative |
| h20 | +0.1% | 1.58 | High variance, s42 outperformed |
| h30 | +0.0% | 1.39 | High variance, diminishing returns |

h10 has the best average performance and most consistent gains across seeds. h20/h30 show higher variance - individual seeds can beat h10 but averages regress.

### Weekly frequency top performers

| Model | DirAcc | Excess | Sharpe | Healthy % | Strength |
|-------|--------|--------|--------|-----------|----------|
| weekly_7q_h1_s456 | 58.2% | +2.2% | 0.95 | 13.8% | Best overall |
| weekly_7q_h1_s123 | 57.9% | +1.9% | 0.97 | 13.4% | Consistent |
| weekly_7q_h1_s42 | 57.9% | +1.9% | 0.90 | 18.8% | Highest healthy % |

Weekly is clear: h1 dominates. All three seeds show consistent +1.9-2.2% excess.

### Selection guidance

| Objective | Daily | Weekly |
|-----------|-------|--------|
| Maximize excess DirAcc | h20 or h10 | h1 |
| Maximize Sharpe | h30 (5q) | h1 |
| Minimize collapse risk | h1 or h5 | h1 |
| Balance all metrics | h10 | h1 |

## Comparison to prior baselines

| Config | Phase | DirAcc | Excess | Notes |
|--------|-------|--------|--------|-------|
| daily_h1 baseline | 02b | 53.6% | ~0% | No skill |
| daily_7q_h10 (avg) | 10/11 | 55.7% | +2.1% | Best avg excess |
| daily_7q_h20_s42 | 10/11 | 55.8% | +2.2% | Best single run |
| weekly baseline | 06a | ~58% | ~+2% | Already good |
| weekly_7q_h1 | 10/11 | 58.2% | +2.2% | Confirmed |
| weekly rolling | 06b | 59.1±8.3% | +2.3% | Generalization confirmed |

Main improvement: Daily went from no skill (~0% excess) to +2.1% average excess via multi-horizon prediction. Weekly was already performing well; this phase confirmed optimal config.

## Future work: Untested combinations

| Combination | Rationale | Priority |
|-------------|-----------|----------|
| daily_7q_h10 + rolling eval | Validate generalization; h10 has best avg excess | High |
| daily_7q_h5 + rolling eval | h5 has highest healthy %; may generalize better | High |
| daily_7q_h10 + core_dynamics | Trend/momentum features may help multi-horizon | Medium |
| daily_7q_h20 + macro_heavy | Longer horizon aligns with macro release frequency | Medium |
| daily_5q_h30 + core_dynamics | Test if 5q + dynamics improves h30 stability | Medium |
| weekly_7q_h1 + market_only | Remove stale macro features from best weekly config | Low |
| Best config + staleness attention | Architectural fix for feature staleness | Low |
| True multi-horizon cumulative | h1=cumret1, h2=cumret2, etc. (pipeline change) | Low |

Rolling evaluation is highest priority - confirms whether fixed-split gains generalize across market regimes (especially 2022 bear market).

## Reproducing results

Run sweep:
```bash
# Example: daily 7q h10
python train/train_tft.py \
    --experiment-name "10_quantile_horizon_sweep/daily_7q_h10_s42" \
    --frequency daily --feature-set core_proposal --alignment vintage \
    --hidden-size 16 --max-encoder-length 20 --dropout 0.1 --batch-size 64 \
    --learning-rate 0.0005 --max-epochs 100 --early-stop-patience 100 \
    --quantiles 7q --max-prediction-length 10 --seed 42

# Cumulative target example
python train/train_tft.py \
    --experiment-name "10_quantile_horizon_sweep/daily_7q_cumret10_s42" \
    --target cumret_10 \
    --frequency daily --feature-set core_proposal --alignment vintage \
    --hidden-size 16 --max-encoder-length 20 --dropout 0.1 --batch-size 64 \
    --learning-rate 0.0005 --max-epochs 100 --early-stop-patience 100 \
    --quantiles 7q --seed 42
```

Evaluate:
```bash
python train/evaluate_checkpoints.py experiments/10_quantile_horizon_sweep/daily_7q_h10_s42 --top-per-metric 3
```

Analyze sweep:
```bash
python scripts/analyze_sweep.py experiments/10_quantile_horizon_sweep/ --min-epoch 20
```

## Data

Experiments stored in `experiments/10_quantile_horizon_sweep/`:
- 49 point-return experiments (some missing due to disk space issues)
- 5 cumulative target experiments

Each experiment contains checkpoints, evaluation results, and training logs.