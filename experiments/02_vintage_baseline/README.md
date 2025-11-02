# Phase 2: Vintage Alignment Validation (Nov 1-2, 2025)

To test whether realistic release date alignment (vintage) affects baseline and staleness feature performance compared to fixed-date preprocessing.

## Motivation
Phase 0 and Phase 1 used fixed release dates (e.g., CPI always 14 days old) for clean hyperparameter comparisons. Before proceeding to architectural modifications, validate whether using realistic release dates (vintage alignment) changes model behavior or collapse patterns.

## Key findings
Vintage alignment improves baseline model quality compared to fixed alignment:
- Reduced unidirectional behavior: 23-43% (vintage) vs 36-72% (fixed)
- Increased healthy predictions: 50-62% (vintage) vs 21-48% (fixed)
- Higher prediction diversity: std 0.037-0.064 (vintage) vs 0.030-0.048 (fixed)
- Directional accuracy slightly lower but within noise: 52-54% (vintage) vs 54-55% (fixed)

Staleness features still collapse on vintage alignment:
- All 3 staleness experiments show unidirectional behavior (73-96%)
- Confirms Phase 1 findings: staleness incompatibility is architectural, not data-related

## Baseline experiments (no staleness)

### baseline_vintage_h16_drop0.15_lr5
Phase 0 counterpart: sweep_hidden_16 (h=16, drop=0.15, lr=0.0005)
- Directional Accuracy: 52.3% (vs 54.5% fixed)
- Sharpe: 0.759, AUC-ROC: 0.514
- Quality: 62.0% healthy, 22.9% unidirectional (vs 45.3% healthy, 35.9% unidirectional fixed)
- Prediction std: 0.064 (vs 0.048 fixed) - more diverse predictions

Best performing vintage baseline. Higher prediction diversity and reduced unidirectional bias compared to fixed alignment, though directional accuracy drops 2.2 percentage points.

### baseline_vintage_h16_drop0.25_lr5
Phase 0 counterpart: sweep2_h16_drop_0.25 (h=16, drop=0.25, lr=0.0005)
- Directional Accuracy: 53.7% (vs 53.9% fixed)
- Sharpe: 0.959, AUC-ROC: 0.500
- Quality: 49.8% healthy, 43.3% unidirectional (vs 20.9% healthy, 72.3% unidirectional fixed)
- Prediction std: 0.037 (vs 0.030 fixed)
- Weak collapse detected (0.4% of period)

Dramatic improvement over fixed alignment: unidirectional behavior reduced from 72% to 43%, healthy periods increased from 21% to 50%. Minor weak collapse episode detected but much healthier overall.

### baseline_vintage_h16_drop0.10_lr5
Phase 0 counterpart: sweep2_h16_drop_0.1 (h=16, drop=0.1, lr=0.0005)
- Directional Accuracy: 53.3% (vs 53.6% fixed)
- Sharpe: 0.723, AUC-ROC: 0.510
- Quality: 61.2% healthy, 25.9% unidirectional (vs 48.2% healthy, 37.7% unidirectional fixed)
- Prediction std: 0.055 (vs 0.056 fixed)

Consistent improvement: more healthy predictions, less unidirectional behavior. Prediction diversity maintained while quality increased.

## Staleness experiments (with staleness features)

### staleness_vintage_h16_drop0.25_lr5
Phase 1 counterpart: staleness_fixed_h16_drop0.25 (h=16, drop=0.25, lr=0.0005)
- Directional Accuracy: 52.3% (vs 53.7% fixed)
- Sharpe: 0.819, AUC-ROC: 0.502
- Quality: 0.9% healthy, 96.3% unidirectional (vs 2.3% healthy, 96.1% unidirectional fixed)
- Prediction std: 0.022 (vs 0.024 fixed)

Nearly identical collapse pattern to fixed alignment. Confirms staleness features cause systematic collapse regardless of release date alignment.

### staleness_vintage_h14_drop0.15_lr5
Phase 1 counterpart: sweep_stale_h14_drop0.15_lr5 (h=14, drop=0.15, lr=0.0005)
- Directional Accuracy: 53.4% (vs 52.5% fixed)
- Sharpe: 0.778, AUC-ROC: 0.512
- Quality: 9.1% healthy, 73.3% unidirectional (vs 60.5% healthy, 17.7% unidirectional fixed)
- Prediction std: 0.037 (vs 0.090 fixed)

Worse on vintage: healthy period drops from 60% to 9%, unidirectional increases from 18% to 73%. Suggests h=14 barely avoided collapse on fixed alignment but fails on vintage.

### staleness_vintage_h14_drop0.20_lr5
Phase 1 counterpart: sweep_stale_h14_drop0.20_lr5 (h=14, drop=0.2, lr=0.0005)
- Directional Accuracy: 53.1% (vs 53.6% fixed)
- Sharpe: 0.744, AUC-ROC: 0.502
- Quality: 7.3% healthy, 83.6% unidirectional (vs 49.5% healthy, 25.1% unidirectional fixed)
- Prediction std: 0.032 (vs 0.082 fixed)

Similar degradation pattern. Both h=14 staleness models collapse more severely on vintage than fixed, indicating smaller capacity cannot handle realistic release date complexity with staleness features.

## Data used
Training/Evaluation: Vintage release dates
- Train: data/splits/vintage/core_proposal_daily_vintage_train.csv
- Test: data/splits/vintage/core_proposal_daily_vintage_test.csv

Vintage alignment uses actual release dates for macroeconomic indicators (CPI released monthly on specific calendar dates, varies 12-16 days lag). This is realistic for deployment and introduces temporal jitter in data availability.

Sample size differences from Phase 0/1:
- Train: 6074 samples (vs 6066 fixed) - vintage has 8 more samples due to alignment differences
- Test: 1302 samples (vs 1300 fixed)

## Evaluation methodology
Same 4-mode quality detection as Phase 0/1:
- HEALTHY: Predictions vary with good directional accuracy and non-negative correlation
- DEGRADED: Predictions vary but poor quality (unidirectional, anticorrelated, low accuracy)
- WEAK_COLLAPSE: 2/3 structural methods detect reduced variation
- STRONG_COLLAPSE: 3/3 structural methods detect near-constant predictions

See individual experiment evaluation/ directories for diagnostic plots.

## Implications for Phase 3
Vintage alignment provides natural regularization against unidirectional drift:
- Baseline models healthier on vintage (50-62% healthy vs 21-48% fixed)
- Temporal jitter from realistic release dates disrupts simple drift exploitation
- All future work should use vintage alignment

Staleness features remain fundamentally incompatible:
- Collapse severity unchanged (h=16) or worse (h=14) on vintage
- Confirms architectural issue, not data preprocessing artifact
- Custom TFT implementation required to handle mixed-frequency data

## Reproducibility
All experiments in this directory can be evaluated using:
```bash
python train/evaluate_tft.py --experiment-name 02_vintage_baseline/experiment_name
```

For batch evaluation:
```bash
python summarize_experiments.py --phase 02_vintage_baseline
```

These models were trained with vintage alignment. Do not evaluate with fixed test data - results will not be meaningful.

## Next
Phase 3 attempted distribution-aware loss (anti-drift, anti-collapse penalties) via monkey-patching but encountered checkpoint serialization failures. All future work will use custom TFT implementation where loss functions and architectural modifications are fully controlled.
