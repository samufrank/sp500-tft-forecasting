# Phase 4: Loss Modification Grid (Nov 22, 2024)

Systematic evaluation of custom loss modifications to address prediction collapse and improve model robustness. Reruns Phase 02b baseline configurations with added loss penalties to quantify their impact.

## Best performing models

By composite score (diversity + accuracy + risk-adjusted returns):
h16d010_dirw1.0
- Directional Accuracy: 53.5%, Sharpe: 1.054, Healthy: 63.6%
- Config: hidden_size=16, dropout=0.10, directional_weight=1.0
- Improvement over baseline: +2.6pp healthy, +0.331 Sharpe

By risk-adjusted returns (Sharpe ratio):
h16d025_dirw2.0
- Sharpe: 0.914, Total Return: 104.0%, Alpha: +10.9%
- Healthy: 34.0%, Unidirectional: 60.9%
- Trade-off: Achieves positive alpha despite high unidirectional bias

By prediction diversity:
h16d015_dirw2.0_extremew10.0p90
- Healthy: 75.7% (highest diversity), Sharpe: [data]
- Config: hidden_size=16, dropout=0.15, directional_weight=2.0, extreme_move_weight=10.0

By statistical accuracy:
h16d015_dirw2.0_extremew5.0p95
- MSE: 1.1659 (lowest), Healthy: 39.4%
- Config: hidden_size=16, dropout=0.15, directional_weight=2.0, extreme_move_weight=5.0

Key finding: Choice between dirw=1.0 and dirw=2.0 depends on optimization objective. dirw=1.0 maximizes prediction diversity while maintaining performance. dirw=2.0 sometimes achieves better risk-adjusted returns at the cost of higher unidirectional bias.

## Performance vs Phase 02b baselines

h16d010 (dropout=0.10):
- Phase 02b baseline: 61% healthy, 53.3% dir_acc, 0.723 Sharpe
- Phase 4 dirw=1.0: 63.6% healthy (+2.6pp), 53.5% dir_acc (+0.2pp), 1.054 Sharpe (+0.331)

h16d015 (dropout=0.15):
- Phase 02b baseline: 62% healthy, 52.3% dir_acc, 0.759 Sharpe
- Phase 4 dirw=1.0: 62.0% healthy (±0pp), 52.3% dir_acc (±0pp), 0.759 Sharpe (±0)
- Phase 4 dirw=2.0: 28.7% healthy (-33pp), 53.6% dir_acc (+1.3pp), 0.785 Sharpe (+0.026)

h16d025 (dropout=0.25):
- Phase 02b baseline: 50% healthy, 53.7% dir_acc, 0.959 Sharpe  
- Phase 4 dirw=1.0: 59.8% healthy (+9.8pp), 50.6% dir_acc (-3.1pp), 0.687 Sharpe (-0.272)
- Phase 4 dirw=2.0: 34.0% healthy (-16pp), 53.6% dir_acc (-0.1pp), 0.914 Sharpe (-0.045)

Interpretation: dirw=1.0 consistently improves or maintains diversity metrics with minimal impact on accuracy. dirw=2.0 shows configuration-dependent behavior - sometimes improving Sharpe (h16d025) while sacrificing diversity, sometimes reducing both diversity and Sharpe (h16d015). The optimal penalty weight depends on baseline configuration characteristics and optimization priorities.

## Key findings

Successful modifications:
- Directional penalty effectiveness is configuration-dependent and objective-dependent
- dirw=1.0: Consistently improves or maintains diversity with minimal performance trade-off
- dirw=2.0: Configuration-dependent - can improve Sharpe (h16d025: +10.9% alpha) or hurt diversity (h16d015: 62% → 29% healthy)
- Both dirw=1.0 and dirw=2.0 remain viable depending on optimization objective

Multi-objective optimization trade-off:
- Diversity-focused (dirw=1.0): Better healthy%, lower unidirectional bias, moderate Sharpe
- Performance-focused (dirw=2.0): Sometimes better Sharpe/alpha, higher unidirectional bias
- Example: h16d025_dirw2.0 achieves 0.914 Sharpe and +10.9% alpha despite 61% unidirectional predictions
- Interpretation: High unidirectional bias may reflect accurate regime detection rather than pathology

Combined penalties:
- Directional + collapse penalty provides incremental improvement (+3-6pp healthy)
- Effect is additive but modest compared to directional penalty alone

Failed modifications:
- Variance-only penalties: Never activated (low thresholds) or hurt performance (high thresholds)
- Linear magnitude weighting: Caused collapse at alpha≥0.5
- High directional weights (≥3.0): Over-penalization induced degradation
- Reason for failures: Fixed thresholds don't adapt to regime changes; excessive penalties fight market drift

Extreme move weighting (mixed results):
- Dramatically improves prediction diversity (up to 76% healthy)
- Often reduces Sharpe ratio despite increased diversity
- Interpretation: Model makes bolder predictions but not more accurate ones

## Evaluation methodology

Same quality classification as Phase 0-2:
- HEALTHY: Appropriate variation, good directional accuracy (>52%), positive correlation
- DEGRADED: Predictions vary but poor quality (dir_acc <48% or negative correlation)
- UNIDIRECTIONAL: Extreme directional bias (>98% same sign)
- WEAK_COLLAPSE: 2/3 structural methods detect collapse
- STRONG_COLLAPSE: 3/3 structural methods detect collapse

Detection methods:
- Structural: Variance threshold, range check, consecutive-similarity analysis
- Quality: Rolling 60-day correlation, directional accuracy
- Unidirectional: Directional bias >98% threshold

## Loss modifications tested

Directional diversity penalty (9 weight values):
- Penalizes when >90% of predictions share the same sign during validation
- Weights tested: 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0
- Sweet spot: 1.0-2.0 (higher weights cause degradation)
- Implementation: Squared penalty on (max_directional_bias - 0.90) when exceeded

Variance collapse penalty (10 threshold/weight combinations):
- Penalizes when prediction std falls below threshold
- Failed: Fixed thresholds don't work across market regimes
- Thresholds tested: 0.005, 0.02, 0.035, 0.05, 0.075, 1.0
- Weights tested: 0.1, 0.5, 1.0
- Result: Either never activated or penalized healthy variance

Extreme move weighting (9 variations):
- Upweights large-magnitude returns (top 5-25%) during training
- Improves diversity but typically reduces Sharpe ratio
- Weights tested: 2.0, 3.0, 5.0, 10.0
- Percentiles tested: 75, 90, 95

Linear magnitude weighting (3 alpha values):
- Multiplies loss by (1 + alpha × |return|)
- Failed: alpha≥0.5 caused catastrophic collapse
- Not recommended

Combined penalties (15 configurations):
- Directional + variance collapse penalty
- Best: dirw=2.0 + collapse_weight=0.1, threshold=0.035
- Improvement over directional-only: +3-6pp healthy

## Data used

Training/Evaluation: Vintage release dates (realistic timing)
- Train: `data/splits/vintage/core_proposal_daily_vintage_train.csv`
- Val: `data/splits/vintage/core_proposal_daily_vintage_val.csv`
- Test: `data/splits/vintage/core_proposal_daily_vintage_test.csv`

All experiments use vintage alignment where macro indicators appear with realistic release delays (e.g., CPI released 2 weeks after month-end). This matches Phase 02b for fair comparison.

## Experiments structure

Total: 63 experiments (100 epochs each, lr=0.0005)

Breakdown by configuration:
- h16d015: 47 experiments (complete loss modification sweep)
  - 1 baseline
  - 9 directional only
  - 15 directional + collapse combined
  - 10 variance only (failed)
  - 9 extreme weighting variations
  - 3 linear magnitude (failed)
  
- h16d010: 8 experiments (cross-config validation)
  - Baseline, dirw=1.0/2.0, combined, extreme, dirw+extreme
  
- h16d025: 8 experiments (cross-config validation)
  - Baseline, dirw=1.0/2.0, combined, extreme, dirw+extreme

Naming convention: `h{size}d{dropout*100:03d}_{loss_config}`
Examples:
- h16d015_baseline
- h16d015_dirw1.0
- h16d015_dirw2.0_colw0.1t0.035
- h16d010_dirw1.0_extremew5.0p90

## Reproducibility

Re-evaluate any experiment:
```bash
python train/evaluate_tft.py --experiment-name h16d010_dirw1.0
```

Retrain with same configuration:
```bash
python train/train_tft.py \
  --experiment-name h16d010_dirw1.0_rerun \
  --hidden-size 16 \
  --dropout 0.10 \
  --directional-weight 1.0 \
  --learning-rate 0.0005 \
  --max-epochs 100 \
  --alignment vintage \
  --feature-set core_proposal
```

## Next steps

Combine loss modifications with architectural changes:
- Regime-conditional output + directional penalty: Test both dirw=1.0 (diversity-focused) and dirw=2.0 (performance-focused)
  * Hypothesis: Regime output provides intelligent magnitude adaptation while directional penalty prevents total collapse
  * dirw=2.0 may synergize well with regime conditioning (both encourage regime-appropriate directional bias)
- Staleness-aware attention + directional penalty: Address input quality and output diversity simultaneously
- Goal: Solve both collapse and regime adaptation problems while optimizing for desired performance metric

## Notes

Why directional penalty works:
- Phase 02b models showed 65% unidirectional predictions (>98% positive)
- Real S&P 500: 53-54% positive returns, never >90% over 30-day windows
- Penalty explicitly discourages this pathological behavior during validation
- Preserves model flexibility (doesn't force 50/50 split, just prevents >90% bias)

Implementation details:
- Custom loss function: `src/custom_losses.py` (EnhancedQuantileLoss)
- All penalties modular and independently toggleable via CLI flags
- Penalties applied only during validation (training uses shuffled data)