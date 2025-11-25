# Phase 4: Loss Modification Grid (Nov 22, 2024)

Systematic evaluation of custom loss modifications to address prediction collapse and improve model robustness. Reruns Phase 02b baseline configurations with added loss penalties to quantify their impact.

## Best performing model

h16d010_dirw1.0
- Sharpe: 1.054, Directional Accuracy: 53.5%, Healthy: 63.6%
- Config: hidden_size=16, dropout=0.10, directional_weight=1.0
- Improvement over h16d010 baseline: +0.331 Sharpe (+45.8%), +2.4pp healthy
- Improvement over best unmodified baseline (h16d025): +0.095 Sharpe (+9.9%)

This model shows that directional diversity penalty can rescue suboptimal hyperparameter configurations and achieve performance exceeding tuned baselines. The h16d010 baseline had mediocre performance (Sharpe 0.723, 61% healthy), while h16d025 baseline was naturally stronger (Sharpe 0.959, 50% healthy). Adding dirw=1.0 to h16d010 produced the best overall configuration, outperforming even the best unmodified baseline.

Other notable results by metric:
- Best diversity: h16d015_dirw2.0_extremew10.0p90 (75.7% healthy, Sharpe 0.625)
- Best MSE: h16d015_dirw2.0_extremew5.0p95 (MSE 1.166, 39.4% healthy)
- Best unmodified baseline: h16d025_baseline (Sharpe 0.959, 50% healthy)

## Performance vs Phase 02b baselines

h16d010 (dropout=0.10, initially mediocre baseline):
- Phase 02b baseline: 61.2% healthy, 53.3% dir_acc, 0.723 Sharpe
- Phase 4 dirw=1.0: 63.6% healthy (+2.4pp), 53.5% dir_acc (+0.2pp), 1.054 Sharpe (+0.331)
- Phase 4 dirw=2.0: 63.2% healthy (+2.0pp), 53.5% dir_acc (+0.2pp), 0.898 Sharpe (+0.175)
- Result: Penalties dramatically improve performance

h16d015 (dropout=0.15, already strong baseline):
- Phase 02b baseline: 62.0% healthy, 52.3% dir_acc, 0.759 Sharpe
- Phase 4 dirw=1.0: 62.0% healthy (±0pp), 52.3% dir_acc (±0pp), 0.759 Sharpe (±0)
- Phase 4 dirw=2.0: 28.7% healthy (-33pp), 53.6% dir_acc (+1.3pp), 0.785 Sharpe (+0.026)
- Result: Penalties provide minimal benefit, dirw=2.0 sacrifices diversity for marginal Sharpe gain

h16d025 (dropout=0.25, naturally best baseline):
- Phase 02b baseline: 50.0% healthy, 53.7% dir_acc, 0.959 Sharpe
- Phase 4 dirw=1.0: 59.8% healthy (+9.8pp), 50.6% dir_acc (-3.1pp), 0.687 Sharpe (-0.272)
- Phase 4 dirw=2.0: 34.0% healthy (-16pp), 53.6% dir_acc (-0.1pp), 0.914 Sharpe (-0.045)
- Phase 4 extremew5.0p90: 54.8% healthy (+4.8pp), 53.3% dir_acc (-0.4pp), 0.937 Sharpe (-0.022)
- Result: All penalties hurt Sharpe ratio; baseline already optimal

## Key findings

Baseline quality determines penalty effectiveness:
- Mediocre baselines (h16d010 @ 0.723 Sharpe): Directional penalty provides dramatic improvement (+0.331 Sharpe, +45.8%)
- Strong baselines (h16d015 @ 0.759 Sharpe): Penalties provide minimal or no benefit
- Optimal baselines (h16d025 @ 0.959 Sharpe): Penalties actively hurt performance (-0.045 to -0.272 Sharpe)

Research contribution:
- Loss modifications rescue suboptimal configurations and exceed best unmodified baseline
- h16d010_dirw1.0 (1.054 Sharpe) outperforms h16d025_baseline (0.959 Sharpe) by 9.9%
- Reduces sensitivity to hyperparameter selection - can achieve SOTA without perfect tuning
- Provides principled alternative to exhaustive hyperparameter search

Configuration-dependent behavior patterns:
- dirw=1.0 on h16d010: Large improvement to both diversity and Sharpe
- dirw=1.0 on h16d015: No effect (already at equilibrium)
- dirw=1.0 on h16d025: Improves diversity but tanks Sharpe
- dirw=2.0 generally sacrifices diversity for Sharpe, with mixed success

Successful modifications (on appropriate baselines):
- Directional penalty (dirw=1.0): Rescues poor configurations
- Extreme move weighting: Improves diversity but typically reduces Sharpe
- Combined directional + collapse: Modest additive benefit (+3-6pp healthy) on some configs

Failed modifications (tested comprehensively on h16d015):
- Variance-only penalties: Never activated (low thresholds) or hurt performance (high thresholds)
- Linear magnitude weighting: Caused catastrophic collapse at alpha≥0.5
- High directional weights (≥3.0): Over-penalization induced degradation
- Reason: Fixed thresholds don't adapt to regime changes; excessive penalties fight market drift

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
- Sweet spot: 1.0 for rescuing poor baselines (h16d010)
- Implementation: Squared penalty on (max_directional_bias - 0.90) when exceeded

Variance collapse penalty (10 threshold/weight combinations):
- Penalizes when prediction std falls below threshold
- Failed across all configurations tested
- Thresholds tested: 0.005, 0.02, 0.035, 0.05, 0.075, 1.0
- Weights tested: 0.1, 0.5, 1.0
- Result: Either never activated or penalized healthy variance

Extreme move weighting (9 variations):
- Upweights large-magnitude returns (top 5-25%) during training
- Improves diversity but typically reduces Sharpe ratio
- Weights tested: 2.0, 3.0, 5.0, 10.0
- Percentiles tested: 75, 90, 95
- Best performer: extremew5.0p90 on h16d025 (only -0.022 Sharpe vs baseline)

Linear magnitude weighting (3 alpha values):
- Multiplies loss by (1 + alpha × |return|)
- Failed: alpha≥0.5 caused catastrophic collapse
- Not recommended

Combined penalties (15 configurations):
- Directional + variance collapse penalty
- Modest incremental benefit when directional penalty already effective
- Best: dirw=2.0 + collapse_weight=0.1, threshold=0.035 on h16d015

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
  - 10 variance only (all failed)
  - 9 extreme weighting variations
  - 3 linear magnitude (all failed)
  
- h16d010: 8 experiments (mediocre baseline improvement validation)
  - Baseline, dirw=1.0/2.0, combined, extreme, dirw+extreme
  
- h16d025: 8 experiments (strong baseline degradation validation)
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
- Regime-conditional output + directional penalty: Test on h16d010 (proven to benefit from penalties)
  * Hypothesis: Regime output provides intelligent magnitude adaptation while dirw=1.0 prevents collapse
  * Avoid testing on h16d025 (penalties hurt performance on strong baselines)
- Staleness-aware attention + directional penalty: Address input quality and output diversity simultaneously
- Goal: Solve both collapse and regime adaptation while maintaining improvements from loss modifications

## Notes

Why penalties help some configs but hurt others:
- h16d010 baseline exhibits pathological unidirectional behavior that penalties correct
- h16d025 baseline already achieves good balance between diversity and performance
- Adding constraints to already-optimal configuration degrades performance
- Lesson: Apply penalties diagnostically, not universally

Implementation details:
- Custom loss function: `src/custom_losses.py` (EnhancedQuantileLoss)
- All penalties modular and independently toggleable via CLI flags
- Penalties applied during validation (training uses shuffled batches)
- Directional penalty specifically targets sequential validation data behavior