# Phase 02b: Extended Vintage Baseline Sweep (Nov 22, 2025)

Systematic hyperparameter sweep on vintage alignment to identify stable configurations for Phase 4 loss modifications.

## Motivation

Phase 02 validated that vintage alignment improves baseline stability (6 experiments). Phase 02b expands the search across hidden_size=[10,12,14,16,18,20] and dropout=[0.10,0.15,0.20,0.25,0.30] to comprehensively map the stable hyperparameter region before testing custom losses.

## Key findings

**Capacity constraints:** Only h14-h16 range produces stable models on vintage data
- h10, h12: 100% collapse rate (10/10 experiments)
- h14: 40% working (2/5), narrow stability window at dropout=0.10-0.20
- h16: 60% working (3/5), stable at dropout=0.10-0.25 (except 0.20 marginal at 17% healthy)
- h18, h20: 100% collapse rate (10/10 experiments)

**Validated stable configs for Phase 4:**
1. h16_drop0.10: 61% healthy, 53.3% dir_acc (replicated Phase 02)
2. h16_drop0.15: 62% healthy, 52.3% dir_acc (replicated Phase 02)  
3. h16_drop0.25: 50% healthy, 53.7% dir_acc (replicated Phase 02)
4. h14_drop0.10: 40% healthy, 51.0% dir_acc (marginal, new)

**Phase 02 replication:** Five configs (h14 dropout [0.15, 0.20], h16 dropout [0.10, 0.15, 0.25]) exactly replicated Phase 02 results, validating experimental reproducibility across independent runs.

**Collapse rate:** 22/30 experiments collapsed or severely degraded (73%), with 17/30 
showing strong collapse (57%). Confirms vintage alignment is challenging and only 
h14-h16 range produces genuinely healthy models.

## Experiments

30 baseline experiments (no staleness features):
- hidden_size: 10, 12, 14, 16, 18, 20
- dropout: 0.10, 0.15, 0.20, 0.25, 0.30
- Fixed params: lr=5e-4, encoder_length=20, attention_heads=2, batch_size=64

All experiments used --no-staleness flag to establish clean baseline performance.

See experiments_summary.csv for complete results.

## Comparison with Phase 02

Phase 02b validates and extends Phase 02 findings:
- Replicated configs (5): Identical performance, confirming reproducibility
- New stable configs (1): h14_drop0.10 discovered as marginal baseline
- Confirmed collapse zones: h10, h12, h18, h20 all fail as predicted
- Dropout sensitivity: dropout=0.20 is unstable (marginal 17% healthy at h16)

Phase 02 baseline performance (no staleness):
- h16_drop0.10: 61% healthy, 53.3% dir_acc
- h16_drop0.15: 62% healthy, 52.3% dir_acc  
- h16_drop0.25: 50% healthy, 53.7% dir_acc

Phase 02 staleness performance:
- h14_drop0.15 + staleness: 59% healthy (only working staleness config across all phases)
- h14_drop0.20 + staleness: 29% healthy (borderline)
- h16_drop0.25 + staleness: 2% healthy (collapsed)

## Implications for Phase 4

Primary configs for loss testing:
1. h16_drop0.10 (most stable, 61% healthy)
2. h16_drop0.15 (best AUC, 62% healthy)
3. h16_drop0.25 (best dir_acc, 50% healthy)

Extended validation (if Phase 4a succeeds):
4. h14_drop0.15 + staleness (from Phase 02, only working staleness config)

Avoid testing: h10, h12, h18, h20 collapse universally and won't benefit from loss modifications.

Strategy: Test custom losses on h16_drop0.10 first to validate approach, then expand to other stable configs if successful.

## Data used

Same as Phase 02: vintage alignment with realistic release dates
- Train: data/splits/vintage/core_proposal_daily_vintage_train.csv (6074 samples)
- Test: data/splits/vintage/core_proposal_daily_vintage_test.csv (1282 samples)

Vintage alignment uses actual FRED release dates for macroeconomic indicators, introducing natural temporal jitter compared to fixed-date preprocessing.

## Evaluation methodology

Same 4-mode quality detection as Phase 02:
- HEALTHY: Predictions vary with good directional accuracy and non-negative correlation
- DEGRADED: Predictions vary but poor quality (unidirectional, anticorrelated, low accuracy)
- WEAK_COLLAPSE: 2/3 structural methods detect reduced variation
- STRONG_COLLAPSE: 3/3 structural methods detect near-constant predictions

Quality assessed over entire 5-year test period (2020-2025), with temporal breakdown showing percentage of time in each mode.

See individual experiment evaluation/ directories for comprehensive diagnostic plots.

## Reproducibility

Run all 30 experiments (runs multiple in parallel):

```bash
bash scripts/run_phase_02b.sh
```

Analyze results:

```bash
# Quick summary
python scripts/analyze_experiments.py --phases 02b_vintage_sweep

# Compare with Phase 02
python scripts/analyze_experiments.py \
    --phases 02_vintage_baseline 02b_vintage_sweep \
    --no-strong-collapse --min-healthy 40

# Detailed analysis of top 5
python scripts/analyze_experiments.py \
    --phases 02b_vintage_sweep \
    --detailed --top 5

# Generate summary CSV
python scripts/summarize_experiments.py \
    --phase 02b_vintage_sweep \
    --output-dir experiments/02b_vintage_sweep \
    --evaluated-only
```