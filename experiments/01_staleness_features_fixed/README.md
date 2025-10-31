# Phase 1: Staleness Features (Oct 21-Oct 22, 2024)

Investigating staleness features with TFT architecture, using fixed-date preprocessing.

## Motivation
Test whether explicitly encoding macro indicator staleness (days since last update, freshness flags) helps TFT models handle mixed-frequency data in financial forecasting.

## Key findings
Staleness features fundamentally incompatible with standard TFT architecture:
- 60/64 staleness experiments (94%) showed severe unidirectional collapse
- **100% positive predictions** across nearly all staleness models (vs 75-85% in Phase 0)
- Log-transform normalization (to address scale mismatch) did not resolve collapse
- No improvement over Phase 0 baseline: similar val_loss (~0.40), dir_acc (~53.7%)
- Extremely low prediction diversity (std: 0.024-0.060 vs 0.04-0.07 in Phase 0)

Root causes identified:
1. Architectural incompatibility: Even with log-transform normalization to [0,1] range, staleness features cause systematic collapse, suggesting the issue is deeper than simple scale mismatch
2. Static feature dominance: Forward-filled macro values + incrementing staleness → contradictory signals
3. Attention saturation: Self-attention collapses when staleness features create degenerate patterns
4. Information redundancy: Model may extract staleness information from forward-filled patterns, making explicit encoding harmful

## Evaluation methodology
4-mode temporal quality detection reveals staleness impact:
- HEALTHY: Predictions vary with good quality (rare in staleness experiments)
- DEGRADED: Unidirectional predictions, anticorrelated, or poor directional accuracy (dominant mode)
- WEAK_COLLAPSE: Reduced prediction variation (2/3 structural methods flagged)
- STRONG_COLLAPSE: Near-constant predictions (3/3 structural methods flagged)

Comparison with Phase 0 baseline (hidden_size=16):
- Phase 0: 56% healthy, 34% degraded, 10% collapse
- Phase 1 staleness: <10% healthy, >80% degraded/collapsed

See individual experiment `evaluation/` directories for:
- Rolling correlation plots (staleness experiments show sustained negative correlation)
- Directional accuracy over time (often <50% - worse than random)
- Prediction magnitude trends (many <0.02% - overly timid)
- Mode-shaded performance plots showing quality degradation timeline

## Data used
Training/Evaluation: fixed release dates
- Train: `data/splits/fixed/core_proposal_daily_fixed_train.csv`  
- Test: `data/splits/fixed/core_proposal_daily_fixed_test.csv`

Important: Like phase 0, uses fixed release date assumption (i.e. 14-day CPI lag). Not realistic for deployment but isolates staleness feature effects from release date modeling complexity.


## Staleness features tested
- `days_since_CPI_update` (log-transformed: log(1+days)/log(1+max_days))
- `CPI_is_fresh` (binary indicator: updated in last 7 days)

Normalization approach: Log transformation applied to compress staleness features to [0,1] range before GroupNormalizer, attempting to prevent scale mismatch that dominated early experiments. Despite proper normalization, collapse persisted.

Additional staleness features for other macroeconomic indicators will be included when those feature configs are tested.


## Results summary
See `experiments_summary.csv` for complete results across all experiments and `experiments_summary_key_metrics.csv` for a condensed version. Compare `healthy_pct`, `degraded_pct`, `weak_collapse_pct`, and `strong_collapse_pct` between Phase 0 and Phase 1 to quantify staleness feature impact.

Key metrics show systematic degradation:
- Directional Accuracy: Most models ~50% (random)
- Sharpe Ratio: Near zero or negative
- AUC-ROC: ~0.5 (no discriminative power)
- Correlation: Often negative (magnitude inversion)
- Unidirectional: Many >95% positive predictions

## Implications for phase 2
Standard TFT architecture fundamentally incompatible with staleness features. The comprehensive negative results provide strong motivation and clear design constraints for Phase 2 architectural modifications:

1. Staleness-aware attention: Separate attention mechanisms for fresh vs stale features to prevent attention saturation
2. Custom normalization: Feature-specific normalization to prevent staleness features from dominating input space
3. Distribution-aware regularization: Encode domain knowledge (returns ≈ N(0, 1%)) to prevent unidirectional predictions
4. Anti-collapse mechanisms: Monitor and penalize unidirectional behavior, low prediction variance, and attention degeneracy during training
5. Regime-conditional attention: Phase 0 results show regime-dependent behavior (2023+ improvement); Phase 2 should explore conditional mechanisms

Alternative hypothesis for Phase 2: TFT may implicitly learn staleness from forward-filled patterns, making explicit staleness features redundant or harmful. Test architecture modifications that help model *use* implicit staleness rather than adding explicit features.

## Reproducibility
All experiments in this directory can be re-evaluated using:
```bash
python train/evaluate_tft.py --experiment-name 01_staleness_features/experiment_name
```

For batch re-evaluation:
```bash
bash scripts/archive/evaluate_all.sh # will reevaluate every experiment in all phases!
python scripts/summarize_experiments.py --all
```

Do not evaluate with vintage test data - these models were trained with fixed dates.

## Next
Phase 2 will implement custom TFT architecture informed by Phase 0 baseline patterns and Phase 1 failure modes. The comprehensive negative results from Phase 1, combined with detailed quality diagnostics, provide strong empirical foundation for architectural design decisions.
