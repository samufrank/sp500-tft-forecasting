# Phase 0: Baseline Exploration (Oct 6-20, 2025)

Initial hyperparameter search without staleness features, using fixed-date preprocessing (i.e. 14-day CPI lag for core proposal feature set).

## Example model
sweep2_h16_drop_0.25
- Directional Accuracy: 52.4%, Sharpe: 0.54, AUC-ROC: 0.544
- Config: hidden_size=16, dropout=0.25, learning_rate=0.0005
- Quality: 56% healthy, 34% degraded, 7% weak collapse, 4% strong collapse

This model demonstrates the baseline capability of standard TFT architecture. Despite 56% of test period classified as healthy with genuine directional skill (52-65% accuracy), the model exhibits magnitude inversion (negative correlation -0.1 to -0.3) throughout 2020-2023 while maintaining directional edge. Prediction confidence increases 2023+ (0.05% → 0.15% magnitude) with improved correlation, suggesting regime-dependent behavior that motivates Phase 2's regime-conditional attention experiments.

Temporal quality breakdown (see evaluation/ for detailed plots):
- 2020-2023: Predictions too timid (<0.05%), negative correlation, but 52-58% directional accuracy
- Q4 2022 - Q1 2023: Brief collapse episodes (10.5% weak, 3.8% strong)
- 2023+: Improved confidence and correlation, sustained 57%+ directional accuracy

## Key findings
- Models with hidden_size > 18 exhibit prediction collapse
- hidden_size = 16-18 avoid strong collapse but still show periodic quality degradation
- Early experiments (exp001) showed 36% healthy with 0% strong collapse - simplicity may yield robustness
- Dropout and learning rate have minimal impact within tested ranges
- Many models stuck predicting only positive returns (unidirectional behavior) despite varying prediction magnitudes

## Evaluation methodology
Enhanced 4-mode quality detection system:
- HEALTHY: Predictions vary appropriately with good directional accuracy and non-negative correlation
- DEGRADED: Predictions vary but exhibit poor quality (unidirectional, anticorrelated, or low accuracy)
- WEAK_COLLAPSE: 2/3 structural methods detect reduced variation (range, variance, consecutive changes)
- STRONG_COLLAPSE: 3/3 structural methods detect near-constant predictions

Quality metrics:
- Structural: Variance, range, consecutive-similarity checks
- Predictive: Rolling correlation (60-day), directional accuracy, unidirectional detection (>98% same sign)
- Combined threshold: Correlation <0 AND dir_acc <52% flags degradation

See individual experiment `evaluation/` directories for comprehensive diagnostic plots showing rolling correlation, directional accuracy, and prediction magnitude over time.

## Data used
Training/Evaluation: Fixed release dates
- Train: `data/splits/fixed/core_proposal_daily_fixed_train.csv`
- Test: `data/splits/fixed/core_proposal_daily_fixed_test.csv`

Important: These experiments assume all macro indicators are available with fixed lag (e.g., CPI always 14 days old). This is NOT realistic for real-world deployment but provides clean hyperparameter comparisons.

## Not included
- Staleness features (added in Phase 1)
- Vintage release dates (realistic timing of macro data availability)
- Mixed-frequency attention mechanisms (Phase 2)

## Reproducibility
All experiments in this directory can be re-evaluated using:
```bash
python train/evaluate_tft.py --experiment-name 00_baseline_exploration/experiment_name
```

For batch re-evaluation with updated diagnostics:
```bash
bash scripts/archive/reevaluate_all.sh  # Evaluates all experiments with latest evaluation framework
python scripts/summarize_experiments.py --all  # Generates experiments_summary.csv with mode statistics
```

Do not evaluate with vintage test data; these models were trained with fixed dates and results will not be meaningful.
