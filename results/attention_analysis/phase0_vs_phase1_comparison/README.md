## Attention pattern analysis results

Experiments analyzed: 121 models (Phase 0: 57, Phase 1: 64)  
Analysis: Yearly attention patterns with multi-signal shift detection

See scripts: `analyze_attention_by_period.py`, `summarize_attention_patterns.py`

---

### Directory structure

```
results/attention_analysis/
├── phase0_vs_phase1_comparison/
│   ├── README.md                              # This file
│   ├── attention_summary.csv                  # Per-experiment entropy/concentration stats
│   ├── attention_shifts.csv                   # Detected attention shifts with signal breakdown
│   ├── attention_with_collapse.csv            # Merged attention + collapse metrics
│   ├── attention_analysis_report.txt          # Human-readable summary report
│   ├── attention_metrics_by_phase.png         # Entropy/concentration distributions
│   ├── attention_shift_timeline.png           # Shift frequency by year transition
│   └── attention_collapse_correlation.png     # Correlation heatmap
```

---
### Experiments included
- Phase 0 (Baseline Exploration): 57 experiments from `00_baseline_exploration/`
  - Various hidden sizes (8-64), dropout rates, learning rates
  - No staleness features
  
- Phase 1 (Staleness Features): 64 experiments from `01_staleness_features_fixed/`
  - Same hyperparameter ranges
  - Added raw staleness features (caused collapse)

All experiments include fixed-shift alignment of release dates.

#### Method
For each experiment:
1. Loaded trained model checkpoint
2. Extracted temporal attention weights across the test set (2020-2025)
3. Grouped predictions by year (script also enables setting a custom period)
4. Computed attention statistics per period:
   - Entropy: measure of attention diffusion (higher = more spread out)
   - Concentration: fraction of attention on top-3 timesteps
   - Top time step: most-attended position in lookback window
5. Detected shifts between consecutive periods using 4 signals

#### Shift Detection Methodology
Multi-signal approach - shift detected if any of the following fire:

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| Cosine similarity | < 0.96 | Attention direction changed |
| L2 distance | > 0.025 | Attention magnitude changed significantly |
| Entropy change | > ±0.05 | Focus sharpened or diffused |
| Top timestep shift | > 5 positions | Strategy changed (e.g., t-1 → t-20) |

High-confidence shifts: 2+ signals fired  
Very strong shifts: 3+ signals fired

---

### Key Findings

#### 1. Attention Shifts Align with Market Regimes

2022 to 2023 shows massive shift (43 experiments):
- Corresponds to Fed pivot from rate hikes to pause
- Most common transition across all experiments
- Primary signals: `top_timestep` (21), `l2_distance` (18)

Other significant transitions:
- 2023 to 2024: 32 experiments (post-pivot adjustment)
- 2024 to 2025: 26 experiments (recent period)
- 2021 to 2022: 24 experiments (end of COVID stimulus)

This implies models do detect regime changes (attention patterns shift) but cannot adapt predictions appropriately (still collapse).


#### 2. Attention does not predict collapse

Correlation analysis (attention metrics vs model performance):

| Metric Pair                 | Correlation | Interpretation                                  |
| --------------------------- | ----------- | ----------------------------------------------- |
| std_entropy + healthy_pct   | -0.29       | Variable attention --> less healthy             |
| entropy_trend + healthy_pct | +0.31       | Sharpening attention --> worse outcomes         |
| std_entropy + pred_std      | +0.26       | Variable attention --> variable predictions     |
| avg_entropy + dir_acc       | ~0.00       | Attention doesn’t predict accuracy              |
| avg_entropy + sharpe_ratio  | ~0.00       | Attention doesn’t predict risk-adjusted returns |

All correlations weak (|r| < 0.31), explaining <10% of variance.

Attention patterns appear to be a symptom of model behavior, not a cause of collapse. The model’s attention mechanism detects important changes but the output layer cannot adapt appropriately.


#### 3. Phase Comparison: Staleness Features Impact

Entropy (attention diffusion):
- Phase 0: Mean = 3.09 ± 0.37 (wide variance, many outliers)
- Phase 1: Mean = 2.97 ± 0.06 (tight distribution, almost no variance)
- Finding: Staleness features reduce entropy variance by 6x

Concentration (focus on top timesteps):
- Phase 0: Mean = 0.047 ± 0.015
- Phase 1: Mean = 0.051 ± 0.004
- Finding: Nearly identical, but Phase 1 more consistent

Interpretation: 
- Staleness features make attention more consistent (less variance)
- But this consistency doesn't help performance - Phase 1 has worse collapse rates
- Over-regularization may prevent adaptive attention behavior


#### 4. Multi-Signal Shift Detection Results

Overall statistics:
- 142 total shifts detected across 55 experiments (45% of models)
- 64 high-confidence shifts (2+ signals)
- 15 very strong shifts (3+ signals)

Signal effectiveness (% of shifts where signal fired):
- Top timestep: Most informative for strategy changes
- L2 distance: Good for magnitude changes
- Entropy: Captures focus sharpening/diffusion
- Cosine: Least sensitive (high baseline similarity)

Most notable shifts (3+ signals):
- Typically occur at major regime boundaries (2022, 2023)
- Involve large top-timestep deltas (e.g., t-1 → t-20)
- Correspond to markets where models struggled most

---

### Implications for Model Development

1. Attention mechanism works as designed
   - Detects important market changes
   - Shifts patterns in response to regime changes
   - No evidence of attention "breaking"

2. The problem is post-attention
   - Models attend appropriately but predictions still collapse
   - Output layer cannot translate attention insights into stable forecasts
   - Need output-level interventions, not attention modifications

3. Staleness features are counterproductive
   - Reduce attention variance dramatically
   - Don't improve collapse rates (make them worse)
   - Over-constrain the attention mechanism
   - Recommendation: Abandon raw staleness features

4. Regime awareness is insufficient
   - Models detect regimes (attention shifts) but can't adapt
   - Need explicit regime-conditional output strategies
   - Or output regularization to prevent collapse


These findings motivate the following:

Phase 2a: Test Log-Transformed Staleness
- Quick test to see if scale mismatch was the issue
- Provides staleness info without dominating input space
- Low risk, high potential reward

Phase 2b: Distribution-Aware Regularization
- Target the actual problem (output instability)
- Domain-aware constraints on prediction statistics
- Independent of attention architecture

Phase 3: Custom Architecture (only if needed)
- Mixed-frequency attention with architectural staleness encoding
- Only pursue if Phases 2a/2b insufficient
- Significant implementation effort

---

### Files Reference

#### attention_summary.csv
Per-experiment attention statistics averaged across all periods.

Columns:
- `experiment_name`: full path (e.g., `00_baseline_exploration/sweep2_h16`)
- `period_type`: yearly/custom
- `n_periods`: number of periods analyzed (typically 6 for 2020-2025)
- `avg_entropy`: mean attention entropy across periods
- `std_entropy`: std dev of entropy across periods
- `min_entropy`, `max_entropy`: entropy range
- `avg_concentration`: mean attention concentration (top-3 timesteps)
- `entropy_trend`: linear slope of entropy over time (negative = sharpening)
- `top_timestep`: most-attended time step overall

Use this to compare attention characteristics across experiments and identify outliers

#### attention_shifts.csv
Detected attention shifts between consecutive periods with signal breakdown.

Columns:
- `experiment_name`: Experiment path
- `period1`, `period2`: Consecutive periods being compared (e.g., "2022", "2023")
- `cosine_similarity`: Cosine similarity of attention vectors (1=identical, 0=orthogonal)
- `l2_distance`: Euclidean distance between attention vectors
- `entropy_change`: Change in entropy (positive = more diffuse)
- `concentration_change`: Change in top-3 concentration
- `top_timestep1`, `top_timestep2`: Most-attended timesteps in each period (e.g., "t-1", "t-20")
- `top_timestep_delta`: Absolute position change
- `signal_cosine`, `signal_l2`, `signal_entropy`, `signal_top_timestep`: Boolean flags for each signal
- `n_signals_fired`: Total signals that detected a shift (0-4)

Use to identify when/where attention strategies changed and understand shift types

#### attention_with_collapse.csv
Merged dataset combining attention metrics with model performance/collapse data.

**Columns**: All from `attention_summary.csv` plus:
- Performance: `dir_acc`, `sharpe_ratio`, `test_rmse`, `test_mae`, `pred_std`, `composite_score`
- Collapse percentages: `healthy_pct`, `degraded_pct`, `unidirectional_pct`, `weak_collapse_pct`, `strong_collapse_pct`
- Collapse flags: `has_any_collapse`, `has_strong_collapse`, `has_degradation`, `has_unidirectional`

Use for correlation analysis and identifying relationships between attention and performance

#### attention_analysis_report.txt
Readable summary report with:
- Summary statistics (entropy/concentration distributions)
- Multi-signal shift detection results
- Signal effectiveness breakdown
- Most common shift transitions
- Notable high-confidence shifts
- Attention-collapse correlations
- Experiments with most/least entropy

Use as a quick reference for key findings

---

### Visualizations

attention_metrics_by_phase.png: 
- 2×2 grid comparing Phase 0 vs Phase 1
- Top row: Entropy (bar chart with error bars + violin plot)
- Bottom row: Concentration (same format)
- Shows how phase 1 has much tighter distributions

attention_shift_timeline.png:
- Bar chart of shift frequency by consecutive year transitions
- Clearly shows 2022→2023 as dominant shift period
- Height = number of experiments with detected shift

attention_collapse_correlation.png:
- Heatmap of correlations between attention metrics and performance
- Rows: attention metrics (avg_entropy, std_entropy, avg_concentration, entropy_trend)
- Columns: performance metrics (collapse percentages, dir_acc, sharpe, etc.)
- Color scale: -0.3 to +0.3 (adjusted for weak correlations)
- Annotations show exact correlation values

---

### To reproduce

#### Generate attention analysis (per experiment)
```bash
python scripts/analyze_attention_by_period.py \
    experiments/00_baseline_exploration/sweep2_h16/ \
    --period yearly \
    --output attention_analysis_year
```

#### Summarize across experiments
```bash
python scripts/summarize_attention_patterns.py \
    experiments/ \
    --output results/attention_analysis/phase0_vs_phase1_comparison/ \
    --experiments-csv results/experiments_summary.csv
```

#### Adjust shift detection thresholds
```bash
python scripts/summarize_attention_patterns.py \
    experiments/ \
    --cosine-threshold 0.98 \
    --l2-threshold 0.03 \
    --entropy-threshold 0.08 \
    --top-timestep-threshold 8 \
    --min-signals 2  # Only report high-confidence shifts
```

---
### Technical notes

#### Attention extraction
- Uses `model.log_attention()` from pytorch-forecasting
- Extracts encoder attention (variable selection + temporal)
- Averages across batch dimension and attention heads
- Shape: `[time_steps, encoder_length]` → `[encoder_length]` (mean attention per position)

#### Period grouping
- Default: yearly (2020, 2021, 2022, 2023, 2024, 2025)
- Custom periods supported via prediction date metadata
- Minimum 10 samples per period required for statistics

#### Statistical measures
- Entropy: `-sum(p * log(p))` where p = attention weights (normalized)
  - Range: 0 (focused) to log(encoder_length) (uniform)
  - Typical: 2.5-4.0 for encoder_length=20
  
- Concentration: `sum(top_3_weights)`
  - Range: 0-1 (1 = all attention on 3 time steps)
  - Typical: 0.03-0.10 for diffuse attention

#### Shift thresholds
Chosen based on empirical analysis of baseline variance:
- Cosine < 0.96: Conservative (attention vectors highly similar by default)
- L2 > 0.025: ~2.5% magnitude change
- Entropy > ±0.05: ~2% relative change from mean
- Top time step > 5: Strategy changed by >= 5 days look back

---

### Caveats

1. Correlation does not equal causation
   - Weak correlations don't prove attention is irrelevant
   - May have nonlinear relationships not captured by Pearson r
   - Confounding variables possible

2. Temporal aggregation
   - Yearly grouping may mask shorter-term dynamics
   - Models may shift attention more frequently than captured
   - Monthly analysis would provide finer resolution

3. Attention averaging
   - Averaging across batches/heads may hide important patterns
   - Some heads may specialize on different features
   - Per-head analysis could reveal more structure

4. Test set only
   - Analysis conducted on test set (2020-2025)
   - Training dynamics not examined
   - May not reflect how attention evolved during training

5. Single architecture
   - All models are TFT variants
   - Findings may not generalize to other architectures
   - Different attention mechanisms may behave differently

---