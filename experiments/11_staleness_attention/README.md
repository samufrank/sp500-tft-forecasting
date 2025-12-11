# Phase 11: Staleness-Aware Attention (Dec 9-11, 2025)

Investigation of staleness-aware mechanisms to help TFT appropriately weight mixed-frequency data. Tested both staleness features (input space) and staleness attention penalties (attention modification).

## Summary

Staleness information causes model collapse regardless of how it's introduced. Continuous staleness counters (`days_since_CPI_update`) create an exploitable recency signal that leads to 97% unidirectional predictions. The attention penalty mechanism amplifies this problem rather than solving it.

Sparse binary release-day flags (`CPI_is_fresh` only) avoid collapse but provide no predictive value—results match baseline.

Root cause: `days_since_CPI_update` correlates monotonically with recency across the encoder window. Any mechanism that "discounts stale data" mathematically becomes "attend to recent timesteps," which the model exploits for collapse.

Recommendation: Do not use staleness features or attention penalties. Baseline without staleness remains optimal.

## Implementation

### Staleness Features (`data_utils.py`)

Two features computed per macro indicator:

```python
# days_since_CPI_update: continuous counter (0 on release day, increments daily)
# CPI_is_fresh: binary flag (1 on release day, 0 otherwise)
```

Controlled by `--staleness-mode` argument:
- `all`: Both features (default, causes collapse)
- `days_only`: Only continuous counter
- `fresh_only`: Only binary flag (avoids collapse, neutral effect)

Change detection uses threshold: `feature.diff().abs() > 0.01`

### Staleness Attention Penalty (`src/staleness_attention.py`)

Modifies attention scores before softmax:

```python
# In forward pass:
attn_logits = Q @ K.T / sqrt(d_k)
staleness_penalty = staleness_weight * decay(staleness_values)
attn_logits = attn_logits - staleness_penalty  # Penalize stale timesteps
attn_weights = softmax(attn_logits)
```

Decay functions available:
- `prenormalized`: Uses log-normalized [0,1] staleness values directly (recommended)
- `linear`: `1 - (staleness / max_staleness)`
- `exponential`: `exp(-staleness / tau)`
- `log`: `1 / (1 + log(1 + staleness))`
- `step`: Binary threshold

Gradient scaling: 100x required for `staleness_weight` parameter. Attention weights are downstream—gradients are weak without scaling. Matches regime attention gates from Phase 07.

Signal threading: Staleness values extracted from batch, passed through model via patched forward method. Normalized consistently across train/val/test.

### Key Files

- `src/staleness_attention.py`: `StalenessAwareInterpretableMultiHeadAttention` class
- `train/staleness_attention_training.py`: Training integration, signal threading
- `collapse_monitor.py`: Updated with staleness diagnostics logging
- `train/train_tft.py`: CLI arguments (`--staleness-attention`, `--staleness-weight`, `--staleness-mode`)

## Experiments

### Controls

| Experiment | Config | Result |
|------------|--------|--------|
| `staleness_feature_only_d10` | Both staleness features, no penalty | 97% unidirectional, 0 negatives by epoch 2 |
| `baseline_no_staleness` | No staleness features | 40-50% healthy, ~200 negative predictions |
| `weekly_baseline_no_staleness` | Weekly, no staleness | Reference for weekly comparisons |

### Attention Penalty Variants

| Experiment | Config | Result |
|------------|--------|--------|
| `attn_prenorm_d10_w05` | Penalty weight=0.5, prenorm decay | 98% unidirectional, worse than features alone |
| `attn_weak_w01` | Penalty weight=0.1 | 98% unidirectional, weight strength irrelevant |
| `attn_regime_combined` | Penalty + regime attention | Partial rescue (46% healthy at best), inconsistent |
| `attn_fresh_only` | Penalty with fresh_only features | Same as fresh_flag_only, penalty has no effect |

### Sparse Flag Variants

| Experiment | Config | Result |
|------------|--------|--------|
| `fresh_flag_only` | CPI_is_fresh only, no days_since | 54.8% healthy at epoch 13, reconverges to 93% unidirectional |
| `fresh_flag_dropout25` | Fresh flag + dropout 0.25 | Still reconverges, dropout doesn't help |
| `fresh_flag_regime` | Fresh flag + regime attention | Overcorrects—47% dir_acc, predicts mostly negative |
| `fresh_flag_weekly` | Fresh flag, weekly frequency | 59.4% dir_acc at best, matches baseline |
| `fresh_flag_regime_weekly` | Fresh flag + regime, weekly | 56.7% dir_acc, combination hurts |

## Key Findings

### 1. Continuous Staleness Features Cause Collapse

Evidence:
- `staleness_feature_only`: 97% unidirectional by epoch 2
- `baseline_no_staleness`: 40-50% healthy throughout training

Mechanism: In a 20-day encoder window, `days_since_CPI_update` values look like:
```
t-20: ~30 days stale (high)
t-19: ~29 days stale
...
t-1:  ~11 days stale (low)
```

This monotonic gradient means "discount stale" = "attend to recent." The model exploits this shortcut.

### 2. Attention Penalty Makes Collapse Worse

Evidence:
- Without penalty (features only): 97% unidirectional
- With penalty (attn_prenorm): 98% unidirectional, attention entropy 1.37 vs 2.48 baseline

The penalty amplifies recency bias. Instead of learning "discount stale macro while preserving signal," the model learns "attend exclusively to t-1."

### 3. VSN Weight Analysis

| Condition | Inflation_YoY | Staleness Features | VIX (2022) |
|-----------|--------------|-------------------|------------|
| Baseline | 7-13% | N/A | 67% |
| Both staleness features | 2-4% | 30-40% | 39% |
| Fresh flag only | 2.3-9.8% | 4-37% (varies) | 17-63% |

Staleness features crowd out actual macro signal. Inflation_YoY weight drops from 7-13% to 2-4% when staleness features are present.

### 4. Attention Pattern Analysis

| Condition | t-1 weight | t-20 weight | Entropy |
|-----------|-----------|-------------|---------|
| Baseline | 0.06-0.09 | 0.02-0.04 | 2.48 |
| Both staleness features | 0.35-0.47 | 0.001-0.006 | 1.37 |
| Fresh flag only | 0.07-0.13 | 0.01-0.02 | ~2.0 |

Staleness features cause attention to collapse from distributed pattern (~3x recency bias) to near-exclusive t-1 focus (~50x recency bias).

### 5. Sparse Binary Flag Is Neutral

`CPI_is_fresh` (1 on ~12 days/year, 0 otherwise) avoids the monotonic recency problem but adds no predictive value:

| Config | Dir Acc | vs Baseline |
|--------|---------|-------------|
| Weekly baseline (Phase 06b) | 59.1% ± 8.3% | — |
| fresh_flag_weekly | 59.4% peak | Matches |
| Daily baseline (Phase 06b) | 53.3% ± 5.2% | — |
| fresh_flag_only (daily) | 53.7% | Matches |

### 6. Combinations Hurt

| Combination | Result | Explanation |
|-------------|--------|-------------|
| fresh_flag + regime attention (daily) | 47% dir_acc, mostly negative | Overcorrected |
| fresh_flag + regime attention (weekly) | 56.7% dir_acc | Worse than either alone |
| staleness penalty + regime attention | Inconsistent, partial rescue | Mechanisms fight |

Regime attention alone works (Phase 07). Fresh flag alone is neutral. Combined, they interfere.

## Comparison to Baselines

### Daily (vs Phase 06b rolling: 53.3% ± 5.2%)

| Experiment | Dir Acc | Sharpe | Verdict |
|------------|---------|--------|---------|
| baseline_no_staleness | 53.7% | 0.84 | Matches baseline |
| staleness_feature_only | 53.7%* | 0.84* | Collapsed, metrics misleading |
| fresh_flag_only | 53.7% | 0.84 | Matches baseline |

*Collapsed models achieve similar metrics by exploiting market drift

### Weekly (vs Phase 06b rolling: 59.1% ± 8.3%)

| Experiment | Dir Acc | Sharpe | Verdict |
|------------|---------|--------|---------|
| fresh_flag_weekly | 59.4% | 1.17 | Matches baseline |
| fresh_flag_regime_weekly | 56.7% | 0.88 | Worse |

Note: Phase 11 used fixed splits; Phase 06b used rolling evaluation (9 folds). Direct comparison is approximate.

## Conclusions

1. Do not use staleness features. Both continuous counters and attention penalties cause or worsen collapse.

2. Sparse release flags are safe but useless. `CPI_is_fresh` avoids collapse but doesn't improve predictions.

3. The problem is structural. Staleness correlates with recency in time-series data. Any staleness signal becomes a recency signal.

4. TFT implicitly handles staleness. The baseline model learns that Inflation_YoY is slow-moving without explicit staleness encoding.

5. Don't combine modifications blindly. Regime attention + fresh flag performed worse than either alone.

## Future Work

| Idea | Worth pursuing? | Rationale |
|------|-----------------|-----------|
| Surprise encoding (`Inflation_surprise = abs(diff) * is_fresh`) | Maybe | Encodes information content, not timing. But same sparsity as fresh flag—likely neutral. |
| Freshness bonus (flip penalty sign) | No | Only affects ~1 timestep per window. Minimal impact. |
| Different decay functions | No | Problem is fundamental, not implementation details. |
| Higher gradient scaling | No | Already at 100x, weight barely moved. Not a gradient flow issue. |
| Per-feature staleness weights | Maybe | Let VSN learn separate weights for each macro's staleness. Requires architecture change. |
| Feature-level attention penalty | Maybe | Penalize attention to stale features not stale timesteps. Requires VSN modification. |
| Multi-macro staleness (unemployment, fed rate) | Low priority | CPI staleness was already neutral/harmful. More staleness features unlikely to help. |
| Change-point detection architecture | Interesting | Explicit detection of "something changed" moments rather than continuous staleness. Research direction. |
| Multi-rate encoder (separate daily/monthly paths) | Interesting | Process features at native frequencies, fuse representations. Closer to MIDAS literature. Significant architecture change. |

## Experiments Directory Structure

```
experiments/11_staleness_attention/
├── staleness_feature_only_d10/     # Control: both features, no penalty
├── baseline_no_staleness/          # Control: no staleness at all
├── weekly_baseline_no_staleness/   # Control: weekly, no staleness
├── attn_prenorm_d10_w05/           # Penalty weight=0.5
├── attn_weak_w01/                  # Penalty weight=0.1
├── attn_regime_combined/           # Penalty + regime attention
├── attn_fresh_only/                # Penalty with fresh_only mode
├── fresh_flag_only/                # Binary flag only, daily
├── fresh_flag_dropout25/           # Binary flag + higher dropout
├── fresh_flag_regime/              # Binary flag + regime attention
├── fresh_flag_weekly/              # Binary flag, weekly frequency
└── fresh_flag_regime_weekly/       # Binary flag + regime, weekly
```

## Reproducing Key Results

```bash
# Collapse demonstration (staleness features)
python train/train_tft.py --experiment-name 11_staleness_attention/staleness_feature_only_d10 \
    --staleness --staleness-mode all --hidden-size 16 --dropout 0.10 \
    --max-epochs 100 --alignment vintage --frequency daily

# Baseline comparison
python train/train_tft.py --experiment-name 11_staleness_attention/baseline_no_staleness \
    --hidden-size 16 --dropout 0.10 --max-epochs 100 --alignment vintage --frequency daily

# Fresh flag only (avoids collapse)
python train/train_tft.py --experiment-name 11_staleness_attention/fresh_flag_only \
    --staleness --staleness-mode fresh_only --hidden-size 16 --dropout 0.10 \
    --max-epochs 100 --alignment vintage --frequency daily

# Attention penalty (makes collapse worse)
python train/train_tft.py --experiment-name 11_staleness_attention/attn_prenorm_d10_w05 \
    --staleness --staleness-mode all --staleness-attention --staleness-weight 0.5 \
    --hidden-size 16 --dropout 0.10 --max-epochs 100 --alignment vintage --frequency daily

# Weekly comparison (correct hyperparameters)
python train/train_tft.py --experiment-name 11_staleness_attention/fresh_flag_weekly \
    --staleness --staleness-mode fresh_only --hidden-size 16 --dropout 0.15 \
    --batch-size 16 --max-epochs 100 --alignment vintage --frequency weekly --max-encoder-length 12

# Evaluation
python evaluate_checkpoints.py --experiment-dir experiments/11_staleness_attention/fresh_flag_only
python analyze_attention_by_period.py --experiment-dir experiments/11_staleness_attention/fresh_flag_only --checkpoint best
```
