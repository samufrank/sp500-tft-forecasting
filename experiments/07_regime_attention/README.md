# Phase 07: Regime-Aware Attention (Nov 30-Dec 01 2025)

Per-head attention gating conditioned on VIX regime. Each attention head gets a learned scalar gate per regime that amplifies/dampens its contribution.

## Summary

**Fixed-split:** Weekly enc12 shows +1.5% dir_acc (59.4% vs 57.9%) and better Sharpe (1.22 vs 1.05). Gates learn meaningful differentiation - high-vol regime amplifies attention (0.57), low-vol dampens (0.46).

**Rolling eval:** No improvement over baseline. Fixed-split gains didn't generalize across 9 folds.

**Most interesting finding:** Regime attention indirectly changes VSN behavior. During 2022 bear market, VIX weight jumps to 0.74 (vs 0.43 baseline). Model learns "when scared, watch VIX almost exclusively."

## Implementation

- 4 new parameters total (2 regimes × 2 heads)
- VIX threshold at 25 defines regime (deterministic)
- Required 100x gradient scaling - gates are downstream, get weak signal
- Files: `src/regime_attention.py`, `train/regime_attention_training.py`

## Results

### Fixed-split

| Config | Dir Acc | Sharpe | Gate Spread | Notes |
|--------|---------|--------|-------------|-------|
| daily h16/enc20 | 54.1% | 1.24 | 0.38 | Gates diverged well, perf mixed |
| weekly enc8 | 56.6% | 0.92 | 0.06 | Worse than baseline |
| weekly enc12 | **59.4%** | **1.22** | 0.10 | Best result |

### Rolling (9 folds, 2016-2024)

| Config | Baseline | + Regime Attn |
|--------|----------|---------------|
| weekly enc8 | 58.5% ± 7.6% | 58.5% ± 7.6% |
| weekly enc12 | 59.1% ± 8.3% | 59.1% ± 8.3% |

No difference. 2022 still fails at 40% for all variants.

## Key findings

- Gates learn consistently: high-vol amplifies, low-vol dampens
- Daily gates diverge more (0.29 vs 0.67) than weekly (0.46 vs 0.57) - more samples = more learning
- enc8 config doesn't benefit, possibly too short for attention gating to help
- VSN adaptation is dramatic - 2022 concentrates 74% on VIX
- Rolling eval washes out fixed-split gains - improvement may be noise

## Limitations

- VIX threshold (25) not tuned
- Rolling eval shows no benefit
- enc8 hurt rather than helped
- 2022 bear market still fails regardless

## Next steps

- Threshold sweep (20, 25, 30)
- 3-regime config (low/medium/high vol at 15/25)
- Combine with regime output or directional penalty
- Try higher gradient scaling for weekly (200x)

## Experiments structure
```
07_regime_attention/
├── test_regime_attention_daily/
├── test_regime_attention_weekly_enc8/
├── test_regime_attention_weekly_full_enc8/
├── test_regime_attention_weekly_full_enc12/  # best
├── rolling_daily/                            # incomplete
├── rolling_weekly_enc8/
├── rolling_weekly_enc12/
└── *.png, *.csv                              # comparison plots/data
```
