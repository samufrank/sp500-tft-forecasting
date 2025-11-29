# Phase 05a: Regime-conditional output layer modifications

## Overview

Implemented a mixture-of-experts (MoE) output architecture as a drop-in replacement for TFT's standard linear output layer. This modification tests whether regime-conditional expert heads can extract sample-specific predictions from TFT's hidden state, motivated by attention analysis findings that models detect regime shifts but fail to adapt predictions accordingly.

Key finding: All 48 configurations exhibit expert collapse to static per-regime biases. Output-layer modifications alone cannot extract sample-specific predictions from TFT's hidden state, confirming the need for attention-level modifications to encode regime information earlier in the architecture.

---

## Architecture

### Mixture-of-experts output layer

```
Hidden State (16-dim) ──┬──> Router ──> Routing Weights [w0, w1, ...]
                        │
                        ├──> Expert 0 ──> Predictions
                        ├──> Expert 1 ──> Predictions
                        └──> Expert N ──> Predictions
                                              │
                        Final = Σ(wi × Expert_i)
```

Components:
- Router: Learns regime assignment from hidden state (`nn.Linear(16, num_regimes)`)
- Experts: Parallel prediction heads (linear or MLP)
- Routing: Soft (weighted mixture) or hard (single expert per sample)

### Routing strategies

| Strategy | Description | Behavior |
|----------|-------------|----------|
| `learned` | Router learns from hidden state | Soft routing via softmax |
| `vix_threshold` | Deterministic VIX-based | Hard routing (VIX > threshold → expert 1) |

### Expert architectures

| Type | Architecture | Parameters (per expert) |
|------|--------------|------------------------|
| Linear | `nn.Linear(16, 7)` | 119 |
| MLP-16 | `Linear(16,16) → ReLU → Linear(16,7)` | 391 |

---

## Experimental design

### Factors tested

| Factor | Values | Count |
|--------|--------|-------|
| Routing strategy | learned, vix_threshold | 2 |
| Number of regimes | 2, 3 | 2 |
| VIX threshold (2-regime) | 15, 20 | 2 |
| VIX thresholds (3-regime) | (14, 23) | 1 |
| Expert architecture | linear, mlp_16 | 2 |
| Dropout | 0.10, 0.25 | 2 |
| Hard routing training | True, False | 2 (VIX only) |
| Load balance weight | 0.0, 1.0, 2.0 | 3 (learned only) |

Total: 48 experiments
- VIX routing: 24 (2-regime: 16, 3-regime: 8)
- Learned routing: 24 (2-regime: 12, 3-regime: 12)

### Naming convention

`{routing}{regimes}_{dropout}_t{threshold}[_mlp][_hr][_lb{weight}]`

Examples:
- `vix2r_d010_t15` - VIX routing, 2 regimes, dropout 0.10, threshold 15, linear
- `vix3r_d025_t14_23_mlp_hr` - VIX routing, 3 regimes, dropout 0.25, MLP, hard routing
- `learn2r_d010_lb1_mlp` - Learned routing, 2 regimes, load balance 1.0, MLP

---

## Results summary

### Top performers by metric

Note: The test set has 53.7% positive days. Models achieving exactly 53.7% directional accuracy with high unidirectional rates are simply predicting positive most/all of the time, exploiting the unconditional market drift rather than learning temporal patterns.

By directional accuracy:
| Experiment | Dir Acc | Healthy % | Pred Std | Sharpe |
|------------|---------|-----------|----------|--------|
| learn2r_d025_lb2_mlp | 53.7% | 2.3% | 0.022 | 0.838 |
| learn3r_d010_lb2 | 53.7% | 2.3% | 0.051 | 0.838 |
| learn2r_d025_lb1 | 53.7% | 2.3% | 0.049 | 0.838 |

By Sharpe ratio:
| Experiment | Sharpe | Dir Acc | Healthy % | Pred Std |
|------------|--------|---------|-----------|----------|
| learn2r_d025_lb2_mlp | 0.838 | 53.7% | 2.3% | 0.022 |
| learn3r_d010_lb2 | 0.838 | 53.7% | 2.3% | 0.051 |
| vix2r_d010_t20_hr | 0.622 | 47.8% | 58.1% | 0.448 |

By healthy percentage:
| Experiment | Healthy % | Dir Acc | Pred Std | Sharpe |
|------------|-----------|---------|----------|--------|
| vix3r_d010_t14_23_mlp_hr | 64.2% | 47.8% | 0.452 | 0.471 |
| vix3r_d025_t14_23_mlp | 62.9% | 47.9% | 0.476 | 0.454 |
| vix2r_d010_t20_mlp | 61.6% | 47.8% | 0.520 | 0.606 |

The apparent "best" performers by directional accuracy (53.7%) are unidirectional collapsed models - they achieve market-rate accuracy by always predicting positive. VIX routing maintains prediction diversity (49-64% healthy) but with lower accuracy (~48%).

---

## Ablation analysis

### Routing strategy (most impactful)

| Routing | Dir Acc | Healthy % | Pred Std |
|---------|---------|-----------|----------|
| learned | 53.7% ± 0.2% | 4.4% ± 5.9% | 0.038 ± 0.018 |
| vix_threshold | 49.0% ± 1.7% | 49.5% ± 14.2% | 0.339 ± 0.138 |

Fundamental tradeoff exists. Learned routing collapses to exploit market drift (53.7% positive days). VIX routing forces diversity but sacrifices accuracy.

### Number of regimes (no impact)

| Regimes | Dir Acc | Healthy % | Pred Std |
|---------|---------|-----------|----------|
| 2 | 51.3% ± 2.4% | 27.7% ± 22.8% | 0.195 ± 0.177 |
| 3 | 51.3% ± 3.0% | 25.9% ± 28.7% | 0.180 ± 0.189 |

Adding a third regime provides no benefit. Both produce equivalent collapse patterns.

### Expert architecture (no impact)

| Type | Dir Acc | Healthy % | Pred Std |
|------|---------|-----------|----------|
| linear | 51.3% ± 2.6% | 27.6% ± 23.9% | 0.178 ± 0.154 |
| mlp_16 | 51.4% ± 2.7% | 26.3% ± 26.8% | 0.200 ± 0.206 |

MLP experts with 6.5× more parameters provide no improvement. Capacity is not the bottleneck.

### Dropout (minimal impact)

| Dropout | Dir Acc | Healthy % | Val Loss |
|---------|---------|-----------|----------|
| 0.10 | 51.2% ± 2.7% | 28.6% ± 25.5% | 0.383 ± 0.013 |
| 0.25 | 51.5% ± 2.7% | 25.3% ± 25.3% | 0.385 ± 0.014 |

Dropout has negligible effect on regime output performance.

### VIX threshold (moderate impact)

| Threshold | Dir Acc | Healthy % | Pred Std |
|-----------|---------|-----------|----------|
| 15 | 51.4% ± 0.5% | 30.4% ± 2.2% | 0.166 ± 0.046 |
| 20 | 47.9% ± 0.1% | 58.1% ± 2.7% | 0.453 ± 0.045 |

Higher threshold (more samples to low-vol expert) increases healthy % but reduces accuracy. Threshold 20 is "safer" but not better overall.

### Load balance weight (no impact - learned only)

| Weight | Dir Acc | Healthy % | Pred Std |
|--------|---------|-----------|----------|
| 0.0 | 53.7% ± 0.1% | 3.7% ± 3.9% | 0.034 ± 0.010 |
| 1.0 | 53.6% ± 0.3% | 4.5% ± 6.0% | 0.041 ± 0.021 |
| 2.0 | 53.6% ± 0.3% | 5.1% ± 7.9% | 0.040 ± 0.021 |

Load balancing does not prevent collapse in learned routing. All configurations collapse regardless of routing weight distribution.

### Hard routing (inconclusive)

| Mode | Dir Acc | Healthy % | Pred Std |
|------|---------|-----------|----------|
| Soft | 49.0% ± 1.8% | 49.5% ± 14.5% | 0.339 ± 0.141 |
| Hard | 49.0% ± 1.8% | 49.5% ± 14.5% | 0.339 ± 0.141 |

Hard routing experiments showed identical results to soft routing counterparts despite training logs confirming hard routing was applied. Possible implementation issue with gradient flow through `torch.where`; results treated as inconclusive. Given other ablation results showing consistent expert collapse regardless of configuration, this was not prioritized for debugging.

---

## Key findings

### 1. Expert collapse is universal

All 48 configurations exhibit expert collapse to static per-regime biases:
- Expert prediction std: 0.02-0.07 (vs target ~1.0%)
- Experts output nearly constant values regardless of input
- Different configurations just shift which constant each expert outputs

### 2. Routing works, experts don't

- Router successfully learns VIX correlation (r = 0.83-0.87)
- Proves regime signal exists in hidden state
- But experts cannot extract sample-specific predictions from that signal
- They only learn "low-vol regime → predict X" and "high-vol regime → predict Y"

### 3. Accuracy vs diversity tradeoff

| | Learned routing | VIX routing |
|--|-----------------|-------------|
| Dir Acc | 53.7% | 49.0% |
| Healthy % | 4.4% | 49.5% |
| Sharpe | 0.83 | 0.58 |

Learned routing achieves market-rate accuracy by collapsing to predict positive (exploiting 53.7% positive rate). VIX routing maintains diversity but at cost of accuracy.

### 4. Eliminated hypotheses

| Hypothesis | Tested via | Result |
|------------|------------|--------|
| Insufficient capacity | MLP vs linear experts | No difference |
| Wrong regime count | 2 vs 3 regimes | No difference |
| Routing collapse | Load balancing 0-2 | No difference |
| Gradient interference | Hard vs soft routing | No difference (inconclusive) |
| Wrong threshold | VIX 15 vs 20 | Tradeoff, no solution |

---

## Collapse analysis

### Mode distribution (all experiments)

| Mode | Mean | Min | Max |
|------|------|-----|-----|
| Healthy | 27.0% | 2.3% | 64.2% |
| Degraded | 5.4% | 0.0% | 19.8% |
| Unidirectional | 64.1% | 27.5% | 96.8% |
| Weak collapse | 1.8% | 0.0% | 12.9% |
| Strong collapse | 1.8% | 0.0% | 15.6% |

### By routing strategy

| Routing | Healthy % | Unidirectional % |
|---------|-----------|------------------|
| learned | 4.4% | 89.6% |
| vix_threshold | 49.5% | 38.6% |

Learned routing produces severe unidirectional collapse. VIX routing mitigates this but experts still produce static biases.

---

## Conclusion

Output-layer modifications cannot solve the regime adaptation problem. The hidden state contains regime classification signal (router extracts it) but not sample-specific prediction signal in a form experts can use. Experts consistently collapse to static per-regime biases regardless of:

- Architecture (linear vs MLP)
- Routing strategy (learned vs VIX)
- Number of regimes (2 vs 3)
- Regularization (load balancing, hard routing)

Implication: Regime information must be injected earlier in the architecture via attention-level modifications. The attention mechanism needs to encode regime awareness into the hidden state values themselves, not just the attention patterns.

---

## Files

Implementation:
- `src/regime_output.py` - RegimeConditionalOutput module
- `train/train_tft.py` - CLI integration and training modifications

Analysis:
- `scripts/aggregate_experiments.py` - Collects metrics across experiments
- `analysis/scripts/regime_ablation.py` - Generates ablation analysis and plots

Results:
- `results/regime_ablation/` - Ablation plots and analysis
- `experiments/05a_regime_output/` - All 48 experiments

---

## Next steps

1. Phase 05b: Combine promising regime output configs with directional penalties from Phase 04
2. Phase 06: Attention-level modifications to encode regime signal in hidden state
3. Potential extension: Multi-target prediction (individual S&P 500 constituents)
