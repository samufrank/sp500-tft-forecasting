# Analysis Scripts Reference

## Experiment Summaries

| When you want to... | Command |
|---------------------|---------|
| Generate CSV summary for a phase | `python summarize_experiments.py --phase 04_custom_losses` |
| Rank/filter from that CSV | `python analyze_experiments.py --phases 04_custom_losses --sort-by dir_acc` |
| Deep snapshot with regime/attention data | `python aggregate_experiments.py --experiments-dir experiments/04_custom_losses` |

**Typical flow:**
```bash
python summarize_experiments.py --phase 04_custom_losses
python analyze_experiments.py --phases 04_custom_losses --no-collapse --top 10
```

**Differences:**
- `summarize_experiments.py` → creates `experiments_summary.csv` in phase dir, has `--split-working`
- `aggregate_experiments.py` → more comprehensive (regime/attention fields), timestamped output
- `analyze_experiments.py` → reads existing CSVs, doesn't extract anything

---

## Checkpoint Analysis

| When you want to... | Command |
|---------------------|---------|
| Compare checkpoint selection strategies | `python aggregate_checkpoints.py experiments/04_custom_losses` |
| Quick single-phase comparison table | `python compare_experiments.py experiments/04_custom_losses` |
| Factorial sweep breakdown | `python analyze_sweep.py experiments/10_overnight_sweep` |

Note: `analyze_sweep.py` expects naming like `weekly_3q_h1_s42` (freq_quant_horizon_seed).

---

## Attention Analysis

| When you want to... | Command |
|---------------------|---------|
| Analyze one model's attention over time | `python analyze_attention_by_period.py --experiment 04_custom_losses/exp001` |
| Run on entire phase | `python batch_analyze_attention.py --phase 04_custom_losses` |
| Aggregate metrics, detect regime shifts | `python summarize_attention_patterns.py experiments/04_custom_losses/` |
| A/B test between two phases | `python compare_phase_attention.py --baseline experiments/00_baseline --treatment experiments/04_custom_losses` |

**Typical flow:**
```bash
python batch_analyze_attention.py --phase 04_custom_losses
python summarize_attention_patterns.py experiments/04_custom_losses/
```

---

## Utilities

| When you want to... | Command |
|---------------------|---------|
| See what hyperparams vary | `python summarize_configs.py experiments/04_custom_losses` |
| Check early stopping / epochs | `python summarize_epochs.py experiments/04_custom_losses` |
| Analyze rolling validation | `python analyze_rolling.py experiments/06b_rolling/daily_baseline` |

---

## Compatibility Notes

All scripts handle missing architectural features gracefully (regime attention, classification head, etc.). They use `.get()` patterns and will produce `None`/empty columns for experiments without those features.

**Only constraint:** `analyze_rolling.py` requires rolling evaluation structure (`fold_*/` or `rolling_results_full.csv`).

---

## Decision Tree

```
Training just finished?
  → summarize_experiments.py --phase X

Need to rank/filter?
  → analyze_experiments.py --phases X

Comparing checkpoint selection methods?
  → aggregate_checkpoints.py

Want attention patterns?
  → batch_analyze_attention.py → summarize_attention_patterns.py

Comparing two architectural variants?
  → compare_phase_attention.py
```
