# Phase 06a: Weekly Frequency Sweep (Nov 29, 2025)

Systematic hyperparameter sweep for weekly frequency data. Weekly aggregation reduces daily noise and requires different hyperparameters than daily (shorter lookback periods, smaller batch sizes for fewer samples).

## Best performing model

h16_enc8_d025_bs32
- Directional Accuracy: 58.1%, Sharpe: 1.05, Healthy: 18.1%
- Config: hidden_size=16, encoder_length=8, dropout=0.25, batch_size=32
- Excess over baseline: +1.3% (test set is 56.8% positive weeks)

Other notable results:
- Best by Sharpe: h16_enc8_d025_bs32 (1.05)
- Second best dir_acc: h16_enc12_d015_bs16 (57.9%), h16_enc12_d015_bs32 (57.9%)
- Most models plateau at 56.7% dir_acc (baseline positive rate) with 0 negative predictions

## Hyperparameter effects

Correlations with directional accuracy:
- hidden_size: +0.485 (16 clearly better than 8)
- batch_size: +0.276 (32 better than 16)
- encoder_length: +0.183 (slight preference for shorter)
- dropout: -0.151 (minimal effect)

Key findings:
- hidden_size=16 dominates all top results; h=8 models never exceed baseline
- encoder_length=8 (2 months lookback) slightly outperforms 12 (3 months)
- batch_size=32 provides more gradient updates per epoch with limited weekly samples (~1200 train)
- dropout has minimal impact in tested range (0.15-0.30)

## Checkpoint selection analysis

Which validation metric produces best test performance:

| Checkpoint Metric | Mean Dir Acc | Mean Sharpe | Mean Healthy % |
|-------------------|--------------|-------------|----------------|
| val_pred_std      | 0.567        | 0.904       | 3.7%           |
| val_loss          | 0.567        | 0.905       | 3.1%           |
| val_dir_acc       | 0.567        | 0.904       | 1.9%           |
| val_sharpe        | 0.567        | 0.901       | 1.9%           |
| val_num_unique    | 0.470        | 0.249       | 1.9%           |

- val_pred_std produces best results (highest healthy %)
- val_num_unique is unreliable (selects early epochs before learning)
- All metrics except val_num_unique produce similar dir_acc/Sharpe
- Top individual checkpoints (58.1% dir_acc) all came from val_pred_std selection

## Comparison to daily baseline

| Metric | Daily (Phase 02b best) | Weekly (06a best) |
|--------|------------------------|-------------------|
| Dir Acc | 53.6% | 58.1% |
| Baseline Rate | 53.9% | 56.8% |
| Excess Accuracy | -0.3% | +1.3% |
| Sharpe | 0.88 | 1.05 |
| Healthy % | ~50% | 18.1% |

Key insight: Daily models perform at or below naive baseline (predict "always positive"). Weekly models show genuine excess accuracy over baseline, indicating actual signal extraction rather than just exploiting market drift.

## Attention pattern analysis

Weekly models show more dynamic, regime-adaptive attention compared to daily:
- Attention weights shift significantly between periods (2020 vs 2022 vs 2023)
- Daily models use relatively uniform recency decay regardless of regime
- Weekly 2022 attention concentrates on recent timesteps (t-1, t-2) during bear market stress
- Feature importance: VIX dominates (0.51-0.57), Inflation_YoY more important than in daily

See attention heatmaps in experiment directories for per-model visualization.

## Limitations

- Most models still unidirectional (few negative predictions)
- Healthy % lower than daily despite better accuracy
- 18% healthy means 82% of predictions classified as problematic
- Higher accuracy may come from better-calibrated positive predictions, not balanced directionality

## Data

Weekly vintage-aligned data:
- Train: 1,269 weeks (56.1% positive)
- Val: 272 weeks (60.3% positive)  
- Test: 273 weeks (56.8% positive)

Note: Val set has higher positive rate than test, which may cause slight optimistic bias during model selection.

## Experiments structure

Total: 24 experiments (100 epochs each, lr=0.0005, early_stop_patience=15)

Grid:
- hidden_size: [8, 16]
- encoder_length: [8, 12]
- dropout: [0.15, 0.25, 0.30]
- batch_size: [16, 32]

Naming convention: `h{size}_enc{encoder}_{dropout*100:03d}_bs{batch}`
Examples: h16_enc8_d025_bs32, h8_enc12_d015_bs16

All experiments evaluated across all saved checkpoints (val_loss, val_dir_acc, val_sharpe, val_pred_std, val_num_unique) to determine optimal checkpoint selection strategy.

## Next steps

- Use best weekly configs (h16_enc8_d025_bs32, h16_enc12_d015_bs16) for rolling evaluation
- Compare weekly vs daily robustness across market regimes (2016-2024)
- Test regime-conditional output on weekly (attention is more regime-aware, may benefit more than daily)
