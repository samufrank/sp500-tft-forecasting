# Phase 0: Baseline Exploration (Oct 6-20, 2025)

Initial hyperparameter search using fixed-date preprocessing.

**Best model:** sweep2_h16_drop_0.25
- Sharpe: 1.02, Dir Acc: 53.7%
- Config: h=16, dropout=0.25, lr=0.0005

**Key findings:**
- Models with h>18 collapse (see COLLAPSE_INVESTIGATION_README.md)
- h=16-18 stable and performant
- This phase used fixed 14-day CPI lag (before vintage dates)

**Note:** Results valid for relative comparisons (hyperparameter ranking), but final evaluation uses vintage dates for temporal accuracy.
