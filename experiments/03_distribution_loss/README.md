# Phase 3: Distribution-Aware Loss (Nov 1-2, 2025)

An attempt to prevent collapse and drift via distribution-aware penalties (anti-collapse, anti-drift) using monkey-patched loss functions.

## Motivation
Phase 1 showed staleness features cause systematic collapse (100% positive predictions). Before implementing custom TFT architecture, test whether distribution-aware regularization can enforce realistic S&P 500 return statistics (mean approximately 0, std approximately 1%) and prevent unidirectional predictions.

## Approach
Monkey-patched pytorch-forecasting's QuantileLoss to add distribution penalties:
- Anti-drift penalty: (pred_mean - target_mean)^2, target_mean = 0.0003 (daily S&P 500 mean)
- Anti-collapse penalty: (pred_std - target_std)^2, target_std = 0.01 (daily S&P 500 std)
- Implemented via runtime method replacement in train/loss_wrapper.py

Tested 9 configurations:
- Control: no penalties (mean_weight=0.0, std_weight=0.0)
- Anti-collapse only: std_weight in [0.1, 0.2]
- Anti-drift only: mean_weight in [0.1, 0.2]
- Balanced: mean_weight = std_weight in [0.05, 0.1]
- Unbalanced: emphasize one penalty over the other

Base configuration: h=16, drop=0.15, lr=0.0005 (best Phase 0 baseline), vintage alignment, no staleness features

## Critical failure: Checkpoint serialization
All 9 experiments trained successfully but cannot be evaluated. Models are unrecoverable from checkpoints.

Root cause: Python's pickle serialization saved monkey-patched QuantileLoss objects with runtime-added methods. During deserialization, pickle attempts to restore these attributes on fresh QuantileLoss instances (which lack patched methods), causing AttributeError before any user code can intervene.

The failure occurs inside torch.load() during unpickling, before checkpoint manipulation code can execute. Multiple attempted fixes all failed:
- Load with strict=False (pytorch-forecasting doesn't support this parameter)
- Manual checkpoint surgery (can't load checkpoint to access state dict)
- Custom unpickler to intercept QuantileLoss (pickle restores attributes after instantiation)
- Modify config.json to signal re-patching (still requires loading checkpoint first)

## Lessons learned
Monkey-patching is fundamentally incompatible with PyTorch checkpoint serialization:
- Runtime method replacement modifies object instances, not class definitions
- Pickle serializes modified instance state but deserializes from original class definition
- This creates permanent checkpoint corruption - no post-hoc recovery possible
- Monkey-patching suitable only for prototyping, never for experiment tracking

For any non-trivial modifications to pytorch-forecasting models (loss functions, attention mechanisms, architectural changes), custom implementation is necessary.

## Experiments (trained but not evaluated)

### Control
- mean_weight=0.0, std_weight=0.0
- Should replicate baseline_vintage_h16_drop0.15_lr5 from Phase 2
- Purpose: Sanity check that distribution loss framework doesn't break standard training

### Anti-collapse only
- std_01: mean_weight=0.0, std_weight=0.1
- std_02: mean_weight=0.0, std_weight=0.2
- Purpose: Test if penalizing low variance prevents unidirectional collapse

### Anti-drift only
- mean_01: mean_weight=0.1, std_weight=0.0
- mean_02: mean_weight=0.2, std_weight=0.0
- Purpose: Test if penalizing mean drift prevents positive-only predictions

### Balanced penalties
- balanced_005: mean_weight=0.05, std_weight=0.05
- balanced_01: mean_weight=0.1, std_weight=0.1
- Purpose: Test if joint regularization maintains S&P 500 distribution statistics

### Unbalanced penalties
- emphasize_std: mean_weight=0.1, std_weight=0.2
- emphasize_mean: mean_weight=0.2, std_weight=0.1
- Purpose: Test whether collapse or drift prevention is more important

## Data used
Training: Vintage release dates
- Train: data/splits/vintage/core_proposal_daily_vintage_train.csv
- Test: data/splits/vintage/core_proposal_daily_vintage_test.csv
- No staleness features (baseline architecture only)

## Why distribution-aware loss matters
S&P 500 daily returns have known statistical properties:
- Mean approximately 0.03% (slight positive drift)
- Standard deviation approximately 1%
- Roughly symmetric distribution around mean

TFT models often predict only positive returns (exploiting market drift) or constant values (collapse). Distribution-aware loss encodes domain knowledge as training constraint, forcing model to learn temporal dynamics rather than exploiting simple statistics.

## Implications for custom TFT
Distribution-aware loss is theoretically sound and should prevent collapse, but implementation requires full control over loss computation:
- Cannot use monkey-patching (breaks checkpoint serialization)
- Cannot modify pytorch-forecasting's Metric interface (too restrictive)
- Must own loss function in custom TFT implementation

Custom TFT will natively support distribution penalties:
- Add penalties directly in loss computation (no external patching)
- Toggle via hyperparameters (mean_weight, std_weight)
- Properly serialize in checkpoints (penalties part of model definition)

## Alternative investigation: Darts library
Investigated whether Darts (Unit8's time series framework) could provide easier extensibility than pytorch-forecasting. Conclusion: No.

Darts documentation states: "The internal sub models are adopted from pytorch-forecasting's TemporalFusionTransformer implementation". Darts is a thin wrapper around pytorch-forecasting's TFT with cleaner API but identical architectural constraints.

For modifications needed (staleness-aware attention, regime-conditional mechanisms, distribution-aware loss), custom TFT implementation from scratch is the only viable path.

## Reproducibility
These experiments cannot be evaluated due to checkpoint corruption. Training logs are preserved in logs/phase_03/ for reference.

For future distribution-aware loss testing, use custom TFT implementation where penalties are part of model definition, not runtime patches.

## Next steps
Implement custom TFT from scratch with:
1. Native distribution-aware loss (trivial to add once we control the architecture)
2. Staleness-aware attention mechanisms (modify decoder to weight attention by staleness)
3. Regime-conditional attention heads (separate mechanisms for different volatility regimes)
4. Full control over serialization, gradient flow, and architectural modifications
