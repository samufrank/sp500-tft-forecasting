#!/usr/bin/env python3
"""
Direct comparison of custom QuantileLoss vs pytorch-forecasting QuantileLoss.

This isolates whether the loss function is causing the 3x difference in initial loss.

Expected outcome:
- If losses match → VSN/forward pass structure is the problem
- If losses differ significantly → QuantileLoss implementation is the problem
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.tft_components import QuantileLoss as CustomQuantileLoss

# Try to import pytorch-forecasting QuantileLoss
try:
    from pytorch_forecasting.metrics import QuantileLoss as BaselineQuantileLoss
except ImportError:
    print("ERROR: pytorch-forecasting not installed")
    print("Install with: pip install pytorch-forecasting")
    sys.exit(1)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)

def test_quantile_loss_basic():
    """Test with simple synthetic data."""
    print("="*80)
    print("TEST 1: Basic Quantile Loss Comparison")
    print("="*80)
    
    set_seed(42)
    
    # Create test data
    batch_size = 64
    time_steps = 1
    num_quantiles = 7
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    
    predictions = torch.randn(batch_size, time_steps, num_quantiles)
    targets = torch.randn(batch_size, time_steps)
    
    print(f"\nInput shapes:")
    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  quantiles: {quantiles}")
    
    # Custom loss
    custom_loss = CustomQuantileLoss(quantiles=quantiles)
    custom_result = custom_loss(predictions, targets)
    
    # Baseline loss
    baseline_loss = BaselineQuantileLoss(quantiles=quantiles)
    
    # Baseline returns per-sample losses, need to match dimensions
    # pytorch-forecasting expects: loss(y_pred, target) where both are [batch, time, ...]
    # It returns [batch, time, num_quantiles]
    baseline_per_sample = baseline_loss.loss(predictions, targets.unsqueeze(-1))
    baseline_result = baseline_per_sample.mean()
    
    print(f"\nResults:")
    print(f"  Custom loss:   {custom_result.item():.6f}")
    print(f"  Baseline loss: {baseline_result.item():.6f}")
    print(f"  Ratio: {custom_result.item() / baseline_result.item():.4f}x")
    print(f"  Absolute diff: {abs(custom_result.item() - baseline_result.item()):.6f}")
    
    # Check if they're close
    relative_error = abs(custom_result.item() - baseline_result.item()) / baseline_result.item()
    
    if relative_error < 0.01:
        print(f"\n✓ PASS: Losses match within 1% (relative error: {relative_error:.4%})")
        return True
    else:
        print(f"\n✗ FAIL: Losses differ significantly (relative error: {relative_error:.4%})")
        return False

def test_quantile_loss_realistic():
    """Test with realistic S&P 500 return distributions."""
    print("\n" + "="*80)
    print("TEST 2: Realistic Financial Returns")
    print("="*80)
    
    set_seed(42)
    
    # Simulate realistic S&P 500 daily returns
    # Real distribution: mean ~0.03% daily, std ~1.2%
    batch_size = 1280  # Full validation set
    time_steps = 1
    num_quantiles = 7
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    
    # Random model predictions (untrained) - will be overconfident
    predictions = torch.randn(batch_size, time_steps, num_quantiles) * 0.01
    
    # Real returns - normal distribution approximation
    targets = torch.randn(batch_size, time_steps) * 0.012 + 0.0003
    
    print(f"\nRealistic return statistics:")
    print(f"  Target mean: {targets.mean().item()*100:.4f}%")
    print(f"  Target std: {targets.std().item()*100:.4f}%")
    print(f"  Prediction mean: {predictions.mean().item()*100:.4f}%")
    print(f"  Prediction std: {predictions.std().item()*100:.4f}%")
    
    # Custom loss
    custom_loss = CustomQuantileLoss(quantiles=quantiles)
    custom_result = custom_loss(predictions, targets)
    
    # Baseline loss
    baseline_loss = BaselineQuantileLoss(quantiles=quantiles)
    baseline_per_sample = baseline_loss.loss(predictions, targets.unsqueeze(-1))
    baseline_result = baseline_per_sample.mean()
    
    print(f"\nResults:")
    print(f"  Custom loss:   {custom_result.item():.6f}")
    print(f"  Baseline loss: {baseline_result.item():.6f}")
    print(f"  Ratio: {custom_result.item() / baseline_result.item():.4f}x")
    
    relative_error = abs(custom_result.item() - baseline_result.item()) / baseline_result.item()
    
    if relative_error < 0.01:
        print(f"\n✓ PASS: Losses match within 1% (relative error: {relative_error:.4%})")
        return True
    else:
        print(f"\n✗ FAIL: Losses differ significantly (relative error: {relative_error:.4%})")
        return False

def test_quantile_loss_edge_cases():
    """Test edge cases that might expose implementation differences."""
    print("\n" + "="*80)
    print("TEST 3: Edge Cases")
    print("="*80)
    
    quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    
    test_cases = [
        ("Perfect predictions", torch.ones(10, 1, 7) * 0.5, torch.ones(10, 1) * 0.5),
        ("All zeros", torch.zeros(10, 1, 7), torch.zeros(10, 1)),
        ("Large positive errors", torch.ones(10, 1, 7) * 2.0, torch.ones(10, 1) * 0.1),
        ("Large negative errors", torch.ones(10, 1, 7) * 0.1, torch.ones(10, 1) * 2.0),
    ]
    
    custom_loss = CustomQuantileLoss(quantiles=quantiles)
    baseline_loss = BaselineQuantileLoss(quantiles=quantiles)
    
    all_passed = True
    
    for name, predictions, targets in test_cases:
        custom_result = custom_loss(predictions, targets)
        baseline_per_sample = baseline_loss.loss(predictions, targets.unsqueeze(-1))
        baseline_result = baseline_per_sample.mean()
        
        relative_error = abs(custom_result.item() - baseline_result.item()) / (baseline_result.item() + 1e-8)
        
        status = "✓" if relative_error < 0.01 else "✗"
        print(f"\n{status} {name}:")
        print(f"    Custom: {custom_result.item():.6f}, Baseline: {baseline_result.item():.6f}, Error: {relative_error:.4%}")
        
        if relative_error >= 0.01:
            all_passed = False
    
    return all_passed

def analyze_implementation_differences():
    """Analyze the mathematical differences between implementations."""
    print("\n" + "="*80)
    print("IMPLEMENTATION ANALYSIS")
    print("="*80)
    
    print("\nCustom QuantileLoss (tft_components.py):")
    print("  - Vectorized computation: max(q*error, (q-1)*error)")
    print("  - Scaling: 2x multiplier")
    print("  - Reduction: .mean() across all dimensions")
    print("  - No masking for variable sequence lengths")
    
    print("\nBaseline QuantileLoss (pytorch-forecasting):")
    print("  - Loop over quantiles: for each q, compute loss separately")
    print("  - Scaling: 2x multiplier")
    print("  - Returns per-sample losses (no automatic reduction)")
    print("  - Has masking support via mask_losses() method")
    print("  - Inherits from TorchMetrics.Metric")
    
    print("\nKey differences:")
    print("  1. Computation method: vectorized vs loop")
    print("  2. Return format: scalar vs per-sample array")
    print("  3. Masking: not supported vs supported")
    
    print("\nFor initialization testing (random weights, full sequences):")
    print("  → Masking shouldn't matter (all sequences same length)")
    print("  → Vectorized vs loop should give same result")
    print("  → Different return format, but mean() should equalize")

def main():
    print("="*80)
    print("QUANTILE LOSS COMPARISON TEST")
    print("="*80)
    print("\nPurpose: Isolate whether QuantileLoss is causing 3x loss difference")
    print("Expected baseline epoch 0 loss: ~0.69 train, ~0.60 val")
    print("Your custom epoch 0 loss: 0.22 (3x too low)")
    
    # Run tests
    test1_pass = test_quantile_loss_basic()
    test2_pass = test_quantile_loss_realistic()
    test3_pass = test_quantile_loss_edge_cases()
    
    analyze_implementation_differences()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✓ ALL TESTS PASSED")
        print("\nConclusion: QuantileLoss implementations match.")
        print("The 3x loss difference is NOT from QuantileLoss.")
        print("→ Investigate forward pass structure (VSN, skip connections, LSTM)")
    else:
        print("\n✗ TESTS FAILED")
        print("\nConclusion: QuantileLoss implementations differ.")
        print("The 3x loss difference IS from QuantileLoss.")
        print("→ Fix: Use pytorch-forecasting's QuantileLoss")
        print("   Change in tft_model.py line 117:")
        print("   from pytorch_forecasting.metrics import QuantileLoss")

if __name__ == "__main__":
    main()