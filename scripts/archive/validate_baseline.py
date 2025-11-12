#!/usr/bin/env python3
"""
Baseline Validation Test Script
================================

Tests the fixed custom TFT against the exact sweep2_h16_drop_0.25 baseline
configuration to verify architectural fixes are correct.

Expected Results:
- Total parameters: ~22.6K (22,604 after VSN fixes)
- Initial loss: ~0.21 (NOT 0.60 - that was a typo in analysis!)
- Smooth monotonic decrease
- No collapse or oscillation

Reference: BASELINE_CONFIG_REFERENCE.md
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tft_model import TemporalFusionTransformer


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model):
    """Count parameters by module."""
    total = 0
    breakdown = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            breakdown[name] = params
            total += params
    
    return total, breakdown


def create_baseline_model():
    """Create model with exact sweep2_h16_drop_0.25 configuration."""
    
    print("Creating baseline model (sweep2_h16_drop_0.25 config)...")
    print()
    
    model = TemporalFusionTransformer(
        num_encoder_features=7,      # 5 base + relative_time_idx + encoder_length
        num_decoder_features=2,      # relative_time_idx + encoder_length
        hidden_size=16,
        hidden_continuous_size=16,
        lstm_layers=1,
        num_attention_heads=2,       # NOT 4! Baseline uses 2
        dropout=0.25,                # NOT 0.1! Baseline uses 0.25
        max_encoder_length=20,
        max_prediction_length=1,
        quantiles=[0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98],
        learning_rate=0.0005,        # NOT 0.001! Baseline uses 0.0005
    )
    
    return model


def validate_architecture(model):
    """Validate model architecture matches baseline."""
    
    print("="*80)
    print("ARCHITECTURE VALIDATION")
    print("="*80)
    print()
    
    # Count parameters
    total_params = count_parameters(model)
    total_detailed, breakdown = count_parameters_detailed(model)
    
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Expected count
    expected_params = 22604  # After VSN fixes
    diff = total_params - expected_params
    
    if abs(diff) < 10:  # Allow small rounding differences
        print(f"✓ PASS: Parameter count matches baseline ({expected_params:,})")
    else:
        print(f"✗ FAIL: Parameter count mismatch!")
        print(f"  Expected: {expected_params:,}")
        print(f"  Got:      {total_params:,}")
        print(f"  Diff:     {diff:+,}")
        print()
        print("VSN fixes may not have been applied correctly.")
        return False
    
    print()
    print("Parameter breakdown by module:")
    for name, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:40s}: {count:6,} params")
    
    print()
    
    # Check specific components
    checks = []
    
    # Encoder VSN should be ~6.4K after fixes (was ~7.9K)
    encoder_vsn_params = sum(p.numel() for p in model.encoder_variable_selection.parameters())
    checks.append(("Encoder VSN", encoder_vsn_params, 6400, 200))
    
    # Decoder VSN should be ~1.2K after fixes (was ~3.2K)  
    decoder_vsn_params = sum(p.numel() for p in model.decoder_variable_selection.parameters())
    checks.append(("Decoder VSN", decoder_vsn_params, 1200, 200))
    
    print("Component validation:")
    all_pass = True
    for name, actual, expected, tolerance in checks:
        diff = actual - expected
        if abs(diff) <= tolerance:
            print(f"  ✓ {name:20s}: {actual:5,} params (expected ~{expected:,})")
        else:
            print(f"  ✗ {name:20s}: {actual:5,} params (expected ~{expected:,}, diff: {diff:+,})")
            all_pass = False
    
    print()
    return all_pass


def validate_forward_pass(model):
    """Test forward pass with dummy data."""
    
    print("="*80)
    print("FORWARD PASS VALIDATION")
    print("="*80)
    print()
    
    # Create dummy batch
    batch_size = 4
    encoder_length = 20
    decoder_length = 1
    
    batch = {
        'encoder_cont': torch.randn(batch_size, encoder_length, 7),
        'decoder_cont': torch.randn(batch_size, decoder_length, 2),
        'encoder_target': torch.randn(batch_size, encoder_length, 1),
        'decoder_target': torch.randn(batch_size, decoder_length, 1),
    }
    
    print("Input shapes:")
    for key, val in batch.items():
        print(f"  {key:20s}: {list(val.shape)}")
    print()
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        
        print("✓ Forward pass successful")
        print()
        print("Output shape:", list(outputs.shape))
        print(f"Expected: [{batch_size}, {decoder_length}, 7] (7 quantiles)")
        
        # Check output shape
        expected_shape = (batch_size, decoder_length, 7)
        if outputs.shape == expected_shape:
            print("✓ PASS: Output shape correct")
        else:
            print(f"✗ FAIL: Output shape mismatch")
            print(f"  Expected: {expected_shape}")
            print(f"  Got:      {outputs.shape}")
            return False
        
        # Check for NaN/Inf
        if torch.isnan(outputs).any():
            print("✗ FAIL: NaN in outputs")
            return False
        if torch.isinf(outputs).any():
            print("✗ FAIL: Inf in outputs")
            return False
        
        print("✓ No NaN/Inf in outputs")
        
        # Compute loss
        loss = model.loss_fn(outputs, batch['decoder_target'])
        print()
        print(f"Initial loss (random weights): {loss.item():.4f}")
        
        # Loss sanity check
        if loss.item() < 0.1 or loss.item() > 2.0:
            print(f"⚠ WARNING: Initial loss seems unusual")
            print(f"  Expected range: ~0.15 - 0.30 for random initialization")
            print(f"  This may indicate architectural issues")
        else:
            print("✓ Loss in reasonable range for random initialization")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Forward pass failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run baseline validation."""
    
    print()
    print("="*80)
    print("BASELINE VALIDATION TEST")
    print("="*80)
    print()
    print("Configuration: sweep2_h16_drop_0.25")
    print("Reference: BASELINE_CONFIG_REFERENCE.md")
    print()
    
    # Create model
    model = create_baseline_model()
    print()
    
    # Validate architecture
    arch_pass = validate_architecture(model)
    print()
    
    # Validate forward pass
    forward_pass = validate_forward_pass(model)
    print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()
    
    if arch_pass and forward_pass:
        print("✓ ALL TESTS PASSED")
        print()
        print("Next steps:")
        print("1. Run parameter audit: python audit_parameters.py")
        print("2. Train for 5 epochs to verify convergence behavior")
        print("3. Compare to sweep2_h16_drop_0.25 results")
        print()
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print()
        if not arch_pass:
            print("Architecture issues:")
            print("- Parameter count doesn't match baseline")
            print("- Check that VSN fixes were applied correctly")
            print("- See FIXES_APPLIED_SUMMARY.txt")
        if not forward_pass:
            print("Forward pass issues:")
            print("- Model forward() failed or produced invalid outputs")
            print("- Check for shape mismatches or NaN/Inf values")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())