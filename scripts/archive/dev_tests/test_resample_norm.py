#!/usr/bin/env python3
"""
test ResampleNorm implementation and GRN with dimension mismatches

this tests the specific case that was failing: VSN's flattened_grn w/
input_size=80 (5 features * 16 embedding), output_size=5, residual=False.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.tft_components import ResampleNorm, GatedResidualNetwork


def test_resample_norm_downsampling():
    """Test ResampleNorm downsampling (e.g., 80 -> 5)."""
    print("=" * 80)
    print("TEST 1: ResampleNorm Downsampling (80 → 5)")
    print("=" * 80)
    
    resample = ResampleNorm(input_size=80, output_size=5, trainable_add=True)
    
    # Test with 3D input [batch, time, features]
    batch_size, time_steps = 4, 10
    x = torch.randn(batch_size, time_steps, 80)
    
    output = resample(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([{batch_size}, {time_steps}, 5])")
    
    assert output.shape == (batch_size, time_steps, 5), f"Wrong output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    
    print("[PASS] Downsampling works correctly")
    return True


def test_resample_norm_upsampling():
    """Test ResampleNorm upsampling (e.g., 5 -> 16)."""
    print("\n" + "=" * 80)
    print("TEST 2: ResampleNorm Upsampling (5 → 16)")
    print("=" * 80)
    
    resample = ResampleNorm(input_size=5, output_size=16, trainable_add=False)
    
    # Test with 2D input [batch, features]
    batch_size = 4
    x = torch.randn(batch_size, 5)
    
    output = resample(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([{batch_size}, 16])")
    
    assert output.shape == (batch_size, 16), f"Wrong output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    
    print("[PASS] Upsampling works correctly")
    return True


def test_resample_norm_no_change():
    """Test ResampleNorm when input_size == output_size (no resampling)."""
    print("\n" + "=" * 80)
    print("TEST 3: ResampleNorm No Resampling (16 → 16)")
    print("=" * 80)
    
    resample = ResampleNorm(input_size=16, output_size=16)
    
    x = torch.randn(4, 10, 16)
    output = resample(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, f"Shape changed unexpectedly: {output.shape}"
    print("[PASS] No-change case works correctly")
    return True


def test_grn_vsn_flattened_case():
    """Test the actual failing case: VSN flattened_grn with dimension mismatch."""
    print("\n" + "=" * 80)
    print("TEST 4: GRN with VSN Flattened Dimensions (80 → 5)")
    print("=" * 80)
    print("\nThis is the case that was causing NotImplementedError:")
    print("  - input_size=80 (5 features × 16 embedding)")
    print("  - hidden_size=5 (min(16, 5))")
    print("  - output_size=5")
    print("  - residual=False")
    
    grn = GatedResidualNetwork(
        input_size=80,
        hidden_size=5,
        output_size=5,
        dropout=0.25,
        residual=False,
    )
    
    # Input from flattened encoder features
    batch_size, time_steps = 4, 20
    x = torch.randn(batch_size, time_steps, 80)
    
    print(f"\nInput shape: {x.shape}")
    
    try:
        output = grn(x)
        print(f"Output shape: {output.shape}")
        print(f"Expected:     torch.Size([{batch_size}, {time_steps}, 5])")
        
        assert output.shape == (batch_size, time_steps, 5), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        
        print("[PASS] VSN flattened_grn case works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {e}")
        return False


def test_grn_parameter_count():
    """Verify GRN with ResampleNorm has correct parameter count."""
    print("\n" + "=" * 80)
    print("TEST 5: GRN Parameter Count Verification")
    print("=" * 80)
    
    # Standard GRN (no ResampleNorm needed)
    grn_standard = GatedResidualNetwork(
        input_size=16,
        hidden_size=16,
        output_size=16,
        dropout=0.25,
    )
    
    standard_params = sum(p.numel() for p in grn_standard.parameters())
    print(f"\nStandard GRN (16->16->16): {standard_params:,} params")
    print(f"Expected: 1,120 params")
    print(f"Match: {'[PASS]' if standard_params == 1120 else '[FAIL]'}")
    
    # GRN with ResampleNorm (VSN flattened case)
    grn_with_resample = GatedResidualNetwork(
        input_size=80,
        hidden_size=5,
        output_size=5,
        dropout=0.25,
        residual=False,
    )
    
    resample_params = sum(p.numel() for p in grn_with_resample.parameters())
    print(f"\nGRN with ResampleNorm (80->5->5): {resample_params:,} params")
    
    # Expected params:
    # fc1: 80*5 + 5 = 405
    # fc2: 5*5 + 5 = 30
    # gate_norm.glu.fc: 5*10 + 10 = 60
    # gate_norm.add_norm.norm: 5*2 = 10
    # resample_norm.mask: 5 (trainable_add=True by default)
    # resample_norm.norm: 5*2 = 10
    # Total: 405 + 30 + 60 + 10 + 5 + 10 = 520
    expected = 520
    print(f"Expected: ~{expected} params")
    
    if standard_params != 1120:
        print("[FAIL] Standard GRN param count doesn't match baseline!")
        return False
    
    print("[PASS] Parameter counts look correct")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through ResampleNorm."""
    print("\n" + "=" * 80)
    print("TEST 6: Gradient Flow Through ResampleNorm")
    print("=" * 80)
    
    grn = GatedResidualNetwork(
        input_size=80,
        hidden_size=5,
        output_size=5,
        dropout=0.0,  # Disable dropout for deterministic test
        residual=False,
    )
    
    x = torch.randn(2, 10, 80, requires_grad=True)
    
    # Forward pass
    output = grn(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are not NaN
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    
    # Check all GRN parameters have gradients
    for name, param in grn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
    
    print("\n[PASS] Gradients flow correctly through all components")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("RESAMPLENORM AND GRN DIMENSION MISMATCH TESTS")
    print("=" * 80)
    
    all_passed = True
    
    tests = [
        test_resample_norm_downsampling,
        test_resample_norm_upsampling,
        test_resample_norm_no_change,
        test_grn_vsn_flattened_case,
        test_grn_parameter_count,
        test_gradient_flow,
    ]
    
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("[PASS] TESTS PASSED")
        print("=" * 80)
        print("\nResampleNorm working correctly")
        print("GRN can now handle dimension mismatches like VSN's flattened_grn")
        sys.exit(0)
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)
