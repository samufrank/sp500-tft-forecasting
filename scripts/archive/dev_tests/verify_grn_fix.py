#!/usr/bin/env python3
"""
verify that the fixed GRN implementation matches baseline parameter count

this:
1. Creates a GRN with baseline config (hidden_size=16)
2. Counts parameters and verifies it matches expected 1,120
3. Shows detailed breakdown by component
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from models.tft_components import GatedResidualNetwork, GateAddNorm

def count_grn_params():
    """Create GRN and count parameters."""
    print("=" * 80)
    print("GRN PARAMETER VERIFICATION")
    print("=" * 80)
    
    # Create GRN with baseline config
    grn = GatedResidualNetwork(
        input_size=16,
        hidden_size=16,
        output_size=16,
        dropout=0.25,
    )
    
    # Count total parameters
    total_params = sum(p.numel() for p in grn.parameters())
    
    print(f"\nTotal GRN parameters: {total_params:,}")
    print(f"Expected (baseline):  1,120")
    print(f"Match: {'[PASS]' if total_params == 1120 else '✗'}")
    
    if total_params != 1120:
        diff = total_params - 1120
        print(f"\nDifference: {diff:+,} params")
        print("WARNING: parameter count does not match baseline")
        return False
    
    # Show detailed breakdown
    print("\n" + "=" * 80)
    print("PARAMETER BREAKDOWN")
    print("=" * 80)
    
    for name, param in grn.named_parameters():
        print(f"{name:50s}: {param.shape!s:20s} = {param.numel():6,} params")
    
    # Verify GateAddNorm structure
    print("\n" + "=" * 80)
    print("GATEADDNORM VERIFICATION")
    print("=" * 80)
    
    gate_norm_params = sum(p.numel() for p in grn.gate_norm.parameters())
    print(f"\nGateAddNorm total params: {gate_norm_params:,}")
    
    # Expected breakdown:
    # GLU: Linear(16 -> 32) = 16*32 + 32 = 544
    # AddNorm: LayerNorm(16) = 16*2 = 32
    # Total: 576
    expected_gate_norm = 576
    print(f"Expected: {expected_gate_norm}")
    print(f"Match: {'[PASS]' if gate_norm_params == expected_gate_norm else '✗'}")
    
    return total_params == 1120

def test_forward_pass():
    """Test that forward pass works."""
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)
    
    grn = GatedResidualNetwork(
        input_size=16,
        hidden_size=16,
        output_size=16,
        dropout=0.25,
    )
    
    # dummy input
    batch_size = 4
    time_steps = 10
    x = torch.randn(batch_size, time_steps, 16)
    
    try:
        output = grn(x)
        print(f"\nInput shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Match: {'✓' if output.shape == x.shape else '✗'}")
        
        # Check for NaNs
        has_nan = torch.isnan(output).any()
        print(f"Contains NaN: {has_nan}")
        
        if has_nan:
            print("WARNING: Output contains NaN values!")
            return False
        
        return True
    except Exception as e:
        print(f"\nERROR during forward pass: {e}")
        return False

if __name__ == '__main__':
    success = True
    
    # Test parameter count
    if not count_grn_params():
        success = False
    
    # Test forward pass
    if not test_forward_pass():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
