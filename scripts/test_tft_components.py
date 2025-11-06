"""
Test script for TFT components.

Run this in your pytorch environment:
    python test_tft_components.py

This will validate that all components work correctly before building
the full TFT architecture on top of them.
"""

import sys
import torch

# Import components from the same directory
from models.tft_components import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    QuantileLoss,
    run_all_tests
)


def test_device_compatibility():
    """Test that components work on both CPU and CUDA if available."""
    print("\nTesting device compatibility...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create small GRN
    grn = GatedResidualNetwork(
        input_size=10,
        hidden_size=16,
        output_size=8,
        dropout=0.1
    ).to(device)
    
    # Test forward pass
    x = torch.randn(2, 5, 10, device=device)
    output = grn(x)
    
    assert output.device == device, f"Output not on correct device: {output.device}"
    print(f"[ PASS ] Components work on {device}")
    
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through all components."""
    print("\nTesting gradient flow...")
    
    # Test GRN
    grn = GatedResidualNetwork(10, 16, 8)
    grn.train()  # Explicitly set to train mode for gradient testing
    x = torch.randn(2, 5, 10, requires_grad=True)
    loss = grn(x).sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    print("[ PASS ] GRN gradients OK")
    
    # Test VSN
    vsn = VariableSelectionNetwork({'a': 5, 'b': 3}, 16)
    vsn.train()
    x_dict = {
        'a': torch.randn(2, 5, 5, requires_grad=True),
        'b': torch.randn(2, 5, 3, requires_grad=True)
    }
    output, weights = vsn(x_dict)
    loss = output.sum()
    loss.backward()
    assert x_dict['a'].grad is not None and not torch.isnan(x_dict['a'].grad).any()
    print("[ PASS ] VSN gradients OK")
    
    # Test Attention
    attn = InterpretableMultiHeadAttention(64, 4)
    attn.train()
    x = torch.randn(2, 5, 64, requires_grad=True)
    output, _ = attn(x, x, x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    print("[ PASS ] Attention gradients OK")
    
    return True


def test_edge_cases():
    """Test edge cases and potential failure modes."""
    print("\nTesting edge cases...")
    
    # Single time step
    grn = GatedResidualNetwork(10, 16, 8)
    x = torch.randn(2, 1, 10)  # Only 1 time step
    output = grn(x)
    assert output.shape == (2, 1, 8)
    print("[ PASS ] Single time step works")
    
    # Very small batch
    x = torch.randn(1, 5, 10)  # Batch size 1
    output = grn(x)
    assert output.shape == (1, 5, 8)
    print("[ PASS ] Batch size 1 works")
    
    # Large sequence
    x = torch.randn(2, 1000, 10)  # Long sequence
    output = grn(x)
    assert output.shape == (2, 1000, 8)
    print("[ PASS ] Long sequences work")
    
    return True


def test_serialization():
    """Test that models can be saved and loaded."""
    print("\nTesting serialization...")
    
    import tempfile
    import os
    
    # Create and save model
    grn = GatedResidualNetwork(10, 16, 8)
    grn.eval()  # Put in eval mode for deterministic output
    
    x = torch.randn(2, 5, 10)
    with torch.no_grad():  # Disable gradients for comparison
        output_before = grn(x)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(grn.state_dict(), f.name)
        temp_path = f.name
    
    # Load model
    grn_loaded = GatedResidualNetwork(10, 16, 8)
    grn_loaded.load_state_dict(torch.load(temp_path))
    grn_loaded.eval()  # Put in eval mode
    
    # Verify same output
    with torch.no_grad():
        output_after = grn_loaded(x)
    
    assert torch.allclose(output_before, output_after, atol=1e-6), \
        "Loaded model produces different output"
    
    os.remove(temp_path)
    print("[ PASS ] Serialization works")
    
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("TFT Components - Test Suite")
    print("="*70)
    
    all_passed = True
    
    # Run basic component tests
    try:
        all_passed &= run_all_tests()
    except Exception as e:
        print(f"[ FAIL ] Basic tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Run additional tests
    try:
        all_passed &= test_device_compatibility()
    except Exception as e:
        print(f"[ FAIL ] Device test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_gradient_flow()
    except Exception as e:
        print(f"[ FAIL ] Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_edge_cases()
    except Exception as e:
        print(f"[ FAIL ] Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_serialization()
    except Exception as e:
        print(f"[ FAIL ] Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Numerical equivalence against pytorch-forecasting (informational)
    print("\n" + "="*70)
    print("Numerical Equivalence vs pytorch-forecasting")
    print("="*70)
    try:
        from tft_components import test_numerical_equivalence
        test_numerical_equivalence()
    except Exception as e:
        print(f"[ WARN ] Could not run numerical equivalence: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    if all_passed:
        print("[ PASS ] ALL TESTS PASSED - Components ready")
    else:
        print("[ FAIL ] SOME TESTS FAILED - Review errors above")
        return 1
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
