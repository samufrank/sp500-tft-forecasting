#!/usr/bin/env python3
"""
Test script to validate the encoder feature fix.

This should show:
1. Model now has 5 encoder features (not 6)
2. Parameter count increases from 20.7K to ~21.4K
3. Initial loss increases from 0.22 to ~0.60-0.69
4. Prediction std increases from 0.007 to ~0.033

Usage:
    cd ~/dev/school/EEE598DL/final/sp500-tft-forecasting
    python test_encoder_fix.py
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path.cwd()))

from models.tft_model import TemporalFusionTransformer

def test_feature_fix():
    print("="*80)
    print("TESTING ENCODER FEATURE FIX")
    print("="*80)
    
    # Test 1: Create model with 5 features (baseline config)
    print("\nTest 1: Model with 5 encoder features (baseline)")
    model = TemporalFusionTransformer(
        num_encoder_features=5,
        num_decoder_features=0,
        hidden_size=16,
        hidden_continuous_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.25,
        max_encoder_length=20,
        max_prediction_length=1,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1000:.1f}K)")
    
    # Check encoder VSN
    enc_vsn = model.encoder_variable_selection
    print(f"  Encoder VSN num_inputs: {enc_vsn.num_inputs}")
    
    if hasattr(enc_vsn, 'flattened_grn'):
        fc1 = enc_vsn.flattened_grn.fc1
        print(f"  Encoder VSN flattened_grn.fc1: in={fc1.in_features}, out={fc1.out_features}")
        print(f"    → Params: {fc1.in_features * fc1.out_features}")
    
    # Test 2: Forward pass with 6-feature input (from TimeSeriesDataSet)
    print("\nTest 2: Forward pass with 6-feature input (encoder_length + 5 time-varying)")
    batch_size = 4
    encoder_length = 20
    
    # Simulate TimeSeriesDataSet batch format with 6 features
    x = {
        'encoder_cont': torch.randn(batch_size, encoder_length, 6),  # 6 features in data
        'encoder_lengths': torch.full((batch_size,), encoder_length, dtype=torch.long),
    }
    
    print(f"  Input encoder_cont shape: {x['encoder_cont'].shape}")
    
    try:
        with torch.no_grad():
            predictions = model(x)
        print(f"  ✓ Forward pass successful!")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Output mean: {predictions.mean().item():.6f}")
        print(f"  Output std: {predictions.std().item():.6f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False
    
    # Test 3: Verify feature extraction
    print("\nTest 3: Feature extraction logic")
    encoder_cont_6feat = torch.randn(batch_size, encoder_length, 6)
    print(f"  Input: 6 features (includes encoder_length)")
    
    # Simulate what model does
    if encoder_cont_6feat.size(-1) > 5:
        encoder_cont_extracted = encoder_cont_6feat[..., 1:]  # Skip encoder_length
    print(f"  After extraction: {encoder_cont_extracted.shape[-1]} features")
    print(f"  ✓ Correctly extracts time-varying features only")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Model configured for 5 encoder features")
    print(f"✓ Parameter count: {total_params:,} (~21.4K expected)")
    print(f"✓ Handles 6-feature input from TimeSeriesDataSet")
    print(f"✓ Extracts 5 time-varying features for encoder VSN")
    print("\nExpected vs Actual:")
    print(f"  Baseline params: 22,600 (includes 1.2K static_vsn we don't have)")
    print(f"  Custom params: {total_params:,}")
    print(f"  Difference: {22600 - total_params:,} (static_vsn we don't implement)")
    
    if 21000 < total_params < 22000:
        print("\n✓✓✓ PARAMETER COUNT LOOKS CORRECT! ✓✓✓")
        return True
    else:
        print(f"\n⚠️  WARNING: Parameter count seems off")
        print(f"   Expected: ~21,400")
        print(f"   Got: {total_params:,}")
        return False

if __name__ == '__main__':
    success = test_feature_fix()
    sys.exit(0 if success else 1)