#!/usr/bin/env python3
"""
extract feature configuration from an existing baseline checkpoint.

usage:
    python check_baseline_checkpoint.py
"""

import torch
import sys
from pathlib import Path

# ckeckpoint path - using best checkpoint from sweep2_h16_drop_0.25
CHECKPOINT_PATH = 'experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/tft-epoch=47-val_loss=0.4023.ckpt'

def main():
    if not Path(CHECKPOINT_PATH).exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("\nAvailable checkpoints:")
        import os
        for root, dirs, files in os.walk('experiments'):
            for f in files:
                if f.endswith('.ckpt'):
                    print(f"  {os.path.join(root, f)}")
        return
    
    print("="*80)
    print("BASELINE CHECKPOINT ANALYSIS")
    print("="*80)
    print(f"Loading: {CHECKPOINT_PATH}\n")
    
    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Extract hyperparameters
    hparams = ckpt['hyper_parameters']
    
    print("FEATURE CONFIGURATION FROM HYPERPARAMETERS:")
    print("-" * 80)
    
    # Check what features were configured
    feature_keys = [
        'time_varying_reals_encoder',
        'time_varying_reals_decoder', 
        'time_varying_known_reals',
        'time_varying_unknown_reals',
        'static_reals',
        'x_reals',  # All reals
    ]
    
    for key in feature_keys:
        if key in hparams:
            value = hparams[key]
            print(f"{key}:")
            print(f"  {value}")
            print(f"  Count: {len(value) if value else 0}")
            print()
    
    print("\nENCODER VSN ANALYSIS:")
    print("-" * 80)
    
    # Check encoder VSN weights
    state_dict = ckpt['state_dict']
    
    # Find encoder VSN flattened_grn.fc1 weights
    enc_vsn_fc1_key = 'encoder_variable_selection.flattened_grn.fc1.weight'
    
    if enc_vsn_fc1_key in state_dict:
        fc1_weight = state_dict[enc_vsn_fc1_key]
        out_features, in_features = fc1_weight.shape
        print(f"encoder_variable_selection.flattened_grn.fc1.weight:")
        print(f"  Shape: {fc1_weight.shape}")
        print(f"  in_features: {in_features}")
        print(f"  out_features: {out_features}")
        print(f"  Total params: {in_features * out_features}")
        print()
        
        # Reverse engineer number of inputs
        # For VSN with hidden_size=16:
        # in_features = num_inputs * hidden_continuous_size
        # If hidden_continuous_size = 16 (typical), then:
        hidden_continuous_size = 16  # Assumption - verify from hparams
        if 'hidden_continuous_size' in hparams:
            hidden_continuous_size = hparams['hidden_continuous_size']
            print(f"  hidden_continuous_size from hparams: {hidden_continuous_size}")
        
        # VSN flattened GRN input is sum of all variable embeddings
        # Each variable is embedded to hidden_continuous_size
        # So: in_features = num_encoder_variables * hidden_continuous_size
        num_encoder_variables = in_features // hidden_continuous_size
        print(f"  -> inferred num_encoder_variables: {num_encoder_variables}")
        print()
        
        if num_encoder_variables == 5:
            print(" CONFIRMED: Baseline encoder VSN processes 5 features")
            print("  This means encoder_length is NOT in encoder VSN")
        elif num_encoder_variables == 6:
            print(" Baseline encoder VSN processes 6 features")
            print("  This means encoder_length IS in encoder VSN")
        else:
            print(f"? Unexpected: {num_encoder_variables} features")
    else:
        print(f"ERROR: Could not find {enc_vsn_fc1_key} in checkpoint")
        print("\nAvailable encoder_variable_selection keys:")
        for key in state_dict.keys():
            if 'encoder_variable_selection' in key:
                print(f"  {key}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    
    if enc_vsn_fc1_key in state_dict:
        fc1_weight = state_dict[enc_vsn_fc1_key]
        in_features = fc1_weight.shape[1]
        hidden_continuous_size = hparams.get('hidden_continuous_size', 16)
        num_vars = in_features // hidden_continuous_size
        
        if 'time_varying_reals_encoder' in hparams:
            enc_reals = hparams['time_varying_reals_encoder']
            print(f"Hyperparameters say encoder has: {enc_reals}")
            print(f"Actual VSN processes: {num_vars} variables")
            
            if len(enc_reals) == num_vars:
                print(" CONSISTENT: hyperparameters match VSN architecture")
            else:
                print(f" MISMATCH: hparams ({len(enc_reals)}) != VSN ({num_vars})")
            
            if 'encoder_length' in enc_reals:
                print("\n-> encoder_length IS in time_varying_reals_encoder")
            else:
                print("\n-> encoder_length is not in time_varying_reals_encoder")
                print("  (it is static metadata not a time-varying VSN input)")

if __name__ == '__main__':
    main()
