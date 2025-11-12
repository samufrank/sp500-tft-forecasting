#!/usr/bin/env python3
"""
Compare GRN layer structure between baseline and custom to find param differences.
"""

import torch

BASELINE = 'experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/tft-epoch=47-val_loss=0.4023.ckpt'
CUSTOM = 'experiments/test_exact_baseline/checkpoints/last.ckpt'

def compare_component(component_name, baseline_state, custom_state):
    """Compare a specific component between models."""
    print(f"\n{'='*80}")
    print(f"{component_name.upper()} COMPARISON")
    print(f"{'='*80}")
    
    # Get all layers for this component
    baseline_keys = [k for k in baseline_state.keys() if component_name in k]
    custom_keys = [k for k in custom_state.keys() if component_name in k]
    
    # Count params
    baseline_params = sum(baseline_state[k].numel() for k in baseline_keys)
    custom_params = sum(custom_state[k].numel() for k in custom_keys)
    
    print(f"\nTotal params: Baseline={baseline_params:,}, Custom={custom_params:,}, Diff={custom_params-baseline_params:+,}")
    
    print("\nBaseline layers:")
    for k in sorted(baseline_keys):
        if '.weight' in k or '.bias' in k:
            shape = baseline_state[k].shape
            params = baseline_state[k].numel()
            short_name = k.replace('model.', '').replace(component_name + '.', '')
            print(f"  {short_name:50s} {str(shape):20s} {params:6,}")
    
    print("\nCustom layers:")
    for k in sorted(custom_keys):
        if '.weight' in k or '.bias' in k:
            shape = custom_state[k].shape
            params = custom_state[k].numel()
            short_name = k.replace('loss_fn.', '').replace(component_name + '.', '')
            print(f"  {short_name:50s} {str(shape):20s} {params:6,}")
    
    # Find differences
    baseline_layer_names = {k.split('.')[-2:][0] for k in baseline_keys if '.weight' in k or '.bias' in k}
    custom_layer_names = {k.split('.')[-2:][0] for k in custom_keys if '.weight' in k or '.bias' in k}
    
    baseline_only = baseline_layer_names - custom_layer_names
    custom_only = custom_layer_names - baseline_layer_names
    
    if baseline_only:
        print(f"\n⚠️  Layers in BASELINE only:")
        for layer in sorted(baseline_only):
            print(f"  - {layer}")
    
    if custom_only:
        print(f"\n⚠️  Layers in CUSTOM only:")
        for layer in sorted(custom_only):
            print(f"  - {layer}")
    
    if not baseline_only and not custom_only:
        print("\n✓ Same layer structure")

if __name__ == '__main__':
    baseline_ckpt = torch.load(BASELINE, map_location='cpu')
    custom_ckpt = torch.load(CUSTOM, map_location='cpu')
    
    baseline_state = baseline_ckpt['state_dict']
    custom_state = custom_ckpt['state_dict']
    
    # Compare key components with param differences
    components = [
        'static_context_enrichment',  # -240 params
        'encoder_variable_selection',  # -1,240 params
        'static_context_initial_hidden_lstm',  # Part of -720 static context
        'static_enrichment',  # -240 params
    ]
    
    for comp in components:
        compare_component(comp, baseline_state, custom_state)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nKey differences to investigate:")
    print("1. Does baseline have gate_norm structure that custom doesn't?")
    print("2. Are there extra normalization layers in baseline?")
    print("3. Check if baseline has resample_norm that custom doesn't")