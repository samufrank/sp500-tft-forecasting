#!/usr/bin/env python3
"""
CORRECT parameter comparison using actual checkpoints.

Compares baseline checkpoint vs custom checkpoint layer-by-layer.
"""

import torch
from collections import defaultdict

BASELINE_CKPT = 'experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/tft-epoch=47-val_loss=0.4023.ckpt'
CUSTOM_CKPT = 'experiments/test_exact_baseline/checkpoints/last.ckpt'

def extract_params(checkpoint_path, model_name):
    """Extract parameter counts from checkpoint."""
    print(f"\n{'='*80}")
    print(f"{model_name} PARAMETERS")
    print(f"{'='*80}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    # Only parameters (exclude buffers)
    params = {}
    for name, tensor in state_dict.items():
        if '.weight' in name or '.bias' in name:
            # Remove 'model.' prefix if present
            clean_name = name.replace('model.', '').replace('loss_fn.', '')
            params[clean_name] = tensor.numel()
    
    total = sum(params.values())
    print(f"Total parameters: {total:,}\n")
    
    # Group by major component
    categories = defaultdict(lambda: {'params': [], 'total': 0})
    
    for name, count in sorted(params.items()):
        # Determine category
        if 'static_variable_selection' in name and 'context' not in name:
            cat = 'Static VSN'
        elif 'encoder_variable_selection' in name:
            cat = 'Encoder VSN'
        elif 'decoder_variable_selection' in name:
            cat = 'Decoder VSN'
        elif 'static_context' in name and 'variable_selection' not in name:
            cat = 'Static Context GRNs'
        elif 'static_context_variable_selection' in name:
            cat = 'Static Context VSN Context'
        elif 'static_enrichment' in name or 'static_context_enrichment' in name:
            cat = 'Static Enrichment'
        elif 'lstm_encoder' in name:
            cat = 'LSTM Encoder'
        elif 'lstm_decoder' in name:
            cat = 'LSTM Decoder'
        elif 'post_lstm_gate_encoder' in name or 'post_lstm_add_norm_encoder' in name:
            cat = 'Post-LSTM Encoder Gate'
        elif 'post_lstm_gate_decoder' in name or 'post_lstm_add_norm_decoder' in name:
            cat = 'Post-LSTM Decoder Gate'
        elif 'multihead_attn' in name or 'attention' in name.lower():
            cat = 'Attention'
        elif 'post_attn' in name:
            cat = 'Post-Attention Gate'
        elif 'pos_wise_ff' in name:
            cat = 'Position-wise FF'
        elif 'pre_output' in name or 'output_layer' in name:
            cat = 'Output'
        elif 'prescalers' in name or 'embeddings' in name:
            cat = 'Embeddings/Prescalers'
        else:
            cat = 'Other'
        
        categories[cat]['params'].append((name, count))
        categories[cat]['total'] += count
    
    # Print summary
    print("\nComponent Totals:")
    print("-" * 80)
    for cat in sorted(categories.keys(), key=lambda x: categories[x]['total'], reverse=True):
        print(f"{cat:40s} {categories[cat]['total']:6,}")
    
    return categories, total, params

def compare_models():
    """Side-by-side comparison."""
    baseline_cats, baseline_total, baseline_params = extract_params(BASELINE_CKPT, "BASELINE")
    custom_cats, custom_total, custom_params = extract_params(CUSTOM_CKPT, "CUSTOM")
    
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    
    # Get all categories
    all_cats = sorted(set(baseline_cats.keys()) | set(custom_cats.keys()))
    
    print(f"\n{'Component':<40s} {'Baseline':>10s} {'Custom':>10s} {'Diff':>10s}")
    print("-" * 80)
    
    for cat in all_cats:
        b_total = baseline_cats[cat]['total'] if cat in baseline_cats else 0
        c_total = custom_cats[cat]['total'] if cat in custom_cats else 0
        diff = c_total - b_total
        
        diff_str = f"{diff:+,}" if diff != 0 else "0"
        print(f"{cat:<40s} {b_total:>10,} {c_total:>10,} {diff_str:>10s}")
    
    print("-" * 80)
    print(f"{'TOTAL':<40s} {baseline_total:>10,} {custom_total:>10,} {custom_total - baseline_total:+,}")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    # Identify major differences
    major_diffs = []
    for cat in all_cats:
        b_total = baseline_cats[cat]['total'] if cat in baseline_cats else 0
        c_total = custom_cats[cat]['total'] if cat in custom_cats else 0
        diff = abs(c_total - b_total)
        
        if diff > 100:  # Only show differences > 100 params
            major_diffs.append((cat, b_total, c_total, c_total - b_total))
    
    print("\nComponents with >100 param difference:")
    for cat, b, c, diff in sorted(major_diffs, key=lambda x: abs(x[3]), reverse=True):
        sign = "+" if diff > 0 else ""
        print(f"  {cat:40s} {sign}{diff:6,} ({b:,} → {c:,})")
    
    # Check for missing/extra components
    baseline_only = set(baseline_cats.keys()) - set(custom_cats.keys())
    custom_only = set(custom_cats.keys()) - set(baseline_cats.keys())
    
    if baseline_only:
        print("\nComponents in baseline but NOT in custom:")
        for cat in baseline_only:
            print(f"  {cat}: {baseline_cats[cat]['total']:,} params")
    
    if custom_only:
        print("\nComponents in custom but NOT in baseline:")
        for cat in custom_only:
            print(f"  {cat}: {custom_cats[cat]['total']:,} params")

if __name__ == '__main__':
    compare_models()