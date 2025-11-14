#!/usr/bin/env python3
"""
Extract exact parameter breakdown from baseline checkpoint.

This will show us layer-by-layer what baseline has that custom doesn't.
"""

import torch
from collections import defaultdict

CHECKPOINT = 'experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/tft-epoch=47-val_loss=0.4023.ckpt'

def analyze_baseline_checkpoint():
    print("="*80)
    print("BASELINE MODEL PARAMETER BREAKDOWN FROM CHECKPOINT")
    print("="*80)
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = ckpt['state_dict']
    
    # Only count parameters (exclude buffers)
    param_dict = {}
    for name, tensor in state_dict.items():
        if '.weight' in name or '.bias' in name:
            param_dict[name] = tensor.numel()
    
    total = sum(param_dict.values())
    print(f"\nTotal parameters: {total:,}\n")
    
    # Categorize by component (matching your diagnostic output)
    categories = {
        'Input Embeddings': [],
        'Static Variable Selection': [],
        'Static Context GRNs': [],
        'Encoder VSN': [],
        'Decoder VSN': [],
        'LSTM Encoder': [],
        'LSTM Decoder': [],
        'Post-LSTM Gates': [],
        'Attention': [],
        'Static Enrichment': [],
        'Output Layers': [],
        'Other': [],
    }
    
    for name, params in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
        # Categorize
        if 'input_embeddings' in name or 'prescalers' in name:
            categories['Input Embeddings'].append((name, params))
        elif 'static_variable_selection' in name and 'context' not in name:
            categories['Static Variable Selection'].append((name, params))
        elif 'static_context' in name and 'variable_selection' not in name:
            categories['Static Context GRNs'].append((name, params))
        elif 'encoder_variable_selection' in name:
            categories['Encoder VSN'].append((name, params))
        elif 'decoder_variable_selection' in name:
            categories['Decoder VSN'].append((name, params))
        elif 'lstm_encoder' in name:
            categories['LSTM Encoder'].append((name, params))
        elif 'lstm_decoder' in name:
            categories['LSTM Decoder'].append((name, params))
        elif 'post_lstm_gate' in name or 'post_lstm_add_norm' in name:
            categories['Post-LSTM Gates'].append((name, params))
        elif 'multihead_attn' in name or 'attention' in name.lower():
            categories['Attention'].append((name, params))
        elif 'static_enrichment' in name or 'static_context_enrichment' in name:
            categories['Static Enrichment'].append((name, params))
        elif 'output' in name or 'pre_output' in name:
            categories['Output Layers'].append((name, params))
        else:
            categories['Other'].append((name, params))
    
    # Print by category
    category_totals = {}
    for cat, items in categories.items():
        if items:
            print(f"\n{cat}:")
            cat_total = 0
            for name, params in items:
                # Shorten name for readability
                short_name = name.replace('model.', '')
                print(f"  {short_name:70s} {params:6,}")
                cat_total += params
            print(f"  {'SUBTOTAL':70s} {cat_total:6,}")
            category_totals[cat] = cat_total
        else:
            category_totals[cat] = 0
    
    print("\n" + "="*80)
    print("SUMMARY BY COMPONENT")
    print("="*80)
    for cat, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat:40s} {total:6,}")
    
    print(f"\n{'TOTAL':40s} {sum(category_totals.values()):6,}")
    
    return category_totals

def compare_with_custom():
    """Compare baseline vs custom."""
    
    # From your diagnostic output
    custom_breakdown = {
        'Encoder VSN': 5115,
        'Static Context GRNs': 2640,
        'LSTM Encoder': 2176,
        'LSTM Decoder': 2176,
        'Static Enrichment': 1136,
        'Attention': 1088,
        'Decoder VSN': 951,  # You have this!
        'Output Layers': 695,
        'Post-LSTM Gates': 576,
        'Input Embeddings': 160 + 32,  # encoder + decoder
        'Static Variable Selection': 0,  # You don't have this
    }
    
    print("\n" + "="*80)
    print("CUSTOM MODEL BREAKDOWN (for comparison)")
    print("="*80)
    for comp, params in sorted(custom_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"{comp:40s} {params:6,}")
    print(f"\n{'TOTAL':40s} {sum(custom_breakdown.values()):6,}")

if __name__ == '__main__':
    baseline_breakdown = analyze_baseline_checkpoint()
    compare_with_custom()
    
    print("\n" + "="*80)
    print("NEXT STEP")
    print("="*80)
    print("Compare the two breakdowns above to identify exact differences.")
    print("Pay special attention to:")
    print("  1. LSTM param counts (yours: 2.2K each, baseline: ?)")
    print("  2. Static VSN (yours: 0, baseline: ?)")
    print("  3. Decoder VSN (yours: 951, baseline: ?)")
    print("  4. Any 'Other' category items in baseline")
