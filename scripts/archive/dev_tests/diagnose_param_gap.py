#!/usr/bin/env python3
"""
Diagnose the 2.7K parameter gap between custom (18.7K) and expected (21.4K).

Expected breakdown (baseline):
- Encoder VSN: ~400 (5 features)  ✓ We have this
- Decoder VSN: ~0 (no decoder features) ✓ We have this
- Static VSN: ~1,200 (baseline has, we don't) - explains 1.2K
- Embeddings: ~1,280 (5 features × 16 × 16)
- LSTM encoder/decoder: ~4,160 each
- Attention: ~4,672
- Static context GRNs: ~3,000
- Output layers: ~2,000

Where are we missing the other ~1.5K params?
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from models.tft_model import TemporalFusionTransformer

def count_params_by_component(model):
    """Detailed parameter breakdown."""
    
    print("="*80)
    print("CUSTOM MODEL DETAILED PARAMETER COUNT")
    print("="*80)
    
    # Count parameters in each named module
    module_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                module_params[name] = params
    
    # Group by component
    categories = {
        'Encoder Embeddings (prescalers)': [],
        'Decoder Embeddings': [],
        'Static Context GRNs': [],
        'Encoder VSN': [],
        'Decoder VSN': [],
        'LSTM Encoder': [],
        'LSTM Decoder': [],
        'Post-LSTM Gates': [],
        'Attention': [],
        'Static Enrichment': [],
        'Output Layers': [],
    }
    
    for name, params in module_params.items():
        if 'encoder_embeddings' in name:
            categories['Encoder Embeddings (prescalers)'].append((name, params))
        elif 'decoder_embeddings' in name:
            categories['Decoder Embeddings'].append((name, params))
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
        elif 'attention' in name:
            categories['Attention'].append((name, params))
        elif 'static_enrichment' in name:
            categories['Static Enrichment'].append((name, params))
        elif 'output_layer' in name or 'pre_output_gate' in name:
            categories['Output Layers'].append((name, params))
    
    # Print by category
    total_by_cat = {}
    for cat, modules in categories.items():
        if modules:
            print(f"\n{cat}:")
            cat_total = 0
            for name, params in sorted(modules, key=lambda x: x[1], reverse=True):
                print(f"  {name:70s} {params:6,}")
                cat_total += params
            print(f"  {'SUBTOTAL':70s} {cat_total:6,}")
            total_by_cat[cat] = cat_total
        else:
            total_by_cat[cat] = 0
    
    print("\n" + "="*80)
    print("SUMMARY BY COMPONENT")
    print("="*80)
    for cat, total in sorted(total_by_cat.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat:40s} {total:6,}")
    
    model_total = sum(p.numel() for p in model.parameters())
    print(f"\n{'TOTAL MODEL PARAMETERS':40s} {model_total:6,}")
    
    return total_by_cat, model_total

def compare_with_baseline():
    """Compare against known baseline counts."""
    
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE (22.6K)")
    print("="*80)
    
    # Known from checkpoint analysis
    baseline_breakdown = {
        'Encoder VSN flattened_grn.fc1': 400,  # 5*16*5
        'Static VSN (we don\'t have)': 1200,  # Baseline has this
        'Embeddings (5 vars × 16×16)': 1280,  # 5 * (16*16)
        'LSTM Encoder': 4160,  # LSTM with h=16
        'LSTM Decoder': 4160,
        'Attention (h=16, heads=4)': 4672,  # MultiheadAttention
        'Static Context GRNs (4 GRNs)': 3000,  # Rough estimate
        'Output layers': 2000,  # Rough estimate
    }
    
    print("\nBaseline components (estimated):")
    baseline_total = 0
    for comp, params in baseline_breakdown.items():
        print(f"  {comp:40s} {params:6,}")
        baseline_total += params
    print(f"  {'TOTAL':40s} {baseline_total:6,}")
    
    print(f"\nBaseline from checkpoint: 22,600")
    print(f"Estimated breakdown total: {baseline_total:,}")
    print(f"Difference: {22600 - baseline_total:,} (rounding/minor components)")

if __name__ == '__main__':
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
    
    custom_breakdown, custom_total = count_params_by_component(model)
    compare_with_baseline()
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print(f"Custom model: {custom_total:,} params")
    print(f"Expected (baseline - static_vsn): ~21,400 params")
    print(f"Missing: {21400 - custom_total:,} params")
    print("\nPossible causes:")
    print("1. Missing static context initialization GRNs?")
    print("2. Missing gate/norm layers?")
    print("3. Different LSTM configuration?")
    print("4. Check your model initialization against baseline source")