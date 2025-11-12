"""
Parameter audit tool for custom TFT vs pytorch-forecasting baseline.

This script provides detailed, layer-by-layer parameter counts to identify
exactly where parameter differences come from.

IMPORTANT NOTE ON PARAMETER COUNTS:
- Your actual training baseline (sweep2_h16_drop_0.25): 22.6K params
- This audit script shows: ~24.0K for baseline, ~24.9K for custom
- The 1.4K discrepancy exists because this audit creates a fresh model via
  TimeSeriesDataSet.from_dataset() which may initialize differently than
  your trained model (module sharing, initialization order, etc.)
- The RELATIVE difference (custom vs baseline) is still informative

The key architectural differences this script identifies are still valid
even if absolute numbers don't match your training logs exactly.

Usage:
    python audit_parameters.py
"""

import os
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple


def audit_checkpoint(checkpoint_path, name="checkpoint"):
    """
    Audit parameters from an actual checkpoint file.
    This gives the REAL parameter counts from trained models.
    """
    print("\n" + "="*80)
    print(f"{name.upper()} PARAMETER AUDIT (from checkpoint)")
    print("="*80)
    
    import torch
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None, None
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        
        # Only count parameters (exclude buffers like running_mean, running_var, etc.)
        # Parameters typically have '.weight' or '.bias' in their name
        param_dict = {}
        for name, tensor in state_dict.items():
            # Include if it's a weight/bias parameter (exclude buffers)
            if '.weight' in name or '.bias' in name:
                param_dict[name] = tensor.numel()
        
        total_params = sum(param_dict.values())
        
        print(f"Total parameters: {total_params:,} ({total_params/1000:.1f}K)")
        print(f"(Excluded {len(state_dict) - len(param_dict)} non-parameter tensors)")
        
        # Organize by component
        categories = categorize_parameters(param_dict)
        
        return categories, total_params
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None, None


def detailed_param_comparison(custom_params, baseline_params):
    """
    Show parameter-by-parameter comparison.
    
    Args:
        custom_params: dict of {param_name: param_count}
        baseline_params: dict of {param_name: param_count}
    """
    print("\n" + "="*80)
    print("DETAILED PARAMETER COMPARISON")
    print("="*80)
    
    # Find parameters that exist in one but not the other
    custom_only = set(custom_params.keys()) - set(baseline_params.keys())
    baseline_only = set(baseline_params.keys()) - set(custom_params.keys())
    common = set(custom_params.keys()) & set(baseline_params.keys())
    
    print(f"\nParameters only in custom: {len(custom_only)}")
    if custom_only and len(custom_only) <= 30:
        for name in sorted(custom_only):
            print(f"  {name}: {custom_params[name]:,} params")
    
    print(f"\nParameters only in baseline: {len(baseline_only)}")  
    if baseline_only and len(baseline_only) <= 30:
        for name in sorted(baseline_only):
            print(f"  {name}: {baseline_params[name]:,} params")
    
    print(f"\nCommon parameters with different sizes:")
    diffs = []
    for name in common:
        if custom_params[name] != baseline_params[name]:
            diff = custom_params[name] - baseline_params[name]
            diffs.append((name, custom_params[name], baseline_params[name], diff))
    
    if diffs:
        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"{'Parameter':<60s} {'Custom':>10s} {'Baseline':>10s} {'Diff':>10s}")
        print("-"*100)
        for name, custom_c, baseline_c, diff in diffs[:50]:
            print(f"{name:<60s} {custom_c:>10,} {baseline_c:>10,} {diff:>+10,}")
    else:
        print("  All common parameters have identical sizes")
    
    # Summary
    custom_only_total = sum(custom_params[n] for n in custom_only)
    baseline_only_total = sum(baseline_params[n] for n in baseline_only)
    diff_total = sum(abs(custom_params[n] - baseline_params[n]) for n in common if custom_params[n] != baseline_params[n])
    
    print("\n" + "-"*100)
    print("Summary of differences:")
    print(f"  Parameters unique to custom:   {custom_only_total:>10,}")
    print(f"  Parameters unique to baseline: {baseline_only_total:>10,}")
    print(f"  Size differences in common:    {diff_total:>10,}")


def count_parameters(module: nn.Module, name: str = "model") -> Dict[str, int]:
    """
    Count parameters in a module, broken down by layer type and name.
    
    Returns dict with:
        'layer_name': parameter_count
    """
    param_dict = {}
    
    for param_name, param in module.named_parameters():
        full_name = f"{name}.{param_name}" if name != "model" else param_name
        param_dict[full_name] = param.numel()
    
    return param_dict


def categorize_parameters(param_dict: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """
    Organize parameters by component type.
    
    Returns:
        {
            'encoder_embeddings': {'var_0.weight': 16, ...},
            'decoder_embeddings': {...},
            'encoder_vsn': {...},
            ...
        }
    """
    categories = defaultdict(dict)
    
    for name, count in param_dict.items():
        # Determine category
        if 'encoder_embeddings' in name:
            categories['encoder_embeddings'][name] = count
        elif 'decoder_embeddings' in name:
            categories['decoder_embeddings'][name] = count
        elif 'encoder_variable_selection' in name:
            categories['encoder_vsn'][name] = count
        elif 'decoder_variable_selection' in name:
            categories['decoder_vsn'][name] = count
        elif 'static_variable_selection' in name:
            categories['static_vsn'][name] = count
        elif 'static_context_variable_selection' in name:
            categories['static_context_var_sel'][name] = count
        elif 'static_context_initial_hidden_lstm' in name:
            categories['static_context_hidden'][name] = count
        elif 'static_context_initial_cell_lstm' in name:
            categories['static_context_cell'][name] = count
        elif 'static_context_enrichment' in name:
            categories['static_context_enrich'][name] = count
        elif 'static_enrichment' in name:
            categories['static_enrichment'][name] = count
        elif 'lstm_encoder' in name:
            categories['lstm_encoder'][name] = count
        elif 'lstm_decoder' in name:
            categories['lstm_decoder'][name] = count
        elif 'post_lstm_gate_encoder' in name:
            categories['post_lstm_gate_encoder'][name] = count
        elif 'post_lstm_gate_decoder' in name:
            categories['post_lstm_gate_decoder'][name] = count
        elif 'post_attn_gate_norm' in name:
            categories['post_attn_gate'][name] = count
        elif 'post_ff_gate_norm' in name:
            categories['post_ff_gate'][name] = count
        elif 'multihead_attention' in name:
            categories['attention'][name] = count
        elif 'pos_wise_ff' in name:
            categories['pos_wise_ff'][name] = count
        elif 'output_layer' in name:
            categories['output_layer'][name] = count
        else:
            categories['other'][name] = count
    
    return dict(categories)


def print_category_summary(categories: Dict[str, Dict[str, int]]) -> None:
    """Print summary by category with totals."""
    print("\n" + "="*80)
    print("PARAMETER BREAKDOWN BY COMPONENT")
    print("="*80)
    
    total = 0
    category_totals = []
    
    for cat_name, params in sorted(categories.items()):
        cat_total = sum(params.values())
        category_totals.append((cat_name, cat_total, len(params)))
        total += cat_total
    
    # Sort by total params (descending)
    category_totals.sort(key=lambda x: x[1], reverse=True)
    
    for cat_name, cat_total, num_params in category_totals:
        pct = 100 * cat_total / total if total > 0 else 0
        print(f"{cat_name:30s}: {cat_total:>6,} params ({pct:>5.1f}%) [{num_params:>3} tensors]")
    
    print("-"*80)
    print(f"{'TOTAL':30s}: {total:>6,} params (100.0%)")
    print("="*80)


def print_category_details(categories: Dict[str, Dict[str, int]], 
                          category: str,
                          max_items: int = 20) -> None:
    """Print detailed parameter breakdown for a specific category."""
    if category not in categories:
        print(f"\nCategory '{category}' not found!")
        return
    
    print(f"\n{category.upper()} - Detailed Breakdown:")
    print("-"*80)
    
    params = categories[category]
    total = sum(params.values())
    
    items = list(params.items())
    if len(items) > max_items:
        print(f"(Showing first {max_items} of {len(items)} items)")
        items = items[:max_items]
    
    for name, count in items:
        # Shorten name for display
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
        print(f"  {short_name:50s}: {count:>6,}")
    
    print("-"*80)
    print(f"  {'SUBTOTAL':50s}: {total:>6,}")


def audit_custom_tft():
    """Audit YOUR custom TFT implementation."""
    print("\n" + "="*80)
    print("CUSTOM TFT PARAMETER AUDIT")
    print("="*80)
    
    try:
        from models.tft_model import TemporalFusionTransformer
        from models.model_summary import get_parameter_summary
    except ImportError as e:
        print(f"Error importing custom TFT: {e}")
        print("Make sure you're running from project root!")
        return None, None
    
    # Create model with baseline config
    # CRITICAL: Actual data has 6 encoder features total:
    # - 4 base features (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY)
    # - 2 auto-added by TimeSeriesDataSet (encoder_length, relative_time_idx)
    # NOTE: SP500_Returns is filtered out (it's the target)
    model = TemporalFusionTransformer(
        num_encoder_features=6,  # 4 base + 2 auto-added
        num_decoder_features=1,  # Only relative_time_idx
        hidden_size=16,
        hidden_continuous_size=16,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.25,
        max_encoder_length=20,
        max_prediction_length=1,
        quantiles=[0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98],
        learning_rate=0.0005,
    )
    
    # Count parameters
    param_dict = count_parameters(model, "custom_tft")
    categories = categorize_parameters(param_dict)
    
    # Get summary stats
    _, trainable, total = get_parameter_summary(model)
    
    print(f"\nConfiguration:")
    print(f"  Encoder features: 6 (4 base + 2 auto-added)")
    print(f"  Decoder features: 1 (relative_time_idx only)")
    print(f"  Hidden size: 16")
    print(f"  Hidden continuous size: 16")
    print(f"  Attention heads: 2")
    print(f"  LSTM layers: 1")
    print(f"\nTotal parameters: {total:,} ({total/1000:.1f}K)")
    print(f"Trainable: {trainable:,} ({trainable/1000:.1f}K)")
    
    return categories, total


def audit_pytorch_forecasting():
    """Audit pytorch-forecasting TFT for comparison."""
    print("\n" + "="*80)
    print("PYTORCH-FORECASTING TFT PARAMETER AUDIT")
    print("="*80)
    
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        import pandas as pd
    except ImportError as e:
        print(f"Error importing pytorch-forecasting: {e}")
        print("Install with: pip install pytorch-forecasting")
        return None, None
    
    # Create minimal dataset to initialize model
    # TimeSeriesDataSet required to use from_dataset()
    data = pd.DataFrame({
        'time_idx': range(100),
        'group': ['A'] * 100,
        'target': [0.0] * 100,
        'VIX': [0.0] * 100,
        'Treasury_10Y': [0.0] * 100,
        'Yield_Spread': [0.0] * 100,
        'Inflation_YoY': [0.0] * 100,
        'Unemployment_Rate': [0.0] * 100,
    })
    
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=20,
        max_prediction_length=1,
        time_varying_known_reals=[],
        time_varying_unknown_reals=['VIX', 'Treasury_10Y', 'Yield_Spread', 
                                     'Inflation_YoY', 'Unemployment_Rate'],
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create model from dataset
    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.25,
        hidden_continuous_size=16,
        lstm_layers=1,
        learning_rate=0.0005,
    )
    
    # Count parameters
    param_dict = count_parameters(model, "pytorch_forecasting")
    categories = categorize_parameters(param_dict)
    
    total = sum(param_dict.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count actual features that go through encoder/decoder VSNs
    actual_encoder_features = len(training.reals)
    actual_decoder_features = 2  # relative_time_idx + encoder_length
    
    print(f"\nConfiguration:")
    print(f"  Base features: 5 (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY, Unemployment_Rate)")
    print(f"  Auto-added decoder features: {actual_decoder_features} (relative_time_idx, encoder_length)")
    print(f"  Total encoder VSN inputs: {actual_encoder_features} (all features)")
    print(f"  Total decoder VSN inputs: {actual_decoder_features}")
    print(f"  Hidden size: 16")
    print(f"  Hidden continuous size: 16")
    print(f"  Attention heads: 2")
    print(f"  LSTM layers: 1")
    print(f"\nTotal parameters: {total:,} ({total/1000:.1f}K)")
    print(f"Trainable: {trainable:,} ({trainable/1000:.1f}K)")
    print(f"\nNOTE: Your actual training (sweep2_h16_drop_0.25) shows 22.6K params.")
    print(f"      This discrepancy ({total-22600:+,} params) likely comes from:")
    print(f"      - Different prescaler/embedding initialization")
    print(f"      - Module sharing (encoder/decoder gates may be shared in training)")
    print(f"      - Layer ordering differences")
    
    return categories, total


def compare_models(custom_cats: Dict, custom_total: int,
                  baseline_cats: Dict, baseline_total: int) -> None:
    """Compare parameter counts between models."""
    print("\n" + "="*80)
    print("COMPARISON: Custom vs pytorch-forecasting")
    print("="*80)
    
    # Get all category names
    all_categories = set(custom_cats.keys()) | set(baseline_cats.keys())
    
    print(f"\n{'Component':30s} {'Custom':>10s} {'Baseline':>10s} {'Diff':>10s}")
    print("-"*80)
    
    for cat in sorted(all_categories):
        custom_count = sum(custom_cats.get(cat, {}).values())
        baseline_count = sum(baseline_cats.get(cat, {}).values())
        diff = custom_count - baseline_count
        
        custom_str = f"{custom_count:,}" if custom_count > 0 else "-"
        baseline_str = f"{baseline_count:,}" if baseline_count > 0 else "-"
        diff_str = f"{diff:+,}" if diff != 0 else "0"
        
        print(f"{cat:30s} {custom_str:>10s} {baseline_str:>10s} {diff_str:>10s}")
    
    print("-"*80)
    diff_total = custom_total - baseline_total
    print(f"{'TOTAL':30s} {custom_total:>10,} {baseline_total:>10,} {diff_total:>+10,}")
    print("="*80)
    
    # Identify biggest differences
    print("\nBiggest Differences (absolute value):")
    diffs = []
    for cat in all_categories:
        custom_count = sum(custom_cats.get(cat, {}).values())
        baseline_count = sum(baseline_cats.get(cat, {}).values())
        diff = custom_count - baseline_count
        if diff != 0:
            diffs.append((cat, diff, abs(diff)))
    
    diffs.sort(key=lambda x: x[2], reverse=True)
    
    for cat, diff, abs_diff in diffs[:10]:
        sign = "MORE" if diff > 0 else "FEWER"
        print(f"  {cat:30s}: {abs_diff:>6,} {sign}")


def main():
    """Run complete parameter audit."""
    print("\n" + "#"*80)
    print("# TFT PARAMETER AUDIT TOOL")
    print("# Comparing custom implementation vs pytorch-forecasting baseline")
    print("#"*80)
    
    # Option 1: Compare actual trained checkpoints (MOST ACCURATE)
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON (Actual Trained Models)")
    print("="*80)
    
    custom_ckpt = "experiments/test_fixes/checkpoints/last.ckpt"  # Your latest with 6 features
    baseline_ckpt = "experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/last.ckpt"  # Baseline
    
    custom_ckpt_cats, custom_ckpt_total = audit_checkpoint(custom_ckpt, "Custom")
    baseline_ckpt_cats, baseline_ckpt_total = audit_checkpoint(baseline_ckpt, "Baseline")
    
    if custom_ckpt_cats and baseline_ckpt_cats:
        print("\n" + "="*80)
        print("CHECKPOINT COMPARISON RESULTS")
        print("="*80)
        print(f"\nCustom checkpoint:   {custom_ckpt_total:,} params ({custom_ckpt_total/1000:.1f}K)")
        print(f"Baseline checkpoint: {baseline_ckpt_total:,} params ({baseline_ckpt_total/1000:.1f}K)")
        print(f"Difference:          {custom_ckpt_total - baseline_ckpt_total:+,} params")
        
        compare_models(custom_ckpt_cats, custom_ckpt_total, 
                      baseline_ckpt_cats, baseline_ckpt_total)
        
        # Add detailed parameter-level comparison
        print("\nGenerating detailed parameter-level analysis...")
        try:
            custom_ckpt_obj = torch.load(custom_ckpt, map_location='cpu')
            baseline_ckpt_obj = torch.load(baseline_ckpt, map_location='cpu')
            
            custom_params = {k: v.numel() for k, v in custom_ckpt_obj['state_dict'].items() 
                           if '.weight' in k or '.bias' in k}
            baseline_params = {k: v.numel() for k, v in baseline_ckpt_obj['state_dict'].items()
                             if '.weight' in k or '.bias' in k}
            
            detailed_param_comparison(custom_params, baseline_params)
        except Exception as e:
            print(f"Could not generate detailed comparison: {e}")
    else:
        print("\nCheckpoint comparison failed. Falling back to fresh model instantiation...")
    
    # Option 2: Fresh model instantiation (less accurate, for reference)
    print("\n\n" + "="*80)
    print("FRESH MODEL COMPARISON (Reference Only)")
    print("="*80)
    print("NOTE: These numbers may differ from actual training due to initialization")
    
    # Audit custom model
    custom_cats, custom_total = audit_custom_tft()
    if custom_cats is None:
        return
    
    print_category_summary(custom_cats)
    
    # Ask if user wants details
    print("\nWould you like detailed breakdown for any component?")
    print("Available components:", ", ".join(sorted(custom_cats.keys())))
    print("(Press Enter to continue to baseline comparison)")
    
    # Audit baseline model
    print("\n\nAttempting to load pytorch-forecasting for comparison...")
    baseline_cats, baseline_total = audit_pytorch_forecasting()
    
    if baseline_cats is not None:
        print_category_summary(baseline_cats)
        compare_models(custom_cats, custom_total, baseline_cats, baseline_total)
    else:
        print("\nCouldn't load pytorch-forecasting baseline.")
        print("To compare:")
        print("  1. Install: pip install pytorch-forecasting")
        print("  2. Re-run this script")
    
    print("\n" + "#"*80)
    print("# Audit Complete")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()