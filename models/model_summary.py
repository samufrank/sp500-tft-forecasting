"""
Model summary utilities for debugging and validation.

Prints parameter counts in pytorch-forecasting style for easy comparison.
"""

import torch.nn as nn
from typing import Dict, Tuple


def get_parameter_summary(model: nn.Module) -> Tuple[Dict[str, int], int, int]:
    """
    Get detailed parameter counts for each module.
    
    Args:
        model: PyTorch model
        
    Returns:
        param_dict: Dict mapping module names to parameter counts
        trainable_params: Total trainable parameters
        total_params: Total parameters (trainable + non-trainable)
    """
    param_dict = {}
    trainable_params = 0
    total_params = 0
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        param_dict[name] = module_params
        trainable_params += module_trainable
        total_params += module_params
    
    return param_dict, trainable_params, total_params


def print_parameter_summary(model: nn.Module, show_details: bool = True):
    """
    Print parameter summary in pytorch-forecasting style.
    
    Args:
        model: PyTorch model
        show_details: If True, print per-module breakdown
    """
    param_dict, trainable_params, total_params = get_parameter_summary(model)
    
    if show_details:
        print("\n" + "="*80)
        print("MODEL PARAMETER SUMMARY")
        print("="*80)
        print(f"{'Name':<40} {'Type':<35} {'Params':>10}")
        print("-"*80)
        
        for idx, (name, module) in enumerate(model.named_children()):
            module_type = module.__class__.__name__
            module_params = param_dict[name]
            
            # Format parameter count with K suffix if >= 1000
            if module_params >= 1000:
                params_str = f"{module_params / 1000:.1f} K"
            else:
                params_str = str(module_params)
            
            print(f"{idx:<3} | {name:<37} | {module_type:<32} | {params_str:>7}")
        
        print("-"*80)
    
    # Summary totals
    trainable_k = trainable_params / 1000
    total_k = total_params / 1000
    total_mb = total_params * 4 / (1024 * 1024)  # Assume float32 (4 bytes)
    
    print(f"{trainable_k:.1f} K    Trainable params")
    print(f"{(total_params - trainable_params) / 1000:.1f} K         Non-trainable params")
    print(f"{total_k:.1f} K    Total params")
    print(f"{total_mb:.3f}     Total estimated model params size (MB)")
    print("="*80 + "\n")
    
    return trainable_params, total_params


def compare_models(model1: nn.Module, model2: nn.Module, 
                   model1_name: str = "Model 1", model2_name: str = "Model 2"):
    """
    Compare parameter counts between two models.
    
    Args:
        model1: First model (e.g., custom TFT)
        model2: Second model (e.g., pytorch-forecasting TFT)
        model1_name: Name for first model
        model2_name: Name for second model
    """
    print("\n" + "="*80)
    print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
    print("="*80)
    
    params1, trainable1, total1 = get_parameter_summary(model1)
    params2, trainable2, total2 = get_parameter_summary(model2)
    
    # Find common modules
    common_modules = set(params1.keys()) & set(params2.keys())
    only_in_1 = set(params1.keys()) - set(params2.keys())
    only_in_2 = set(params2.keys()) - set(params1.keys())
    
    # Print comparison table
    print(f"\n{'Module':<40} {model1_name:>15} {model2_name:>15} {'Diff':>10}")
    print("-"*80)
    
    for name in sorted(common_modules):
        p1 = params1[name]
        p2 = params2[name]
        diff = p1 - p2
        
        p1_str = f"{p1/1000:.1f}K" if p1 >= 1000 else str(p1)
        p2_str = f"{p2/1000:.1f}K" if p2 >= 1000 else str(p2)
        diff_str = f"{diff:+d}" if abs(diff) < 1000 else f"{diff/1000:+.1f}K"
        
        print(f"{name:<40} {p1_str:>15} {p2_str:>15} {diff_str:>10}")
    
    if only_in_1:
        print(f"\nOnly in {model1_name}:")
        for name in sorted(only_in_1):
            p1 = params1[name]
            p1_str = f"{p1/1000:.1f}K" if p1 >= 1000 else str(p1)
            print(f"  {name:<40} {p1_str:>15}")
    
    if only_in_2:
        print(f"\nOnly in {model2_name}:")
        for name in sorted(only_in_2):
            p2 = params2[name]
            p2_str = f"{p2/1000:.1f}K" if p2 >= 1000 else str(p2)
            print(f"  {name:<40} {p2_str:>15}")
    
    # Total comparison
    print("-"*80)
    print(f"{'TOTAL':<40} {total1/1000:>14.1f}K {total2/1000:>14.1f}K {(total1-total2)/1000:>9.1f}K")
    print(f"{'Difference:':<40} {'':<15} {'':<15} {(total1-total2)/total2*100:>9.1f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    from models.tft_model import TemporalFusionTransformer
    
    model = TemporalFusionTransformer(
        num_encoder_features=4,
        num_decoder_features=2,  # With auto-added features
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.25,
        max_encoder_length=20,
        max_prediction_length=1,
    )
    
    print_parameter_summary(model, show_details=True)
