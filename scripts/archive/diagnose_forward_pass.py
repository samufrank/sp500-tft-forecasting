#!/usr/bin/env python3
"""
Diagnostic script to isolate forward pass differences between custom TFT and baseline.

Tests:
1. Parameter count verification
2. Initial loss with random weights
3. Gradient flow through different components
4. VSN output distributions
5. LSTM hidden state initialization

Usage: python diagnose_forward_pass.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.tft_model import TemporalFusionTransformer

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_dummy_batch(batch_size=64, encoder_len=20, decoder_len=1, 
                       num_encoder_feats=7, num_decoder_feats=1):
    """Create dummy batch matching your data format."""
    encoder_cont = torch.randn(batch_size, encoder_len, num_encoder_feats)
    decoder_cont = torch.randn(batch_size, decoder_len, num_decoder_feats)
    targets = torch.randn(batch_size, decoder_len)
    return encoder_cont, decoder_cont, targets

def analyze_parameter_distribution(model):
    """Analyze weight initialization statistics."""
    print("\n" + "="*80)
    print("PARAMETER INITIALIZATION ANALYSIS")
    print("="*80)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            std = param.data.std().item()
            mean = param.data.mean().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            print(f"{name:50s} | std={std:.6f} | mean={mean:.6f} | range=[{min_val:.3f}, {max_val:.3f}]")

def test_forward_pass_detailed(model, encoder_cont, decoder_cont, targets):
    """Test forward pass with detailed intermediate outputs."""
    print("\n" + "="*80)
    print("FORWARD PASS DETAILED ANALYSIS")
    print("="*80)
    
    # Hook to capture intermediate values
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                # VSN returns (output_tensor, weights_tensor)
                out_tensor = output[0]
                weights_tensor = output[1]
                
                if torch.is_tensor(out_tensor):
                    activations[name] = {
                        'output': out_tensor.detach().cpu(),
                        'weights': weights_tensor.detach().cpu() if torch.is_tensor(weights_tensor) else None
                    }
                else:
                    # Store as-is if not tensor (e.g., LSTM hidden states)
                    activations[name] = output
            else:
                # Single tensor output
                if torch.is_tensor(output):
                    activations[name] = output.detach().cpu()
                else:
                    activations[name] = output
        return hook
    
    # Register hooks
    hooks = []
    hooks.append(model.encoder_variable_selection.register_forward_hook(get_activation('encoder_vsn')))
    hooks.append(model.decoder_variable_selection.register_forward_hook(get_activation('decoder_vsn')))
    hooks.append(model.lstm_encoder.register_forward_hook(get_activation('lstm_encoder')))
    hooks.append(model.multihead_attention.register_forward_hook(get_activation('attention')))
    hooks.append(model.output_layer.register_forward_hook(get_activation('output_layer')))
    
    # Forward pass
    with torch.no_grad():
        predictions = model(encoder_cont, decoder_cont)
        
        # Compute loss manually
        loss = model.loss_fn(predictions, targets)
        
        # Package outputs to match expected format
        outputs = {'loss': loss, 'prediction': predictions}
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze activations
    print("\nIntermediate Activation Statistics:")
    print("-" * 80)
    for name, activation in activations.items():
        if isinstance(activation, dict):
            output = activation['output']
            weights = activation['weights']
            print(f"\n{name}:")
            print(f"  output - shape: {output.shape}, std: {output.std():.6f}, mean: {output.mean():.6f}")
            if weights is not None:
                print(f"  weights - shape: {weights.shape}, std: {weights.std():.6f}, mean: {weights.mean():.6f}")
        else:
            if isinstance(activation, tuple):
                # LSTM returns (output, (h_n, c_n))
                output, (h_n, c_n) = activation
                print(f"\n{name}:")
                print(f"  output - shape: {output.shape}, std: {output.std():.6f}, mean: {output.mean():.6f}")
                print(f"  hidden - shape: {h_n.shape}, std: {h_n.std():.6f}, mean: {h_n.mean():.6f}")
                print(f"  cell - shape: {c_n.shape}, std: {c_n.std():.6f}, mean: {c_n.mean():.6f}")
            else:
                print(f"\n{name}:")
                print(f"  shape: {activation.shape}, std: {activation.std():.6f}, mean: {activation.mean():.6f}")
    
    return outputs

def test_gradient_flow(model, encoder_cont, decoder_cont, targets):
    """Test gradient magnitudes through network."""
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    # Enable gradients
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward + backward
    predictions = model(encoder_cont, decoder_cont)
    loss = model.loss_fn(predictions, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Analyze gradients
    print("\nGradient Norms by Component:")
    print("-" * 80)
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms[name] = grad_norm
            if grad_norm > 0.01:  # Only show significant gradients
                print(f"{name:50s} | norm={grad_norm:.6f}")
    
    # Summary stats
    print("\nGradient Summary:")
    print(f"  Total components with gradients: {len([g for g in gradient_norms.values() if g > 0])}")
    print(f"  Max gradient norm: {max(gradient_norms.values()):.6f}")
    print(f"  Mean gradient norm: {np.mean(list(gradient_norms.values())):.6f}")
    
    return gradient_norms

def compare_quantile_outputs(predictions, targets):
    """Analyze quantile prediction behavior."""
    print("\n" + "="*80)
    print("QUANTILE OUTPUT ANALYSIS")
    print("="*80)
    
    # Check quantile ordering
    print("\nSample predictions (first 5 samples, all quantiles):")
    print(predictions[:5, 0, :])
    
    # Check if quantiles are ordered
    quantile_order_violations = 0
    for i in range(predictions.shape[0]):
        for t in range(predictions.shape[1]):
            pred_quantiles = predictions[i, t, :]
            if not torch.all(pred_quantiles[:-1] <= pred_quantiles[1:]):
                quantile_order_violations += 1
    
    total_predictions = predictions.shape[0] * predictions.shape[1]
    print(f"\nQuantile ordering violations: {quantile_order_violations}/{total_predictions} " +
          f"({100*quantile_order_violations/total_predictions:.1f}%)")
    print("(Note: High violation rate expected at initialization with random weights)")
    
    # Distribution stats
    print(f"\nPrediction statistics:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    # Target stats
    print(f"\nTarget statistics:")
    print(f"  Shape: {targets.shape}")
    print(f"  Mean: {targets.mean():.6f}")
    print(f"  Std: {targets.std():.6f}")
    print(f"  Range: [{targets.min():.6f}, {targets.max():.6f}]")

def main():
    print("="*80)
    print("CUSTOM TFT FORWARD PASS DIAGNOSTIC")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create model with baseline config
    print("\nCreating model with baseline configuration:")
    print("  - num_encoder_features: 7 (5 base + 2 auto-added)")
    print("  - num_decoder_features: 1")
    print("  - hidden_size: 16")
    print("  - num_attention_heads: 2")
    print("  - dropout: 0.25")
    
    model = TemporalFusionTransformer(
        num_encoder_features=7,  # CORRECTED: 5 base + relative_time_idx + encoder_length
        num_decoder_features=1,
        hidden_size=16,
        hidden_continuous_size=16,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.25,
        max_encoder_length=20,
        max_prediction_length=1,
        learning_rate=0.0005,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter count:")
    print(f"  Total: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1000:.1f}K)")
    print(f"  Target (baseline - static_vsn): ~21.4K")
    print(f"  Baseline actual: 22.6K")
    
    # Create dummy batch
    encoder_cont, decoder_cont, targets = create_dummy_batch()
    print(f"\nBatch shapes:")
    print(f"  encoder_cont: {encoder_cont.shape}")
    print(f"  decoder_cont: {decoder_cont.shape}")
    print(f"  targets: {targets.shape}")
    
    # Run diagnostics
    analyze_parameter_distribution(model)
    
    outputs = test_forward_pass_detailed(model, encoder_cont, decoder_cont, targets)
    
    print("\n" + "="*80)
    print("FINAL OUTPUT ANALYSIS")
    print("="*80)
    print(f"\nLoss: {outputs['loss'].item():.6f}")
    print(f"  Expected (baseline epoch 0): ~0.69 train, ~0.60 val")
    print(f"  Your previous custom: 0.22 (TOO LOW)")
    
    predictions = outputs['prediction']
    compare_quantile_outputs(predictions, targets)
    
    gradient_norms = test_gradient_flow(model, encoder_cont, decoder_cont, targets)
    
    print("\n" + "="*80)
    print("KEY DIAGNOSTICS SUMMARY")
    print("="*80)
    print(f"\n✓ Parameter count: {total_params/1000:.1f}K (target: ~21.4K, baseline: 22.6K)")
    print(f"✓ Initial loss: {outputs['loss'].item():.4f} (expected: ~0.60-0.70)")
    print(f"✓ Prediction std: {predictions.std():.6f} (expected: ~0.03)")
    print(f"✓ Output layer gradient: {gradient_norms.get('output_layer.weight', 0):.6f} (expected: ~0.62)")
    
    loss_ratio = outputs['loss'].item() / 0.69
    pred_std_ratio = predictions.std().item() / 0.033
    
    print(f"\nRatios vs baseline:")
    print(f"  Loss ratio: {loss_ratio:.2f}x (should be ~1.0x)")
    print(f"  Pred std ratio: {pred_std_ratio:.2f}x (should be ~1.0x)")
    
    if loss_ratio < 0.5:
        print("\n⚠️  CRITICAL: Loss is too low - check QuantileLoss implementation!")
    if pred_std_ratio < 0.3:
        print("⚠️  CRITICAL: Predictions too confident - likely related to loss calculation!")

if __name__ == "__main__":
    main()