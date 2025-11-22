"""
Callback for monitoring collapse behavior during TFT training.

Tracks prediction diversity, gradient flow, and weight statistics to understand
when and how models collapse to constant predictions.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import json
import os
from pathlib import Path


class CollapseMonitor(Callback):
    """
    Monitor training dynamics to detect and analyze prediction collapse.
    
    Tracks:
    - Prediction diversity (std, range, sign distribution)
    - Gradient magnitudes by layer
    - Weight statistics
    - Variable selection network outputs
    - Attention weight entropy
    """
    
    def __init__(self, val_dataloader, log_dir, log_every_n_epochs=1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        
        # Buffer to store gradients captured during training
        self._current_epoch_gradients = {}
        
        self.history = {
            'epoch': [],
            'prediction_std': [],
            'prediction_range': [],
            'prediction_mean': [],
            'pct_positive': [],
            'pct_negative': [],
            'num_unique_predictions': [],
            'gradient_norms': {},
            'weight_norms': {},
            'weight_stds': {},
            'vsn_output_std': {},
            'attention_entropy': [],
        }
        
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        """Capture gradients before they're cleared by optimizer."""
        # Store gradient norms for this batch
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if name not in self._current_epoch_gradients:
                    self._current_epoch_gradients[name] = []
                    
                self._current_epoch_gradients[name].append(grad_norm)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset temporal state in loss function before validation epoch."""
        # Reset directional penalty buffer and temporal consistency state
        # This ensures validation statistics are computed on fresh, sequential data
        print(f"[DEBUG CollapseMonitor] on_validation_epoch_start called for epoch {trainer.current_epoch}")
        if hasattr(pl_module, 'loss') and hasattr(pl_module.loss, 'reset_temporal_state'):
            pl_module.loss.reset_temporal_state()
            print(f"[DEBUG CollapseMonitor] reset_temporal_state() called on loss function")
        else:
            print(f"[DEBUG CollapseMonitor] WARNING: Could not find reset_temporal_state on loss")
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics at end of each epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        print(f"\n[CollapseMonitor] Epoch {trainer.current_epoch}")
        
        # 1. Prediction diversity metrics
        self._log_prediction_diversity(trainer, pl_module)
        
        # 2. Gradient flow
        self._log_gradient_flow(pl_module)
        
        # 3. Weight statistics
        self._log_weight_statistics(pl_module)
        
        # 4. Variable selection network activity
        self._log_vsn_activity(trainer, pl_module)
        
        # 5. Attention patterns
        self._log_attention_patterns(trainer, pl_module)
        
        # Save to disk
        self._save_history(trainer.current_epoch)
        
        # Reset model state after monitoring
        pl_module.train()
        if hasattr(pl_module, '_current_fx_name'):
            pl_module._current_fx_name = None
        
    def _log_prediction_diversity(self, trainer, pl_module):
        """
        Measure diversity of predictions on validation set.
        
        Metrics:
        - std/range/mean: Basic statistics over all validation predictions
        - Pos/Neg %: Percentage of predictions above/below zero
        - Unique: Number of distinct prediction values across entire val set
          (1 = collapsed to constant, hundreds = healthy diversity)
        """
        pl_module.eval()
        all_predictions = []
        
        with torch.no_grad():
            batch_count = 0
            for batch in self.val_dataloader:
                batch_count += 1
                x, y = batch
                
                # Move batch to device
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get predictions - handle different output formats
                output = pl_module(x)
                if isinstance(output, dict):
                    # pytorch-forecasting TFT model - returns dict with 'prediction' key
                    preds = output['prediction'][:, 0, 3]
                elif hasattr(output, 'prediction'):
                    # pytorch-forecasting namedtuple output
                    preds = output.prediction[:, 0, 3]
                else:
                    # Custom TFT model - returns predictions tensor directly
                    # Shape: [batch, max_prediction_length, num_quantiles]
                    # Extract median quantile (index 3 for 7 quantiles) at first timestep
                    preds = output[:, 0, 3]
                    
                all_predictions.append(preds.cpu().numpy())

        print(f"  [DEBUG] Processed {batch_count} batches from val_dataloader")
           
        predictions = np.concatenate(all_predictions)
        print(f"  [DEBUG] Predictions shape after concat: {predictions.shape}")
        print(f"  [DEBUG] All predictions list length: {len(all_predictions)}")
        print(f"  [DEBUG] First batch shape: {all_predictions[0].shape if all_predictions else 'empty'}")

        # Compute diversity metrics
        pred_std = np.std(predictions)
        pred_range = np.ptp(predictions)
        pred_mean = np.mean(predictions)
        pct_pos = np.mean(predictions > 0) * 100
        pct_neg = np.mean(predictions < 0) * 100
        n_unique = len(np.unique(np.round(predictions, decimals=6)))
        
        self.history['epoch'].append(trainer.current_epoch)
        self.history['prediction_std'].append(float(pred_std))
        self.history['prediction_range'].append(float(pred_range))
        self.history['prediction_mean'].append(float(pred_mean))
        self.history['pct_positive'].append(float(pct_pos))
        self.history['pct_negative'].append(float(pct_neg))
        self.history['num_unique_predictions'].append(int(n_unique))
        
        # Compute anti-collapse penalty if model uses EnhancedQuantileLoss
        collapse_penalty = None
        if hasattr(pl_module.loss, 'collapse_weight') and hasattr(pl_module.loss, 'collapse_threshold'):
            loss_fn = pl_module.loss
            if pred_std < loss_fn.collapse_threshold:
                collapse_penalty = loss_fn.collapse_weight * (loss_fn.collapse_threshold - pred_std) ** 2
            else:
                collapse_penalty = 0.0
            self.history['collapse_penalty'] = self.history.get('collapse_penalty', [])
            self.history['collapse_penalty'].append(float(collapse_penalty))
        
        print(f"  Pred std: {pred_std:.6f}, range: {pred_range:.6f}, "
              f"mean: {pred_mean:.6f}")
        print(f"  Pos: {pct_pos:.1f}%, Neg: {pct_neg:.1f}%, "
              f"Unique: {n_unique}")
        
        # Log to pl_module so ModelCheckpoint can monitor these metrics
        pl_module.log('val_pred_std', float(pred_std), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_pct_positive', float(pct_pos), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_num_unique', int(n_unique), on_step=False, on_epoch=True, prog_bar=False)
        
        # Print collapse penalty if computed
        if collapse_penalty is not None:
            print(f"  Collapse penalty: {collapse_penalty:.6f} "
                  f"(threshold: {loss_fn.collapse_threshold:.6f})")
        
        # Print directional penalty if model uses it
        if hasattr(pl_module.loss, 'directional_weight') and hasattr(pl_module.loss, 'last_directional_penalty'):
            dir_weight = pl_module.loss.directional_weight
            dir_penalty = pl_module.loss.last_directional_penalty
            dir_threshold = pl_module.loss.directional_threshold
            if dir_weight > 0:
                if dir_penalty is not None:
                    print(f"  Directional penalty: {dir_penalty:.6f} "
                          f"(weight: {dir_weight:.2f}, threshold: {dir_threshold:.2f})")
                else:
                    print(f"  Directional penalty: not computed yet "
                          f"(weight: {dir_weight:.2f}, threshold: {dir_threshold:.2f})")
            
        # Save actual predictions for debugging
        pred_save_path = self.log_dir / f'val_predictions_epoch{trainer.current_epoch}.npy'
        np.save(pred_save_path, predictions)
        print(f"  Saved validation predictions to: {pred_save_path}")

    def _log_gradient_flow(self, pl_module):
        """Log gradient magnitudes by layer using stored gradients from training."""
        epoch = len(self.history['epoch']) - 1
        
        # Debug: Print attention parameter names to verify pattern
        attn_params = [n for n, p in pl_module.named_parameters() if 'attention' in n.lower()]
        if epoch == 0 and attn_params:  # Only print on first epoch to avoid spam
            print(f"  [DEBUG] Attention param names: {attn_params[:5]}")  # Show first 5
        
        # Use the gradients we captured during training
        for name, grad_norms in self._current_epoch_gradients.items():
            # Average gradient norm across all batches in this epoch
            avg_grad_norm = np.mean(grad_norms)
            
            if name not in self.history['gradient_norms']:
                self.history['gradient_norms'][name] = []
                
            self.history['gradient_norms'][name].append(float(avg_grad_norm))
        
        # Print summary of key layers
        key_layers = ['lstm_encoder', 'lstm_decoder', 'multihead_attention', 
                      'output_layer']
        print("  Gradient norms:")
        for layer_name in key_layers:
            matching = [k for k in self.history['gradient_norms'].keys() 
                       if layer_name in k]
            if matching:
                norms = [self.history['gradient_norms'][k][-1] for k in matching]
                avg_norm = np.mean(norms)
                print(f"    {layer_name}: {avg_norm:.6f}")
        
        # Clear buffer for next epoch
        self._current_epoch_gradients = {}
                
    def _log_weight_statistics(self, pl_module):
        """Log weight matrix statistics."""
        epoch = len(self.history['epoch']) - 1
        
        for name, param in pl_module.named_parameters():
            if 'weight' in name:
                weight_norm = param.norm().item()
                weight_std = param.std().item()
                
                if name not in self.history['weight_norms']:
                    self.history['weight_norms'][name] = []
                    self.history['weight_stds'][name] = []
                    
                self.history['weight_norms'][name].append(float(weight_norm))
                self.history['weight_stds'][name].append(float(weight_std))
                
    def _log_vsn_activity(self, trainer, pl_module):
        """Log variable selection network output statistics."""
        pl_module.eval()
        
        vsn_outputs = {
            'encoder': [],
        }
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Sample first 10 batches
                    break
                    
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get output - could be dict, namedtuple, or plain tensor
                output = pl_module(x)
                
                # Try to extract encoder_variables using different methods
                encoder_vsn = None
                
                # Method 1: Check if it's a dict with 'encoder_variables' key
                if isinstance(output, dict) and 'encoder_variables' in output:
                    encoder_vsn = output['encoder_variables']
                # Method 2: Check if it's a namedtuple with encoder_variables attribute
                elif hasattr(output, 'encoder_variables'):
                    encoder_vsn = output.encoder_variables
                # Method 3: Custom TFT model - use getter method
                elif hasattr(pl_module, 'get_encoder_vsn_output'):
                    vsn_result = pl_module.get_encoder_vsn_output()
                    if vsn_result is not None:
                        encoder_vsn, _ = vsn_result  # (output, weights) tuple
                
                if encoder_vsn is not None:
                    vsn_outputs['encoder'].append(encoder_vsn.cpu().numpy())
        
        # Compute std for each VSN
        epoch = len(self.history['epoch']) - 1
        print("  VSN output std:")
        for vsn_name, outputs in vsn_outputs.items():
            if outputs:
                concatenated = np.concatenate(outputs)
                vsn_std = np.std(concatenated)
                
                if vsn_name not in self.history['vsn_output_std']:
                    self.history['vsn_output_std'][vsn_name] = []
                    
                self.history['vsn_output_std'][vsn_name].append(float(vsn_std))
                print(f"    {vsn_name}: {vsn_std:.6f}")
            else:
                print(f"    {vsn_name}: (no data captured)")
                if vsn_name not in self.history['vsn_output_std']:
                    self.history['vsn_output_std'][vsn_name] = []
                self.history['vsn_output_std'][vsn_name].append(None)
                
    def _log_attention_patterns(self, trainer, pl_module):
        """
        Log attention weight entropy.
        
        For pytorch-forecasting TFT: Use interpret_output() method
        For custom TFT: Use get_attention_weights() getter method
        """
        pl_module.eval()
        attention_entropies = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Sample first 10 batches
                    break
                    
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get predictions first
                output = pl_module(x)
                
                attn_weights = None
                
                # Method 1: Custom TFT model - use getter method
                if hasattr(pl_module, 'get_attention_weights'):
                    attn_weights = pl_module.get_attention_weights()
                    
                    if attn_weights is not None:
                        # Shape: [batch, num_heads, max_prediction_length, encoder_length + max_prediction_length]
                        # Compute per-head entropy instead of averaging heads first
                        
                        # For single-step prediction, extract first (only) prediction timestep
                        # Shape: [batch, num_heads, encoder_length + max_prediction_length]
                        if attn_weights.size(2) == 1:
                            attn_weights = attn_weights[:, :, 0, :]
                        else:
                            # If multiple prediction steps, use first one
                            attn_weights = attn_weights[:, :, 0, :]
                        
                        # Move to CPU and convert to numpy
                        attn = attn_weights.cpu().numpy()
                        
                        # Compute entropy per head, then average
                        # attn shape: [batch, num_heads, encoder_length + max_prediction_length]
                        per_head_entropies = []
                        num_heads = attn.shape[1]
                        
                        for h in range(num_heads):
                            # Extract attention for this head: [batch, seq_len]
                            head_attn = attn[:, h, :]
                            
                            # Normalize (should already be normalized, but verify)
                            head_attn_norm = head_attn / (head_attn.sum(axis=-1, keepdims=True) + 1e-10)
                            
                            # Compute entropy: [batch]
                            head_entropy = -np.sum(head_attn_norm * np.log(head_attn_norm + 1e-10), axis=-1)
                            per_head_entropies.append(head_entropy)
                        
                        # Average entropy across heads: [batch]
                        avg_entropy_per_batch = np.mean(per_head_entropies, axis=0)
                        attention_entropies.append(avg_entropy_per_batch)
                        
                # Method 2: pytorch-forecasting TFT model - use interpret_output
                elif hasattr(pl_module, 'interpret_output'):
                    try:
                        interpretation = pl_module.interpret_output(
                            output, 
                            reduction='none',
                            attention_prediction_horizon=0  # Focus on first prediction step
                        )
                        
                        # Extract attention weights from interpretation dict
                        # Note: interpret_output may already average across heads internally
                        # We cannot change pytorch-forecasting's internal behavior, but keep
                        # consistent structure with custom TFT branch above
                        if 'attention' in interpretation:
                            attn = interpretation['attention'].cpu().numpy()
                            # Compute entropy of attention distribution
                            # attn shape: [batch, encoder_length] typically (already head-averaged)
                            # Normalize if not already (attention should sum to 1)
                            attn_norm = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)
                            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)
                            attention_entropies.append(entropy)
                    except Exception as e:
                        # interpret_output might fail for various reasons
                        print(f"    (interpret_output failed: {type(e).__name__})")
                        break
                else:
                    # No method available to extract attention
                    print(f"    (no attention extraction method available)")
                    break
        
        if attention_entropies:
            avg_entropy = np.mean(np.concatenate(attention_entropies))
            self.history['attention_entropy'].append(float(avg_entropy))
            print(f"  Attention entropy: {avg_entropy:.6f}")
        else:
            self.history['attention_entropy'].append(None)
            print(f"  Attention entropy: (no data captured)")
            
    def _save_history(self, epoch):
        """Save monitoring history to disk."""
        save_path = self.log_dir / f'collapse_monitor_epoch{epoch}.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        # Also save latest as separate file for easy access
        latest_path = self.log_dir / 'collapse_monitor_latest.json'
        with open(latest_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"  Saved to: {save_path}")
