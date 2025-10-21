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
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics at end of each epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        print(f"\n[CollapseMonitor] Epoch {trainer.current_epoch}")
        
        # 1. Prediction diversity metrics
        #self._log_prediction_diversity(trainer, pl_module)
        
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
        """Measure diversity of predictions on validation set."""
        pl_module.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                x, y = batch
                
                # Move batch to device
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get predictions - output is dict with 'prediction' key
                output = pl_module(x)
                preds = output['prediction'][:, 0, 3]  # [batch, time=0, quantile=3 (median)]
                    
                all_predictions.append(preds.cpu().numpy())
       
        predictions = np.concatenate(all_predictions)
        
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
        
        print(f"  Pred std: {pred_std:.6f}, range: {pred_range:.6f}, "
              f"mean: {pred_mean:.6f}")
        print(f"  Pos: {pct_pos:.1f}%, Neg: {pct_neg:.1f}%, "
              f"Unique: {n_unique}")
        
        # Warning if collapse detected
        if pred_std < 0.01 or n_unique < 10:
            print(f"  WARNING: Potential collapse detected!")
            
    def _log_gradient_flow(self, pl_module):
        """Log gradient magnitudes by layer."""
        epoch = len(self.history['epoch']) - 1
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if name not in self.history['gradient_norms']:
                    self.history['gradient_norms'][name] = []
                    
                self.history['gradient_norms'][name].append(float(grad_norm))
        
        # Print summary of key layers
        key_layers = ['lstm_encoder', 'lstm_decoder', 'multihead_attn', 
                      'output_layer']
        print("  Gradient norms:")
        for layer_name in key_layers:
            matching = [k for k in self.history['gradient_norms'].keys() 
                       if layer_name in k]
            if matching:
                norms = [self.history['gradient_norms'][k][-1] for k in matching]
                avg_norm = np.mean(norms)
                print(f"    {layer_name}: {avg_norm:.6f}")
                
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
                
                # Access the actual TFT model (nested inside Lightning module)
                tft_model = pl_module
                
                # Get encoder VSN output - encoder_variables is in output dict
                output = pl_module(x)
                if 'encoder_variables' in output and output['encoder_variables'] is not None:
                    encoder_vsn = output['encoder_variables']
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
                
    def _log_attention_patterns(self, trainer, pl_module):
        """Log attention weight entropy."""
        pl_module.eval()
        attention_entropies = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Sample first 10 batches
                    break
                    
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get attention from output dict
                output = pl_module(x)
                if 'attention' in output and output['attention'] is not None:
                    attn = output['attention'].cpu().numpy()
                    # Compute entropy of attention distribution
                    # attn shape: [batch, time, heads] or similar
                    entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1)
                    attention_entropies.append(entropy)
        
        if attention_entropies:
            avg_entropy = np.mean(np.concatenate(attention_entropies))
            self.history['attention_entropy'].append(float(avg_entropy))
            print(f"  Attention entropy: {avg_entropy:.6f}")
        else:
            self.history['attention_entropy'].append(None)
            
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
        
    def on_train_end(self, trainer, pl_module):
        """Final summary on training completion."""
        print("\n" + "="*70)
        print("COLLAPSE MONITOR SUMMARY")
        print("="*70)
        
        epochs = self.history['epoch']
        pred_stds = self.history['prediction_std']
        
        print(f"Total epochs: {len(epochs)}")
        print(f"\nPrediction std over time:")
        print(f"  Initial: {pred_stds[0]:.6f}")
        print(f"  Final: {pred_stds[-1]:.6f}")
        print(f"  Min: {min(pred_stds):.6f} (epoch {epochs[pred_stds.index(min(pred_stds))]})")
        print(f"  Max: {max(pred_stds):.6f} (epoch {epochs[pred_stds.index(max(pred_stds))]})")
        
        # Check for collapse
        if pred_stds[-1] < 0.01:
            print("\n    MODEL COLLAPSED - final prediction std < 0.01")
        elif pred_stds[-1] < 0.1:
            print("\n    MODEL NEAR COLLAPSE - final prediction std < 0.1")
        else:
            print("\n    Model predictions remain diverse")
            
        print("="*70)

