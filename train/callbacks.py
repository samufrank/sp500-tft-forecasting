"""
PyTorch Lightning callbacks for TFT training.

Note: CollapseMonitor is in collapse_monitor.py due to its size/complexity.
"""

import pytorch_lightning as pl


class EpochSummaryCallback(pl.Callback):
    """Print training summary after each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print training loss at end of epoch."""
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', metrics.get('train_loss_epoch'))
        
        if train_loss is not None:
            print(f"Epoch {trainer.current_epoch}: train_loss={train_loss}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation loss at end of epoch."""
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss')
        
        if val_loss is not None:
            print(f"Epoch {trainer.current_epoch}: val_loss={val_loss}")
        else:
            print(f"Epoch {trainer.current_epoch}: val_loss=N/A")


class DistributionLossLogger(pl.Callback):
    """
    Log distribution penalty statistics during training.
    
    Works with DistributionPenaltyWrapper to log prediction statistics.
    """
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print distribution statistics after each training epoch."""
        # Check if using wrapper
        if not hasattr(pl_module, 'is_enabled'):
            return
        
        if not pl_module.is_enabled():
            return
        
        # Print to console for monitoring
        if pl_module.last_pred_mean is not None:
            print(f"  [DistLoss] pred_mean={pl_module.last_pred_mean:.6f} "
                  f"(target={pl_module.target_mean:.6f}), "
                  f"pred_std={pl_module.last_pred_std:.6f} "
                  f"(target={pl_module.target_std:.6f})")
