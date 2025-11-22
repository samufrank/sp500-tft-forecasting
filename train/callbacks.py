"""
PyTorch Lightning callbacks for TFT training.

Note: CollapseMonitor is in collapse_monitor.py due to its size/complexity.
AntiCollapse penalty logging is integrated into CollapseMonitor for consistency.
"""

import torch
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
