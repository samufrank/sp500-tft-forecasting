"""
Alternative approach: Add distribution penalties by monkey-patching the loss method.

This directly replaces the loss computation method without wrapping or subclassing.
"""

import torch


def add_distribution_penalties(model, mean_weight=0.0, std_weight=0.0, 
                                target_mean=0.0003, target_std=0.01):
    """
    Patch a TFT model's loss function to add distribution penalties.
    
    Args:
        model: TemporalFusionTransformer instance
        mean_weight: Weight for anti-drift penalty (0 = disabled)
        std_weight: Weight for anti-collapse penalty (0 = disabled)
        target_mean: Target mean for predictions (S&P 500 daily return)
        target_std: Target std for predictions (S&P 500 daily volatility)
    
    Returns:
        Modified model with penalty tracking attributes
    
    Usage:
        tft = TemporalFusionTransformer.from_dataset(...)
        tft = add_distribution_penalties(tft, mean_weight=0.1, std_weight=0.1)
        trainer.fit(tft, ...)
    """
    # Store penalty config on model
    model.mean_weight = mean_weight
    model.std_weight = std_weight
    model.target_mean = target_mean
    model.target_std = target_std
    
    # For logging
    model.last_pred_mean = None
    model.last_pred_std = None
    model.last_mean_penalty = None
    model.last_std_penalty = None
    
    # Store original loss method
    original_loss_fn = model.loss.loss
    
    def patched_loss(self, y_pred, y_actual):
        """Replacement loss function with distribution penalties."""
        # Compute base loss
        base_loss_value = original_loss_fn(y_pred, y_actual)
        
        # Add penalties if enabled
        if model.mean_weight > 0 or model.std_weight > 0:
            # Handle both dict and tensor inputs
            if isinstance(y_pred, dict):
                predictions = y_pred['prediction']
            else:
                predictions = y_pred
            
            median_pred = predictions[..., 3]
            pred_flat = median_pred.reshape(-1)
            
            total_loss = base_loss_value
            
            if model.mean_weight > 0:
                pred_mean = pred_flat.mean()
                mean_penalty = (pred_mean - model.target_mean) ** 2
                total_loss = total_loss + model.mean_weight * mean_penalty
                model.last_mean_penalty = mean_penalty.detach().item()
            
            if model.std_weight > 0:
                pred_std = pred_flat.std()
                std_penalty = (pred_std - model.target_std) ** 2
                total_loss = total_loss + model.std_weight * std_penalty
                model.last_std_penalty = std_penalty.detach().item()
            
            model.last_pred_mean = pred_flat.mean().detach().item()
            model.last_pred_std = pred_flat.std().detach().item()
            
            return total_loss
        else:
            return base_loss_value
    
    # Monkey-patch the loss method
    import types
    model.loss.loss = types.MethodType(patched_loss, model.loss)
    
    # Add helper method to model
    model.is_enabled = lambda: mean_weight > 0 or std_weight > 0
    
    return model
