"""
Training integration for Staleness-Aware Attention TFT.

Provides hooks to thread staleness signals through the TFT forward pass
during training with pytorch-lightning.

The staleness signal is extracted from encoder_cont for all timesteps and
passed to the attention module to apply staleness penalties.

Usage:
    from staleness_attention import replace_attention_with_staleness
    from staleness_attention_training import patch_forward_for_staleness_training
    
    model = TemporalFusionTransformer.from_dataset(dataset, ...)
    model = replace_attention_with_staleness(model, staleness_mode='penalty', ...)
    model = patch_forward_for_staleness_training(
        model, 
        staleness_feature_name='days_since_CPI_update',
        vix_feature_name='VIX'
    )
    
    # Train normally with pytorch-lightning
    trainer.fit(model, train_dataloader, val_dataloader)

Notes:
    - Staleness features must be present in the data (use --staleness flag)
    - The staleness penalty is applied in addition to any VSN weighting
    - This tests whether staleness-as-penalty helps even with staleness-as-feature
"""

from typing import Optional, Dict, Any, Callable, List
import math
import torch
import torch.nn as nn
import numpy as np
from functools import wraps
import types


def find_feature_index(
    model: nn.Module,
    feature_name: str,
    variations: Optional[List[str]] = None
) -> Optional[int]:
    """
    Find index of a feature in model's continuous features.
    
    Args:
        model: TFT model with hparams
        feature_name: Primary feature name to search
        variations: Alternative names to try
        
    Returns:
        Index in x_reals, or None if not found
    """
    if not hasattr(model, 'hparams'):
        return None
    
    x_reals = getattr(model.hparams, 'x_reals', [])
    
    if feature_name in x_reals:
        return x_reals.index(feature_name)
    
    # Try variations
    if variations:
        for var in variations:
            if var in x_reals:
                return x_reals.index(var)
    
    return None


def find_staleness_index(
    model: nn.Module,
    staleness_feature_name: str = 'days_since_CPI_update'
) -> Optional[int]:
    """Find index of staleness feature in model's continuous features."""
    variations = [
        staleness_feature_name,
        'days_since_CPI_update',
        'days_since_cpi_update',
        'CPI_staleness',
        'staleness_CPI',
    ]
    return find_feature_index(model, staleness_feature_name, variations)


def find_vix_index(
    model: nn.Module,
    vix_feature_name: str = 'VIX'
) -> Optional[int]:
    """Find index of VIX feature in model's continuous features."""
    variations = ['VIX', 'vix', 'VIX_close', 'vix_close', 'VIXCLS']
    return find_feature_index(model, vix_feature_name, variations)


def extract_staleness_from_batch(
    x: Dict[str, torch.Tensor],
    staleness_idx: int,
    denormalize: bool = False,
    staleness_max_days: float = 45.0
) -> Optional[torch.Tensor]:
    """
    Extract staleness values for all key timesteps from batch.
    
    In TFT, keys span encoder + decoder sequence. We need staleness values
    for all key positions to penalize attention appropriately.
    
    Args:
        x: Input batch dictionary
        staleness_idx: Index of staleness feature in continuous features
        denormalize: If True, convert from normalized [0,1] to raw days.
                    If False (recommended), use normalized values directly.
                    Using normalized values avoids frequency-dependent scaling issues.
        staleness_max_days: Max days for denormalization (only used if denormalize=True)
        
    Returns:
        staleness: [batch, key_len] staleness values (normalized or raw), or None
        
    Notes:
        For weekly data, staleness is computed per observation index, not calendar days.
        Using normalized values (denormalize=False) is recommended as it works across
        frequencies without needing to match the exact max_days from preprocessing.
    """
    encoder_cont = x.get('encoder_cont')
    if encoder_cont is None:
        return None
    
    # Extract staleness from encoder
    # encoder_cont shape: [batch, encoder_len, n_features]
    if encoder_cont.dim() != 3 or encoder_cont.size(-1) <= staleness_idx:
        return None
    
    staleness_enc = encoder_cont[:, :, staleness_idx]  # [batch, encoder_len]
    
    # Check if decoder has staleness (for multi-step predictions)
    decoder_cont = x.get('decoder_cont')
    if decoder_cont is not None and decoder_cont.dim() == 3:
        if decoder_cont.size(-1) > staleness_idx:
            staleness_dec = decoder_cont[:, :, staleness_idx]  # [batch, decoder_len]
            staleness_full = torch.cat([staleness_enc, staleness_dec], dim=1)
        else:
            staleness_full = staleness_enc
    else:
        staleness_full = staleness_enc
    
    # Optionally denormalize (not recommended - use normalized values for robustness)
    # In data prep, staleness was transformed: normalized = log(1 + days) / log(1 + max_days)
    # Inverse: days = exp(normalized * log(1 + max_days)) - 1
    if denormalize:
        log_max = math.log(1.0 + staleness_max_days)
        staleness_full = torch.exp(staleness_full * log_max) - 1.0
        # Clamp to reasonable range
        staleness_full = torch.clamp(staleness_full, min=0.0, max=staleness_max_days * 2)
    
    return staleness_full


def extract_vix_from_batch(
    x: Dict[str, torch.Tensor],
    vix_idx: int,
    use_last_timestep: bool = True
) -> Optional[torch.Tensor]:
    """
    Extract VIX values from batch.
    
    Args:
        x: Input batch dictionary
        vix_idx: Index of VIX feature in continuous features
        use_last_timestep: If True, return VIX from last encoder timestep
        
    Returns:
        vix: [batch] or [batch, seq_len], or None
    """
    encoder_cont = x.get('encoder_cont')
    if encoder_cont is None:
        return None
    
    if encoder_cont.dim() != 3 or encoder_cont.size(-1) <= vix_idx:
        return None
    
    if use_last_timestep:
        vix = encoder_cont[:, -1, vix_idx]  # [batch]
    else:
        vix = encoder_cont[:, :, vix_idx]  # [batch, encoder_len]
    
    return vix


def patch_forward_for_staleness_training(
    model: nn.Module,
    staleness_feature_name: str = 'days_since_CPI_update',
    vix_feature_name: str = 'VIX',
    verbose: bool = True
) -> nn.Module:
    """
    Patch model's forward() to extract and pass staleness/VIX signals.
    
    This modifies the model to automatically:
    1. Extract staleness values for all encoder timesteps
    2. Extract VIX values for regime detection (if regime mode enabled)
    3. Set signals on attention module before forward pass
    
    Args:
        model: TFT model with staleness-aware attention
        staleness_feature_name: Name of staleness feature in x_reals
        vix_feature_name: Name of VIX feature in x_reals
        verbose: Print feature discovery info
        
    Returns:
        model: Modified model (in-place)
        
    Notes:
        Staleness values are passed as-is (normalized [0,1] from preprocessing).
        The attention module should use staleness_decay='prenormalized' to work
        with these values directly.
    """
    # Find feature indices
    staleness_idx = find_staleness_index(model, staleness_feature_name)
    vix_idx = find_vix_index(model, vix_feature_name)
    
    if verbose:
        if staleness_idx is not None:
            print(f"[STALENESS TRAINING] Found '{staleness_feature_name}' at index {staleness_idx}")
        else:
            print(f"[STALENESS TRAINING] WARNING: '{staleness_feature_name}' not found in model features")
            print(f"[STALENESS TRAINING] Available features: {getattr(model.hparams, 'x_reals', [])}")
        
        if vix_idx is not None:
            print(f"[STALENESS TRAINING] Found '{vix_feature_name}' at index {vix_idx}")
        else:
            print(f"[STALENESS TRAINING] VIX not found (regime gating disabled)")
    
    # Store indices on model for access in patched forward
    model._staleness_feature_idx = staleness_idx
    model._vix_feature_idx = vix_idx
    
    # Store original forward
    original_forward = model.forward
    
    def patched_forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward with staleness and VIX signal injection."""
        
        # Set staleness signal if available
        if self._staleness_feature_idx is not None and hasattr(self, 'multihead_attn'):
            if hasattr(self.multihead_attn, 'set_staleness_signal'):
                # Extract staleness - use normalized values directly (denormalize=False)
                # This is robust across frequencies (daily vs weekly)
                staleness = extract_staleness_from_batch(
                    x,
                    self._staleness_feature_idx,
                    denormalize=False  # Use normalized [0,1] values directly
                )
                self.multihead_attn.set_staleness_signal(staleness)
        
        # Set VIX/regime signal if available
        if self._vix_feature_idx is not None and hasattr(self, 'multihead_attn'):
            if hasattr(self.multihead_attn, 'set_regime_signal'):
                vix = extract_vix_from_batch(x, self._vix_feature_idx, use_last_timestep=True)
                self.multihead_attn.set_regime_signal(vix)
        
        # Call original forward
        return original_forward(x)
    
    # Replace forward method
    model.forward = types.MethodType(patched_forward, model)
    
    return model


def setup_staleness_attention_for_training(
    model: nn.Module,
    staleness_mode: str = 'penalty',
    staleness_decay: str = 'prenormalized',  # Use prenormalized by default (robust to freq)
    staleness_weight: float = 0.5,
    staleness_learnable: bool = True,
    staleness_max_days: float = 45.0,  # Only used for non-prenormalized decays
    regime_mode: str = 'disabled',
    vix_threshold: float = 25.0,
    staleness_feature_name: str = 'days_since_CPI_update',
    vix_feature_name: str = 'VIX',
    verbose: bool = True
) -> nn.Module:
    """
    Complete setup for staleness-aware TFT training.
    
    Combines attention module replacement and forward patching.
    
    Args:
        model: Base TFT model
        staleness_mode: 'disabled' or 'penalty'
        staleness_decay: Decay function for penalty:
            - 'prenormalized': Use normalized [0,1] staleness directly (recommended)
            - 'linear', 'exponential', 'log', 'step': Transform raw days
        staleness_weight: Penalty magnitude
        staleness_learnable: Whether weight is learnable
        staleness_max_days: Normalization factor (only for non-prenormalized)
        regime_mode: 'disabled' or 'vix_threshold' (can combine)
        vix_threshold: VIX threshold for regime detection
        staleness_feature_name: Name of staleness feature in data
        vix_feature_name: Name of VIX feature in data
        verbose: Print setup information
        
    Returns:
        model: Configured model ready for training
        
    Notes:
        Using staleness_decay='prenormalized' is recommended because:
        1. Works with both daily and weekly frequencies without adjustment
        2. Staleness values are already log-normalized in preprocessing
        3. Avoids need to match max_days between preprocessing and attention
    """
    # Import here to avoid circular imports
    import sys
    import os
    
    # Add the directory containing staleness_attention.py to path if needed
    staleness_module_dir = os.path.dirname(os.path.abspath(__file__))
    if staleness_module_dir not in sys.path:
        sys.path.insert(0, staleness_module_dir)
    
    from staleness_attention import replace_attention_with_staleness
    
    if verbose:
        print("\n" + "="*70)
        print("STALENESS ATTENTION SETUP")
        print("="*70)
    
    # Step 1: Replace attention module
    model = replace_attention_with_staleness(
        model,
        staleness_mode=staleness_mode,
        staleness_decay=staleness_decay,
        staleness_weight=staleness_weight,
        staleness_learnable=staleness_learnable,
        staleness_max_days=staleness_max_days,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold
    )
    
    # Step 2: Patch forward for signal threading
    model = patch_forward_for_staleness_training(
        model,
        staleness_feature_name=staleness_feature_name,
        vix_feature_name=vix_feature_name,
        verbose=verbose
    )
    
    if verbose:
        print("="*70 + "\n")
    
    return model


class StalenessAttentionCallback:
    """
    PyTorch Lightning callback for logging staleness attention diagnostics.
    
    Logs staleness weights, penalties, and attention patterns during training.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log diagnostics after each training batch."""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps != 0:
            return
        
        if not hasattr(pl_module, 'multihead_attn'):
            return
        
        attn = pl_module.multihead_attn
        if not hasattr(attn, 'get_diagnostics'):
            return
        
        diagnostics = attn.get_diagnostics()
        
        # Log staleness weight if learnable
        if diagnostics.get('staleness_mode') != 'disabled':
            weight = diagnostics.get('staleness_weight', 0)
            trainer.logger.log_metrics({
                'staleness/weight': weight
            }, step=trainer.global_step)
        
        # Log regime gate statistics if enabled
        if diagnostics.get('regime_mode') != 'disabled':
            gates = diagnostics.get('regime_gates')
            if gates is not None:
                trainer.logger.log_metrics({
                    'regime/gate_mean': gates.mean().item(),
                    'regime/gate_std': gates.std().item(),
                }, step=trainer.global_step)
