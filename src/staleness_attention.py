"""
Staleness-Aware Attention for TFT with Mixed-Frequency Data.

Extends RegimeAwareInterpretableMultiHeadAttention to add staleness-based
attention penalties. Designed for mixed-frequency financial data where
monthly macro indicators (CPI, unemployment) get forward-filled between
releases and become progressively stale.

Key insight from Phase 1 experiments: Adding staleness as INPUT features
caused 94% model collapse because the model learned pathological patterns
(e.g., 35% VSN weight on days_since_CPI_update in 2022). Instead, staleness
should modify attention SCORES directly - imposing a structural prior that
stale timesteps deserve less attention.

Architecture:
    1. Compute attention logits as normal: scores = QK^T / sqrt(d_k)
    2. Apply staleness penalty to keys: scores -= penalty * staleness[key_idx]
    3. Apply softmax: attn = softmax(scores)
    4. Apply regime gating (if enabled): attn *= regime_gate[head]
    
Staleness penalty is ADDITIVE on logits (before softmax), while regime
gating is MULTIPLICATIVE on weights (after softmax). They're orthogonal:
- Staleness controls WHERE to attend (which timesteps)
- Regime controls HOW MUCH each head contributes (which patterns)

Decay functions:
    - 'linear': penalty = staleness_weight * (days / max_days)
    - 'exponential': penalty = staleness_weight * (1 - exp(-days / half_life))
    - 'log': penalty = staleness_weight * log(1 + days) / log(1 + max_days)
    - 'step': penalty = staleness_weight if days > threshold else 0

Usage:
    from staleness_attention import (
        StalenessAwareInterpretableMultiHeadAttention,
        replace_attention_with_staleness,
        patch_forward_for_staleness
    )
    
    model = TemporalFusionTransformer.from_dataset(dataset, ...)
    model = replace_attention_with_staleness(
        model,
        staleness_mode='penalty',
        staleness_decay='log',
        staleness_weight=0.5,
        regime_mode='vix_threshold',  # Can combine with regime
        vix_threshold=25.0
    )
    model = patch_forward_for_staleness(model, staleness_feature_name='days_since_CPI_update')

References:
    - Original TFT: Lim et al. (2021)
    - Phase 1 failure analysis: staleness as features caused 94% collapse
    - Staleness attention analysis: 35% VSN weight on staleness in 2022
"""

import math
from typing import Optional, Tuple, Dict, List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StalenessAwareInterpretableMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with staleness penalties and optional regime gating.
    
    Combines two orthogonal modifications:
    1. Staleness penalty: Additive penalty on attention logits for stale timesteps
    2. Regime gating: Multiplicative scaling of attention weights per head
    
    Parameters
    ----------
    n_head : int
        Number of attention heads
    d_model : int
        Model dimension (hidden_size)
    dropout : float, default=0.0
        Dropout rate
    
    Staleness parameters:
    ---------------------
    staleness_mode : str, default='disabled'
        - 'disabled': No staleness modification
        - 'penalty': Apply staleness penalty to attention logits
    staleness_decay : str, default='log'
        Decay function for staleness penalty:
        - 'linear': penalty proportional to days stale
        - 'exponential': penalty with exponential saturation
        - 'log': penalty with logarithmic compression (recommended)
        - 'step': binary penalty above threshold
    staleness_weight : float, default=0.5
        Maximum penalty magnitude (learned if staleness_learnable=True)
    staleness_learnable : bool, default=True
        Whether staleness_weight is a learnable parameter
    staleness_max_days : float, default=45.0
        Normalization factor for staleness (max expected days stale)
    staleness_half_life : float, default=14.0
        Half-life for exponential decay (days)
    staleness_threshold : float, default=7.0
        Threshold for step decay (days)
    staleness_grad_scale : float, default=10.0
        Gradient scaling for staleness parameter (if learnable)
    
    Regime parameters (same as RegimeAwareInterpretableMultiHeadAttention):
    -----------------------------------------------------------------------
    regime_mode : str, default='disabled'
    vix_threshold : float, default=25.0
    num_regimes : int, default=2
    gate_init_std : float, default=0.01
    gate_grad_scale : float, default=100.0
    gate_init : str, default='neutral'
    """
    
    def __init__(
        self,
        n_head: int,
        d_model: int,
        dropout: float = 0.0,
        mask_bias: float = -1e9,
        # Staleness parameters
        staleness_mode: str = 'disabled',
        staleness_decay: str = 'log',
        staleness_weight: float = 0.5,
        staleness_learnable: bool = True,
        staleness_max_days: float = 45.0,
        staleness_half_life: float = 14.0,
        staleness_threshold: float = 7.0,
        staleness_grad_scale: float = 10.0,
        # Regime parameters
        regime_mode: str = 'disabled',
        vix_threshold: float = 25.0,
        num_regimes: int = 2,
        gate_init_std: float = 0.01,
        gate_grad_scale: float = 100.0,
        gate_init: str = 'neutral'
    ):
        super().__init__()
        
        # Validate modes
        if staleness_mode not in ['disabled', 'penalty']:
            raise ValueError(f"staleness_mode must be 'disabled' or 'penalty', got '{staleness_mode}'")
        if staleness_decay not in ['linear', 'exponential', 'log', 'step', 'prenormalized']:
            raise ValueError(f"staleness_decay must be 'linear', 'exponential', 'log', 'step', or 'prenormalized'")
        if regime_mode not in ['disabled', 'vix_threshold']:
            raise ValueError(f"regime_mode must be 'disabled' or 'vix_threshold'")
        if gate_init not in ['neutral', 'separated']:
            raise ValueError(f"gate_init must be 'neutral' or 'separated'")
        
        # Core attention parameters
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.mask_bias = mask_bias
        self.dropout = nn.Dropout(p=dropout)
        
        # Staleness parameters
        self.staleness_mode = staleness_mode
        self.staleness_decay = staleness_decay
        self.staleness_max_days = staleness_max_days
        self.staleness_half_life = staleness_half_life
        self.staleness_threshold = staleness_threshold
        self.staleness_grad_scale = staleness_grad_scale
        
        # Regime parameters
        self.regime_mode = regime_mode
        self.vix_threshold = vix_threshold
        self.num_regimes = num_regimes
        self.gate_grad_scale = gate_grad_scale
        self.gate_init = gate_init
        
        # === ATTENTION LAYERS ===
        # Shared value projection (interpretability constraint from TFT)
        self.v_layer = nn.Linear(d_model, self.d_v)
        
        # Per-head Q and K projections
        self.q_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_q) for _ in range(n_head)
        ])
        self.k_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_k) for _ in range(n_head)
        ])
        
        # Output projection
        self.w_h = nn.Linear(self.d_v, d_model, bias=False)
        
        # === STALENESS COMPONENTS ===
        if staleness_mode != 'disabled':
            if staleness_learnable:
                # Learnable staleness weight (initialized to specified value)
                # Use log-space for stability: actual_weight = exp(log_weight)
                self._staleness_log_weight = nn.Parameter(
                    torch.tensor(math.log(staleness_weight))
                )
                # Gradient scaling for learnable weight
                self._staleness_log_weight.register_hook(
                    lambda grad: grad * self.staleness_grad_scale
                )
            else:
                # Fixed staleness weight
                self.register_buffer(
                    '_staleness_weight_fixed',
                    torch.tensor(staleness_weight)
                )
            self.staleness_learnable = staleness_learnable
        
        # === REGIME COMPONENTS ===
        if regime_mode != 'disabled':
            if gate_init == 'separated':
                init_gates = torch.zeros(num_regimes, n_head)
                init_gates[0, :] = -0.5  # Low-vol dampens
                init_gates[1, :] = 0.5   # High-vol amplifies
                self.regime_gates = nn.Parameter(init_gates)
            else:
                self.regime_gates = nn.Parameter(
                    torch.randn(num_regimes, n_head) * gate_init_std
                )
            self.regime_gates.register_hook(lambda grad: grad * self.gate_grad_scale)
        
        # === SIGNAL BUFFERS ===
        # Staleness signal: [batch, seq_len] days since last update
        self.register_buffer('_staleness_signal', None, persistent=False)
        
        # Regime signal: [batch] VIX values
        self.register_buffer('_regime_signal', None, persistent=False)
        self.register_buffer('_current_regime', None, persistent=False)
        
        # === DIAGNOSTIC CACHES ===
        self.register_buffer('_cached_attention_weights', None, persistent=False)
        self.register_buffer('_cached_staleness_penalty', None, persistent=False)
        self.register_buffer('_cached_regime_gates', None, persistent=False)
        self.register_buffer('_cached_head_contributions', None, persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for layer in self.q_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.k_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.v_layer.weight)
        nn.init.zeros_(self.v_layer.bias)
        nn.init.xavier_uniform_(self.w_h.weight)
    
    @property
    def staleness_weight(self) -> float:
        """Get current staleness weight value."""
        if self.staleness_mode == 'disabled':
            return 0.0
        if self.staleness_learnable:
            return torch.exp(self._staleness_log_weight).item()
        return self._staleness_weight_fixed.item()
    
    def set_staleness_signal(self, staleness: Optional[torch.Tensor]):
        """
        Set staleness values for all key timesteps.
        
        Args:
            staleness: Days since last update [batch, key_len] or None
                      Values should be raw days (not normalized)
        """
        self._staleness_signal = staleness
    
    def set_regime_signal(self, vix_values: Optional[torch.Tensor]):
        """
        Set VIX values for regime detection.
        
        Args:
            vix_values: Raw VIX values [batch] or None
        """
        if vix_values is None:
            self._regime_signal = None
            self._current_regime = None
            return
        
        self._regime_signal = vix_values
        
        if self.regime_mode == 'vix_threshold':
            self._current_regime = (vix_values >= self.vix_threshold).long()
        else:
            self._current_regime = None
    
    def _compute_staleness_penalty(
        self,
        staleness: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute attention penalty from staleness values.
        
        Args:
            staleness: Staleness values [batch, key_len]
                      - If staleness_decay='prenormalized': expects [0,1] normalized values
                      - Otherwise: expects raw days since update
            device: Target device
            
        Returns:
            penalty: Attention logit penalty [batch, 1, key_len] for broadcasting
        """
        # Get staleness weight
        if self.staleness_learnable:
            weight = torch.exp(self._staleness_log_weight)
        else:
            weight = self._staleness_weight_fixed
        
        # Compute decay-scaled staleness
        if self.staleness_decay == 'prenormalized':
            # Input is already normalized to [0, 1] (from data preprocessing)
            # Use directly without additional transformation
            # This is robust to different frequencies (daily vs weekly)
            scaled = staleness
            
        elif self.staleness_decay == 'linear':
            # Linear: penalty = weight * (days / max_days)
            scaled = staleness / self.staleness_max_days
            
        elif self.staleness_decay == 'exponential':
            # Exponential: penalty = weight * (1 - exp(-days / half_life))
            scaled = 1.0 - torch.exp(-staleness / self.staleness_half_life)
            
        elif self.staleness_decay == 'log':
            # Logarithmic: penalty = weight * log(1 + days) / log(1 + max_days)
            # Compresses large values, matches data preprocessing
            log_max = math.log(1.0 + self.staleness_max_days)
            scaled = torch.log1p(staleness) / log_max
            
        elif self.staleness_decay == 'step':
            # Step: penalty = weight if days > threshold else 0
            scaled = (staleness > self.staleness_threshold).float()
        
        # Clamp to [0, 1] for stability
        scaled = torch.clamp(scaled, 0.0, 1.0)
        
        # Final penalty: [batch, key_len]
        penalty = weight * scaled
        
        # Add dimension for broadcasting: [batch, 1, key_len]
        penalty = penalty.unsqueeze(1)
        
        # Cache for diagnostics
        self._cached_staleness_penalty = penalty.detach()
        
        return penalty
    
    def _get_regime_gates(
        self,
        batch_size: int,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Get per-sample regime gate values.
        
        Returns:
            gate_weights: [batch_size, n_head] or None
        """
        if self.regime_mode == 'disabled' or self._current_regime is None:
            return None
        
        regime_indices = self._current_regime.to(device)
        raw_gates = self.regime_gates[regime_indices]
        gate_weights = torch.sigmoid(raw_gates)
        
        self._cached_regime_gates = gate_weights.detach()
        
        return gate_weights
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with staleness penalties and regime gating.
        
        Args:
            q: Query tensor [batch, query_len, d_model]
            k: Key tensor [batch, key_len, d_model]
            v: Value tensor [batch, key_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attended output [batch, query_len, d_model]
            attn_weights: Attention weights [batch, n_head, query_len, key_len]
        """
        batch_size = q.size(0)
        key_len = k.size(1)
        device = q.device
        
        # Shared value projection
        v_proj = self.v_layer(v)  # [batch, key_len, d_v]
        
        # Compute staleness penalty if enabled
        staleness_penalty = None
        if self.staleness_mode != 'disabled' and self._staleness_signal is not None:
            staleness_penalty = self._compute_staleness_penalty(
                self._staleness_signal, device
            )
        
        # Get regime gates if enabled
        regime_gates = self._get_regime_gates(batch_size, device)
        
        # Per-head attention computation
        head_outputs = []
        head_attns = []
        
        for h in range(self.n_head):
            # Project Q and K for this head
            q_h = self.q_layers[h](q)  # [batch, query_len, d_q]
            k_h = self.k_layers[h](k)  # [batch, key_len, d_k]
            v_h = v_proj  # [batch, key_len, d_v]
            
            # Attention scores: [batch, query_len, key_len]
            attn_scores = torch.bmm(q_h, k_h.transpose(1, 2))
            attn_scores = attn_scores / math.sqrt(self.d_k)
            
            # Apply staleness penalty (BEFORE softmax)
            # Subtracts from logits: stale keys get lower scores
            if staleness_penalty is not None:
                attn_scores = attn_scores - staleness_penalty
            
            # Apply mask
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, self.mask_bias)
            
            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply regime gating (AFTER softmax)
            if regime_gates is not None:
                gate_h = regime_gates[:, h].unsqueeze(-1).unsqueeze(-1)
                attn_weights = attn_weights * gate_h
            
            # Compute attended values
            head_out = torch.bmm(attn_weights, v_h)  # [batch, query_len, d_v]
            
            head_outputs.append(head_out)
            head_attns.append(attn_weights)
        
        # Average head outputs (TFT interpretability constraint)
        stacked = torch.stack(head_outputs, dim=0)  # [n_head, batch, query_len, d_v]
        averaged = stacked.mean(dim=0)  # [batch, query_len, d_v]
        
        # Cache for analysis
        self._cached_head_contributions = stacked.detach()
        
        # Final output projection
        output = self.w_h(averaged)  # [batch, query_len, d_model]
        
        # Stack attention weights
        attn_weights_stacked = torch.stack(head_attns, dim=1)
        self._cached_attention_weights = attn_weights_stacked.detach()
        
        return output, attn_weights_stacked
    
    def get_diagnostics(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get diagnostic information for analysis."""
        diagnostics = {
            'staleness_mode': self.staleness_mode,
            'regime_mode': self.regime_mode,
            'staleness_signal': self._staleness_signal,
            'staleness_penalty': self._cached_staleness_penalty,
            'regime_signal': self._regime_signal,
            'current_regime': self._current_regime,
            'attention_weights': self._cached_attention_weights,
        }
        
        if self.staleness_mode != 'disabled':
            diagnostics['staleness_weight'] = self.staleness_weight
            diagnostics['staleness_decay'] = self.staleness_decay
        
        if self.regime_mode != 'disabled':
            diagnostics['regime_gates'] = self._cached_regime_gates
            diagnostics['raw_gates'] = self.regime_gates.detach().cpu()
        
        return diagnostics
    
    def extra_repr(self) -> str:
        parts = [
            f"n_head={self.n_head}",
            f"d_model={self.d_model}",
            f"staleness_mode='{self.staleness_mode}'",
        ]
        if self.staleness_mode != 'disabled':
            parts.extend([
                f"staleness_decay='{self.staleness_decay}'",
                f"staleness_weight={self.staleness_weight:.3f}",
                f"staleness_learnable={self.staleness_learnable}",
            ])
        parts.append(f"regime_mode='{self.regime_mode}'")
        if self.regime_mode != 'disabled':
            parts.append(f"vix_threshold={self.vix_threshold}")
        return ", ".join(parts)


def replace_attention_with_staleness(
    model: nn.Module,
    staleness_mode: str = 'penalty',
    staleness_decay: str = 'log',
    staleness_weight: float = 0.5,
    staleness_learnable: bool = True,
    staleness_max_days: float = 45.0,
    staleness_grad_scale: float = 100.0,
    regime_mode: str = 'disabled',
    vix_threshold: float = 25.0,
    num_regimes: int = 2,
    gate_grad_scale: float = 100.0,
    gate_init: str = 'neutral'
) -> nn.Module:
    """
    Replace TFT's attention module with staleness-aware version.
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        TFT model instance
    staleness_mode : str
        'disabled' or 'penalty'
    staleness_decay : str
        'linear', 'exponential', 'log', or 'step'
    staleness_weight : float
        Penalty magnitude
    staleness_learnable : bool
        Whether weight is learnable
    staleness_max_days : float
        Normalization factor
    regime_mode : str
        'disabled' or 'vix_threshold'
    vix_threshold : float
        VIX threshold for regime detection
    num_regimes : int
        Number of regimes
    gate_grad_scale : float
        Gradient scaling for regime gates
    gate_init : str
        'neutral' or 'separated'
        
    Returns
    -------
    model : Modified model (in-place)
    """
    if not hasattr(model, 'multihead_attn'):
        raise ValueError("Model must have 'multihead_attn' attribute")
    
    old_attn = model.multihead_attn
    
    # Extract parameters from existing attention
    n_head = old_attn.n_head
    d_model = old_attn.d_model
    dropout = old_attn.dropout.p if hasattr(old_attn.dropout, 'p') else 0.0
    mask_bias = getattr(old_attn, 'mask_bias', -1e9)
    
    # Create staleness-aware replacement
    new_attn = StalenessAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=dropout,
        mask_bias=mask_bias,
        staleness_mode=staleness_mode,
        staleness_decay=staleness_decay,
        staleness_weight=staleness_weight,
        staleness_learnable=staleness_learnable,
        staleness_max_days=staleness_max_days,
        staleness_grad_scale=staleness_grad_scale,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold,
        num_regimes=num_regimes,
        gate_grad_scale=gate_grad_scale,
        gate_init=gate_init
    )
    
    # Copy weights from original attention
    new_attn.v_layer.load_state_dict(old_attn.v_layer.state_dict())
    new_attn.w_h.load_state_dict(old_attn.w_h.state_dict())
    
    for i in range(n_head):
        new_attn.q_layers[i].load_state_dict(old_attn.q_layers[i].state_dict())
        new_attn.k_layers[i].load_state_dict(old_attn.k_layers[i].state_dict())
    
    # Replace in model
    model.multihead_attn = new_attn
    
    # Summary
    print(f"\n[STALENESS ATTENTION] Replaced attention module:")
    print(f"  Staleness mode: {staleness_mode}")
    if staleness_mode != 'disabled':
        print(f"  Staleness decay: {staleness_decay}")
        print(f"  Staleness weight: {staleness_weight} ({'learnable' if staleness_learnable else 'fixed'})")
        print(f"  Max days: {staleness_max_days}")
    print(f"  Regime mode: {regime_mode}")
    if regime_mode != 'disabled':
        print(f"  VIX threshold: {vix_threshold}")
        print(f"  Regime gates: {num_regimes} × {n_head}")
    
    return model


def find_staleness_index(
    model: nn.Module,
    staleness_feature_name: str = 'days_since_CPI_update'
) -> Optional[int]:
    """
    Find index of staleness feature in model's continuous features.
    
    Args:
        model: TFT model with hparams
        staleness_feature_name: Name of staleness feature
        
    Returns:
        Index in x_reals, or None if not found
    """
    if not hasattr(model, 'hparams'):
        return None
    
    x_reals = getattr(model.hparams, 'x_reals', [])
    
    if staleness_feature_name in x_reals:
        return x_reals.index(staleness_feature_name)
    
    # Try common variations
    variations = [
        staleness_feature_name,
        'days_since_CPI_update',
        'days_since_cpi_update',
        'CPI_staleness',
        'staleness_CPI',
    ]
    
    for var in variations:
        if var in x_reals:
            return x_reals.index(var)
    
    return None


def find_vix_index(model: nn.Module, vix_feature_name: str = 'VIX') -> Optional[int]:
    """Find index of VIX feature in model's continuous features."""
    if not hasattr(model, 'hparams'):
        return None
    
    x_reals = getattr(model.hparams, 'x_reals', [])
    
    variations = ['VIX', 'vix', 'VIX_close', 'vix_close', 'VIXCLS']
    for var in [vix_feature_name] + variations:
        if var in x_reals:
            return x_reals.index(var)
    
    return None


def patch_forward_for_staleness(
    model: nn.Module,
    staleness_feature_name: str = 'days_since_CPI_update',
    vix_feature_name: str = 'VIX',
    staleness_is_normalized: bool = True,
    staleness_max_days: float = 45.0
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
        staleness_is_normalized: If True, denormalize from [0,1] to raw days
        staleness_max_days: Max days for denormalization
        
    Returns:
        model: Modified model (in-place)
    """
    # Find feature indices
    staleness_idx = find_staleness_index(model, staleness_feature_name)
    vix_idx = find_vix_index(model, vix_feature_name)
    
    if staleness_idx is not None:
        print(f"[STALENESS PATCH] Found '{staleness_feature_name}' at index {staleness_idx}")
    else:
        print(f"[STALENESS PATCH] WARNING: '{staleness_feature_name}' not found in features")
    
    if vix_idx is not None:
        print(f"[STALENESS PATCH] Found '{vix_feature_name}' at index {vix_idx}")
    else:
        print(f"[STALENESS PATCH] WARNING: '{vix_feature_name}' not found in features")
    
    # Store original forward
    original_forward = model.forward
    
    def patched_forward(x):
        # Extract staleness for all key timesteps
        if staleness_idx is not None and hasattr(model, 'multihead_attn'):
            encoder_cont = x.get('encoder_cont')
            decoder_cont = x.get('decoder_cont')
            
            # Keys span encoder + decoder
            if encoder_cont is not None:
                staleness_enc = encoder_cont[:, :, staleness_idx]  # [batch, enc_len]
                
                if decoder_cont is not None and decoder_cont.size(-1) > staleness_idx:
                    staleness_dec = decoder_cont[:, :, staleness_idx]  # [batch, dec_len]
                    staleness_full = torch.cat([staleness_enc, staleness_dec], dim=1)
                else:
                    staleness_full = staleness_enc
                
                # Denormalize if needed (log transform was applied in preprocessing)
                if staleness_is_normalized:
                    # Inverse of: normalized = log(1 + days) / log(1 + max_days)
                    # days = exp(normalized * log(1 + max_days)) - 1
                    log_max = math.log(1.0 + staleness_max_days)
                    staleness_full = torch.exp(staleness_full * log_max) - 1.0
                
                model.multihead_attn.set_staleness_signal(staleness_full)
        
        # Extract VIX for regime detection (last encoder timestep)
        if vix_idx is not None and hasattr(model, 'multihead_attn'):
            encoder_cont = x.get('encoder_cont')
            if encoder_cont is not None:
                vix = encoder_cont[:, -1, vix_idx]
                model.multihead_attn.set_regime_signal(vix)
        
        # Call original forward
        return original_forward(x)
    
    # Replace forward method
    model.forward = patched_forward
    
    return model


# Convenience function for common setup
def setup_staleness_aware_tft(
    model: nn.Module,
    staleness_mode: str = 'penalty',
    staleness_decay: str = 'log',
    staleness_weight: float = 0.5,
    staleness_learnable: bool = True,
    regime_mode: str = 'disabled',
    vix_threshold: float = 25.0,
    staleness_feature_name: str = 'days_since_CPI_update',
    staleness_is_normalized: bool = True
) -> nn.Module:
    """
    Complete setup for staleness-aware TFT.
    
    Combines replace_attention_with_staleness and patch_forward_for_staleness.
    
    Returns:
        model: Fully configured model
    """
    # Replace attention module
    model = replace_attention_with_staleness(
        model,
        staleness_mode=staleness_mode,
        staleness_decay=staleness_decay,
        staleness_weight=staleness_weight,
        staleness_learnable=staleness_learnable,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold
    )
    
    # Patch forward for signal threading
    model = patch_forward_for_staleness(
        model,
        staleness_feature_name=staleness_feature_name,
        staleness_is_normalized=staleness_is_normalized
    )
    
    return model
