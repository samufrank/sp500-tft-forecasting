"""
Regime-Aware Interpretable Multi-Head Attention for TFT.

Drop-in replacement for pytorch-forecasting's InterpretableMultiHeadAttention
that modulates attention based on market regime (detected via VIX threshold).

Architecture:
    Standard InterpretableMultiHeadAttention with regime-conditional scaling:
    1. Compute attention logits as normal: attn = softmax(QK^T / sqrt(d_k))
    2. Detect regime from VIX signal (threshold-based, deterministic)
    3. Apply learned per-head regime gates: attn_scaled = attn * gate[regime, head]
    
    The regime gates are small scalar multipliers per head that learn to
    amplify/dampen specific heads based on current market conditions.

Design Rationale:
    - Your attention analysis shows the encoder already adapts behavior by regime
      (lower entropy in 2022, concentration on recent timesteps during volatility)
    - VIX is already the most-attended feature
    - Hypothesis: explicit regime modulation provides stronger gradient signal
      to the output layer, not that attention isn't adapting at all
    - Minimal parameter addition to avoid capacity-collapse (hidden_size > 16-18)

Usage:
    # Replace attention module after model creation
    from regime_attention import replace_attention_module
    
    model = TemporalFusionTransformer.from_dataset(dataset, ...)
    model = replace_attention_module(
        model,
        regime_mode='vix_threshold',
        vix_threshold=25.0,
        num_regimes=2
    )

References:
    - Original TFT: Lim et al. (2021) "Temporal Fusion Transformers for 
      Interpretable Multi-horizon Time Series Forecasting"
    - Regime-switching: Hamilton (1989) regime-switching models
    - Mixture of Experts gating: Shazeer et al. (2017) "Outrageously Large
      Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
"""

import math
from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (copied from pytorch-forecasting for standalone use).
    
    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = None, scale: bool = True, mask_bias: float = -1e9):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        self.mask_bias = mask_bias
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch * n_heads, query_len, d_k]
            k: Key tensor [batch * n_heads, key_len, d_k]  
            v: Value tensor [batch * n_heads, key_len, d_v]
            mask: Optional attention mask [batch, query_len, key_len]
            
        Returns:
            output: Attended values [batch * n_heads, query_len, d_v]
            attn: Attention weights [batch * n_heads, query_len, key_len]
        """
        # Compute attention scores
        attn = torch.bmm(q, k.permute(0, 2, 1))  # [batch*heads, q_len, k_len]
        
        if self.scale:
            dimension = torch.as_tensor(
                k.size(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension
        
        if mask is not None:
            attn = attn.masked_fill(mask, self.mask_bias)
        
        attn = self.softmax(attn)
        
        if self.dropout is not None:
            attn = self.dropout(attn)
        
        output = torch.bmm(attn, v)
        return output, attn


class RegimeAwareInterpretableMultiHeadAttention(nn.Module):
    """
    Regime-Aware Interpretable Multi-Head Attention.
    
    Drop-in replacement for pytorch-forecasting's InterpretableMultiHeadAttention
    with regime-conditional head gating.
    
    Key difference from standard attention:
        After computing attention weights, applies learned per-head scaling
        factors conditioned on detected market regime:
        
        attn_final[h] = attn[h] * sigmoid(regime_gate[regime, h])
        
    Parameters
    ----------
    n_head : int
        Number of attention heads
    d_model : int  
        Model dimension (hidden_size)
    dropout : float, default=0.0
        Dropout rate
    mask_bias : float, default=-1e9
        Bias for masked positions
    num_regimes : int, default=2
        Number of market regimes (2 = low/high volatility)
    regime_mode : str, default='vix_threshold'
        How to detect regime:
        - 'vix_threshold': Deterministic threshold on VIX feature
        - 'disabled': No regime modulation (baseline behavior)
    vix_threshold : float, default=25.0
        VIX level above which regime=1 (high volatility)
    gate_init_std : float, default=0.01
        Std for regime gate initialization (small = near-identity start)
        
    Notes
    -----
    - Maintains exact same interface as InterpretableMultiHeadAttention
    - VIX must be passed through set_regime_signal() before forward()
    - Regime gates add only n_heads * num_regimes parameters (~8 for typical config)
    - Designed to avoid capacity-collapse by minimal parameter addition
    """
    
    def __init__(
        self,
        n_head: int,
        d_model: int,
        dropout: float = 0.0,
        mask_bias: float = -1e9,
        num_regimes: int = 2,
        regime_mode: str = 'vix_threshold',
        vix_threshold: float = 25.0,
        gate_init_std: float = 0.01
    ):
        super().__init__()
        
        if regime_mode not in ['vix_threshold', 'disabled']:
            raise ValueError(f"regime_mode must be 'vix_threshold' or 'disabled', got '{regime_mode}'")
        
        self.n_head = n_head
        self.d_model = d_model
        self.mask_bias = mask_bias
        self.num_regimes = num_regimes
        self.regime_mode = regime_mode
        self.vix_threshold = vix_threshold
        
        # Head dimensions (same as original InterpretableMultiHeadAttention)
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        
        # Shared value projection (interpretability constraint from original TFT)
        self.v_layer = nn.Linear(d_model, self.d_v)
        
        # Per-head Q and K projections
        self.q_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_q) for _ in range(n_head)
        ])
        self.k_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_k) for _ in range(n_head)
        ])
        
        # Attention computation
        self.attention = ScaledDotProductAttention(mask_bias=mask_bias)
        
        # Output projection
        self.w_h = nn.Linear(self.d_v, d_model, bias=False)
        
        # === REGIME-AWARE COMPONENTS ===
        if regime_mode != 'disabled':
            # Learnable per-head gates for each regime
            # Shape: [num_regimes, n_head]
            # Initialized near zero so sigmoid(gate) ≈ 0.5 (neutral scaling)
            self.regime_gates = nn.Parameter(
                torch.randn(num_regimes, n_head) * gate_init_std
            )

            # Scale up gradients for gates (they're far downstream, get weak signal)
            self.regime_gates.register_hook(lambda grad: grad * 100)
            
            # Optional: per-regime bias for attention logits (more expressive)
            # Disabled by default to keep minimal
            self.use_logit_bias = False
            if self.use_logit_bias:
                self.regime_logit_bias = nn.Parameter(
                    torch.zeros(num_regimes, n_head)
                )
        
        # Buffer for regime signal (set externally before forward)
        self.register_buffer('_regime_signal', None, persistent=False)
        self.register_buffer('_current_regime', None, persistent=False)
        
        # Diagnostic caches
        self.register_buffer('_cached_attention_weights', None, persistent=False)
        self.register_buffer('_cached_regime_gates', None, persistent=False)
        self.register_buffer('_cached_head_contributions', None, persistent=False)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier uniform (matches original TFT)."""
        for name, p in self.named_parameters():
            if "bias" not in name and "regime_gate" not in name:
                torch.nn.init.xavier_uniform_(p)
            elif "bias" in name:
                torch.nn.init.zeros_(p)
    
    def set_regime_signal(self, vix_values: torch.Tensor) -> None:
        """
        Set the regime signal for current batch.
        
        Must be called before forward() when regime_mode='vix_threshold'.
        
        Args:
            vix_values: VIX values, shape [batch] or [batch, seq_len]
                       If 2D, uses the last timestep (most recent VIX)
        """
        if vix_values is None:
            self._regime_signal = None
            self._current_regime = None
            return
            
        # Handle both [batch] and [batch, seq_len] inputs
        if vix_values.dim() == 2:
            # Use last timestep (most recent)
            vix_values = vix_values[:, -1]
        
        self._regime_signal = vix_values.detach()
        
        # Compute regime assignment
        if self.regime_mode == 'vix_threshold':
            # Binary: 0 = low vol (VIX < threshold), 1 = high vol (VIX >= threshold)
            self._current_regime = (vix_values >= self.vix_threshold).long()
        else:
            self._current_regime = torch.zeros_like(vix_values, dtype=torch.long)
    
    def _get_regime_gate_weights(self, regime_indices: torch.Tensor) -> torch.Tensor:
        """
        Get per-head gate weights for given regime indices.
        
        Args:
            regime_indices: [batch] tensor of regime assignments (0 or 1)
            
        Returns:
            gate_weights: [batch, n_head] sigmoid-activated gate values
        """
        # Index into regime gates: [batch, n_head]
        gates = self.regime_gates[regime_indices]  # [batch, n_head]
        
        # Sigmoid activation to get [0, 1] scaling factors
        # With small init, starts near 0.5 (neutral)
        gate_weights = torch.sigmoid(gates)
        
        return gate_weights
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with regime-conditional attention.
        
        Interface matches InterpretableMultiHeadAttention exactly.
        
        Args:
            q: Query tensor [batch, query_len, d_model]
            k: Key tensor [batch, key_len, d_model]
            v: Value tensor [batch, key_len, d_model]
            mask: Optional attention mask [batch, query_len, key_len]
            
        Returns:
            outputs: Attended output [batch, query_len, d_model]
            attn: Attention weights [batch, query_len, n_head, key_len]
        """
        batch_size = q.size(0)
        
        # Compute value projection (shared across heads for interpretability)
        vs = self.v_layer(v)  # [batch, key_len, d_v]
        
        heads = []
        attns = []
        
        for i in range(self.n_head):
            # Per-head Q, K projections
            qs = self.q_layers[i](q)  # [batch, query_len, d_q]
            ks = self.k_layers[i](k)  # [batch, key_len, d_k]
            
            # Compute attention
            head, attn = self.attention(qs, ks, vs, mask)  # attn: [batch, q_len, k_len]
            
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        
        # Stack heads: [batch, query_len, n_head, d_v] and [batch, query_len, n_head, key_len]
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0].unsqueeze(2)
        attn = torch.stack(attns, dim=2)  # [batch, query_len, n_head, key_len]
        
        # === REGIME-CONDITIONAL GATING ===
        if self.regime_mode != 'disabled' and self._current_regime is not None:
            # Get gate weights: [batch, n_head]
            gate_weights = self._get_regime_gate_weights(self._current_regime)
            
            # Cache for diagnostics
            self._cached_regime_gates = gate_weights.detach().cpu()
            
            # Apply gating to attention weights
            # Expand gates: [batch, 1, n_head, 1] to broadcast over query_len and key_len
            gate_weights_expanded = gate_weights.unsqueeze(1).unsqueeze(-1)
            
            # Scale attention weights by regime-specific head gates
            # This modulates which heads contribute more based on regime
            attn_gated = attn * gate_weights_expanded
            
            # Re-normalize attention to sum to 1 (optional, preserves attention semantics)
            # Commented out: let the model learn whether to use normalized or scaled
            # attn_gated = attn_gated / (attn_gated.sum(dim=-1, keepdim=True) + 1e-9)
            
            # Also gate the head outputs for consistent gradient flow
            head_gated = head * gate_weights.unsqueeze(1).unsqueeze(-1)
            
            # Store gated attention for analysis
            self._cached_attention_weights = attn_gated.detach()
            
            # Use gated versions
            attn = attn_gated
            head = head_gated
        else:
            self._cached_attention_weights = attn.detach()
        
        # Combine heads (mean, not concat - matches original TFT)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head.squeeze(2)
        
        # Output projection
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        
        return outputs, attn
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get cached attention weights from last forward pass."""
        return self._cached_attention_weights
    
    def get_regime_diagnostics(self) -> Dict[str, torch.Tensor]:
        """
        Get diagnostic information about regime gating.
        
        Returns:
            dict with:
                - 'regime_signal': VIX values used for regime detection
                - 'current_regime': Detected regime indices
                - 'gate_weights': Per-head gate values applied
                - 'raw_gates': Pre-sigmoid gate parameters
        """
        diagnostics = {
            'regime_signal': self._regime_signal,
            'current_regime': self._current_regime,
        }
        
        if self.regime_mode != 'disabled':
            diagnostics['gate_weights'] = self._cached_regime_gates
            diagnostics['raw_gates'] = self.regime_gates.detach().cpu()
        
        return diagnostics
    
    def extra_repr(self) -> str:
        """String representation for print/debug."""
        return (
            f"n_head={self.n_head}, d_model={self.d_model}, "
            f"regime_mode='{self.regime_mode}', "
            f"num_regimes={self.num_regimes}, "
            f"vix_threshold={self.vix_threshold}"
        )


class RegimeAttentionTFT(nn.Module):
    """
    Wrapper that threads regime signals through TFT forward pass.
    
    This is a thin wrapper around an existing TFT model that:
    1. Extracts VIX values from input batch
    2. Sets regime signal on the attention module
    3. Calls original TFT forward
    
    Usage:
        base_model = TemporalFusionTransformer.from_dataset(...)
        base_model = replace_attention_module(base_model, ...)
        model = RegimeAttentionTFT(base_model, vix_feature_idx=5)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        vix_feature_name: str = 'VIX',
        vix_feature_idx: Optional[int] = None
    ):
        """
        Args:
            base_model: TFT model with RegimeAwareInterpretableMultiHeadAttention
            vix_feature_name: Name of VIX feature in x_reals
            vix_feature_idx: Index of VIX in continuous features (if known)
        """
        super().__init__()
        self.model = base_model
        self.vix_feature_name = vix_feature_name
        self._vix_feature_idx = vix_feature_idx
        
        # Try to find VIX index from model hparams
        if self._vix_feature_idx is None and hasattr(base_model, 'hparams'):
            x_reals = getattr(base_model.hparams, 'x_reals', [])
            if vix_feature_name in x_reals:
                self._vix_feature_idx = x_reals.index(vix_feature_name)
    
    def _extract_vix(self, x: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract VIX values from input batch."""
        if self._vix_feature_idx is None:
            return None
        
        # x['encoder_cont'] shape: [batch, encoder_len, n_cont_features]
        encoder_cont = x.get('encoder_cont')
        if encoder_cont is None:
            return None
        
        # Get VIX from last encoder timestep
        vix = encoder_cont[:, -1, self._vix_feature_idx]
        return vix
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with regime signal injection."""
        # Extract VIX and set regime signal
        vix = self._extract_vix(x)
        
        if hasattr(self.model, 'multihead_attn') and \
           hasattr(self.model.multihead_attn, 'set_regime_signal'):
            self.model.multihead_attn.set_regime_signal(vix)
        
        # Standard forward
        return self.model(x)
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def replace_attention_module(
    model: nn.Module,
    regime_mode: str = 'vix_threshold',
    vix_threshold: float = 25.0,
    num_regimes: int = 2,
    gate_init_std: float = 0.01
) -> nn.Module:
    """
    Replace TFT's attention module with regime-aware version.
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        TFT model instance
    regime_mode : str, default='vix_threshold'
        'vix_threshold' or 'disabled'
    vix_threshold : float, default=25.0
        VIX level for regime switching
    num_regimes : int, default=2
        Number of regimes
    gate_init_std : float, default=0.01
        Initialization std for regime gates
        
    Returns
    -------
    model : TemporalFusionTransformer
        Modified model (in-place)
    """
    if not hasattr(model, 'multihead_attn'):
        raise ValueError("Model must have 'multihead_attn' attribute")
    
    old_attn = model.multihead_attn
    
    # Extract parameters from existing attention
    n_head = old_attn.n_head
    d_model = old_attn.d_model
    dropout = old_attn.dropout.p if hasattr(old_attn.dropout, 'p') else 0.0
    mask_bias = getattr(old_attn, 'mask_bias', -1e9)
    
    # Create regime-aware replacement
    new_attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=dropout,
        mask_bias=mask_bias,
        num_regimes=num_regimes,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold,
        gate_init_std=gate_init_std
    )
    
    # Copy weights from original attention
    new_attn.v_layer.load_state_dict(old_attn.v_layer.state_dict())
    new_attn.w_h.load_state_dict(old_attn.w_h.state_dict())
    
    for i in range(n_head):
        new_attn.q_layers[i].load_state_dict(old_attn.q_layers[i].state_dict())
        new_attn.k_layers[i].load_state_dict(old_attn.k_layers[i].state_dict())
    
    # Replace in model
    model.multihead_attn = new_attn
    
    # Calculate parameter overhead
    regime_params = num_regimes * n_head  # Just the gate parameters
    total_attn_params = sum(p.numel() for p in new_attn.parameters())
    
    print(f"\n[REGIME ATTENTION] Replaced attention module:")
    print(f"  Mode: {regime_mode}")
    print(f"  Regimes: {num_regimes}")
    print(f"  VIX threshold: {vix_threshold}")
    print(f"  Heads: {n_head}")
    print(f"  New parameters: {regime_params} (regime gates only)")
    print(f"  Total attention parameters: {total_attn_params}")
    
    return model


def create_regime_attention_model(
    base_model: nn.Module,
    vix_feature_name: str = 'VIX',
    regime_mode: str = 'vix_threshold',
    vix_threshold: float = 25.0,
    num_regimes: int = 2
) -> RegimeAttentionTFT:
    """
    Create complete regime-aware TFT model.
    
    Convenience function that:
    1. Replaces attention module
    2. Wraps in RegimeAttentionTFT for automatic VIX extraction
    
    Parameters
    ----------
    base_model : TemporalFusionTransformer
        Base TFT model
    vix_feature_name : str
        Name of VIX feature in x_reals
    regime_mode : str
        'vix_threshold' or 'disabled'
    vix_threshold : float
        VIX threshold for regime detection
    num_regimes : int
        Number of regimes
        
    Returns
    -------
    RegimeAttentionTFT
        Wrapped model with regime-aware attention
    """
    # Replace attention module
    model = replace_attention_module(
        base_model,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold,
        num_regimes=num_regimes
    )
    
    # Wrap for automatic VIX extraction
    wrapped = RegimeAttentionTFT(
        model,
        vix_feature_name=vix_feature_name
    )
    
    return wrapped
