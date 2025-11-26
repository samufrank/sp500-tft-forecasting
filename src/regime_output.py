"""
Regime-conditional output layer for Temporal Fusion Transformer.

Implements Mixture-of-Experts architecture for regime-adaptive predictions.
Drop-in replacement for TFT's standard output layer.

Usage:
    # After model creation
    from src.regime_output import RegimeConditionalOutput
    
    model.output_layer = RegimeConditionalOutput(
        hidden_size=model.hparams.hidden_size,
        output_size=model.hparams.output_size,
        num_regimes=2,
        routing_mode='learned'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class RegimeConditionalOutput(nn.Module):
    """
    Mixture-of-Experts output layer for regime-conditional predictions.
    
    Replaces TFT's standard nn.Linear output layer with multiple expert heads
    that specialize in different market regimes (e.g., high/low volatility).
    A learned router dynamically weights expert predictions.
    
    Parameters
    ----------
    hidden_size : int
        TFT hidden dimension (input size)
    output_size : int
        Number of quantiles (typically 7 for QuantileLoss)
    num_regimes : int, default=2
        Number of expert heads (e.g., 2 for normal/volatile regimes)
    routing_mode : str, default='learned'
        Routing strategy:
        - 'learned': Learn regime detection from hidden state
        - 'disabled': Single expert (equivalent to baseline)
    
    Notes
    -----
    - Input shape: [batch, seq_len, hidden_size]
    - Output shape: [batch, seq_len, output_size]
    - Routing happens per timestep in sequence (supports multi-horizon)
    
    Examples
    --------
    >>> # Replace TFT's output layer
    >>> model = TemporalFusionTransformer.from_dataset(dataset)
    >>> model.output_layer = RegimeConditionalOutput(
    ...     hidden_size=16,
    ...     output_size=7,
    ...     num_regimes=2
    ... )
    
    >>> # Extract diagnostics during evaluation
    >>> pred, diagnostics = model.output_layer(hidden_state, return_diagnostics=True)
    >>> routing_weights = diagnostics['routing_weights']  # [batch, seq_len, num_regimes]
    >>> expert_preds = diagnostics['expert_preds']  # List of [batch, seq_len, output_size]
    """
    
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_regimes: int = 2,
        routing_mode: str = 'learned',
        routing_strategy: str = 'learned',
        vix_threshold: float = 25.0
    ):
        super().__init__()
        
        if routing_mode not in ['learned', 'disabled']:
            raise ValueError(
                f"routing_mode must be 'learned' or 'disabled', got '{routing_mode}'"
            )
        
        if routing_strategy not in ['learned', 'vix_threshold']:
            raise ValueError(
                f"routing_strategy must be 'learned' or 'vix_threshold', got '{routing_strategy}'"
            )
        
        if num_regimes < 1:
            raise ValueError(f"num_regimes must be >= 1, got {num_regimes}")
        
        if routing_mode == 'disabled' and num_regimes != 1:
            raise ValueError(
                f"routing_mode='disabled' requires num_regimes=1, got {num_regimes}"
            )
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_regimes = num_regimes
        self.routing_mode = routing_mode
        self.routing_strategy = routing_strategy
        self.vix_threshold = vix_threshold
        
        # Load balancing parameters
        self.load_balance_weight = 2.0  # Auxiliary loss weight (tunable)
        self.register_buffer('_routing_history', None, persistent=False)  # For loss computation
        
        # Create expert heads
        if routing_mode == 'disabled':
            # Single expert (baseline equivalent)
            self.experts = nn.ModuleList([
                nn.Linear(hidden_size, output_size)
            ])
        else:
            # Multiple experts (one per regime)
            self.experts = nn.ModuleList([
                nn.Linear(hidden_size, output_size)
                for _ in range(num_regimes)
            ])
            
            # Router: learns regime from hidden state
            if routing_mode == 'learned':
                self.router = nn.Linear(hidden_size, num_regimes)
                # Initialize to zero for balanced routing at start
                # This prevents random initialization bias that causes winner-takes-all
                nn.init.zeros_(self.router.weight)
                nn.init.zeros_(self.router.bias)
        
        # Diagnostic caches (non-persistent - not saved to checkpoint)
        # These store detached tensors on CPU for monitoring without memory buildup
        self.register_buffer('_cached_routing_weights', None, persistent=False)
        self.register_buffer('_cached_expert_preds_0', None, persistent=False)
        if num_regimes > 1:
            self.register_buffer('_cached_expert_preds_1', None, persistent=False)
        if num_regimes > 2:
            for i in range(2, num_regimes):
                self.register_buffer(f'_cached_expert_preds_{i}', None, persistent=False)
        
        # Cache for load balancing loss (computed during forward pass)
        self.register_buffer('_cached_lb_loss', None, persistent=False)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        vix_values: torch.Tensor = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional diagnostics.
        
        Parameters
        ----------
        hidden_state : torch.Tensor
            TFT's final hidden state, shape [batch, seq_len, hidden_size]
        vix_values : torch.Tensor, optional
            VIX values for VIX-based routing, shape [batch]
        return_diagnostics : bool, default=False
            If True, return (prediction, diagnostics_dict)
        
        Returns
        -------
        prediction : torch.Tensor
            Quantile predictions, shape [batch, seq_len, output_size]
        diagnostics : dict, optional
            Dictionary with keys:
            - 'routing_weights': [batch, seq_len, num_regimes] - soft routing weights
            - 'expert_preds': List of [batch, seq_len, output_size] - individual expert predictions
        """
        if self.routing_mode == 'disabled':
            # Baseline mode: single expert
            pred = self.experts[0](hidden_state)
            
            if return_diagnostics:
                return pred, {
                    'routing_weights': None,
                    'expert_preds': [pred]
                }
            return pred
        
        # Get predictions from all experts
        # Each: [batch, seq_len, hidden_size] -> [batch, seq_len, output_size]
        expert_preds = [expert(hidden_state) for expert in self.experts]
        
        # Compute routing weights based on strategy
        """
        if self.routing_strategy == 'vix_threshold':
            # Check for VIX values (passed as parameter or cached)
            vix = vix_values
        """
        if self.routing_strategy == 'vix_threshold':
            vix = vix_values or getattr(self, '_vix_for_forward', None)
            if vix is not None and not hasattr(self, '_vix_logged_this_epoch'):
                print(f"[DEBUG] VIX routing - batch vix: min={vix.min().item():.1f}, max={vix.max().item():.1f}, mean={vix.mean().item():.1f}")
                self._vix_logged_this_epoch = True

            if vix is None and hasattr(self, '_vix_for_forward'):
                vix = self._vix_for_forward
            
            if vix is not None:
                # Deterministic VIX-based routing
                batch_size = hidden_state.size(0)
                seq_len = hidden_state.size(1)
                
                # VIX threshold routing: high VIX → regime 1, low VIX → regime 0
                # vix shape: [batch]
                is_high_vol = (vix > self.vix_threshold).float()  # [batch]
                
                # Create routing weights: [batch, num_regimes]
                routing_weights_2d = torch.stack([1 - is_high_vol, is_high_vol], dim=1)

                #print(f"[DEBUG] VIX routing - first sample: vix={vix[0].item():.1f}, weights={routing_weights_2d[0].cpu().numpy()}")
                
                # Expand to [batch, seq_len, num_regimes] for multi-step predictions
                routing_weights = routing_weights_2d.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Fallback to learned routing if VIX not available
                router_logits = self.router(hidden_state)
                routing_weights = F.softmax(router_logits, dim=-1)
        else:
            # Learned routing (default)
            router_logits = self.router(hidden_state)  # [batch, seq_len, num_regimes]
            routing_weights = F.softmax(router_logits, dim=-1)  # Soft routing over regimes
        
        # Cache diagnostics for monitoring (detached, on CPU to prevent memory buildup)
        # Only cache during validation (not training) to avoid overhead
        if not self.training:
            self._cached_routing_weights = routing_weights.detach().cpu()
            for i, expert_pred in enumerate(expert_preds):
                setattr(self, f'_cached_expert_preds_{i}', expert_pred.detach().cpu())
        else:
            # During training, compute and cache load balancing loss
            avg_routing = routing_weights.mean(dim=(0, 1))  # [num_regimes]
            target = torch.ones_like(avg_routing) / self.num_regimes
            lb_loss = F.mse_loss(avg_routing, target) * self.load_balance_weight
            self._cached_lb_loss = lb_loss  # Store for loss function to add
        
        # Weighted combination of expert predictions
        # Stack: [batch, seq_len, num_regimes, output_size]
        stacked_preds = torch.stack(expert_preds, dim=2)
        
        # Expand routing weights: [batch, seq_len, num_regimes, 1]
        weights_expanded = routing_weights.unsqueeze(-1)
        
        # Weighted sum: [batch, seq_len, output_size]
        final_pred = torch.sum(stacked_preds * weights_expanded, dim=2)
        
        if return_diagnostics:
            return final_pred, {
                'routing_weights': routing_weights,
                'expert_preds': expert_preds
            }
        
        return final_pred
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Penalizes deviation from uniform expert usage to prevent
        winner-takes-all routing collapse.
        
        Returns
        -------
        torch.Tensor
            Scalar loss term (0.0 if no routing or disabled mode)
        
        Notes
        -----
        Based on Switch Transformer load balancing (Fedus et al. 2021).
        Loss = alpha * sum((f_i - 1/K)^2) where:
        - f_i = fraction of samples routed to expert i
        - K = number of experts
        - alpha = load_balance_weight hyperparameter
        """
        if self.routing_mode == 'disabled' or self._routing_history is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Compute average routing weights across batch and time
        # routing_history shape: [batch, seq_len, num_regimes]
        avg_routing = self._routing_history.mean(dim=(0, 1))  # [num_regimes]
        
        # Target: uniform distribution [1/K, 1/K, ...]
        target = torch.ones_like(avg_routing) / self.num_regimes
        
        # L2 penalty on deviation from uniform
        loss = F.mse_loss(avg_routing, target)
        
        return self.load_balance_weight * loss
    
    def get_expert_parameters(self) -> Dict[str, int]:
        """
        Get parameter counts for analysis.
        
        Returns
        -------
        dict
            Parameter counts: {'total', 'experts', 'router'}
        """
        expert_params = sum(
            p.numel() for expert in self.experts for p in expert.parameters()
        )
        
        router_params = 0
        if hasattr(self, 'router'):
            router_params = sum(p.numel() for p in self.router.parameters())
        
        return {
            'total': expert_params + router_params,
            'experts': expert_params,
            'router': router_params
        }
    
    def extra_repr(self) -> str:
        """String representation for print/debug."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, "
            f"num_regimes={self.num_regimes}, "
            f"routing_mode='{self.routing_mode}'"
        )


def replace_output_layer(
    model: nn.Module,
    num_regimes: int = 2,
    routing_mode: str = 'learned',
    routing_strategy: str = 'learned',
    vix_threshold: float = 25.0
) -> nn.Module:
    """
    Replace TFT's output layer with RegimeConditionalOutput.
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        TFT model instance
    num_regimes : int, default=2
        Number of expert heads
    routing_mode : str, default='learned'
        'learned' or 'disabled'
    routing_strategy : str, default='learned'
        'learned' (MLP router) or 'vix_threshold' (deterministic VIX routing)
    vix_threshold : float, default=25.0
        VIX threshold for deterministic routing (only used if routing_strategy='vix_threshold')
    
    Returns
    -------
    model : TemporalFusionTransformer
        Modified model (in-place)
    
    Examples
    --------
    >>> # Learned routing
    >>> model = TemporalFusionTransformer.from_dataset(dataset)
    >>> model = replace_output_layer(model, num_regimes=2, routing_strategy='learned')
    >>> 
    >>> # VIX-based routing
    >>> model = replace_output_layer(model, num_regimes=2, routing_strategy='vix_threshold', vix_threshold=25.0)
    """
    if not hasattr(model, 'output_layer'):
        raise ValueError("Model must have 'output_layer' attribute")
    
    if not hasattr(model.hparams, 'hidden_size') or not hasattr(model.hparams, 'output_size'):
        raise ValueError("Model must have hparams.hidden_size and hparams.output_size")
    
    # Handle multi-target case (not yet implemented - for future)
    if isinstance(model.output_layer, nn.ModuleList):
        raise NotImplementedError(
            "Multi-target regime output not yet implemented. "
            "Currently only supports single-target TFT."
        )
    
    # Replace with regime-conditional output
    model.output_layer = RegimeConditionalOutput(
        hidden_size=model.hparams.hidden_size,
        output_size=model.hparams.output_size,
        num_regimes=num_regimes,
        routing_mode=routing_mode,
        routing_strategy=routing_strategy,
        vix_threshold=vix_threshold
    )
    
    print(f"\n[REGIME OUTPUT] Replaced output layer:")
    print(f"  Mode: {routing_mode}")
    print(f"  Strategy: {routing_strategy}")
    if routing_strategy == 'vix_threshold':
        print(f"  VIX threshold: {vix_threshold}")
    print(f"  Experts: {num_regimes}")
    print(f"  Parameters: {model.output_layer.get_expert_parameters()}")
    
    return model
