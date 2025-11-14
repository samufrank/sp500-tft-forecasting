"""
Custom Temporal Fusion Transformer (TFT) implementation.

This module provides a from-scratch TFT implementation that exactly matches
pytorch-forecasting's architecture for baseline validation, while allowing
future architectural modifications for:
- Regime-conditional output layers
- Distribution-aware loss functions
- Staleness-aware attention mechanisms

Architecture follows Lim et al. (2021) "Temporal Fusion Transformers for 
Interpretable Multi-horizon Time Series Forecasting" and is validated against
pytorch-forecasting source code.

Author: Sam Ehrle 
Date: November 2025
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple

from models.tft_components import (
    GatedResidualNetwork,
    VariableSelectionNetwork, 
    InterpretableMultiHeadAttention,
    GateAddNorm,
    # QuantileLoss  # TEMPORARILY disabled - testing with pytorch-forecasting version
)

# TEMPORARY: Test if QuantileLoss is causing 0.45 vs 0.60 difference
from pytorch_forecasting.metrics import QuantileLoss


class TemporalFusionTransformer(pl.LightningModule):
    """
    Custom TFT implementation for single-step S&P 500 return forecasting.
    
    Architecture matches pytorch-forecasting's TFT exactly for baseline validation.
    Designed for future modifications:
    - Output layer is separate for regime-conditional variants
    - Attention module can be swapped for staleness-aware versions
    - Loss function is modular for distribution-aware losses
    
    Parameters
    ----------
    num_encoder_features : int
        Number of time-varying unknown features in encoder (e.g., 4 for baseline:
        VIX, Treasury_10Y, Yield_Spread, Inflation_YoY)
    num_decoder_features : int
        Number of time-varying known features in decoder (0 for baseline - no
        known future inputs)
    hidden_size : int
        Hidden layer dimension throughout network (16 for baseline)
    lstm_layers : int
        Number of LSTM layers (1 for baseline, following pytorch-forecasting default)
    num_attention_heads : int
        Number of attention heads (4 for baseline)
    dropout : float
        Dropout rate (0.25 for baseline)
    max_encoder_length : int
        Number of historical timesteps (20 for baseline)
    max_prediction_length : int
        Prediction horizon (always 1 for single-step forecasting)
    quantiles : list of float, optional
        Quantiles to predict. Default: [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default: 0.001
        
    Notes
    -----
    Static covariates are not implemented in baseline (not used in S&P 500 
    forecasting task). TODOs mark where static covariate handling would be added.
    
    Input format:
        encoder_cont: [batch, max_encoder_length, num_encoder_features]
        decoder_cont: [batch, max_prediction_length, num_decoder_features] or None
        
    Output format:
        predictions: [batch, max_prediction_length, num_quantiles]
    """
    
    def __init__(
        self,
        num_encoder_features: int,
        num_decoder_features: int = 0,
        hidden_size: int = 16,
        hidden_continuous_size: int = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.25,
        max_encoder_length: int = 20,
        max_prediction_length: int = 1,
        quantiles: Optional[list] = None,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture parameters
        self.num_encoder_features = num_encoder_features
        self.num_decoder_features = num_decoder_features
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.learning_rate = learning_rate
        
        # Quantile configuration
        if quantiles is None:
            quantiles = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Loss function
        self.loss_fn = QuantileLoss(quantiles=quantiles)
        
        # ====================================================================
        # FEATURE EMBEDDINGS (prescalers)
        # ====================================================================
        
        # Embed raw continuous features to hidden_continuous_size
        # This matches pytorch-forecasting's prescaler behavior
        # Each feature gets its own embedding layer: 1 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ hidden_continuous_size
        self.encoder_embeddings = nn.ModuleDict({
            f'var_{i}': nn.Linear(1, hidden_continuous_size)
            for i in range(num_encoder_features)
        })
        
        if num_decoder_features > 0:
            self.decoder_embeddings = nn.ModuleDict({
                f'var_{i}': nn.Linear(1, hidden_continuous_size)
                for i in range(num_decoder_features)
            })
        else:
            self.decoder_embeddings = None
        
        # ====================================================================
        # VARIABLE SELECTION NETWORKS
        # ====================================================================
        
        # Encoder VSN: Selects from embedded encoder features
        # Input: dictionary of encoder features, each [batch, encoder_length, hidden_continuous_size]
        # Output: [batch, encoder_length, hidden_size]
        encoder_input_sizes = {f'var_{i}': hidden_continuous_size for i in range(num_encoder_features)}
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,  # Receives context from static enrichment
        )
        
        # Decoder VSN: Selects from embedded decoder features
        # For baseline: num_decoder_features=0, VSN returns zeros shaped by context
        if num_decoder_features > 0:
            decoder_input_sizes = {f'var_{i}': hidden_continuous_size for i in range(num_decoder_features)}
            self.decoder_variable_selection = VariableSelectionNetwork(
                input_sizes=decoder_input_sizes,
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=hidden_size,
            )
        else:
            # Empty VSN for baseline (returns zeros)
            self.decoder_variable_selection = VariableSelectionNetwork(
                input_sizes={},  # No inputs
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=hidden_size,
            )
        
        # TODO: Static variable selection network would go here for static covariates
        # self.static_variable_selection = VariableSelectionNetwork(...)
        
        # ====================================================================
        # STATIC CONTEXT ENCODING
        # ====================================================================
        
        # These GRNs transform static embedding into various contexts
        # For baseline: static_embedding = zeros, but GRNs learn meaningful
        # transformations via biases
        
        # Context for variable selection (used by encoder/decoder VSNs)
        # NOTE: This GRN has NO context input itself (context_size=None)
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None,  # No context for this GRN
        )
        
        # Context for enrichment layer (transforms static info for temporal enrichment)
        # NOTE: This GRN has NO context input itself
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )
        
        # Context for LSTM initial hidden state
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )
        
        # Context for LSTM initial cell state
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )
        
        # ====================================================================
        # LSTM ENCODER/DECODER
        # ====================================================================
        
        # Encoder LSTM: Processes historical time-varying features
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )
        
        # Decoder LSTM: Processes known future features (zeros for baseline)
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )
        
        # Skip connection #1: LSTM output -> Add & Norm with VSN output
        # CRITICAL: Share gate module between encoder and decoder (pytorch-forecasting line 390)
        # This saves 592 params and ensures consistent gating behavior
        self.post_lstm_gate_encoder = GateAddNorm(
            input_size=hidden_size,
            hidden_size=hidden_size,
            skip_size=hidden_size,
            trainable_add=False,  # Match baseline
            dropout=dropout,
        )
        
        # Share the same gate module for decoder (do NOT create separate instance)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        
        # ====================================================================
        # STATIC ENRICHMENT
        # ====================================================================
        
        # Static enrichment GRN: Enriches temporal features with static context
        # NOTE: This GRN DOES receive context (from static_context_enrichment)
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,  # Receives static context
        )
        
        # ====================================================================
        # TEMPORAL SELF-ATTENTION
        # ====================================================================
        
        # Multi-head attention mechanism
        # Query: decoder positions only [batch, 1, hidden_size]
        # Keys/Values: all positions [batch, 21, hidden_size]
        # NOTE: This is where staleness-aware attention would be implemented
        self.multihead_attention = InterpretableMultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Skip connection #2: Attention output -> Add & Norm with attention INPUT
        # (connects to ÃƒÅ½Ã‚Â¸_dec, the enriched decoder features)
        self.post_attn_gate_norm = GateAddNorm(
            input_size=hidden_size,
            hidden_size=hidden_size,
            skip_size=hidden_size,
            trainable_add=False,  # Match baseline
            dropout=dropout,
        )
        
        # ====================================================================
        # POSITION-WISE FEED-FORWARD
        # ====================================================================
        
        # Position-wise feed-forward layer (GRN without context)
        self.pos_wise_ff = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )
        
        # Skip connection #3: FFN output -> Add & Norm with ÃƒÂÃ¢â‚¬Â ÃƒÅ’Ã†â€™_dec
        # (connects back to post-LSTM decoder output)
        self.post_ff_gate_norm = GateAddNorm(
            input_size=hidden_size,
            hidden_size=hidden_size,
            skip_size=hidden_size,
            trainable_add=False,  # Match baseline
            dropout=dropout,
        )
        
        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================
        
        # Skip connection #4: Pre-output gate+norm (wraps output layer)
        # Adds phi_tilde_decoder (post-LSTM decoder) as skip before final projection
        self.pre_output_gate_norm = GateAddNorm(
            input_size=hidden_size,
            hidden_size=hidden_size,
            skip_size=hidden_size,
            trainable_add=False,
            dropout=None,  # No dropout before output
        )
        
        # Final linear projection to quantiles
        # NOTE: This is where regime-conditional output would be implemented
        # Current: Simple linear layer
        # Future: Mixture of experts, regime-specific heads, etc.
        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)
        
    def forward(
        self,
        encoder_cont: torch.Tensor,
        decoder_cont: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through TFT.
        
        Parameters
        ----------
        encoder_cont : torch.Tensor or dict
            Encoder features, shape [batch, max_encoder_length, num_encoder_features]
            If dict (from pytorch-forecasting dataloader), will extract 'encoder_cont' key
        decoder_cont : torch.Tensor or None
            Decoder features, shape [batch, max_prediction_length, num_decoder_features]
            For baseline (no known future inputs), can be None
            
        Returns
        -------
        predictions : torch.Tensor
            Quantile predictions, shape [batch, max_prediction_length, num_quantiles]
        """
        # Handle dict input from pytorch-forecasting dataloaders
        if isinstance(encoder_cont, dict):
            x = encoder_cont
            encoder_cont = x['encoder_cont']
            decoder_cont = x.get('decoder_cont', None)
            
            # CRITICAL FIX: encoder_cont from TimeSeriesDataSet has 6 features (includes encoder_length)
            # But encoder VSN should only process 5 time-varying features (excludes encoder_length)
            # encoder_length is in x['encoder_lengths'] as metadata, not a time-varying feature
            #
            # TimeSeriesDataSet feature order (for core_proposal baseline):
            # 0: encoder_length (STATIC - exclude from VSN)
            # 1: relative_time_idx (time-varying)
            # 2: VIX (time-varying)
            # 3: Treasury_10Y (time-varying)
            # 4: Yield_Spread (time-varying)
            # 5: Inflation_YoY (time-varying)
            #
            # For encoder VSN, we want features 1-5 (skip encoder_length at index 0)
            if encoder_cont.size(-1) > self.num_encoder_features:
                # encoder_cont has more features than expected - likely includes encoder_length
                # encoder_length is always first in x_reals ordering, so skip it
                encoder_cont = encoder_cont[..., 1:]  # Skip first feature (encoder_length)
            
            # Verify shape matches expectation
            if encoder_cont.size(-1) != self.num_encoder_features:
                raise ValueError(
                    f"encoder_cont has {encoder_cont.size(-1)} features after extraction, "
                    f"but model expects {self.num_encoder_features}. "
                    f"Check TimeSeriesDataSet feature configuration."
                )
        
        batch_size = encoder_cont.size(0)
        encoder_length = encoder_cont.size(1)
        device = encoder_cont.device
        dtype = encoder_cont.dtype
        
        debug = hasattr(self, 'debug_mode') and self.debug_mode
        
        if debug:
            print(f"\n[FORWARD PASS MAGNITUDES]")
            print(f"  Input encoder_cont: shape={encoder_cont.shape}, "
                  f"mean={encoder_cont.mean().item():.6f}, "
                  f"std={encoder_cont.std().item():.6f}, "
                  f"min={encoder_cont.min().item():.6f}, "
                  f"max={encoder_cont.max().item():.6f}")
            if decoder_cont is not None:
                print(f"  Input decoder_cont: mean={decoder_cont.mean().item():.6f}, "
                      f"std={decoder_cont.std().item():.6f}")
        
        # ====================================================================
        # STATIC CONTEXT ENCODING
        # ====================================================================
        
        # For baseline: no static features, use learned embedding
        # Initialize from zeros (will be enriched by GRNs)
        static_input = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        
        # Process through static context GRNs
        # These learn to encode temporal position and global context
        static_context_variable_selection = self.static_context_variable_selection(static_input)
        static_context_enrichment = self.static_context_enrichment(static_input)
        static_context_initial_hidden_lstm = self.static_context_initial_hidden_lstm(static_input)
        static_context_initial_cell_lstm = self.static_context_initial_cell_lstm(static_input)
        
        if debug:
            print(f"  Static context variable selection: mean={static_context_variable_selection.mean().item():.6f}, "
                  f"std={static_context_variable_selection.std().item():.6f}")
            print(f"  Static context enrichment: mean={static_context_enrichment.mean().item():.6f}, "
                  f"std={static_context_enrichment.std().item():.6f}")
            print(f"  Static LSTM hidden init: mean={static_context_initial_hidden_lstm.mean().item():.6f}, "
                  f"std={static_context_initial_hidden_lstm.std().item():.6f}")
            print(f"  Static LSTM cell init: mean={static_context_initial_cell_lstm.mean().item():.6f}, "
                  f"std={static_context_initial_cell_lstm.std().item():.6f}")
        
        # Initialize LSTM hidden states
        # Expand to [num_layers, batch, hidden_size]
        h0 = static_context_initial_hidden_lstm.unsqueeze(0).expand(
            self.lstm_layers, -1, -1
        ).contiguous()
        c0 = static_context_initial_cell_lstm.unsqueeze(0).expand(
            self.lstm_layers, -1, -1
        ).contiguous()
        
        # ====================================================================
        # FEATURE EMBEDDING - ENCODER
        # ====================================================================
        
        # Embed raw features: 1D ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ hidden_continuous_size
        # This matches pytorch-forecasting's prescaler behavior
        encoder_features_embedded = {
            f'var_{i}': self.encoder_embeddings[f'var_{i}'](encoder_cont[..., i:i+1])
            for i in range(self.num_encoder_features)
        }
        # Each embedded feature: [batch, encoder_length, hidden_continuous_size]
        
        if debug:
            for i in range(min(3, self.num_encoder_features)):  # Show first 3 features
                feat = encoder_features_embedded[f'var_{i}']
                print(f"  Encoder feature {i} embedded: mean={feat.mean().item():.6f}, "
                      f"std={feat.std().item():.6f}")
        
        # Expand context to time dimension for VSN
        # [batch, hidden_size] -> [batch, encoder_length, hidden_size]
        context_encoder = static_context_variable_selection.unsqueeze(1).expand(
            -1, encoder_length, -1
        )
        
        # Apply VSN to select relevant features
        # Returns: [batch, encoder_length, hidden_size]
        xi_tilde_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            encoder_features_embedded,
            context_encoder
        )
        
        # Store for monitoring/interpretability
        self._last_encoder_vsn_output = xi_tilde_encoder
        self._last_encoder_sparse_weights = encoder_sparse_weights
        
        if debug:
            print(f"  After encoder VSN: mean={xi_tilde_encoder.mean().item():.6f}, "
                  f"std={xi_tilde_encoder.std().item():.6f}, "
                  f"min={xi_tilde_encoder.min().item():.6f}, "
                  f"max={xi_tilde_encoder.max().item():.6f}")
            print(f"  Encoder VSN weights: mean={encoder_sparse_weights.mean().item():.6f}, "
                  f"std={encoder_sparse_weights.std().item():.6f}")
        
        # ====================================================================
        # FEATURE EMBEDDING - DECODER
        # ====================================================================
        
        # Embed decoder features if they exist
        if decoder_cont is None or self.num_decoder_features == 0:
            decoder_features_embedded = {}
        else:
            decoder_features_embedded = {
                f'var_{i}': self.decoder_embeddings[f'var_{i}'](decoder_cont[..., i:i+1])
                for i in range(self.num_decoder_features)
            }
        
        if debug and decoder_features_embedded:
            for i in range(min(2, self.num_decoder_features)):
                feat = decoder_features_embedded[f'var_{i}']
                print(f"  Decoder feature {i} embedded: mean={feat.mean().item():.6f}, "
                      f"std={feat.std().item():.6f}")
        
        # Expand context to decoder time dimension
        context_decoder = static_context_variable_selection.unsqueeze(1).expand(
            -1, self.max_prediction_length, -1
        )
        
        # Apply VSN (returns zeros for empty input)
        # Returns: [batch, max_prediction_length, hidden_size]
        xi_tilde_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            decoder_features_embedded,
            context_decoder
        )
        
        # Store for monitoring/interpretability
        self._last_decoder_vsn_output = xi_tilde_decoder
        self._last_decoder_sparse_weights = decoder_sparse_weights
        
        if debug:
            print(f"  After decoder VSN: mean={xi_tilde_decoder.mean().item():.6f}, "
                  f"std={xi_tilde_decoder.std().item():.6f}")
        
        # ====================================================================
        # LSTM ENCODER-DECODER
        # ====================================================================
        
        # Process encoder sequence
        # Input: [batch, encoder_length, hidden_size]
        # Output: [batch, encoder_length, hidden_size]
        phi_encoder, (hidden, cell) = self.lstm_encoder(
            xi_tilde_encoder,
            (h0, c0)
        )
        
        # Process decoder sequence (uses final encoder states)
        # Input: [batch, max_prediction_length, hidden_size]
        # Output: [batch, max_prediction_length, hidden_size]
        phi_decoder, _ = self.lstm_decoder(
            xi_tilde_decoder,
            (hidden, cell)  # Continue from encoder final states
        )
        
        # Store LSTM outputs for monitoring
        self._last_lstm_encoder_output = phi_encoder
        self._last_lstm_decoder_output = phi_decoder
        self._last_lstm_hidden_state = hidden
        self._last_lstm_cell_state = cell
        
        if debug:
            print(f"  After LSTM encoder: mean={phi_encoder.mean().item():.6f}, "
                  f"std={phi_encoder.std().item():.6f}, "
                  f"min={phi_encoder.min().item():.6f}, "
                  f"max={phi_encoder.max().item():.6f}")
            print(f"  After LSTM decoder: mean={phi_decoder.mean().item():.6f}, "
                  f"std={phi_decoder.std().item():.6f}")
            print(f"  LSTM hidden state: mean={hidden.mean().item():.6f}, "
                  f"std={hidden.std().item():.6f}")
        
        # ====================================================================
        # GATED SKIP CONNECTION (LSTM encoder)
        # ====================================================================
        
        # Apply gating mechanism: combines input and LSTM output
        # GLU(input, output) + LayerNorm(residual)
        # CORRECT: Gate the LSTM output, add VSN output as skip
        phi_tilde_encoder = self.post_lstm_gate_encoder(
            phi_encoder,  # Gate this (LSTM output)
            xi_tilde_encoder  # Add as skip (VSN output)
        )
        
        if debug:
            print(f"  After encoder gate+skip: mean={phi_tilde_encoder.mean().item():.6f}, "
                  f"std={phi_tilde_encoder.std().item():.6f}, "
                  f"min={phi_tilde_encoder.min().item():.6f}, "
                  f"max={phi_tilde_encoder.max().item():.6f}")
        
        # ====================================================================
        # GATED SKIP CONNECTION (LSTM decoder)
        # ====================================================================
        
        # Same pattern: gate LSTM output, add VSN as skip
        phi_tilde_decoder = self.post_lstm_gate_decoder(
            phi_decoder,  # Gate this (LSTM output)
            xi_tilde_decoder  # Add as skip (VSN output)
        )
        
        if debug:
            print(f"  After decoder gate+skip: mean={phi_tilde_decoder.mean().item():.6f}, "
                  f"std={phi_tilde_decoder.std().item():.6f}")
        
        # ====================================================================
        # TEMPORAL FUSION
        # ====================================================================
        
        # Concatenate encoder and decoder outputs
        # [batch, encoder_length + max_prediction_length, hidden_size]
        phi_tilde = torch.cat([phi_tilde_encoder, phi_tilde_decoder], dim=1)
        
        if debug:
            print(f"  After concat (phi_tilde): mean={phi_tilde.mean().item():.6f}, "
                  f"std={phi_tilde.std().item():.6f}")
        
        # Expand static enrichment context to time dimension
        context_enrichment = static_context_enrichment.unsqueeze(1).expand(
            -1, encoder_length + self.max_prediction_length, -1
        )
        
        # Apply static enrichment GRN
        # Input: concatenated LSTM outputs
        # Context: learned static context
        # Output: ÃƒÅ½Ã‚Â¸ (enriched temporal representations)
        theta = self.static_enrichment(
            phi_tilde,
            context_enrichment
        )  # [batch, encoder_length + max_prediction_length, hidden_size]
        
        # Store enriched representations
        self._last_static_enrichment_output = theta
        
        if debug:
            print(f"  After static enrichment (theta): mean={theta.mean().item():.6f}, "
                  f"std={theta.std().item():.6f}, "
                  f"min={theta.min().item():.6f}, "
                  f"max={theta.max().item():.6f}")
        
        # ====================================================================
        # SELF-ATTENTION
        # ====================================================================
        
        # Extract decoder portion for queries
        # Shape: [batch, max_prediction_length, hidden_size]
        theta_decoder = theta[:, -self.max_prediction_length:, :]
        
        # Self-attention: decoder attends to full sequence
        # Q: decoder positions
        # K, V: all positions (encoder + decoder)
        
        # For single-step prediction (max_prediction_length=1):
        # - Query: last position [batch, 1, hidden]
        # - Keys/Values: all positions [batch, encoder_length + 1, hidden]
        # - Output: attended representation [batch, 1, hidden]
        
        # Lim et al. (2021) use "interpretable multi-head attention" 
        # with additive attention scores rather than scaled dot-product
        
        # 1. Apply attention to get enriched decoder representations
        # 2. Attention weights can be analyzed for interpretability
        # 3. Use staleness as additional context for attention computation
        
        beta, attention_weights = self.multihead_attention(
            theta_decoder,  # Query: [batch, max_prediction_length, hidden]
            theta,          # Keys: [batch, encoder_length + max_prediction_length, hidden]
            theta,          # Values: [batch, encoder_length + max_prediction_length, hidden]
            None,           # No mask needed for single-step prediction
        )
        # beta: [batch, max_prediction_length, hidden_size]
        # attention_weights: [batch, num_heads, max_prediction_length, encoder_length + max_prediction_length]
        
        # Store attention weights for analysis
        self._last_attention_weights = attention_weights
        
        if debug:
            print(f"  After attention (beta): mean={beta.mean().item():.6f}, "
                  f"std={beta.std().item():.6f}, "
                  f"min={beta.min().item():.6f}, "
                  f"max={beta.max().item():.6f}")
            print(f"  Attention weights: mean={attention_weights.mean().item():.6f}, "
                  f"std={attention_weights.std().item():.6f}")
        
        # ====================================================================
        # GATED SKIP CONNECTION (Attention)
        # ====================================================================
        
        # Add & norm with gating
        # Skip connection target: decoder portion of theta (attention INPUT)
        # Per TFT paper: ÃƒÂÃ‹â€  = LN(GLU(ÃƒÅ½Ã‚Â²) + ÃƒÅ½Ã‚Â¸_dec)
        psi = self.post_attn_gate_norm(
            beta,  # Gate attention output
            theta_decoder  # Add attention INPUT as residual (not phi_tilde!)
        )
        
        if debug:
            print(f"  After attention gate+skip (psi): mean={psi.mean().item():.6f}, "
                  f"std={psi.std().item():.6f}, "
                  f"min={psi.min().item():.6f}, "
                  f"max={psi.max().item():.6f}")
        
        # ====================================================================
        # POSITION-WISE FEED-FORWARD
        # ====================================================================
        
        # Apply position-wise GRN
        # No context for this layer (as per Lim et al.)
        psi_hat = self.pos_wise_ff(psi)
        
        if debug:
            print(f"  After position-wise FF (psi_hat): mean={psi_hat.mean().item():.6f}, "
                  f"std={psi_hat.std().item():.6f}, "
                  f"min={psi_hat.min().item():.6f}, "
                  f"max={psi_hat.max().item():.6f}")
        
        # ====================================================================
        # FINAL GATED SKIP CONNECTION
        # ====================================================================
        
        # Final skip connection: gate FF output, add post-attention output
        # Per TFT paper: ÃƒÂÃ‹â€ ÃƒÅ’Ã†â€™ = LN(GLU(ÃƒÂÃ‹â€ ÃƒÅ’Ã¢â‚¬Å¡) + ÃƒÂÃ‹â€ )
        psi_tilde = self.post_ff_gate_norm(
            psi_hat,  # Gate feed-forward output
            psi  # Add post-attention output as residual
        )
        
        if debug:
            print(f"  After final gate+skip (psi_tilde): mean={psi_tilde.mean().item():.6f}, "
                  f"std={psi_tilde.std().item():.6f}, "
                  f"min={psi_tilde.min().item():.6f}, "
                  f"max={psi_tilde.max().item():.6f}")
        
        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================
        
        # Apply 4th skip connection: gate final features, add decoder LSTM output
        # This provides a direct path from LSTM decoder to output predictions
        output_input = self.pre_output_gate_norm(
            psi_tilde,  # Gate the enriched features after attention+FFN
            phi_tilde_decoder  # Add decoder LSTM output as skip
        )
        
        # Project to quantile predictions
        # Input: [batch, max_prediction_length, hidden_size]
        # Output: [batch, max_prediction_length, num_quantiles]
        predictions = self.output_layer(output_input)
        
        if debug:
            print(f"  Final predictions: mean={predictions.mean().item():.6f}, "
                  f"std={predictions.std().item():.6f}, "
                  f"min={predictions.min().item():.6f}, "
                  f"max={predictions.max().item():.6f}")
            # Show per-quantile stats
            for q_idx, q in enumerate(self.quantiles):
                q_pred = predictions[..., q_idx]
                print(f"    Quantile {q:.2f}: mean={q_pred.mean().item():.6f}, "
                      f"std={q_pred.std().item():.6f}")
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        x, y = batch
        
        # Debug logging
        if hasattr(self, 'debug_mode') and self.debug_mode and batch_idx % 10 == 0:
            print(f"[TRAINING_STEP] Epoch {self.current_epoch}, Batch {batch_idx}")
            print(f"  encoder_cont shape: {x['encoder_cont'].shape}")
            if 'decoder_cont' in x:
                print(f"  decoder_cont shape: {x['decoder_cont'].shape}")
            print(f"  target shape: {y[0].shape}")
        
        # Unpack batch (pytorch-forecasting format)
        encoder_cont = x['encoder_cont']  # [batch, encoder_length, num_features]
        decoder_cont = x.get('decoder_cont', None)  # None for baseline
        
        # Target is already shaped correctly: [batch, max_prediction_length]
        # For single-step prediction: [batch, 1]
        
        # Forward pass
        predictions = self(encoder_cont, decoder_cont)
        
        # Debug: Check prediction stats
        if hasattr(self, 'debug_mode') and self.debug_mode and batch_idx % 10 == 0:
            print(f"  predictions shape: {predictions.shape}")
            print(f"  predictions mean: {predictions.mean().item():.6f}, std: {predictions.std().item():.6f}")
        
        # Compute quantile loss
        loss = self.loss_fn(predictions, y[0])  # y is tuple (target, weight)
        
        # Debug: Track loss
        if hasattr(self, 'debug_mode') and self.debug_mode and batch_idx % 10 == 0:
            print(f"  batch_loss: {loss.item():.6f}")
        
        # Log metrics
        #self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        x, y = batch
        
        # Debug logging
        if hasattr(self, 'debug_mode') and self.debug_mode and batch_idx == 0:
            print(f"[VALIDATION_STEP] Epoch {self.current_epoch}, Batch {batch_idx}")
            print(f"  encoder_cont shape: {x['encoder_cont'].shape}")
            if 'decoder_cont' in x:
                print(f"  decoder_cont shape: {x['decoder_cont'].shape}")
        
        encoder_cont = x['encoder_cont']
        decoder_cont = x.get('decoder_cont', None)
        
        predictions = self(encoder_cont, decoder_cont)
        loss = self.loss_fn(predictions, y[0])
        
        # Debug: Check validation stats
        if hasattr(self, 'debug_mode') and self.debug_mode and batch_idx == 0:
            print(f"  val_loss: {loss.item():.6f}")
            print(f"  predictions mean: {predictions.mean().item():.6f}, std: {predictions.std().item():.6f}")
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning."""
        x, y = batch
        
        encoder_cont = x['encoder_cont']
        decoder_cont = x.get('decoder_cont', None)
        
        predictions = self(encoder_cont, decoder_cont)
        loss = self.loss_fn(predictions, y[0])
        
        self.log('test_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer (Adam) and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            cooldown=4,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Retrieve attention weights from last forward pass.
        
        Returns
        -------
        attention_weights : torch.Tensor or None
            Shape: [batch, num_heads, max_prediction_length, encoder_length + max_prediction_length]
        """
        return getattr(self, '_last_attention_weights', None)
    
    def get_encoder_vsn_output(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve encoder VSN outputs from last forward pass.
        
        Returns
        -------
        tuple of (output, weights) or None
            output: [batch, encoder_length, hidden_size]
            weights: [batch, encoder_length, num_encoder_features]
        """
        output = getattr(self, '_last_encoder_vsn_output', None)
        weights = getattr(self, '_last_encoder_sparse_weights', None)
        if output is not None and weights is not None:
            return output, weights
        return None
    
    def get_decoder_vsn_output(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve decoder VSN outputs from last forward pass.
        
        Returns
        -------
        tuple of (output, weights) or None
            output: [batch, max_prediction_length, hidden_size]
            weights: [batch, max_prediction_length, num_decoder_features]
        """
        output = getattr(self, '_last_decoder_vsn_output', None)
        weights = getattr(self, '_last_decoder_sparse_weights', None)
        if output is not None and weights is not None:
            return output, weights
        return None
    
    def get_lstm_outputs(self) -> Optional[dict]:
        """
        Retrieve LSTM encoder/decoder outputs from last forward pass.
        
        Returns
        -------
        dict or None
            'encoder_output': [batch, encoder_length, hidden_size]
            'decoder_output': [batch, max_prediction_length, hidden_size]
            'hidden_state': [num_layers, batch, hidden_size]
            'cell_state': [num_layers, batch, hidden_size]
        """
        encoder = getattr(self, '_last_lstm_encoder_output', None)
        decoder = getattr(self, '_last_lstm_decoder_output', None)
        hidden = getattr(self, '_last_lstm_hidden_state', None)
        cell = getattr(self, '_last_lstm_cell_state', None)
        
        if encoder is not None:
            return {
                'encoder_output': encoder,
                'decoder_output': decoder,
                'hidden_state': hidden,
                'cell_state': cell,
            }
        return None
    
    def get_static_enrichment_output(self) -> Optional[torch.Tensor]:
        """
        Retrieve static enrichment output (theta) from last forward pass.
        
        Returns
        -------
        theta : torch.Tensor or None
            Shape: [batch, encoder_length + max_prediction_length, hidden_size]
        """
        return getattr(self, '_last_static_enrichment_output', None)
    
    def get_all_internal_states(self) -> dict:
        """
        Retrieve all internal states from last forward pass.
        
        Useful for comprehensive analysis and debugging.
        
        Returns
        -------
        dict with keys:
            'encoder_vsn': tuple of (output, weights) or None
            'decoder_vsn': tuple of (output, weights) or None
            'lstm': dict of lstm outputs or None
            'static_enrichment': tensor or None
            'attention_weights': tensor or None
        """
        return {
            'encoder_vsn': self.get_encoder_vsn_output(),
            'decoder_vsn': self.get_decoder_vsn_output(),
            'lstm': self.get_lstm_outputs(),
            'static_enrichment': self.get_static_enrichment_output(),
            'attention_weights': self.get_attention_weights(),
        }
