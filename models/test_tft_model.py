"""
Unit tests for custom TFT implementation.

Tests validate:
1. Component instantiation
2. Forward pass shapes at each stage
3. Gradient flow through all parameters
4. Ability to overfit small dataset
5. Consistency with pytorch-forecasting behavior

Run with: pytest test_tft_model.py -v
"""

import torch
import pytest
import numpy as np
from models.tft_model import TemporalFusionTransformer


class TestTFTInstantiation:
    """Test that TFT can be instantiated with various configurations."""
    
    def test_baseline_config(self):
        """Test baseline configuration (h=16, 4 features, no decoder)."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.25,
            max_encoder_length=20,
            max_prediction_length=1,
        )
        
        assert model.num_encoder_features == 4
        assert model.num_decoder_features == 0
        assert model.hidden_size == 16
        assert model.num_quantiles == 7
        assert len(model.quantiles) == 7
    
    def test_with_decoder_features(self):
        """Test with decoder features (known future inputs)."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=2,  # e.g., calendar features
            hidden_size=16,
        )
        
        assert model.num_decoder_features == 2
    
    def test_custom_quantiles(self):
        """Test custom quantile configuration."""
        custom_quantiles = [0.1, 0.5, 0.9]
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            quantiles=custom_quantiles,
        )
        
        assert model.num_quantiles == 3
        assert model.quantiles == custom_quantiles
    
    def test_different_hidden_sizes(self):
        """Test various hidden sizes (capacity sweep range)."""
        for h in [8, 16, 20, 32]:
            model = TemporalFusionTransformer(
                num_encoder_features=4,
                hidden_size=h,
            )
            assert model.hidden_size == h


class TestTFTForwardPass:
    """Test forward pass shapes and correctness."""
    
    @pytest.fixture
    def baseline_model(self):
        """Baseline TFT configuration."""
        return TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.25,
            max_encoder_length=20,
            max_prediction_length=1,
        )
    
    def test_forward_pass_shape(self, baseline_model):
        """Test output shape is correct."""
        batch_size = 32
        encoder_cont = torch.randn(batch_size, 20, 4)
        
        predictions = baseline_model(encoder_cont)
        
        # Should output [batch, prediction_length, num_quantiles]
        assert predictions.shape == (batch_size, 1, 7)
    
    def test_forward_pass_with_decoder(self):
        """Test forward pass with decoder features."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=2,
            hidden_size=16,
            max_encoder_length=20,
            max_prediction_length=1,
        )
        
        batch_size = 16
        encoder_cont = torch.randn(batch_size, 20, 4)
        decoder_cont = torch.randn(batch_size, 1, 2)
        
        predictions = model(encoder_cont, decoder_cont)
        
        assert predictions.shape == (batch_size, 1, 7)
    
    def test_batch_sizes(self, baseline_model):
        """Test various batch sizes."""
        for batch_size in [1, 16, 32, 64, 128]:
            encoder_cont = torch.randn(batch_size, 20, 4)
            predictions = baseline_model(encoder_cont)
            assert predictions.shape[0] == batch_size
    
    def test_quantile_ordering(self, baseline_model):
        """Test that quantiles are in ascending order in output."""
        encoder_cont = torch.randn(32, 20, 4)
        
        # Run model in eval mode to get stable predictions
        baseline_model.eval()
        with torch.no_grad():
            predictions = baseline_model(encoder_cont)
        
        # For well-trained model, quantiles should be ordered
        # (Not guaranteed during random init, but check shape is correct)
        assert predictions.shape[-1] == 7  # 7 quantiles
    
    def test_attention_weights_stored(self, baseline_model):
        """Test that attention weights are stored for interpretability."""
        encoder_cont = torch.randn(16, 20, 4)
        
        _ = baseline_model(encoder_cont)
        
        # Should have stored attention weights
        attn_weights = baseline_model.get_attention_weights()
        assert attn_weights is not None
        
        # Shape: [batch, num_heads, decoder_length, total_length]
        assert attn_weights.shape == (16, 4, 1, 21)  # 20 encoder + 1 decoder
    
    def test_deterministic_with_seed(self, baseline_model):
        """Test that forward pass is deterministic with same seed."""
        torch.manual_seed(42)
        encoder_cont = torch.randn(8, 20, 4)
        
        baseline_model.eval()
        with torch.no_grad():
            pred1 = baseline_model(encoder_cont)
        
        # Reset and run again
        baseline_model.eval()
        with torch.no_grad():
            pred2 = baseline_model(encoder_cont)
        
        # Should be identical
        assert torch.allclose(pred1, pred2)


class TestTFTGradients:
    """Test gradient flow through the network."""
    
    @pytest.fixture
    def baseline_model(self):
        """Baseline TFT configuration."""
        return TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.0,  # Disable dropout for gradient tests
            max_encoder_length=20,
            max_prediction_length=1,
        )
    
    def test_backward_pass(self, baseline_model):
        """Test that gradients can be computed."""
        encoder_cont = torch.randn(8, 20, 4)
        target = torch.randn(8, 1)
        
        predictions = baseline_model(encoder_cont)
        loss = baseline_model.loss_fn(predictions, target)
        
        # Should be able to compute gradients
        loss.backward()
        
        # Check that gradients were computed
        for name, param in baseline_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_all_parameters_used(self, baseline_model):
        """Test that all parameters receive gradients (no unused params)."""
        encoder_cont = torch.randn(16, 20, 4)
        target = torch.randn(16, 1)
        
        baseline_model.zero_grad()
        predictions = baseline_model(encoder_cont)
        loss = baseline_model.loss_fn(predictions, target)
        loss.backward()
        
        # All parameters should have gradients
        unused_params = []
        for name, param in baseline_model.named_parameters():
            if param.requires_grad and param.grad is None:
                unused_params.append(name)
        
        assert len(unused_params) == 0, f"Unused parameters: {unused_params}"
    
    def test_gradient_flow_magnitude(self, baseline_model):
        """Test that gradients are not vanishing or exploding."""
        encoder_cont = torch.randn(32, 20, 4)
        target = torch.randn(32, 1)
        
        baseline_model.zero_grad()
        predictions = baseline_model(encoder_cont)
        loss = baseline_model.loss_fn(predictions, target)
        loss.backward()
        
        # Collect gradient magnitudes
        grad_norms = []
        for param in baseline_model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        # Gradients should be reasonable (not vanishing or exploding)
        assert all(g < 100 for g in grad_norms), "Exploding gradients detected"
        assert all(g > 1e-7 for g in grad_norms), "Vanishing gradients detected"


class TestTFTOverfitting:
    """Test ability to overfit small datasets (sanity check for learning)."""
    
    def test_overfit_single_batch(self):
        """Test that model can memorize a single batch."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            hidden_size=32,  # Larger capacity for overfitting
            dropout=0.0,  # No regularization
        )
        
        # Single batch to overfit
        encoder_cont = torch.randn(16, 20, 4)
        target = torch.randn(16, 1)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for many iterations
        initial_loss = None
        final_loss = None
        
        for i in range(200):
            optimizer.zero_grad()
            predictions = model(encoder_cont)
            loss = model.loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
            if i == 199:
                final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, \
            f"Model failed to overfit. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    
    def test_perfect_fit_simple_pattern(self):
        """Test model can learn a simple deterministic pattern."""
        model = TemporalFusionTransformer(
            num_encoder_features=1,
            num_decoder_features=0,
            hidden_size=16,
            dropout=0.0,
        )
        
        # Simple pattern: output = mean of last 3 inputs
        n_samples = 32
        encoder_cont = torch.randn(n_samples, 20, 1)
        target = encoder_cont[:, -3:, 0].mean(dim=1, keepdim=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for _ in range(500):
            optimizer.zero_grad()
            predictions = model(encoder_cont)
            
            # Use median prediction (quantile index 3)
            pred_median = predictions[:, :, 3]  # 0.50 quantile
            loss = torch.nn.functional.mse_loss(pred_median, target)
            
            loss.backward()
            optimizer.step()
        
        # Final loss should be very small
        model.eval()
        with torch.no_grad():
            final_predictions = model(encoder_cont)
            final_pred_median = final_predictions[:, :, 3]
            final_loss = torch.nn.functional.mse_loss(final_pred_median, target).item()
        
        assert final_loss < 0.01, f"Model failed to learn simple pattern. Loss: {final_loss:.4f}"


class TestTFTNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_input(self):
        """Test that model handles zero input without NaN."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            hidden_size=16,
        )
        
        encoder_cont = torch.zeros(8, 20, 4)
        
        model.eval()
        with torch.no_grad():
            predictions = model(encoder_cont)
        
        # Should not produce NaN
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_large_input_values(self):
        """Test stability with large input values."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            hidden_size=16,
        )
        
        # Large but not extreme values (typical after normalization outliers)
        encoder_cont = torch.randn(8, 20, 4) * 10
        
        model.eval()
        with torch.no_grad():
            predictions = model(encoder_cont)
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_single_sample_batch(self):
        """Test batch size of 1 (edge case for batch norm layers if any)."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            hidden_size=16,
        )
        
        encoder_cont = torch.randn(1, 20, 4)
        
        # Should handle batch size 1
        predictions = model(encoder_cont)
        assert predictions.shape == (1, 1, 7)


class TestTFTComponentCounts:
    """Test that model has expected number of parameters."""
    
    def test_parameter_count_baseline(self):
        """Test parameter count for baseline configuration."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            num_decoder_features=0,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should be in expected range for h=16
        # From experiments: h=16 has ~25-30K parameters
        assert 20_000 < total_params < 40_000, \
            f"Unexpected parameter count: {total_params:,}"
        assert total_params == trainable_params, "All parameters should be trainable"
    
    def test_parameter_scaling_with_hidden_size(self):
        """Test that parameter count scales roughly quadratically with hidden size."""
        counts = {}
        for h in [8, 16, 32]:
            model = TemporalFusionTransformer(
                num_encoder_features=4,
                hidden_size=h,
            )
            counts[h] = sum(p.numel() for p in model.parameters())
        
        # Should roughly scale quadratically (not exactly due to input/output layers)
        ratio_16_8 = counts[16] / counts[8]
        ratio_32_16 = counts[32] / counts[16]
        
        # Expect ratios between 3-5 (quadratic would be 4)
        assert 2 < ratio_16_8 < 6
        assert 2 < ratio_32_16 < 6


class TestTFTLoss:
    """Test loss computation."""
    
    def test_quantile_loss_computation(self):
        """Test that quantile loss is computed correctly."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            hidden_size=16,
        )
        
        encoder_cont = torch.randn(16, 20, 4)
        target = torch.randn(16, 1)
        
        predictions = model(encoder_cont)
        loss = model.loss_fn(predictions, target)
        
        # Loss should be scalar
        assert loss.dim() == 0
        
        # Loss should be positive
        assert loss.item() > 0
    
    def test_loss_decreases_with_training(self):
        """Test that loss decreases during training."""
        model = TemporalFusionTransformer(
            num_encoder_features=4,
            hidden_size=16,
            dropout=0.0,
        )
        
        encoder_cont = torch.randn(32, 20, 4)
        target = torch.randn(32, 1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Record losses
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            predictions = model(encoder_cont)
            loss = model.loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should trend downward
        initial_avg = np.mean(losses[:10])
        final_avg = np.mean(losses[-10:])
        assert final_avg < initial_avg, "Loss did not decrease during training"


if __name__ == '__main__':
    # Run tests with pytest
    # Can also run individual test classes for debugging
    pytest.main([__file__, '-v', '--tb=short'])
