"""
Train custom Temporal Fusion Transformer on S&P 500 return prediction.

This script trains the custom TFT implementation for baseline validation against
pytorch-forecasting. Uses the same data pipeline (TimeSeriesDataSet) to ensure
feature parity.

Usage:
    # Baseline validation (h=16)
    python train/train_tft_custom.py --experiment-name custom_tft_h16_baseline \\
        --hidden-size 16 --dropout 0.25
    
    # Debug mode with parameter summary
    python train/train_tft_custom.py --experiment-name custom_tft_debug \\
        --hidden-size 16 --debug
"""


import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import os
import sys
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import custom TFT and utilities
from models.tft_model import TemporalFusionTransformer
from models.model_summary import print_parameter_summary
from src.feature_configs import FEATURE_SETS
from train.collapse_monitor import CollapseMonitor
from train.callbacks import EpochSummaryCallback, DistributionLossLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train custom TFT for S&P 500 return prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name for this experiment (creates output directory)')
    parser.add_argument('--feature-set', type=str, default='core_proposal',
                        choices=['core_proposal', 'core_plus_credit', 'macro_heavy', 
                                 'market_only', 'kitchen_sink'],
                        help='Feature set configuration')
    parser.add_argument('--frequency', type=str, default='daily',
                        choices=['daily', 'monthly'],
                        help='Data frequency')
    parser.add_argument('--alignment', type=str, default='fixed',
                        choices=['fixed', 'vintage'],
                        help='Release date alignment mode')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # TFT architecture
    parser.add_argument('--max-encoder-length', type=int, default=20,
                        help='Lookback window length')
    parser.add_argument('--hidden-size', type=int, default=16,
                        help='Hidden layer size')
    parser.add_argument('--hidden-continuous-size', type=int, default=16,
                        help='Embedding size for continuous features')
    parser.add_argument('--lstm-layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--attention-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--gradient-clip', type=float, default=0.1,
                        help='Gradient clipping value')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Debug and validation
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed logging')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test run (10 epochs, small data)')
    
    # Data paths
    parser.add_argument('--splits-dir', type=str, 
                        default='data/splits',
                        help='Directory containing train/val/test splits')
    parser.add_argument('--output-dir', type=str,
                        default='experiments',
                        help='Base directory for experiment outputs')
    
    return parser.parse_args()

def get_features(feature_set):
    """
    Get feature lists based on configuration.
    Matches baseline's get_features() but simplified (no staleness for now).
    
    Returns:
    --------
    dict with keys:
        'all': list of encoder feature names (excludes target)
        'features': original feature list from config (for reference)
    """
    from src.feature_configs import FEATURE_SETS
    
    config = FEATURE_SETS[feature_set]
    
    # Filter out target from encoder features
    # This matches baseline train_tft.py line 160
    encoder_features = [f for f in config['features'] if f != 'SP500_Returns']
    
    return {
        'all': encoder_features,  # For TimeSeriesDataSet
        'features': config['features'],  # Original list (for reference)
    }


def load_data(args):
    """Load and prepare training/validation data using TimeSeriesDataSet."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Get feature configuration using baseline's logic
    features = get_features(args.feature_set)
    
    print(f"Feature set: {args.feature_set}")
    print(f"Encoder features: {features['all']}")
    print(f"Target: SP500_Returns")
    
    # Load splits
    split_prefix = f"{args.feature_set}_{args.frequency}_{args.alignment}"
    split_dir = os.path.join(args.splits_dir, args.alignment)
    
    train_path = os.path.join(split_dir, f"{split_prefix}_train.csv")
    val_path = os.path.join(split_dir, f"{split_prefix}_val.csv")
    
    print(f"\nLoading from: {split_dir}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    
    # Quick test mode - use subset
    if args.quick_test:
        print("\n[QUICK TEST MODE] Using subset of data")
        train_df = train_df.iloc[:500]
        val_df = val_df.iloc[:200]
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
    
    # Add time index and group (required by TimeSeriesDataSet)
    train_df['time_idx'] = range(len(train_df))
    val_df['time_idx'] = range(len(train_df), len(train_df) + len(val_df))
    train_df['group'] = 'SP500'
    val_df['group'] = 'SP500'
    
    # DEBUG: Print raw feature statistics BEFORE normalization (matches baseline)
    print("\n" + "="*70)
    print("FEATURE STATISTICS - BEFORE NORMALIZATION")
    print("="*70)
    feature_cols = [c for c in train_df.columns if c in features['all']]
    for col in feature_cols:
        data = train_df[col]
        print(f"{col:30s}  mean={data.mean():8.4f}  std={data.std():8.4f}  "
              f"min={data.min():8.4f}  max={data.max():8.4f}")
    print("="*70)
    
    # Create TimeSeriesDataSet
    print("\nCreating TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=1,
        time_varying_known_reals=[],  # No explicit known future inputs
        time_varying_unknown_reals=features['all'],  # Matches baseline exactly
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # DEBUG: Check what GroupNormalizer did (sample a batch) - matches baseline
    print("\n" + "="*70)
    print("FEATURES AFTER GROUPNORMALIZER (first batch, last timestep)")
    print("="*70)
    dataloader_debug = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    x_debug, y_debug = next(iter(dataloader_debug))
    
    # Extract feature values from batch
    encoder_cont = x_debug['encoder_cont']  # Shape: [batch, time, features]
    # Get first sample, last timestep
    sample = encoder_cont[0, -1, :].cpu().numpy()
    
    # DEBUG: Show what TimeSeriesDataSet actually created
    print(f"DEBUG: encoder_cont shape: {encoder_cont.shape}")
    print(f"DEBUG: training.reals: {training.reals}")
    print(f"DEBUG: Sample values: {sample}")
    print()
    
    # Print normalized values using same order as passed to TimeSeriesDataSet
    # encoder_cont features are in same order as features['all'] list
    for i, col in enumerate(features['all']):
        print(f"{col:30s}  normalized_value={sample[i]:8.4f}")
    print("="*70 + "\n")
    
    # DEBUG: Verify actual features after auto-add
    print(f"\nActual features in TimeSeriesDataSet:")
    print(f"  training.reals (all time-varying): {training.reals}")
    print(f"  Total encoder features: {len(training.reals)}")
    print(f"  Decoder features: {training.time_varying_known_reals}")
    print(f"  Static covariates: {training.static_categoricals + training.static_reals}")
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True
    )
    
    # Calculate number of encoder and decoder features
    # CRITICAL: Use correct TimeSeriesDataSet attributes
    # The dataset doesn't expose time_varying_reals_encoder directly,
    # but we can infer it from the data structure
    
    print(f"\nDataset feature categorization:")
    print(f"  training.reals (all reals): {training.reals}")
    print(f"    Count: {len(training.reals)}")
    print(f"  training.time_varying_known_reals: {training.time_varying_known_reals}")
    print(f"    Count: {len(training.time_varying_known_reals)}")
    print(f"  training.time_varying_unknown_reals: {training.time_varying_unknown_reals}")
    print(f"    Count: {len(training.time_varying_unknown_reals)}")
    print(f"  training.static_reals: {training.static_reals}")
    print(f"    Count: {len(training.static_reals)}")
    
    # Encoder gets: time_varying_unknown_reals + time_varying_known_reals (excluding target)
    # But encoder_length is in static_reals, so it's NOT in the encoder VSN inputs
    # The encoder VSN should process: training.reals - static_reals
    encoder_vsn_features = [f for f in training.reals if f not in training.static_reals]
    
    num_encoder_features = len(encoder_vsn_features)
    num_decoder_features = len(training.time_varying_known_reals)
    
    print(f"\nModel configuration:")
    print(f"  Encoder VSN features: {encoder_vsn_features}")
    print(f"  Encoder VSN will process: {num_encoder_features} features")
    print(f"  Decoder VSN will process: {num_decoder_features} features")
    print(f"  Static features (not in VSN): {training.static_reals}")
    
    return training, validation, features, num_encoder_features, num_decoder_features


def create_model(args, num_encoder_features, num_decoder_features):
    """Create custom TFT model."""
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = TemporalFusionTransformer(
        num_encoder_features=num_encoder_features,
        num_decoder_features=num_decoder_features,
        hidden_size=args.hidden_size,
        hidden_continuous_size=args.hidden_continuous_size,
        lstm_layers=args.lstm_layers,
        num_attention_heads=args.attention_heads,
        dropout=args.dropout,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=1,
        learning_rate=args.learning_rate,
    )
    
    if args.debug:
        print_parameter_summary(model, show_details=True)
    else:
        # Just print total
        from models.model_summary import get_parameter_summary
        _, trainable, total = get_parameter_summary(model)
        print(f"Total parameters: {total:,} ({total/1000:.1f}K)")
        print(f"Trainable parameters: {trainable:,} ({trainable/1000:.1f}K)")
    
    return model


def save_config(args, features, num_encoder_features, num_decoder_features, output_dir):
    """Save experiment configuration."""
    config = {
        'experiment_name': args.experiment_name,
        'created_at': datetime.now().isoformat(),
        'model_type': 'custom_tft',
        'random_seed': args.seed,
        'feature_set': args.feature_set,
        'frequency': args.frequency,
        'alignment': args.alignment,
        'architecture': {
            'hidden_size': args.hidden_size,
            'hidden_continuous_size': args.hidden_continuous_size,
            'lstm_layers': args.lstm_layers,
            'attention_heads': args.attention_heads,
            'dropout': args.dropout,
            'max_encoder_length': args.max_encoder_length,
            'max_prediction_length': 1,
            'num_encoder_features': num_encoder_features,  # Passed from training.reals
            'num_decoder_features': num_decoder_features,
        },
        'training': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'gradient_clip': args.gradient_clip,
            'early_stop_patience': args.early_stop_patience,
        },
        'features': features,
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {config_path}")


def set_all_seeds(seed):
    """Set seeds for complete reproducibility (matches baseline)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set random seed with comprehensive control
    set_all_seeds(args.seed)
    
    # Load data
    # Returns total encoder features (including auto-added relative_time_idx, encoder_length)
    training, validation, features, num_encoder_features, num_decoder_features = load_data(args)
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        num_workers=0 if args.debug else 2,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=0 if args.debug else 2,
    )
    
    # Create model with total encoder features (including auto-added)
    model = create_model(args, num_encoder_features, num_decoder_features)
    
    # Set debug mode flag for verbose logging
    if args.debug:
        model.debug_mode = True
        print("\n[DEBUG MODE ENABLED] Verbose logging active")
    
    # Setup experiment directory with proper structure
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories to match baseline structure
    collapse_dir = os.path.join(output_dir, 'collapse_monitoring')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(collapse_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    save_config(args, features, num_encoder_features, num_decoder_features, output_dir)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # Save to checkpoints/ subdirectory
        filename='tft-epoch={epoch:02d}-val_loss={val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,  # Also save last checkpoint
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stop_patience,
        mode='min',
    )
    
    collapse_monitor = CollapseMonitor(
        val_dataloader=val_dataloader,
        log_dir=collapse_dir,  # Save to collapse_monitoring/ subdirectory
        log_every_n_epochs=1,
    )
    epoch_summary = EpochSummaryCallback()
    
    # CRITICAL: Match baseline callback order exactly
    # Baseline: [early_stop, checkpoint, EpochSummaryCallback, CollapseMonitor]
    callbacks = [
        early_stop_callback,
        checkpoint_callback,
        epoch_summary,
        collapse_monitor,
    ]
    
    if args.debug:
        callbacks.append(DistributionLossLogger())
    
    # Setup trainer
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    max_epochs = 10 if args.quick_test else args.max_epochs
    
    # Match baseline trainer configuration
    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        logger=CSVLogger(output_dir, name='logs'),
        deterministic=False,  # Matches baseline (CUDA determinism set via set_all_seeds)
        strategy="auto",
        enable_progress_bar=False,
        enable_model_summary=True,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save final metrics (needed for evaluation scripts)
    final_metrics = {
        'best_model_path': checkpoint_callback.best_model_path,  # Match baseline key name
        'best_val_loss': float(checkpoint_callback.best_model_score),
        'total_epochs': trainer.current_epoch + 1,
        'early_stopped': early_stop_callback.stopped_epoch > 0,
    }
    
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.6f}")
    print(f"Total epochs: {trainer.current_epoch + 1}")
    print(f"Early stopped: {early_stop_callback.stopped_epoch > 0}")
    
    if args.debug:
        print("\n[DEBUG] Final model state:")
        print_parameter_summary(model, show_details=False)
    
    print(f"\nExperiment outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
