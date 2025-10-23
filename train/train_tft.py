"""
Train Temporal Fusion Transformer on S&P 500 return prediction.

Includes full experiment tracking, checkpointing, and reproducibility controls.

Usage:
    # Basic usage with defaults
    python train_tft.py --experiment-name tft_baseline
    
    # Custom hyperparameters
    python train_tft.py --experiment-name tft_large \\
        --hidden-size 64 --attention-heads 4 --max-epochs 100
    
    # Different feature set
    python train_tft.py --experiment-name tft_macro \\
        --feature-set macro_heavy --frequency monthly
"""

import os
import platform
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import json
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- enable Tensor Core optimization for RTX 3070 ---
torch.set_float32_matmul_precision('medium')  # high for even faster, but slightly less precise


# --- device setup (macOS compatibility) ---
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    print("\n[INFO] MPS detected but upsample ops unsupported — using CPU for stability.")
    device = "cpu"
else:
    device = "auto"  # let Lightning pick CUDA or CPU
    print(f"\n[INFO] Using {device.upper()} device selection.")


# --- plot safety patch: prevent early-epoch plots from crashing on negative yerr ---
try:
    from matplotlib.axes import _axes as _mpl_axes

    _orig_errorbar = _mpl_axes.Axes.errorbar
    def _safe_errorbar(self, x, y, yerr=None, *args, **kwargs):
        # matplotlib expects yerr as magnitudes; if PF hands us negatives, make them magnitudes
        if yerr is not None:
            yerr_arr = np.asarray(yerr)
            if (yerr_arr < 0).any():
                yerr = np.abs(yerr_arr)
        return _orig_errorbar(self, x, y, yerr=yerr, *args, **kwargs)

    _mpl_axes.Axes.errorbar = _safe_errorbar
    print("\n[INFO] Patched matplotlib.Axes.errorbar to abs() negative yerr for robustness.")
except Exception as e:
    print(f"\n[WARN] Could not patch matplotlib errorbar: {e}")
    

# ============================================================================
# Callbacks
# ============================================================================

from pytorch_lightning.callbacks import Callback

class EpochSummaryCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get('train_loss_epoch', metrics.get('train_loss', 'N/A'))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}" if isinstance(train_loss, float) else f"Epoch {epoch}: train_loss={train_loss}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get('val_loss', 'N/A')
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}" if isinstance(val_loss, float) else f"Epoch {epoch}: val_loss={val_loss}")


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Temporal Fusion Transformer for S&P 500 return prediction',
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
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # TFT architecture
    parser.add_argument('--max-encoder-length', type=int, default=20,
                        help='Lookback window length')
    parser.add_argument('--hidden-size', type=int, default=32,
                        help='Hidden layer size')
    parser.add_argument('--attention-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--hidden-continuous-size', type=int, default=16,
                        help='Hidden size for continuous features')
    
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

    # Include staleness features
    parser.add_argument('--no-staleness', action='store_true',
                        help='Disable staleness features (for baseline comparison)')    
    
    # Paths
    parser.add_argument('--splits-dir', type=str, default='data/splits',
                        help='Directory containing data splits')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Base directory for experiment outputs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing experiment directory')
    
    return parser.parse_args()


# ============================================================================
# FEATURE DEFINITIONS (based on frequency and feature set)
# ============================================================================

def get_features(feature_set, frequency, include_staleness=True):
    """
    Get feature lists based on configuration.
    Must match feature_configs.py exactly.
    
    Parameters:
    -----------
    feature_set : str
        Name of feature set from FEATURE_SETS
    frequency : str
        'daily' or 'monthly'
    include_staleness : bool
        If True, add staleness features for low-frequency variables
        
    Returns:
    --------
    dict with keys:
        'high_freq': list of high-frequency feature names
        'low_freq': list of low-frequency feature names
        'all': list of all feature names (including staleness if enabled)
        'staleness': list of staleness feature names (empty if not enabled)
    """
    from src.feature_configs import FEATURE_SETS, get_staleness_features
    
    config = FEATURE_SETS[feature_set]
    if config['features'] == 'all':
        # Kitchen sink - use everything available
        high_freq = ["VIX", "Treasury_10Y", "Yield_Spread"]
        low_freq = ["Inflation_YoY", "Unemployment", "Fed_Rate", 
                    "Consumer_Sentiment", "Industrial_Production"]
        all_features = high_freq + low_freq + ["SP500_Volatility"]
    else:
        # Use specified features from config
        all_features = [f for f in config['features'] if f != 'SP500_Returns']
        
        # Categorize as high/low frequency
        high_freq = [f for f in all_features if f in 
                     ["VIX", "Treasury_10Y", "Yield_Spread", "Credit_HY", "Credit_IG"]]
        low_freq = [f for f in all_features if f in 
                    ["Inflation_YoY", "Unemployment", "Fed_Rate", 
                     "Consumer_Sentiment", "Industrial_Production"]]
    
    # Add staleness features if requested
    staleness_info = get_staleness_features(all_features)
    staleness_features = staleness_info['staleness_features'] if include_staleness else []
    
    return {
        'high_freq': high_freq,
        'low_freq': low_freq,
        'all': all_features + staleness_features,
        'staleness': staleness_features,
    }


# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_all_seeds(seed):
    """Set seeds for complete reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA LOADING
# ============================================================================

def load_splits(splits_dir, feature_set, frequency):
    """Load pre-created train/val/test splits."""
    split_prefix = f"{feature_set}_{frequency}"
    train = pd.read_csv(
        os.path.join(splits_dir, f"{split_prefix}_train.csv"),
        index_col='Date',
        parse_dates=True
    )
    val = pd.read_csv(
        os.path.join(splits_dir, f"{split_prefix}_val.csv"),
        index_col='Date',
        parse_dates=True
    )
    test = pd.read_csv(
        os.path.join(splits_dir, f"{split_prefix}_test.csv"),
        index_col='Date',
        parse_dates=True
    )
    return train, val, test

def prepare_tft_data(train_df, val_df, args, features, add_staleness=True):
    """Prepare data in TimeSeriesDataSet format for TFT."""
    
    # add staleness
    if add_staleness and len(features['staleness']) > 0:
        print("\nAdding staleness features to data...")
        from src.data_utils import add_staleness_features
        
        train_df = add_staleness_features(train_df, use_vintage=False, verbose=True)
        val_df = add_staleness_features(val_df, use_vintage=False, verbose=True)

    # DROP SOURCE COLUMNS (used for staleness detection only, not model features)
    source_cols_to_drop = []
    from src.feature_configs import FEATURE_METADATA
    for feature in train_df.columns:
        if feature in FEATURE_METADATA:
            continue  # Keep actual features
        # Check if this is a source column for another feature
        for feat, meta in FEATURE_METADATA.items():
            if meta.get('source_column') == feature:
                source_cols_to_drop.append(feature)
                break
    
    if source_cols_to_drop:
        print(f"\nDropping source columns (used for staleness only): {source_cols_to_drop}")
        train_df = train_df.drop(columns=source_cols_to_drop)
        val_df = val_df.drop(columns=source_cols_to_drop)
    
    print(f"\nFinal features for TFT: {list(train_df.columns)}")

    # Reset index and add required columns
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    
    # Add time index (sequential integers)
    train_df['time_idx'] = range(len(train_df))
    val_df['time_idx'] = range(len(train_df), len(train_df) + len(val_df))
    
    # Add group identifier (single time series)
    train_df['group'] = 'SP500'
    val_df['group'] = 'SP500'
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=1,  # Always predict 1 step ahead
        time_varying_known_reals=[],  # No known future inputs
        time_varying_unknown_reals=features['all'],
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset (uses training stats)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True
    )
    
    return training, validation

# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(training_dataset, args):
    """Initialize TFT model."""
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=7,  # 7 quantiles for probabilistic forecasting
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return tft

# ============================================================================
# EXPERIMENT TRACKING SETUP
# ============================================================================

def save_config(args, features, output_dir):
    """Save all hyperparameters and configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'experiment_name': args.experiment_name,
        'created_at': datetime.now().isoformat(),
        'random_seed': args.seed,
        'feature_set': args.feature_set,
        'frequency': args.frequency,
        'data': {
            'splits_dir': args.splits_dir,
            'split_prefix': f"{args.feature_set}_{args.frequency}",
            'train_size': len(pd.read_csv(os.path.join(
                args.splits_dir, 
                f"{args.feature_set}_{args.frequency}_train.csv"
            ))),
            'val_size': len(pd.read_csv(os.path.join(
                args.splits_dir, 
                f"{args.feature_set}_{args.frequency}_val.csv"
            ))),
        },
        'architecture': {
            'max_encoder_length': args.max_encoder_length,
            'max_prediction_length': 1,
            'hidden_size': args.hidden_size,
            'attention_head_size': args.attention_heads,
            'dropout': args.dropout,
            'hidden_continuous_size': args.hidden_continuous_size,
        },
        'training': {
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'learning_rate': args.learning_rate,
            'gradient_clip_val': args.gradient_clip,
            'early_stop_patience': args.early_stop_patience,
        },
        'features': features,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config

# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training loop."""
    # Parse arguments
    args = parse_args()
    
    # Set seeds for reproducibility
    set_all_seeds(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    # Check for existing experiment
    if os.path.exists(output_dir) and not args.overwrite:
        print(f"\nERROR: Experiment directory already exists: {output_dir}")
        print("Options:")
        print("  1. Use --overwrite flag to overwrite")
        print("  2. Use a different --experiment-name")
        print("  3. Manually delete the directory")
        return None, None
    
    # Get features for this configuration
    features = get_features(args.feature_set, args.frequency, include_staleness=not args.no_staleness)
    
    print("="*70)
    print(f"Training TFT: {args.experiment_name}")
    print("="*70)
    
    # Save configuration
    config = save_config(args, features, output_dir)
    
    # Load data
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_splits(
        args.splits_dir, 
        args.feature_set, 
        args.frequency
    )
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare TFT datasets
    print("\nPreparing TimeSeriesDataSet...")
    training, validation = prepare_tft_data(train_df, val_df, args, features)
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for reproducibility
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Initialize model
    print("\nInitializing model...")
    tft = create_model(training, args)
    print(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")
    
    # Setup callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stop_patience,
        mode="min"
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    # Setup both loggers so you get numeric metrics + figures
    csv_logger = CSVLogger("experiments", name=args.experiment_name)
    tb_logger = TensorBoardLogger("experiments", name=args.experiment_name)
    # Ensure tb_logger is first so figure logging is prioritized
    logger = [tb_logger, csv_logger]
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator=device,
        devices=1,
        gradient_clip_val=args.gradient_clip,
        callbacks=[early_stop, checkpoint, EpochSummaryCallback()],
        deterministic=False,
        strategy="auto",
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # Debug prints to inspect logger setup
    """
    print("Logger(s):", trainer.logger)
    if hasattr(trainer.logger, "loggers"):
        for lg in trainer.logger.loggers:
            exp = getattr(lg, "experiment", None)
            print(
                f"  Logger: {type(lg)}, experiment type: {type(exp)}, "
                f"has add_figure? {hasattr(exp, 'add_figure')}"
            )
    else:
        exp = getattr(trainer.logger, "experiment", None)
        print(
            f"  Single logger: {type(trainer.logger)}, experiment: {type(exp)}, "
            f"has add_figure? {hasattr(exp, 'add_figure')}"
        )
    """

    # Patch fallback for ExperimentWriter if it lacks add_figure
    if hasattr(trainer.logger, "loggers"):
        for lg in trainer.logger.loggers:
            exp = getattr(lg, "experiment", None)
            if exp is not None and not hasattr(exp, "add_figure"):
                def _dummy_add_figure(*args, **kwargs):
                    print("\n[WARN] add_figure called on experiment without support. Skipping.")
                setattr(exp, "add_figure", _dummy_add_figure)
                print(f"\n[INFO] Patched add_figure on {type(exp)}")
    else:
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and not hasattr(exp, "add_figure"):
            def _dummy_add_figure(*args, **kwargs):
                print("\n[WARN] add_figure called on experiment without support. Skipping.")
            setattr(exp, "add_figure", _dummy_add_figure)
            print(f"\n[INFO] Patched add_figure on {type(exp)}")

    # Finally, run training
    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {output_dir}/checkpoints/")
    print(f"Logs will be saved to: {output_dir}/logs/")
    print()
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print()
    print("\n" + "="*70)
    print("Training complete")
    print("="*70)
    print(f"\nBest model checkpoint: {checkpoint.best_model_path}")
    print(f"Best validation loss: {checkpoint.best_model_score:.6f}")


    # --- Determine if early stopping triggered ---
    stopped_early = False
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            # check attributes that may exist depending on version
            if getattr(cb, "stopped", False) or getattr(cb, "stopped_epoch", None) not in (None, 0):
                stopped_early = True
            break
    
    # Save final metrics
    metrics = {
        'best_model_path': checkpoint.best_model_path,
        'best_val_loss': float(checkpoint.best_model_score),
        'total_epochs': trainer.current_epoch,
        'early_stopped': stopped_early,
    }
    
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return tft, trainer

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train()
