"""
Evaluate trained Temporal Fusion Transformer on test set.

Comprehensive evaluation including statistical metrics, financial metrics,
and diagnostic outputs with full experiment logging.

Usage:
    # Basic evaluation (automatically uses best checkpoint)
    python train/evaluate_tft.py --experiment-name exp004
    
    # Specific checkpoint
    python train/evaluate_tft.py \\
        --experiment-name exp004 \\
        --checkpoint experiments/exp004/checkpoints/tft-epoch=00-val_loss=0.1191.ckpt
    
    # Custom test split
    python train/evaluate_tft.py \\
        --experiment-name exp004 \\
        --test-split data/splits/core_proposal_daily_test.csv
"""

import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained TFT model on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name of experiment (for loading config)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if None, uses best from training)')
    parser.add_argument('--test-split', type=str, default=None,
                        help='Path to test CSV (if None, infers from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (if None, uses experiment dir)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(experiment_name):
    """Load experiment configuration from training run."""
    config_path = os.path.join('experiments', experiment_name, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(config, test_split_path=None):
    """Load test split and prepare for evaluation."""
    # Determine test split path
    if test_split_path is None:
        split_prefix = f"{config['feature_set']}_{config['frequency']}"
        test_split_path = f"data/splits/{split_prefix}_test.csv"
    
    # Load test data
    test_df = pd.read_csv(test_split_path, index_col='Date', parse_dates=True)
    
    # Also need training data for TimeSeriesDataSet creation
    # (TFT needs training stats for normalization)
    split_prefix = f"{config['feature_set']}_{config['frequency']}"
    train_path = f"data/splits/{split_prefix}_train.csv"
    train_df = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    
    return train_df, test_df


def prepare_test_dataset(train_df, test_df, config):
    """Prepare test dataset in TimeSeriesDataSet format."""
    # Reset index and add required columns
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    # Combine train and test for continuous time index
    # This ensures test samples have enough history for encoder
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['time_idx'] = range(len(combined_df))
    combined_df['group'] = 'SP500'
    
    # Get feature list from config
    features = config['features']['all']
    
    # Create training dataset for normalization parameters (train only)
    train_df['time_idx'] = range(len(train_df))
    train_df['group'] = 'SP500'
    
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=config['architecture']['max_encoder_length'],
        max_prediction_length=1,
        time_varying_known_reals=[],
        time_varying_unknown_reals=features,
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create test dataset from combined data (uses train normalization)
    # Only samples from test period will be predicted, but with train history available
    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        combined_df,
        predict=False,  # false for rolling predictions
        stop_randomization=True
    )
    
    # Filter dataset to only test period indices
    # Indices in combined_df: 0 to len(train_df)-1 are training, rest are test
    test_start_idx = len(train_df)
    test_indices = list(range(test_start_idx, len(combined_df)))

    # DEBUG
    #print(f"test_dataset.index columns: {test_dataset.index.columns.tolist()}")
    #print(f"test_dataset.index shape: {test_dataset.index.shape}")
    #print(f"First few rows:\n{test_dataset.index.head()}")

    # Manually filter the dataset's index
    test_dataset.index = test_dataset.index[test_dataset.index['time'] >= test_start_idx]

    # DEBUG
    #print(f"Filtered test_dataset size: {len(test_dataset.index)}")
    
    return test_dataset, test_df, test_start_idx


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path):
    """Load trained TFT model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    return model


# ============================================================================
# PREDICTION
# ============================================================================

def generate_predictions(model, test_dataset, batch_size=128):
    """Generate predictions on test set."""
    # Create dataloader
    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0
    )
    
    # DEBUG
    #print(f"Test dataloader batches: {len(test_dataloader)}")
    #print(f"Test dataset size: {len(test_dataset)}")
    
    # Generate predictions
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Unpack batch - dataloader returns (x_dict, y_tuple)
            x, y = batch
            
            # Get predictions
            pred = model(x)
            
            # DEBUG check what pred actually is
            #print(f"pred type: {type(pred)}")
            #print(f"pred: {pred if not isinstance(pred, torch.Tensor) else pred.shape}")
            
            # Extract point prediction (median quantile)
            # TFT returns a named tuple with 'prediction' attribute
            # Shape: [batch_size, prediction_length, num_quantiles]
            if hasattr(pred, 'prediction'):
                pred_tensor = pred.prediction
            elif isinstance(pred, tuple):
                pred_tensor = pred[0]
            elif isinstance(pred, dict) and 'prediction' in pred:
                pred_tensor = pred['prediction']
            else:
                pred_tensor = pred
            
            # For prediction_length=1, quantiles=[0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
            # Index 3 is the median (0.50 quantile)
            point_pred = pred_tensor[:, 0, 3]
            
            # Extract actual values - y is also a tuple (target, weight) or just target
            if isinstance(y, tuple):
                y_actual = y[0]
            else:
                y_actual = y
            
            # DEBUG
            #print(f"Batch {i}: predictions shape={point_pred.shape}, actuals shape={y_actual.shape}")
            
            predictions.append(point_pred.cpu().numpy())
            actuals.append(y_actual[:, 0].cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    print(f"Total predictions: {len(predictions)}, Total actuals: {len(actuals)}")
    
    return predictions, actuals


# ============================================================================
# STATISTICAL METRICS
# ============================================================================

def compute_statistical_metrics(predictions, actuals):
    """Compute standard statistical forecast metrics."""
    # Basic error metrics
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    metrics = {
        'mse': float(np.mean(squared_errors)),
        'rmse': float(np.sqrt(np.mean(squared_errors))),
        'mae': float(np.mean(abs_errors)),
        'mape': float(np.mean(np.abs(errors / (actuals + 1e-10))) * 100),
    }
    
    # R-squared (out-of-sample)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    metrics['r2'] = float(1 - (ss_res / ss_tot))
    
    # Mean and std of errors
    metrics['mean_error'] = float(np.mean(errors))
    metrics['std_error'] = float(np.std(errors))
    
    return metrics


# ============================================================================
# FINANCIAL METRICS
# ============================================================================

def compute_financial_metrics(predictions, actuals):
    """Compute financial performance metrics."""
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = float(np.mean(pred_direction == actual_direction))
    
    # Simple long-only strategy (trade when predicted return > 0)
    strategy_returns = np.where(predictions > 0, actuals, 0)
    cumulative_returns = np.cumprod(1 + strategy_returns / 100) - 1
    
    # Sharpe ratio (annualized, assuming daily data)
    if len(strategy_returns) > 1:
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe = float((mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns / 100)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = float(np.min(drawdown))
    
    # Hit rate (percentage of profitable trades)
    profitable_trades = strategy_returns > 0
    hit_rate = float(np.mean(profitable_trades)) if len(strategy_returns) > 0 else 0.0
    
    metrics = {
        'directional_accuracy': directional_accuracy,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 0.0,
        'hit_rate': hit_rate,
        'num_trades': int(np.sum(predictions > 0)),
    }
    
    # DEBUG
    print(f"Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
    print(f"Negative predictions: {np.sum(predictions < 0)}/{len(predictions)}")
    
    return metrics, strategy_returns


# ============================================================================
# DIAGNOSTIC OUTPUTS
# ============================================================================

def compute_residual_diagnostics(predictions, actuals):
    """Compute residual diagnostics."""
    errors = actuals - predictions
    
    # Normality test
    _, normality_p = stats.normaltest(errors)
    
    # Autocorrelation at lag 1
    if len(errors) > 1:
        acf_lag1 = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
    else:
        acf_lag1 = 0.0
    
    diagnostics = {
        'normality_p_value': float(normality_p),
        'acf_lag1': acf_lag1,
        'skewness': float(stats.skew(errors)),
        'kurtosis': float(stats.kurtosis(errors)),
    }
    
    return diagnostics


def create_diagnostic_plots(predictions, actuals, dates, output_dir):
    """Create diagnostic plots."""
    errors = actuals - predictions
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[0, 0].plot([actuals.min(), actuals.max()], 
                     [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Returns (%)')
    axes[0, 0].set_ylabel('Predicted Returns (%)')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals over time
    axes[0, 1].plot(dates, errors, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error (%)')
    axes[0, 1].set_title('Residuals Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual distribution
    axes[1, 0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_plots.png'), dpi=150)
    plt.close()


def create_performance_plots(actuals, strategy_returns, dates, output_dir):
    """Create performance plots."""
    # Cumulative returns
    cumulative_strategy = np.cumprod(1 + strategy_returns / 100)
    cumulative_buy_hold = np.cumprod(1 + actuals / 100)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. Cumulative returns comparison
    axes[0].plot(dates, cumulative_buy_hold, label='Buy & Hold', linewidth=2)
    axes[0].plot(dates, cumulative_strategy, label='TFT Strategy', linewidth=2)
    axes[0].set_ylabel('Cumulative Return (Growth of $1)')
    axes[0].set_title('Strategy Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe ratio (60-day window)
    window = min(60, len(strategy_returns) // 4)
    if window > 5:
        rolling_mean = pd.Series(strategy_returns).rolling(window).mean()
        rolling_std = pd.Series(strategy_returns).rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        axes[1].plot(dates, rolling_sharpe, linewidth=2)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(f'Rolling Sharpe ({window}-day)')
        axes[1].set_title('Rolling Sharpe Ratio')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_plots.png'), dpi=150)
    plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(predictions, actuals, dates, metrics_stat, metrics_fin, 
                 diagnostics, output_dir):
    """Save all evaluation results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions CSV
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': actuals - predictions,
    })
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Combine all metrics
    all_metrics = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'statistical_metrics': metrics_stat,
        'financial_metrics': metrics_fin,
        'residual_diagnostics': diagnostics,
    }
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


def print_summary(metrics_stat, metrics_fin):
    """Print evaluation summary to console."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nSTATISTICAL METRICS:")
    print(f"  MSE:              {metrics_stat['mse']:.6f}")
    print(f"  RMSE:             {metrics_stat['rmse']:.6f}")
    print(f"  MAE:              {metrics_stat['mae']:.6f}")
    print(f"  Out-of-sample R²: {metrics_stat['r2']:.4f}")
    
    print("\nFINANCIAL METRICS:")
    print(f"  Directional Acc:  {metrics_fin['directional_accuracy']:.2%}")
    print(f"  Sharpe Ratio:     {metrics_fin['sharpe_ratio']:.4f}")
    print(f"  Total Return:     {metrics_fin['total_return']:.2%}")
    print(f"  Max Drawdown:     {metrics_fin['max_drawdown']:.2%}")
    print(f"  Hit Rate:         {metrics_fin['hit_rate']:.2%}")
    print(f"  Number of Trades: {metrics_fin['num_trades']}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate():
    """Main evaluation pipeline."""
    args = parse_args()
    
    print("="*70)
    print(f"Evaluating TFT: {args.experiment_name}")
    print("="*70)
    
    # Determine checkpoint path
    if args.checkpoint is None:
        # Auto-load best checkpoint from training
        metrics_path = os.path.join('experiments', args.experiment_name, 'final_metrics.json')
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"No final_metrics.json found at {metrics_path}.\n"
                f"Either specify --checkpoint manually or ensure training completed successfully."
            )
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        checkpoint_path = metrics['best_model_path']
        print(f"\nUsing best checkpoint from training:")
        print(f"  {checkpoint_path}")
        print(f"  Validation loss: {metrics['best_val_loss']:.6f}")
    else:
        checkpoint_path = args.checkpoint
        print(f"\nUsing specified checkpoint:")
        print(f"  {checkpoint_path}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(args.experiment_name)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join('experiments', args.experiment_name, 'evaluation')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    train_df, test_df = load_test_data(config, args.test_split)
    print(f"Test samples: {len(test_df)}")
    
    # Prepare test dataset
    print("Preparing test dataset...")
    test_dataset, test_df_indexed, test_start_idx = prepare_test_dataset(train_df, test_df, config)
    
    # Count actual test predictions
    # Need to account for encoder length requirement
    max_encoder_length = config['architecture']['max_encoder_length']
    expected_predictions = len(test_df) - max_encoder_length + 1
    print(f"Expected test predictions: {expected_predictions} (test samples - encoder length + 1)")
    
    # Load model
    print(f"Loading model from checkpoint...")
    model = load_model(checkpoint_path)
    
    # Generate predictions
    print("Generating predictions...")
    predictions, actuals = generate_predictions(model, test_dataset, args.batch_size)
    
    # Get dates for plotting
    dates = test_df_indexed['Date'].values[-len(predictions):]
    
    # Compute metrics
    print("Computing metrics...")
    metrics_stat = compute_statistical_metrics(predictions, actuals)
    metrics_fin, strategy_returns = compute_financial_metrics(predictions, actuals)
    diagnostics = compute_residual_diagnostics(predictions, actuals)
    
    # Create plots
    print("Creating diagnostic plots...")
    create_diagnostic_plots(predictions, actuals, dates, output_dir)
    create_performance_plots(actuals, strategy_returns, dates, output_dir)
    
    # Save results
    print("Saving results...")
    all_metrics = save_results(predictions, actuals, dates, metrics_stat, 
                                metrics_fin, diagnostics, output_dir)
    
    # Add checkpoint info to saved metrics
    all_metrics['checkpoint_used'] = checkpoint_path
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    print_summary(metrics_stat, metrics_fin)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - predictions.csv")
    print(f"  - evaluation_metrics.json")
    print(f"  - diagnostic_plots.png")
    print(f"  - performance_plots.png")
    
    return all_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    evaluate()
