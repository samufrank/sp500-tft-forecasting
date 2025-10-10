"""
Summarize TFT experiment results across multiple runs.

Reads config.json and final_metrics.json from each experiment directory
and creates a comparison table.

Usage:
    python summarize_results.py
    python summarize_results.py --experiments-dir experiments
    python summarize_results.py --sort-by val_loss
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize TFT experiment results')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directory containing experiment subdirectories')
    parser.add_argument('--sort-by', type=str, default='val_loss',
                        choices=['val_loss', 'epochs', 'experiment'],
                        help='Column to sort results by')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to CSV file (optional)')
    return parser.parse_args()


def load_experiment_results(exp_dir):
    """Load config and metrics from a single experiment directory."""
    exp_path = Path(exp_dir)
    
    # Load config
    config_path = exp_path / 'config.json'
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load final metrics
    metrics_path = exp_path / 'final_metrics.json'
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract relevant information
    result = {
        'experiment': exp_path.name,
        'val_loss': metrics.get('best_val_loss', float('inf')),
        'epochs': metrics.get('total_epochs', 0),
        'early_stopped': metrics.get('early_stopped', False),
        'hidden_size': config['architecture'].get('hidden_size', '-'),
        'attention_heads': config['architecture'].get('attention_head_size', '-'),
        'encoder_length': config['architecture'].get('max_encoder_length', '-'),
        'learning_rate': config['training'].get('learning_rate', '-'),
        'batch_size': config['training'].get('batch_size', '-'),
        'dropout': config['architecture'].get('dropout', '-'),
        'feature_set': config.get('feature_set', '-'),
        'frequency': config.get('frequency', '-'),
    }
    
    return result


def main():
    args = parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"Error: Directory {experiments_dir} does not exist")
        return
    
    # Collect results from all experiments
    results = []
    
    for exp_path in sorted(experiments_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        
        result = load_experiment_results(exp_path)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No experiment results found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by specified column
    if args.sort_by == 'val_loss':
        df = df.sort_values('val_loss')
    elif args.sort_by == 'epochs':
        df = df.sort_values('epochs', ascending=False)
    elif args.sort_by == 'experiment':
        df = df.sort_values('experiment')
    
    # Format for display
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print()
    
    # Create display table with key columns first
    display_cols = [
        'experiment', 'val_loss', 'epochs', 'early_stopped',
        'hidden_size', 'attention_heads', 'encoder_length',
        'learning_rate', 'batch_size', 'dropout',
        'feature_set', 'frequency'
    ]
    
    # Print summary statistics
    print(f"Total experiments: {len(df)}")
    print(f"Best validation loss: {df['val_loss'].min():.6f} ({df.loc[df['val_loss'].idxmin(), 'experiment']})")
    print(f"Average validation loss: {df['val_loss'].mean():.6f}")
    print(f"Early stopped: {df['early_stopped'].sum()} / {len(df)}")
    print()
    
    # Print full table
    print(df[display_cols].to_string(index=False))
    print()
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
        print()
    
    # Print top 3 performers
    print("="*100)
    print("TOP 3 EXPERIMENTS BY VALIDATION LOSS")
    print("="*100)
    top3 = df.nsmallest(3, 'val_loss')
    for idx, row in top3.iterrows():
        print(f"\n{row['experiment']}:")
        print(f"  Val Loss: {row['val_loss']:.6f}")
        print(f"  Epochs: {row['epochs']} {'(early stopped)' if row['early_stopped'] else ''}")
        print(f"  Config: hidden={row['hidden_size']}, attn_heads={row['attention_heads']}, "
              f"lr={row['learning_rate']}, batch={row['batch_size']}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
