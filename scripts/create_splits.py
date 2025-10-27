"""
Create and save train/validation/test splits for experiments.

This script ensures all experiments use identical data splits.
Run once after data preprocessing is complete.

Usage:
    python create_splits.py --feature-set core_proposal --frequency daily
    python create_splits.py --feature-set macro_heavy --frequency monthly
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from src.data_utils import load_feature_set, create_train_val_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits for financial forecasting experiments'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        default='core_proposal',
        choices=['core_proposal', 'core_plus_credit', 'macro_heavy', 'market_only', 'kitchen_sink'],
        help='Feature set configuration from feature_configs.py'
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default='daily',
        choices=['daily', 'monthly'],
        help='Data frequency'
    )
    parser.add_argument(
        '--train-pct',
        type=float,
        default=0.7,
        help='Training set proportion (default: 0.7)'
    )
    parser.add_argument(
        '--val-pct',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='.',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--timestamp',
        action='store_true',
        help='Add timestamp to output filenames (prevents overwriting)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Optional version suffix (e.g., "v2" for core_proposal_daily_v2_train.csv)'
    )
    parser.add_argument(
        '--data-version',
        type=str,
        default='fixed',
        choices=['fixed', 'vintage'],
        help='Data version to use: fixed (fixed-shift alignment) or vintage (ALFRED alignment)'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print("Creating Data Splits")
    print("="*70)
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load feature set
    print(f"\nLoading feature set: {args.feature_set}")
    print(f"Frequency: {args.frequency}")
    print(f"Data version: {args.data_version}")
    df = load_feature_set(
        config_name=args.feature_set,
        frequency=args.frequency,
        version=args.data_version,
        #data_path=args.data_path,
        verbose=True
    )
    
    # Create temporal splits
    print(f"\nCreating temporal splits (train={args.train_pct}, val={args.val_pct})...")
    train, val, test = create_train_val_test_split(
        df, 
        train_pct=args.train_pct, 
        val_pct=args.val_pct,
        verbose=True
    )
    
    # Save splits with data version in filename and subdirectory
    # Create subdirectory for data version (unless already specified in output-dir)
    if args.output_dir.endswith(args.data_version):
        # User already specified version in path (e.g., data/splits/vintage)
        version_output_dir = args.output_dir
    else:
        # Add version subdirectory (e.g., data/splits -> data/splits/vintage)
        version_output_dir = os.path.join(args.output_dir, args.data_version)
    
    os.makedirs(version_output_dir, exist_ok=True)
    
    split_prefix = f"{args.feature_set}_{args.frequency}_{args.data_version}"
    if args.version:
        # Optional additional version suffix (e.g., "v2" for experiments)
        split_prefix = f"{split_prefix}_{args.version}"
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_prefix = f"{split_prefix}_{timestamp}"
    
    train_path = os.path.join(version_output_dir, f"{split_prefix}_train.csv")
    val_path = os.path.join(version_output_dir, f"{split_prefix}_val.csv")
    test_path = os.path.join(version_output_dir, f"{split_prefix}_test.csv")
    
    train.to_csv(train_path)
    val.to_csv(val_path)
    test.to_csv(test_path)
    
    print(f"\nSaved splits to {version_output_dir}/")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'feature_set': args.feature_set,
        'frequency': args.frequency,
        'data_version': args.data_version,
        'train_pct': args.train_pct,
        'val_pct': args.val_pct,
        'test_pct': 1 - args.train_pct - args.val_pct,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'train_dates': {
            'start': str(train.index[0]),
            'end': str(train.index[-1])
        },
        'val_dates': {
            'start': str(val.index[0]),
            'end': str(val.index[-1])
        },
        'test_dates': {
            'start': str(test.index[0]),
            'end': str(test.index[-1])
        },
        'features': list(df.columns),
    }
    
    metadata_path = os.path.join(version_output_dir, f"{split_prefix}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print("\n" + "="*70)
    print("Split creation complete!")
    print("="*70)

if __name__ == "__main__":
    main()
