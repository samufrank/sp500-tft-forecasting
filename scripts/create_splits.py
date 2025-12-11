"""
Create and save train/validation/test splits for experiments.

This script ensures all experiments use identical data splits.
Run once after data preprocessing is complete.

Usage:
    # Percentage-based split (default)
    python create_splits.py --feature-set core_proposal --frequency daily
    python create_splits.py --feature-set macro_heavy --frequency monthly
    
    # Date-based split (for rolling/walk-forward evaluation)
    python create_splits.py --feature-set core_proposal --frequency daily \\
        --train-end 2015-12-31 --val-end 2017-12-31 --test-end 2019-12-31
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from src.data_utils import (
    load_feature_set, 
    create_train_val_test_split, 
    create_split_by_dates,
    add_cumulative_returns,
    CUMRET_HORIZONS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits for financial forecasting experiments'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        default='core_proposal',
        choices=['core_proposal', 'core_plus_credit', 'macro_heavy', 'market_only', 'kitchen_sink', 'core_dynamics'],
        help='Feature set configuration from feature_configs.py'
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default='daily',
        choices=['daily', 'weekly', 'monthly'],
        help='Data frequency'
    )
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Use enhanced dataset with technical features'
    )
    parser.add_argument(
        '--train-pct',
        type=float,
        default=0.7,
        help='Training set proportion (default: 0.7). Ignored if date boundaries are specified.'
    )
    parser.add_argument(
        '--val-pct',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15). Ignored if date boundaries are specified.'
    )
    # Date boundary arguments for rolling/walk-forward evaluation
    parser.add_argument(
        '--train-start',
        type=str,
        default=None,
        help='Training start date (YYYY-MM-DD). If not specified, uses start of data.'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        default=None,
        help='Training end date (YYYY-MM-DD). Enables date-based splitting.'
    )
    parser.add_argument(
        '--val-end',
        type=str,
        default=None,
        help='Validation end date (YYYY-MM-DD). If not specified, no validation split.'
    )
    parser.add_argument(
        '--test-start',
        type=str,
        default=None,
        help='Test start date (YYYY-MM-DD). If not specified, starts after val-end or train-end.'
    )
    parser.add_argument(
        '--test-end',
        type=str,
        default=None,
        help='Test end date (YYYY-MM-DD). If not specified, uses end of data.'
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
    parser.add_argument('--lookback-buffer', type=int, default=0,
                    help='Number of rows from val/train to prepend to test for context')
    parser.add_argument('--cumret', action='store_true', default=True,
                    help='Add cumulative return targets (cumret_5, cumret_10, cumret_20, cumret_30). '
                         'Enabled by default. Use --no-cumret to disable.')
    parser.add_argument('--no-cumret', dest='cumret', action='store_false',
                    help='Disable cumulative return target computation')
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
    print(f"Enhanced: {args.enhanced}")
    df = load_feature_set(
        config_name=args.feature_set,
        frequency=args.frequency,
        version=args.data_version,
        enhanced=args.enhanced,
        #data_path=args.data_path,
        verbose=True
    )
    
    # Add cumulative return targets (computed on full dataset before splitting)
    cumret_stats = None
    max_horizon = 0
    if args.cumret:
        print(f"\nComputing cumulative return targets...")
        print(f"  Horizons: {CUMRET_HORIZONS} periods")
        print(f"  Note: For {args.frequency} data, cumret_N means N-{args.frequency[:-2] if args.frequency.endswith('ly') else args.frequency} forward return")
        df, cumret_stats = add_cumulative_returns(
            df, 
            return_col='SP500_Returns',
            horizons=CUMRET_HORIZONS,
            verbose=True
        )
        max_horizon = max(CUMRET_HORIZONS)
    
    # Create temporal splits - use date-based if dates provided, else percentage-based
    use_date_split = args.train_end is not None
    
    if use_date_split:
        print(f"\nCreating date-based splits...")
        print(f"  Train: {args.train_start or 'start'} to {args.train_end}")
        print(f"  Val:   {args.train_end} to {args.val_end or 'none'}")
        print(f"  Test:  {args.test_start or 'auto'} to {args.test_end or 'end'}")
        
        train, val, test = create_split_by_dates(
            df,
            train_start=args.train_start,
            train_end=args.train_end,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=args.test_end,
            verbose=True
        )
    else:
        print(f"\nCreating percentage-based splits (train={args.train_pct}, val={args.val_pct})...")
        train, val, test = create_train_val_test_split(
            df, 
            train_pct=args.train_pct, 
            val_pct=args.val_pct,
            verbose=True
        )
    
    # Drop trailing NaN rows from cumret computation
    # Each split loses its last max_horizon rows where cumret_30 would be NaN
    if args.cumret and max_horizon > 0:
        print(f"\nDropping trailing NaN rows from cumret computation...")
        cumret_cols = [f'cumret_{h}' for h in CUMRET_HORIZONS]
        
        # For train: drop last max_horizon rows
        train_before = len(train)
        train = train.iloc[:-max_horizon] if len(train) > max_horizon else train.iloc[0:0]
        print(f"  Train: {train_before} -> {len(train)} (dropped {train_before - len(train)} rows)")
        
        # For val: drop last max_horizon rows
        val_before = len(val)
        if len(val) > max_horizon:
            val = val.iloc[:-max_horizon]
        elif len(val) > 0:
            # Val set too small - this is a warning condition
            print(f"  WARNING: Validation set ({val_before} rows) smaller than max horizon ({max_horizon})")
            val = val.iloc[0:0]  # Empty
        print(f"  Val:   {val_before} -> {len(val)} (dropped {val_before - len(val)} rows)")
        
        # For test: drop last max_horizon rows
        test_before = len(test)
        test = test.iloc[:-max_horizon] if len(test) > max_horizon else test.iloc[0:0]
        print(f"  Test:  {test_before} -> {len(test)} (dropped {test_before - len(test)} rows)")
        
        # Verify no NaNs remain in cumret columns
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            if len(split_df) > 0:
                for col in cumret_cols:
                    if col in split_df.columns:
                        nan_count = split_df[col].isna().sum()
                        if nan_count > 0:
                            print(f"  WARNING: {split_name} still has {nan_count} NaN in {col}")
    
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
    if args.enhanced:
        split_prefix = f"{split_prefix}_enhanced"
    if args.version:
        # Optional additional version suffix (e.g., "v2" for experiments)
        split_prefix = f"{split_prefix}_{args.version}"
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_prefix = f"{split_prefix}_{timestamp}"
    
    if args.lookback_buffer and args.lookback_buffer > 0:
        buffer_source = val if len(val) > 0 else train
        if len(buffer_source) >= args.lookback_buffer:
            buffer_rows = buffer_source.tail(args.lookback_buffer)
            test = pd.concat([buffer_rows, test])
            print(f"  Added {args.lookback_buffer} lookback rows to test (now {len(test)} total)")
        else:
            print(f"  WARNING: Not enough rows for lookback buffer")
    
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
        'enhanced': args.enhanced,
        'split_method': 'date' if use_date_split else 'percentage',
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'features': list(df.columns),
        'cumret_enabled': args.cumret,
    }
    
    # Add cumret-specific metadata
    if args.cumret:
        metadata['cumret'] = {
            'horizons': [int(h) for h in CUMRET_HORIZONS],
            'columns': [f'cumret_{h}' for h in CUMRET_HORIZONS],
            'max_horizon': int(max_horizon),
            'rows_dropped_per_split': int(max_horizon),
            'valid_targets': ['SP500_Returns'] + [f'cumret_{h}' for h in CUMRET_HORIZONS],
        }
        if cumret_stats:
            # Convert numpy types to native Python for JSON serialization
            def to_native(v):
                if hasattr(v, 'item'):  # numpy scalar
                    return v.item()
                return v
            
            metadata['cumret']['statistics'] = {
                col: {k: to_native(v) for k, v in stats.items()}
                for col, stats in cumret_stats.items()
            }
    
    # Add split parameters based on method
    if use_date_split:
        metadata['date_boundaries'] = {
            'train_start': args.train_start,
            'train_end': args.train_end,
            'val_end': args.val_end,
            'test_start': args.test_start,
            'test_end': args.test_end,
        }
    else:
        metadata['percentages'] = {
            'train_pct': args.train_pct,
            'val_pct': args.val_pct,
            'test_pct': 1 - args.train_pct - args.val_pct,
        }
    
    # Add actual date ranges
    if len(train) > 0:
        metadata['train_dates'] = {
            'start': str(train.index[0]),
            'end': str(train.index[-1])
        }
    if len(val) > 0:
        metadata['val_dates'] = {
            'start': str(val.index[0]),
            'end': str(val.index[-1])
        }
    if len(test) > 0:
        metadata['test_dates'] = {
            'start': str(test.index[0]),
            'end': str(test.index[-1])
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