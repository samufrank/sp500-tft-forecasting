"""
Utilities for loading and preparing datasets for modeling.
"""

import os
import pandas as pd
from src.feature_configs import FEATURE_SETS, FEATURE_METADATA, TARGET, get_constituent_columns


def load_constituents(frequency='daily', version='vintage_2005', data_path='data', verbose=True):
    """
    Load constituent returns from separate CSV file.
    
    Parameters:
    -----------
    frequency : str
        'daily' (weekly/monthly not typically used for constituents)
    version : str
        Version suffix, e.g., 'vintage_2005' -> constituents_daily_vintage_2005.csv
    data_path : str
        Path to data directory
    verbose : bool
        Print loading information
        
    Returns:
    --------
    pd.DataFrame or None
        Constituent returns indexed by date, or None if file doesn't exist
    """
    filename = f"{data_path}/constituents_{frequency}_{version}.csv"
    
    if not os.path.exists(filename):
        if verbose:
            print(f"Constituent file not found: {filename}")
            print("Run collect_constituents.py to generate this file.")
        return None
    
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    if verbose:
        print(f"Loaded constituents: {df.shape[1]} tickers, {len(df)} observations")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def load_feature_set(config_name='core_proposal', 
                     frequency='daily',
                     version='fixed',
                     data_path='data',
                     verbose=True):
    """
    Load a specific feature set configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
    frequency : str
        'daily', 'weekly', or 'monthly'
    version : str
        'fixed' or 'vintage'
    data_path : str
        Path to data directory
    verbose : bool
        Print loading information
        
    Returns:
    --------
    pd.DataFrame
        Dataset with selected features and date filtering applied.
        If config has include_constituents=True, constituent columns are joined.
    """
    # Validate config
    if config_name not in FEATURE_SETS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(FEATURE_SETS.keys())}")
    
    config = FEATURE_SETS[config_name]
    
    # Load full dataset
    filename = f"{data_path}/financial_dataset_{frequency}_{version}.csv"
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading Feature Set: {config_name}")
        print(f"{'='*70}")
        print(f"Description: {config['description']}")
    
    # Select features
    if config['features'] == 'all':
        # Kitchen sink: use everything except derived features we'll recalculate
        selected = df.copy()
    else:
        # Start with requested features
        features_to_load = config['features'].copy()
        
        # Add source columns needed for staleness computation
        for feature in config['features']:
            if feature in FEATURE_METADATA:
                metadata = FEATURE_METADATA[feature]
                if metadata.get('needs_staleness', False):
                    source_col = metadata.get('source_column', feature)
                    if source_col != feature and source_col not in features_to_load:
                        features_to_load.append(source_col)
                        if verbose:
                            print(f"Including source column '{source_col}' for staleness computation of '{feature}'")
        
        selected = df[features_to_load].copy()
    
    # Apply date filtering if specified
    if config.get('min_date') is not None:
        original_len = len(selected)
        selected = selected[selected.index >= config['min_date']]
        if verbose:
            print(f"Date filter: >={config['min_date']}")
            print(f"  Samples before: {original_len}")
            print(f"  Samples after: {len(selected)}")
    
    # Join constituent returns if requested
    if config.get('include_constituents', False):
        constituent_count = config.get('constituent_count', 50)
        constituent_version = config.get('constituent_version', 'vintage_2005')
        
        if verbose:
            print(f"\nLoading constituent returns (top {constituent_count}, version: {constituent_version})...")
        
        constituents_df = load_constituents(
            frequency=frequency,
            version=constituent_version,
            data_path=data_path,
            verbose=verbose
        )
        
        if constituents_df is not None:
            # Get column names for requested count
            requested_cols = get_constituent_columns(constituent_count)
            
            # Filter to available columns (some may have failed coverage screening)
            available_cols = [col for col in requested_cols if col in constituents_df.columns]
            
            if verbose:
                print(f"Requested: {len(requested_cols)} constituents")
                print(f"Available: {len(available_cols)} constituents")
            
            # Join on date index
            constituents_subset = constituents_df[available_cols]
            selected = selected.join(constituents_subset, how='left')
            
            if verbose:
                print(f"Joined constituent columns: {len(available_cols)}")
        else:
            if verbose:
                print("WARNING: Constituent data not available. Proceeding without.")
    
    # Remove any remaining NaNs
    original_len = len(selected)
    selected = selected.dropna()
    
    if verbose:
        print(f"\nFinal dataset:")
        print(f"  Shape: {selected.shape}")
        print(f"  Date range: {selected.index[0].date()} to {selected.index[-1].date()}")
        print(f"  Years: {(selected.index[-1] - selected.index[0]).days / 365.25:.1f}")
        print(f"  Features: {list(selected.columns)}")
        if original_len != len(selected):
            print(f"  Rows dropped (NaN): {original_len - len(selected)}")
    
    return selected


def create_train_val_test_split(df, train_pct=0.7, val_pct=0.15, verbose=True):
    """
    Create temporal train/validation/test splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with DatetimeIndex
    train_pct : float
        Proportion for training (default 0.7)
    val_pct : float
        Proportion for validation (default 0.15)
    verbose : bool
        Print split information
        
    Returns:
    --------
    train, val, test : tuple of pd.DataFrame
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Train/Validation/Test Split")
        print(f"{'='*70}")
        print(f"Train: {train.index[0].date()} to {train.index[-1].date()} "
              f"({len(train):,} obs, {train_pct*100:.0f}%)")
        print(f"Val:   {val.index[0].date()} to {val.index[-1].date()} "
              f"({len(val):,} obs, {val_pct*100:.0f}%)")
        print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} "
              f"({len(test):,} obs, {(1-train_pct-val_pct)*100:.0f}%)")
    
    return train, val, test


def add_staleness_features(df, use_vintage=False, verbose=True):
    """
    Add staleness indicators for low-frequency features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and financial features
    use_vintage : bool
        If True, compute staleness from actual release dates (vintage)
        If False, use typical lag patterns (fixed)
    verbose : bool
        Print information about staleness computation
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with additional staleness features
    """
    
    df = df.copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Adding Staleness Features")
        print(f"{'='*70}")
        print(f"Mode: {'Vintage (ALFRED)' if use_vintage else 'Fixed lag'}")
    
    for feature, metadata in FEATURE_METADATA.items():
        if not metadata['needs_staleness']:
            continue
            
        if feature not in df.columns:
            continue
       
        # Use source_column if specified, otherwise use feature itself
        source_col = metadata.get('source_column', feature)
        
        if source_col not in df.columns:
            if verbose:
                print(f"\nSkipping {feature}: source column {source_col} not found")
            continue
        
        if verbose:
            print(f"\nProcessing: {feature} (detecting from {source_col})")
        
        # Detect updates from source column
        feature_values = df[source_col]
        
        # Detect updates: value changed from previous day
        # For CPI/inflation, changes should be substantial (at least 0.01%)
        updates = (feature_values.diff().abs() > 0.01)

        # For first observation, assume it's an update
        updates.at[updates.index[0]] = True
        
        # Days since last update
        days_since_update = pd.Series(0, index=df.index)
        last_update_idx = 0
        
        for i in range(len(df)):
            if updates.iloc[i]:
                last_update_idx = i
                days_since_update.iloc[i] = 0
            else:
                days_since_update.iloc[i] = i - last_update_idx
        
        # Binary freshness indicator
        is_fresh = updates.astype(int)
        
        # Add to dataframe with appropriate names
        staleness_col_name = metadata['staleness_features'][0]  # e.g., 'days_since_CPI_update'
        freshness_col_name = metadata['staleness_features'][1]  # e.g., 'CPI_is_fresh'
        
        df[staleness_col_name] = days_since_update
        df[freshness_col_name] = is_fresh
        
        if verbose:
            num_updates = is_fresh.sum()
            avg_staleness = days_since_update.mean()
            max_staleness = days_since_update.max()
            print(f"  Updates detected: {num_updates}")
            print(f"  Avg days stale: {avg_staleness:.1f}")
            print(f"  Max days stale: {max_staleness}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Total staleness features added: {len([c for c in df.columns if 'days_since' in c or 'is_fresh' in c])}")
        print(f"{'='*70}\n")
    
    return df


def get_constituent_target_columns(df):
    """
    Extract constituent return columns from a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset that may contain constituent return columns
        
    Returns:
    --------
    list
        Column names ending with '_Returns' (excluding SP500_Returns)
    """
    return [col for col in df.columns 
            if col.endswith('_Returns') and col != 'SP500_Returns']


if __name__ == "__main__":
    print("Testing feature set loading...\n")
    
    # Test each config
    for config_name in FEATURE_SETS.keys():
        try:
            # Skip multitask if constituents not available
            if 'multitask' in config_name:
                print(f"\n{config_name}: Skipping (requires constituent data)")
                continue
                
            data = load_feature_set(config_name, frequency='daily', version='vintage', verbose=False)
            train, val, test = create_train_val_test_split(data, verbose=False)
            print(f"\n{config_name}: {len(data)} total samples "
                  f"(train={len(train)}, val={len(val)}, test={len(test)})")
        except Exception as e:
            print(f"\n{config_name}: Error - {e}")