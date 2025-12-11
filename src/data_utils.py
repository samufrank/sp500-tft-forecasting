"""
Utilities for loading and preparing datasets for modeling.
"""

import pandas as pd
from src.feature_configs import (
    FEATURE_SETS, FEATURE_METADATA, TARGET, 
    DEFAULT_CHANGE_THRESHOLD, get_change_threshold
)

def load_feature_set(config_name='core_proposal', 
                     frequency='daily',
                     version='fixed',
                     enhanced=False,
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
    enhanced : bool
        If True, load enhanced dataset with technical features
    data_path : str
        Path to data directory
    verbose : bool
        Print loading information
        
    Returns:
    --------
    pd.DataFrame
        Dataset with selected features and date filtering applied
    """
    # Validate config
    if config_name not in FEATURE_SETS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(FEATURE_SETS.keys())}")
    
    # Validate frequency
    valid_frequencies = ['daily', 'weekly', 'monthly']
    if frequency not in valid_frequencies:
        raise ValueError(f"Unknown frequency: {frequency}. "
                        f"Available: {valid_frequencies}")
    
    config = FEATURE_SETS[config_name]
    
    # Load full dataset
    if enhanced:
        filename = f"{data_path}/financial_dataset_{frequency}_{version}_enhanced.csv"
    else:
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
        from src.feature_configs import FEATURE_METADATA
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
    if config['min_date'] is not None:
        original_len = len(selected)
        selected = selected[selected.index >= config['min_date']]
        if verbose:
            print(f"Date filter: >={config['min_date']}")
            print(f"  Samples before: {original_len}")
            print(f"  Samples after: {len(selected)}")
    
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
    Create temporal train/validation/test splits by percentage.
    
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


def create_split_by_dates(df, train_start=None, train_end=None, val_end=None, 
                          test_start=None, test_end=None, verbose=True):
    """
    Create temporal train/validation/test splits by date boundaries.
    
    For rolling/walk-forward evaluation. Dates are inclusive.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with DatetimeIndex
    train_start : str or None
        Training start date (YYYY-MM-DD). None = start of data.
    train_end : str
        Training end date (YYYY-MM-DD). Required.
    val_end : str or None
        Validation end date. None = no validation split (val will be empty).
    test_start : str or None
        Test start date. None = day after val_end (or train_end if no val).
    test_end : str or None
        Test end date. None = end of data.
    verbose : bool
        Print split information
        
    Returns:
    --------
    train, val, test : tuple of pd.DataFrame
        Note: val may be empty DataFrame if val_end not specified
    """
    if train_end is None:
        raise ValueError("train_end is required for date-based splitting")
    
    # Convert to timestamps
    train_start = pd.Timestamp(train_start) if train_start else df.index[0]
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end) if val_end else None
    test_start = pd.Timestamp(test_start) if test_start else None
    test_end = pd.Timestamp(test_end) if test_end else df.index[-1]
    
    # Create splits
    train = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    
    if val_end:
        # Validation: from day after train_end to val_end
        val = df[(df.index > train_end) & (df.index <= val_end)].copy()
        # Test: from test_start (or day after val_end) to test_end
        if test_start is None:
            test_start = val_end + pd.Timedelta(days=1)
        test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    else:
        # No validation split
        val = df.iloc[0:0].copy()  # Empty DataFrame with same columns
        # Test: from day after train_end to test_end
        if test_start is None:
            test_start = train_end + pd.Timedelta(days=1)
        test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Train/Validation/Test Split (Date-Based)")
        print(f"{'='*70}")
        if len(train) > 0:
            print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} obs)")
        else:
            print(f"Train: EMPTY")
        if len(val) > 0:
            print(f"Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val):,} obs)")
        else:
            print(f"Val:   EMPTY (no validation split)")
        if len(test) > 0:
            print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} obs)")
        else:
            print(f"Test:  EMPTY")
    
    return train, val, test


def add_staleness_features(df, use_vintage=False, staleness_mode='all', verbose=True):
    """
    Add staleness indicators for low-frequency features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and financial features
    use_vintage : bool
        If True, compute staleness from actual release dates (vintage)
        If False, use typical lag patterns (fixed)
    staleness_mode : str
        Which staleness features to add:
        - 'all': both days_since_* and *_is_fresh (default)
        - 'days_only': only days_since_* (continuous counter)
        - 'fresh_only': only *_is_fresh (sparse binary flag)
    verbose : bool
        Print information about staleness computation
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with additional staleness features
    """
    valid_modes = ['all', 'days_only', 'fresh_only']
    if staleness_mode not in valid_modes:
        raise ValueError(f"Invalid staleness_mode: {staleness_mode}. Must be one of {valid_modes}")
    
    df = df.copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Adding Staleness Features")
        print(f"{'='*70}")
        print(f"Vintage mode: {'Vintage (ALFRED)' if use_vintage else 'Fixed lag'}")
        print(f"Staleness mode: {staleness_mode}")
    
    for feature, metadata in FEATURE_METADATA.items():
        if not metadata.get('needs_staleness', False):
            continue
            
        if feature not in df.columns:
            continue
       
        # Use source_column if specified, otherwise use feature itself
        source_col = metadata.get('source_column', feature)
        
        if source_col not in df.columns:
            if verbose:
                print(f"\nSkipping {feature}: source column {source_col} not found")
            continue
        
        # Get feature-specific threshold
        threshold = get_change_threshold(feature)
        
        if verbose:
            print(f"\nProcessing: {feature} (detecting from {source_col}, threshold={threshold})")
        
        # Detect updates from source column
        feature_values = df[source_col]
        
        # Detect updates: value changed from previous day by more than threshold
        updates = (feature_values.diff().abs() > threshold)

        # For first observation, assume it's an update
        updates.iloc[0] = True
        
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
        
        # Add to dataframe based on staleness_mode
        staleness_col_name = metadata['staleness_features'][0]  # e.g., 'days_since_CPI_update'
        freshness_col_name = metadata['staleness_features'][1]  # e.g., 'CPI_is_fresh'
        
        if staleness_mode in ['all', 'days_only']:
            df[staleness_col_name] = days_since_update
        if staleness_mode in ['all', 'fresh_only']:
            df[freshness_col_name] = is_fresh
        
        if verbose:
            num_updates = is_fresh.sum()
            avg_staleness = days_since_update.mean()
            max_staleness = days_since_update.max()
            # Sanity check: expected updates for monthly data
            years = (df.index[-1] - df.index[0]).days / 365.25
            expected_monthly = years * 12
            expected_irregular = years * 10  # Fed Funds effective rate, empirically ~10/year
            freq = metadata.get('update_frequency', 'monthly')
            expected = expected_irregular if freq == 'irregular' else expected_monthly
            
            print(f"  Updates detected: {num_updates} (expected ~{expected:.0f} for {freq})")
            print(f"  Ratio actual/expected: {num_updates/expected:.2f}x")
            if staleness_mode in ['all', 'days_only']:
                print(f"  Avg days stale: {avg_staleness:.1f}")
                print(f"  Max days stale: {max_staleness}")
    
    if verbose:
        added_features = [c for c in df.columns if 'days_since' in c or 'is_fresh' in c]
        print(f"\n{'='*70}")
        print(f"Total staleness features added: {len(added_features)}")
        print(f"Features: {added_features}")
        print(f"{'='*70}\n")
    
    return df


def validate_staleness_detection(df, feature_name, verbose=True):
    """
    Validate that staleness detection is working correctly for a feature.
    
    Checks:
    1. Number of updates is reasonable for the update frequency
    2. Max staleness doesn't exceed expected bounds
    3. Updates align with known release patterns (if applicable)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with staleness features already added
    feature_name : str
        Base feature name (e.g., 'Unemployment')
    verbose : bool
        Print validation results
        
    Returns:
    --------
    dict
        Validation results with 'passed', 'warnings', and 'stats' keys
    """
    if feature_name not in FEATURE_METADATA:
        return {'passed': False, 'warnings': [f'{feature_name} not in FEATURE_METADATA']}
    
    metadata = FEATURE_METADATA[feature_name]
    if not metadata.get('needs_staleness', False):
        return {'passed': True, 'warnings': ['Feature does not need staleness']}
    
    staleness_col = metadata['staleness_features'][0]
    freshness_col = metadata['staleness_features'][1]
    
    warnings = []
    stats = {}
    
    # Check columns exist
    for col in [staleness_col, freshness_col]:
        if col not in df.columns:
            warnings.append(f"Missing column: {col}")
    
    if warnings:
        return {'passed': False, 'warnings': warnings, 'stats': stats}
    
    # Compute stats
    days_stale = df[staleness_col]
    is_fresh = df[freshness_col]
    
    num_updates = is_fresh.sum()
    years = (df.index[-1] - df.index[0]).days / 365.25
    
    freq = metadata.get('update_frequency', 'monthly')
    if freq == 'monthly':
        expected_updates = years * 12
        # Max staleness can spike during govt shutdowns, unchanged values, holidays
        # Unemployment can stay flat for 4+ months; CPI delayed during shutdowns
        max_expected_staleness = 130  # ~4 months covers edge cases
    elif freq == 'irregular':  # Fed Funds - can be unchanged for years during ZIRP
        expected_updates = years * 10  # ~10 meaningful moves per year empirically
        max_expected_staleness = 300  # ZIRP periods: 2008-2015, 2020-2022
    else:
        expected_updates = years * 12
        max_expected_staleness = 130
    
    stats = {
        'num_updates': int(num_updates),
        'expected_updates': int(expected_updates),
        'ratio': num_updates / expected_updates if expected_updates > 0 else 0,
        'avg_staleness': float(days_stale.mean()),
        'max_staleness': int(days_stale.max()),
        'max_expected_staleness': max_expected_staleness,
    }
    
    # Validate
    # Note: Unemployment can stay unchanged for months (ratio ~0.76 is normal)
    if stats['ratio'] < 0.4:
        warnings.append(f"Too few updates: {num_updates} vs expected {expected_updates:.0f} (ratio={stats['ratio']:.2f})")
    elif stats['ratio'] > 2.0:
        warnings.append(f"Too many updates: {num_updates} vs expected {expected_updates:.0f} (ratio={stats['ratio']:.2f}) - threshold may be too low")
    
    if stats['max_staleness'] > max_expected_staleness:
        warnings.append(f"Max staleness {stats['max_staleness']} exceeds expected {max_expected_staleness}")
    
    passed = len(warnings) == 0
    
    if verbose:
        status = "PASSED" if passed else "✗ FAILED"
        print(f"\nValidation for {feature_name}: {status}")
        print(f"  Updates: {stats['num_updates']} (expected ~{stats['expected_updates']}, ratio={stats['ratio']:.2f})")
        print(f"  Staleness: avg={stats['avg_staleness']:.1f}, max={stats['max_staleness']}")
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")
    
    return {'passed': passed, 'warnings': warnings, 'stats': stats}



if __name__ == "__main__":
    print("Testing feature set loading and staleness detection...\n")
    
    # Test macro_heavy which has multiple low-frequency features
    print("=" * 70)
    print("Test 1: Load macro_heavy feature set")
    print("=" * 70)
    
    try:
        data = load_feature_set('macro_heavy', frequency='daily', version='vintage')
        print(f"\nLoaded {len(data)} observations")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Failed to load: {e}")
        data = None
    
    if data is not None:
        print("\n" + "=" * 70)
        print("Test 2: Add staleness features")
        print("=" * 70)
        
        data_with_staleness = add_staleness_features(data, staleness_mode='all', verbose=True)
        
        print("\n" + "=" * 70)
        print("Test 3: Validate staleness detection")
        print("=" * 70)
        
        for feature in ['Inflation_YoY', 'Unemployment', 'Fed_Rate', 'Consumer_Sentiment', 'Industrial_Production']:
            if feature in data.columns:
                validate_staleness_detection(data_with_staleness, feature, verbose=True)
