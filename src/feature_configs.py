"""
Feature set configurations for different modeling experiments.

Each config specifies:
- features: list of column names to use
- description: what this feature set represents
- min_date: earliest date with all features available (None = use all)
- include_constituents: (optional) whether to join constituent returns for multi-task
- constituent_count: (optional) how many constituents to include (25, 50, 75, 100)
"""

FEATURE_SETS = {
    'core_proposal': {
        'features': [
            'SP500_Returns',      # Target (will be lagged for prediction)
            'VIX',                # Market volatility
            'Treasury_10Y',       # Long-term rate
            'Yield_Spread',       # Yield curve slope (10Y-2Y)
            'Inflation_YoY',      # CPI year-over-year
        ],
        'description': 'Original 5 features from proposal',
        'min_date': None,  # Available since 1990
    },
    
    'core_plus_credit': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
            'Credit_HY',          # High yield spread
            'Credit_IG',          # Investment grade spread
        ],
        'description': 'Core + credit risk indicators',
        'min_date': '1997-01-01',  # Credit spreads start 1997
    },
    'core_dynamics': {
        'features': [
            'SP500_Returns', 'VIX', 'Treasury_10Y', 'Yield_Spread', 'Inflation_YoY',
            'VIX_relative', 'VIX_spike', 'Treasury_10Y_change', 'CPI',
        ],
        'enhanced': True,
        'description': 'Core features + regime dynamics (VIX spikes, rate changes)',
        'min_date': None,
    },
    'macro_heavy': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Inflation_YoY',
            'Unemployment',
            'Fed_Rate',
            'Consumer_Sentiment',
            'Industrial_Production',
        ],
        'description': 'Emphasis on macroeconomic fundamentals',
        'min_date': None,
    },
    
    'market_only': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'SP500_Volatility',
        ],
        'description': 'Pure market-based features, no macro releases',
        'min_date': None,
    },
    
    'kitchen_sink': {
        'features': 'all',  # Special case: use everything
        'description': 'All available features',
        'min_date': '1997-01-01',  # Most restrictive constraint
    },
    
    # =========================================================================
    # Multi-task feature sets (include constituent returns as auxiliary targets)
    # =========================================================================
    
    'multitask_core': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
        ],
        'include_constituents': True,
        'constituent_count': 50,  # Top 50 by market cap
        'constituent_version': 'vintage_2005',  # Which constituent file to use
        'description': 'Core features + top 50 constituent returns for multi-task learning',
        'min_date': None,
    },
    
    # =========================================================================
    # Single-stock prediction (transfer learning targets)
    # =========================================================================
    
    # as an example
    'predict_AAPL': {
        'features': [
            'AAPL_Returns',  # Target
            'SP500_Returns', # Index as feature
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
        ],
        'target': 'AAPL_Returns',  # Override default SP500_Returns
        'include_constituents': False,
        'description': 'Predict AAPL using macro features + index',
        'min_date': '2005-01-01',  # AAPL data reliable from here
    },
}

# Define default target variable (can be overridden per feature set)
TARGET = 'SP500_Returns'

# Define which features are lagged returns (for proper temporal splits)
AUTOREGRESSIVE_FEATURES = ['SP500_Returns']


def get_target(config_name: str) -> str:
    """
    Get target variable for a feature set.
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
        
    Returns:
    --------
    str
        Target column name (default: SP500_Returns, or override if specified)
    """
    config = FEATURE_SETS.get(config_name, {})
    return config.get('target', TARGET)

# Constituent configuration
CONSTITUENT_CONFIG = {
    # Ordered by market cap (highest first)
    # Used to select top-N for multi-task learning
    'tickers': [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'BRK-B', 'TSLA', 'UNH',
        'XOM', 'LLY', 'JPM', 'JNJ', 'V', 'AVGO', 'PG', 'MA', 'HD', 'CVX',
        'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'ADBE', 'WMT', 'MCD', 'CRM', 'TMO',
        'CSCO', 'BAC', 'ACN', 'LIN', 'NFLX', 'AMD', 'ABT', 'NKE', 'DIS', 'TXN',
        'PM', 'WFC', 'ORCL', 'DHR', 'CMCSA', 'INTU', 'VZ', 'COP', 'NEE', 'QCOM',
        'UNP', 'IBM', 'AMGN', 'RTX', 'PFE', 'LOW', 'SPGI', 'HON', 'CAT', 'ELV',
        'UPS', 'MS', 'BA', 'AMAT', 'GE', 'BLK', 'PLD', 'DE', 'SYK', 'LMT',
        'BKNG', 'MDT', 'ADP', 'ADI', 'TJX', 'GILD', 'MDLZ', 'C', 'VRTX', 'MMC',
        'SBUX', 'AMT', 'AXP', 'ISRG', 'REGN', 'CI', 'PGR', 'MO', 'ZTS', 'BDX',
        'SCHW', 'CB', 'ETN', 'BMY', 'SO', 'DUK', 'CVS', 'LRCX', 'NOC', 'BSX'
    ],
    'column_suffix': '_Returns',  # e.g., AAPL_Returns
}


def get_constituent_columns(count: int = 50) -> list:
    """
    Get column names for top-N constituents.
    
    Parameters:
    -----------
    count : int
        Number of top constituents (by market cap) to include.
        
    Returns:
    --------
    list
        Column names like ['AAPL_Returns', 'MSFT_Returns', ...]
    """
    tickers = CONSTITUENT_CONFIG['tickers'][:count]
    suffix = CONSTITUENT_CONFIG['column_suffix']
    return [f"{ticker}{suffix}" for ticker in tickers]


def get_all_targets(config_name: str) -> dict:
    """
    Get all target columns for a feature set (primary + auxiliary).
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
        
    Returns:
    --------
    dict with keys:
        'primary': str - main target (SP500_Returns)
        'auxiliary': list - constituent return columns (if multi-task)
        'all': list - all targets combined
    """
    config = FEATURE_SETS.get(config_name, {})
    
    primary = TARGET
    auxiliary = []
    
    if config.get('include_constituents', False):
        count = config.get('constituent_count', 50)
        auxiliary = get_constituent_columns(count)
    
    return {
        'primary': primary,
        'auxiliary': auxiliary,
        'all': [primary] + auxiliary,
    }


# Default threshold for change detection (used if not specified per-feature)
DEFAULT_CHANGE_THRESHOLD = 1e-6

# To create staleness features
FEATURE_METADATA = {
    # =========================================================================
    # High-frequency features (daily updates, no staleness needed)
    # =========================================================================
    'VIX': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'Treasury_10Y': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'Yield_Spread': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'SP500_Returns': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'SP500_Volatility': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    
    # =========================================================================
    # Low-frequency features (monthly updates with lag, need staleness)
    # =========================================================================
    # 
    # change_threshold: minimum absolute change to count as a "new release"
    #   - Set to ~50% of smallest expected real change to filter float noise
    #   - Run diagnose_staleness_thresholds.py to calibrate empirically
    #
    # typical_lag_days: days between reference period end and release date
    #   - CPI: ~14 days (mid-month release for prior month)
    #   - Unemployment: ~7 days (first Friday for prior month)
    #   - Fed_Rate: 0 days (immediate after FOMC)
    #   - Consumer_Sentiment: 0 days (preliminary mid-month, final end-month)
    #   - Industrial_Production: ~14 days (mid-month for prior month)
    # =========================================================================
    
    'Inflation_YoY': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 14,
        'staleness_features': ['days_since_CPI_update', 'CPI_is_fresh'],
        'source_column': 'CPI',  # Detect from CPI since Inflation_YoY is derived
        'change_threshold': 0.002,  # Empirical: min_change=0.004, catches all 431 updates
    },
    'Unemployment': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 7,
        'staleness_features': ['days_since_unemployment_update', 'unemployment_is_fresh'],
        'source_column': 'Unemployment',
        'change_threshold': 0.05,  # Empirical: min_change=0.1, catches all 325 updates (rate sometimes unchanged)
    },
    'Fed_Rate': {
        'update_frequency': 'irregular',  # FOMC meetings ~8x per year, but effective rate fluctuates daily
        'needs_staleness': True,
        'typical_lag_days': 0,
        'staleness_features': ['days_since_fed_update', 'fed_is_fresh'],
        'source_column': 'Fed_Rate',
        'change_threshold': 0.005,  # Empirical: min_change=0.01, catches 312 updates (effective rate moves)
    },
    'Consumer_Sentiment': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 0,
        'staleness_features': ['days_since_sentiment_update', 'sentiment_is_fresh'],
        'source_column': 'Consumer_Sentiment',
        'change_threshold': 0.05,  # Empirical: min_change=0.1, catches all 423 updates
    },
    'Industrial_Production': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 14,
        'staleness_features': ['days_since_indprod_update', 'indprod_is_fresh'],
        'source_column': 'Industrial_Production',
        'change_threshold': 0.002,  # Empirical: min_change=0.0045, catches all 451 updates (incl. revisions)
    },
}


def get_staleness_features(feature_list, staleness_mode='all'):
    """
    Given a list of features, return staleness features that should be added.

    Parameters:
    -----------
    feature_list : list
        List of feature names (e.g., ['VIX', 'Inflation_YoY', ...])
    staleness_mode : str
        Which staleness features to include:
        - 'all': both days_since_* and *_is_fresh (default)
        - 'days_only': only days_since_* (continuous counter)
        - 'fresh_only': only *_is_fresh (sparse binary flag)

    Returns:
    --------
    dict with keys:
        'staleness_features': list of staleness feature names to add
        'needs_staleness': dict mapping original features to bool
    """
    valid_modes = ['all', 'days_only', 'fresh_only']
    if staleness_mode not in valid_modes:
        raise ValueError(f"Invalid staleness_mode: {staleness_mode}. Must be one of {valid_modes}")
    
    staleness_features = []
    needs_staleness = {}

    for feature in feature_list:
        if feature in FEATURE_METADATA:
            metadata = FEATURE_METADATA[feature]
            needs_staleness[feature] = metadata['needs_staleness']

            if metadata['needs_staleness']:
                # staleness_features is [days_since_*, *_is_fresh] by convention
                feature_staleness = metadata['staleness_features']
                if staleness_mode == 'all':
                    staleness_features.extend(feature_staleness)
                elif staleness_mode == 'days_only':
                    # First element is days_since_*
                    staleness_features.append(feature_staleness[0])
                elif staleness_mode == 'fresh_only':
                    # Second element is *_is_fresh
                    staleness_features.append(feature_staleness[1])

    return {
        'staleness_features': staleness_features,
        'needs_staleness': needs_staleness,
    }


def get_change_threshold(feature_name: str) -> float:
    """
    Get the change detection threshold for a feature.
    
    Parameters:
    -----------
    feature_name : str
        Feature name to look up
        
    Returns:
    --------
    float
        Threshold for detecting meaningful changes (absolute value)
    """
    if feature_name in FEATURE_METADATA:
        return FEATURE_METADATA[feature_name].get('change_threshold', DEFAULT_CHANGE_THRESHOLD)
    return DEFAULT_CHANGE_THRESHOLD
