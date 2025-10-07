"""
Feature set configurations for different modeling experiments.

Each config specifies:
- features: list of column names to use
- description: what this feature set represents
- min_date: earliest date with all features available (None = use all)
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
}

# Define target variable
TARGET = 'SP500_Returns'

# Define which features are lagged returns (for proper temporal splits)
AUTOREGRESSIVE_FEATURES = ['SP500_Returns']
