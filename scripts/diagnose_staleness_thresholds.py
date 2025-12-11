#!/usr/bin/env python3
"""
Diagnose appropriate staleness detection thresholds by analyzing actual data.
Run from project root: python diagnose_staleness_thresholds.py
"""

import pandas as pd
import numpy as np

def analyze_feature_changes(df, feature_name):
    """Analyze the distribution of day-over-day changes for a feature."""
    if feature_name not in df.columns:
        print(f"  {feature_name}: NOT IN DATASET")
        return None
    
    series = df[feature_name].dropna()
    diffs = series.diff().abs().dropna()
    
    # Filter to non-zero changes only
    nonzero_diffs = diffs[diffs > 0]
    
    if len(nonzero_diffs) == 0:
        print(f"  {feature_name}: NO CHANGES DETECTED (constant)")
        return None
    
    # Count how many "updates" we'd detect at various thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    print(f"\n  {feature_name}:")
    print(f"    Value range: [{series.min():.3f}, {series.max():.3f}]")
    print(f"    Total observations: {len(series)}")
    print(f"    Days with ANY change: {len(nonzero_diffs)} ({100*len(nonzero_diffs)/len(diffs):.1f}%)")
    print(f"    Non-zero change stats: min={nonzero_diffs.min():.4f}, median={nonzero_diffs.median():.4f}, max={nonzero_diffs.max():.4f}")
    
    print(f"    Updates detected at threshold:")
    for thresh in thresholds:
        count = (diffs > thresh).sum()
        pct = 100 * count / len(diffs)
        # Expected monthly updates ~= total_days / 21 trading days
        expected_monthly = len(diffs) / 21
        print(f"      >{thresh:6.3f}: {count:5d} updates ({pct:5.1f}%) - ratio to expected monthly: {count/expected_monthly:.2f}x")
    
    return {
        'min_change': nonzero_diffs.min(),
        'median_change': nonzero_diffs.median(),
        'max_change': nonzero_diffs.max(),
        'num_changes': len(nonzero_diffs),
    }


def main():
    print("=" * 70)
    print("Staleness Threshold Diagnostic")
    print("=" * 70)
    
    # Load daily vintage data
    df = pd.read_csv('data/financial_dataset_daily_vintage.csv', 
                     index_col='Date', parse_dates=True)
    
    print(f"\nDataset: {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")
    
    # Features that need staleness detection
    low_freq_features = [
        'CPI',              # For Inflation_YoY staleness
        'Unemployment',
        'Fed_Rate', 
        'Consumer_Sentiment',
        'Industrial_Production',
    ]
    
    print("\n" + "=" * 70)
    print("Low-Frequency Feature Analysis")
    print("=" * 70)
    
    results = {}
    for feature in low_freq_features:
        results[feature] = analyze_feature_changes(df, feature)
    
    # Also check high-freq features for comparison
    print("\n" + "=" * 70)
    print("High-Frequency Features (for comparison)")
    print("=" * 70)
    
    high_freq_features = ['VIX', 'Treasury_10Y', 'Yield_Spread']
    for feature in high_freq_features:
        analyze_feature_changes(df, feature)
    
    # Recommend thresholds
    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 70)
    print("""
Based on analysis, recommended thresholds should be set to catch genuine
monthly releases while ignoring floating-point noise. Target ~12-15 updates
per year for monthly series, ~8 for Fed_Rate.

Copy these to FEATURE_METADATA in feature_configs.py:
""")
    
    for feature, stats in results.items():
        if stats is None:
            continue
        # Recommend threshold at ~50% of minimum observed change
        # This catches all real updates while filtering noise
        recommended = stats['min_change'] * 0.5
        print(f"  '{feature}': {{'change_threshold': {recommended:.4f}}},  # min_change={stats['min_change']:.4f}")


if __name__ == "__main__":
    main()
