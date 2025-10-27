#!/usr/bin/env python3
"""
Development utility: Verify Inflation_YoY calculation and split consistency

This script validates that Inflation_YoY values are correctly calculated and 
propagated through the data pipeline from raw CSVs to train/val/test splits.

Used during development to debug a bug where vintage splits were incorrectly 
loading fixed-shift data due to missing --data-version parameter in create_splits.py.

Usage:
    python scripts/dev_check_cpi_yoy.py
    
Checks:
1. CPI values at time t and t-252 days (~1 year) in both fixed/vintage datasets
2. Inflation_YoY stored in CSV vs manually calculated from CPI
3. Inflation_YoY values in train splits match their source CSVs
4. Consistency across fixed-shift and vintage-aligned versions

To:
- Validating data preprocessing pipeline changes
- Verifying split creation loads correct source data
- Debugging alignment discrepancies between fixed and vintage methods

Expected:
- CPI should differ btwn fixed/vintage (alignment methods differ)
- Inflation_YoY should differ (calculated from different CPI values)
- Split values should match their respective CSV sources
"""

import pandas as pd

fixed = pd.read_csv('data/financial_dataset_daily_fixed.csv', index_col=0, parse_dates=True)
vintage = pd.read_csv('data/financial_dataset_daily_vintage.csv', index_col=0, parse_dates=True)

# Check CPI on 2005-03-10 AND 252 days before
date1 = '2005-03-10'
date2 = '2004-03-10'  # roughly 252 trading days before

print(f"CPI on {date1}:")
print(f"  Fixed: {fixed.loc[date1, 'CPI']}")
print(f"  Vintage: {vintage.loc[date1, 'CPI']}")

print(f"\nCPI on {date2}:")
print(f"  Fixed: {fixed.loc[date2, 'CPI']}")
print(f"  Vintage: {vintage.loc[date2, 'CPI']}")

print(f"\nInflation_YoY on {date1}:")
print(f"  Fixed CSV: {fixed.loc[date1, 'Inflation_YoY']}")
print(f"  Vintage CSV: {vintage.loc[date1, 'Inflation_YoY']}")

# Manually calculate what it should be
fixed_calc = (fixed.loc[date1, 'CPI'] / fixed.loc[date2, 'CPI'] - 1) * 100
vintage_calc = (vintage.loc[date1, 'CPI'] / vintage.loc[date2, 'CPI'] - 1) * 100

print(f"\nManually calculated:")
print(f"  Fixed should be: {fixed_calc}")
print(f"  Vintage should be: {vintage_calc}")

fixed_split = pd.read_csv('data/splits/fixed/core_proposal_daily_fixed_train.csv', index_col=0, parse_dates=True)
vintage_split = pd.read_csv('data/splits/vintage/core_proposal_daily_vintage_train.csv', index_col=0, parse_dates=True)

print(f"Split Inflation_YoY on 2005-03-10:")
print(f"  Fixed split: {fixed_split.loc['2005-03-10', 'Inflation_YoY']}")
print(f"  Vintage split: {vintage_split.loc['2005-03-10', 'Inflation_YoY']}")
