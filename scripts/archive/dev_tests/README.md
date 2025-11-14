# Development Test Scripts

## dev_check_cpi_yoy.py
One-time validation script used to debug a bug where vintage splits were 
incorrectly loading fixed-shift data due to missing --data-version parameter 
in create_splits.py.

Checked specific date (2005-03-10) and t-252 to validate:
- CPI values differed between fixed/vintage (expected)
- Inflation_YoY calculations were correct
- Split CSVs matched their source datasets

Bug was fixed by adding --data-version to create_splits.py

Kept for reference in case similar alignment issues arise.

TODO: update with additional dev_test scripts
