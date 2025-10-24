import pandas as pd
from src.data_utils import add_staleness_features

# l split that includes CPI
df = pd.read_csv('data/splits/fixed/core_proposal_daily_train.csv', 
                 index_col='Date', parse_dates=True)

print("Train split date range:", df.index[0], "to", df.index[-1])

# use dates from 2014-2015 (end of train split)
df_sample = df.loc['2014-01-01':'2015-05-08']

print("df_sample shape:", df_sample.shape)
print("Columns:", df_sample.columns.tolist())

# add staleness
df_with_staleness = add_staleness_features(df_sample, use_vintage=False, verbose=True)

print(df_with_staleness[['Inflation_YoY', 'CPI', 'days_since_CPI_update', 'CPI_is_fresh']].head(20))
