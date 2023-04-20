import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

# 1) IMPORT DATASET
# DATASET : prices open and close of the hotel sector
#           from 2014 until 31.01.2022
df = pd.read_csv('hotelSector copy.csv')

# 2) RENAME date column
df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

# 3) Use column date as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index()
df['Day'] = df.index.weekday

#0) Create a dataframe with Close Prices and Day of the week
df2 = df.iloc[:, 5:12]

#1) Compute the daily/weekly simple and daily/weekly log returns
### daily returns
simple_returns_daily = df2.iloc[:,:-1].pct_change()
log_returns_daily = np.log(df2.iloc[:,:-1]/df2.iloc[:,:-1].shift(1))

### weekly returns
weekly_price = df2.resample('W-Mon', label='left', closed = 'left').ffill()
weekly_price = weekly_price.iloc[:,:-1]
weekly_price = weekly_price.dropna(how='all')

simple_returns_weekly = weekly_price.pct_change()
log_returns_weekly = np.log(weekly_price/weekly_price.shift(1))

#2) Compute for each of the 4 cases the mean,var,skew,kurto,min,max

grouped = df.groupby(df.index.year)
grouped.size()