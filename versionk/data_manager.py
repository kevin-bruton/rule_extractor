import pandas as pd
import os
from enum import Enum

class Timeframe(Enum):
    M1 = '1min'
    M5 = '5min'
    M15 = '15min'
    M30 = '30min'
    H1 = '60min'
    H4 = '4H'
    D1 = 'D'
    W1 = 'W'

def load_data(filename):
  if os.path.exists(f'./data/parquet/{filename}.parquet'):
    min = pd.read_parquet(f'./data/parquet/{filename}.parquet')
  else:
    min = pd.read_csv(f'./data/{filename}.csv')
    min['Datetime'] = pd.to_datetime(min['Date'] + ' ' + min['Time'])
    min['Volume'] = min['Up'] + min['Down']
    min.drop(columns=['Date', 'Time', 'Up', 'Down'], inplace=True)
    min.set_index('Datetime', inplace=True)
    #min.sort_index(ascending=False, inplace=True)
    min.to_parquet(f'./data/parquet/{filename}.parquet')
  print('Data loaded.')
  return min

def minute_to_timeframe(df_1min: pd.DataFrame, timeframe: Timeframe):
  df_tf = df_1min.resample(timeframe.value).agg({
    "Open": "first", "High": "max", "Low": "min", "Close": "last"
  }).dropna()
  return df_tf