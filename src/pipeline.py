import yfinance as yf
import pandas as pd 
import numpy as np 

def download_from_source(symbol, start_date, end_date):
    return yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval='1d'
    )

symbol = 'INFY.NS'
start_date= '2022-01-01'
end_date= '2023-01-01'

data  = download_from_source(symbol, start_date, end_date)
# data.to_csv('data/raw/stock_data.csv')
print(data.head())

def validate_data(df):
    print(f'Shape: {df.shape}')
    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"\nMissing Values: \n{df.isnull().sum()}")
    print(f"\nData types: \n{df.dtypes}")
    print(f"\nClose Price Stat: \n{df['Close'].describe()}")
    return df

validate_data(data)
