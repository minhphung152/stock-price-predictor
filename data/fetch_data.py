import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date='2015-01-01', end_date='2024-12-31'):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f'data/{ticker}.csv')
    return df