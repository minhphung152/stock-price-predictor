import pandas as pd

def add_features(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Volatility_10d'] = df['Close'].rolling(window=10).std()
    df = df.dropna()
    return df