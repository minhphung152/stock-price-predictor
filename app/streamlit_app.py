import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title('Stock Price Predictor')

ticker = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(ticker, start='2020-01-01')
st.line_chart(df['Close'])