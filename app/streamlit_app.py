import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer_features import add_features
# from models.train_tree import train_xgboost
from models.train_lstm import reshape_for_lstm, build_lstm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title='Stock Price Predictor', layout='wide')

st.title('Stock Price Predictor')
st.markdown('Predict future stock prices using machine learning models')

# Sidebar for user inputs
st.sidebar.header('Configuration')
start_date = st.sidebar.date_input('Start Date', datetime(2020, 1, 1))
prediction_days = st.sidebar.slider('Days to Predict', 1, 30, 5)
model_choice = st.sidebar.selectbox('Select Model', ['LSTM'])

@st.cache_data
def load_data(ticker, start_date):
    try:
        end_date = datetime.now()
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error(f'No data found for ticker {ticker}')
            return None
        return df
    except Exception as e:
        st.error(f'Error loading data: {str(e)}')
        return None
    
@st.cache_data
def prepare_features(df):
    df_features = add_features(df.copy())
    return df_features

def train_lstm_model(df, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values)

    X, y = reshape_for_lstm(scaled_data.flatten(), window_size)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm((window_size, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return model, scaler, preds, y_test_actual, split_idx

def predict_future(model, scaler, last_sequence, days, model_type):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(current_sequence.reshape(1, -1, 1))[0][0]
        predictions.append(pred)
        # Update sequence
        current_sequence = np.append(current_sequence[1:], pred)
    
    if model_type == 'LSTM':
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions.flatten()

with st.container(height=700, border=True):
    ticker = st.text_input('Enter Stock Ticker', 'AAPL')

    if ticker:
        df = load_data(ticker, start_date)

        if df is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Stock Data')
                st.write(f"Data points: {len(df)}")
                latest_price = float(df['Close'].iloc[-1])
                st.write(f"Latest Price: ${latest_price:.2f}")

            with col2:
                st.subheader('Price Chart')
                st.line_chart(df['Close'], use_container_width=True)
                # fig = go.Figure()
                # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
                # fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price (USD)')
                # st.plotly_chart(fig, use_container_width=True)
            
            df_features = prepare_features(df)

            st.subheader('Model Training & Predictions')

            if st.button('Train Model & Predict'):
                with st.spinner('Training model...'):
                    try:
                        model, scaler, preds, y_test_actual, split_idx = train_lstm_model(df)

                        # Calculate metrics
                        mse = mean_squared_error(y_test_actual, preds)
                        mae = mean_absolute_error(y_test_actual, preds)

                        # Future predictions
                        last_sequence = scaler.transform(df[['Close']].tail(60).values).flatten()
                        future_pred = predict_future(model, scaler, last_sequence, prediction_days, 'LSTM')

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric('MSE', f'{mse:.2f}')
                        with col2:
                            st.metric('MAE', f'{mae:.2f}')

                        fig = go.Figure()
                        # fig.add_trace(go.Scatter(
                        #     x=df.index[:split_idx],
                        #     y=df['Close'][:split_idx],
                        #     name='Training Data',
                        #     line=dict(color='blue')
                        # ))

                        # Test data
                        test_dates = df.index[split_idx + 60:]
                        fig.add_trace(go.Scatter(
                            x=test_dates,
                            y=y_test_actual.flatten(),
                            name='Actual Test Data',
                            line=dict(color='orange')
                        ))

                        # Predictions
                        fig.add_trace(go.Scatter(
                            x=test_dates,
                            y=preds.flatten(),
                            name='Predictions',
                            line=dict(color='red', dash='dash')
                        ))

                        # Future predictions
                        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=prediction_days)
                        # st.line_chart(pd.Series(future_pred, index=future_dates))
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_pred,
                            name='Future Predictions',
                            line=dict(color='green', dash='dot')
                        ))

                        fig.update_layout(
                            title=f'{ticker} Stock Price Predictions',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True, height=600)

                        # Display future predictions
                        st.subheader('Future Predictions')
                        future_df = pd.DataFrame({
                            'Date': future_dates.strftime('%m-%d-%Y'),
                            'Predicted Price': future_pred
                        })
                        st.dataframe(future_df)
                    
                    except Exception as e:
                        st.error(f'Error training model: {str(e)}')

st.markdown('---')
st.markdown('**Disclaimer**: This is for educational purposes only. Do not use for factual trading decisions.')