# Stock Price Predictor

This project predicts future stock prices using learning models like XGBoost and LSTM.

## Features
- Data fetching using Yahoo Finance
- Technical indicator feature engineering
- Tree-based and deep leaning models
- Streamlit UI for visualization

## Getting Started
```bash
git clone https://github.com/minhphung152/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
```

## Running the Streamlit app
```bash
streamlit run app/streamlit.py
```

## Example Usage

We will train a LSTM model to predict the stock price of Microsoft (MSFT):

1. **Fetch data:**
- Choose a start date.
- Enter 'MSFT' in the ticker input field and press Enter.

![Stock data and price chart](https://github.com/minhphung152/stock-price-predictor/blob/main/img/fetch-data.png?raw=true)

2. **Train the model:**
- Choose the model type (We will use LSTM in this example).
- Choose the number of days to predict (e.g., 30 days).
- Click the "Train Model & Predict" button.
- The model will train and display the predicted stock price.

3. **View predictions:**
- The predicted stock prices will be displayed in a chart.

![Stock Price Predictions](https://github.com/minhphung152/stock-price-predictor/blob/main/img/predictions.png?raw=true)

- The app will also display the model's performance metrics. The lower the metrics, the better the model's performance.

![Model Performance Metrics](https://github.com/minhphung152/stock-price-predictor/blob/main/img/metrics.png?raw=true)