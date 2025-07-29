from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def train_xgboost(X, y):
    model = XGBRegressor(n_estimators=100)
    tscv = TimeSeriesSplit(n_splits=5)

    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], preds))
        print(f'RMSE: {rmse}')
    
    return model