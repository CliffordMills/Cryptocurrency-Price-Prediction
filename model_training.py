import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_random_forest(data):
    X = data.drop('close', axis=1)
    y = data['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    rf_model = RandomForestRegressor(random_state=0)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print('Random Forest MAE:', mae)
    return rf_model, X_test, y_test, predictions

