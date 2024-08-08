import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R^2 Score:', r2)

    return mae, mse, rmse, r2

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.title('Residual Plot')
    plt.xlabel('Actual Close Price')
    plt.ylabel('Residuals')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

