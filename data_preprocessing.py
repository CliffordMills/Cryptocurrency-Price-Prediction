import pandas as pd
import warnings
from scipy import stats
import numpy as np

warnings.filterwarnings('ignore')

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data

def clean_data(data):
    # Dropping rows with missing target variable (close)
    data.dropna(subset=['close'], inplace=True)
    # Dropping irrelevant columns
    columns_to_drop = ["id", "asset_id", "medium", "youtube"]
    data.drop(columns=columns_to_drop, inplace=True)
    return data

def impute_missing_values(data):
    for column in data.columns:
        if column != 'close':
            data[column] = data.apply(lambda row: row['close'] if pd.isna(row[column]) else row[column], axis=1)
    return data

def remove_outliers(data):
    # Calculate the IQR for each relevant column
    Q1 = data[['open', 'high', 'low', 'close', 'volume']].quantile(0.25)
    Q3 = data[['open', 'high', 'low', 'close', 'volume']].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    data_filtered = data[~((data[['open', 'high', 'low', 'close', 'volume']] < lower_bound) | (data[['open', 'high', 'low', 'close', 'volume']] > upper_bound)).any(axis=1)]
    return data_filtered

def remove_outliers_2(data):
    # Calculate z-scores for relevant columns
    z_scores = np.abs(stats.zscore(data[['open', 'high', 'low', 'close', 'volume']]))

    # Set a threshold for outlier detection (3 standard deviations)
    threshold = 3

    # Filter the DataFrame to remove outliers based on z-scores
    data_filtered_z = data[(z_scores < threshold).all(axis=1)]

    return data_filtered_z



def normalize_data(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'volume']] = scaler.fit_transform(data[['open', 'high', 'low', 'volume']])
    return data


