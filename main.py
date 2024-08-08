import data_preprocessing
import model_training
import model_evaluation
import visualization
import joblib
import pandas as pd
import pickle

# Data Preprocessing
file_path = 'Data.csv'
data = data_preprocessing.load_data(file_path)
data = data_preprocessing.clean_data(data)
data = data_preprocessing.impute_missing_values(data)
data = data_preprocessing.remove_outliers(data)
data = data_preprocessing.remove_outliers_2(data)
data = data_preprocessing.normalize_data(data)
data.to_csv('processed_data.csv', index=False)


# Model Training
model, X_test, y_test, predictions = model_training.train_random_forest(data)
# Save the model using 'with open'
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
pd.DataFrame(predictions, columns=['predictions']).to_csv('predictions.csv', index=False)

# Model Evaluation
model_evaluation.evaluate_model(y_test, predictions)
model_evaluation.plot_residuals(y_test, predictions)

# Save the model again for redundancy if needed
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
