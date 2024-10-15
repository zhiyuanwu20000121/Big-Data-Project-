import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Custom accuracy function that returns error bounds
def custom_accuracy_with_ranges(y_true, y_pred):
    accuracy_count = 0
    upper_bounds = []
    lower_bounds = []

    for true_price, pred_price in zip(y_true, y_pred):
        error = np.abs(true_price - pred_price)
        if true_price <= 10:
            upper_bounds.append(true_price + 5)
            lower_bounds.append(true_price - 5)
            if error <= 5:
                accuracy_count += 1
        elif 10 < true_price <= 20:
            upper_bounds.append(true_price + 7)
            lower_bounds.append(true_price - 7)
            if error <= 7:
                accuracy_count += 1
        elif 20 < true_price <= 50:
            upper_bounds.append(true_price + 9)
            lower_bounds.append(true_price - 9)
            if error <= 9:
                accuracy_count += 1
        else:  # true_price > 50
            upper_bounds.append(true_price + 10)
            lower_bounds.append(true_price - 10)
            if error <= 10:
                accuracy_count += 1

    return accuracy_count / len(y_true), np.array(lower_bounds), np.array(upper_bounds)

# Load the data
file_path_csv = 'wine_updated_normalized.csv'
wine_data_with_normalized_points = pd.read_csv(file_path_csv)

# Select the price column and calculate the 95th percentile
price_quantile_95 = wine_data_with_normalized_points['price'].quantile(0.95)

# Filter data to include prices below the 95th percentile
filtered_data = wine_data_with_normalized_points[wine_data_with_normalized_points['price'] <= price_quantile_95]

# Initialize LabelEncoder for encoding categorical features
label_encoder = LabelEncoder()

# Encode categorical features
filtered_data['variety_encoded'] = label_encoder.fit_transform(filtered_data['variety'])
filtered_data['region_1_encoded'] = label_encoder.fit_transform(filtered_data['region_1'])
filtered_data['winery_encoded'] = label_encoder.fit_transform(filtered_data['winery'])

# Prepare features and target variable
X = filtered_data[['variety_encoded', 'region_1_encoded', 'winery_encoded', 'points_normalized', 'Sentiment_Score']]
y = filtered_data['price'].values

# Apply log transformation to the target variable
y_log = np.log1p(y)  # log1p ensures no log(0)

# Step 1: Split data into 80% training and 20% testing sets
X_train_full, X_test, y_train_full_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Step 2: Further split the 80% training set into 80% training and 20% validation sets
X_train, X_val, y_train_log, y_val_log = train_test_split(X_train_full, y_train_full_log, test_size=0.2, random_state=42)

# Define the Linear Regression model
linear_model = LinearRegression()

# Fit the model on the training set
linear_model.fit(X_train, y_train_log)

# Predict on the validation set
y_val_pred_log = linear_model.predict(X_val)

# Inverse log transformation of the predicted values to revert to the price scale
y_val_pred = np.expm1(y_val_pred_log)
y_val = np.expm1(y_val_log)

# Set the predicted price lower bound to 0 to avoid negative values
y_val_pred = np.maximum(y_val_pred, 0)

# Calculate accuracy within reasonable price ranges using the custom function
custom_acc_val, lower_bounds_val, upper_bounds_val = custom_accuracy_with_ranges(y_val, y_val_pred)

# Step 3: Use the trained model to predict on the second 20% test set
y_test_pred_log = linear_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test = np.expm1(y_test_log)
y_test_pred = np.maximum(y_test_pred, 0)

custom_acc_test, lower_bounds_test, upper_bounds_test = custom_accuracy_with_ranges(y_test, y_test_pred)

# Calculate standard performance metrics for the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)

# Calculate standard performance metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Plot for the validation set
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5, label="Predicted Prices")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel('True Prices (Validation)')
plt.ylabel('Predicted Prices (Validation)')
plt.title('(Linear Regression) True vs Predicted Prices (Validation Set)')
plt.legend()
plt.savefig('true_vs_predicted_prices_val.png')
plt.show()

# Plot for the second 20% test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, label="Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
plt.plot(y_test, lower_bounds_test, 'g-', lw=1, label="Lower Bound (Reasonable Price)")
plt.plot(y_test, upper_bounds_test, 'g-', lw=1, label="Upper Bound (Reasonable Price)")
plt.xlabel('True Prices (Test)')
plt.ylabel('Predicted Prices (Test)')
plt.title('(Linear Regression) True vs Predicted Prices with Reasonable Price Range (Test Set)')
plt.legend()
plt.savefig('true_vs_predicted_prices_test.png')
plt.show()

# Output performance metrics and custom accuracy
print(f"Validation Set - Mean Squared Error: {mse_val}")
print(f"Validation Set - Mean Absolute Error: {mae_val}")
print(f"Validation Set - Accuracy: {custom_acc_val}")

print(f"Test Set - Mean Squared Error: {mse_test}")
print(f"Test Set - Mean Absolute Error: {mae_test}")
print(f"Test Set - Reasonable Price Percentage: {custom_acc_test}")
