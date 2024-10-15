import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import category_encoders as ce
import matplotlib.pyplot as plt

# Load your dataset
print("loading data")
df = pd.read_csv('wine_updated_normalized.csv')  # Replace 'your_file_path.csv' with the actual path to your dataset

print("data processing")

# Drop rows with missing values in relevant columns
df_filtered = df[['country', 'designation', 'points_normalized', 'province', 'region_1', 'Year', 'variety', 'Sentiment_Score', 'price']].dropna()

# Drop rare designation
print("droping rare designation")
designation_counts = df_filtered['designation'].value_counts()
rare_designations = designation_counts[designation_counts < 3].index
df_filtered.loc[df_filtered['designation'].isin(rare_designations), 'designation'] = 'Other'  # Use .loc[] to avoid SettingWithCopyWarning

# Apply target encoding to both 'designation' and 'variety'
encoder = ce.TargetEncoder(cols=['designation', 'variety'])
df_filtered[['designation_encoded', 'variety_encoded']] = encoder.fit_transform(df_filtered[['designation', 'variety']], df_filtered['price'])

# Check the number of unique designations after replacing rare categories
print(f"Number of unique designations: {df_filtered['designation'].nunique()}")

# Drop the original 'designation' and 'variety' columns to avoid issues during training
df_filtered = df_filtered.drop(columns=['designation', 'variety'])

# Now perform one-hot encoding on the remaining categorical columns
df_encoded = pd.get_dummies(df_filtered, columns=['country', 'province', 'region_1'])

print("encoding done")

# Check that no string (object) columns remain
print(df_encoded.dtypes[df_encoded.dtypes == 'object'])  # Should print an empty series if no strings remain

# Define the independent variables (factors) and dependent variable (price)
X = df_encoded.drop(columns=['price'])  # All features except 'price'
y = df_encoded['price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Number of features after encoding: {X_train.shape[1]}")

# Initialize and train the Random Forest model
print("Training RF")
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# print("RF predicting")
# # Predict using the Random Forest model
# y_rf_pred = rf_model.predict(X_test)

# print("RF calculating score")
# # Calculate R-squared and Mean Squared Error for Random Forest
# rf_r2 = r2_score(y_test, y_rf_pred)
# rf_mse = mean_squared_error(y_test, y_rf_pred)

# # Print results
# print(f"Random Forest R-squared: {rf_r2}")
# print(f"Random Forest Mean Squared Error: {rf_mse}")

# ---------------- Feature Importance Grouped by Original Features ----------------
print("Extracting feature importance")

# Get feature importances from the trained Random Forest model
importances = rf_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Group by the original feature categories
print("grouping feature importance")
feature_groups = ['country', 'province', 'region_1', 'points_normalized', 'Year', 'Sentiment_Score', 'designation_encoded', 'variety_encoded']

grouped_importance = feature_importance_df.groupby(
    feature_importance_df['Feature'].apply(lambda x: next((g for g in feature_groups if g in x), x))
)['Importance'].sum().sort_values(ascending=False)

# Display the grouped importances
print("Grouped Feature Importances:")
print(grouped_importance)

# Visualize the grouped feature importances
print("visualizing")
plt.figure(figsize=(10, 6))
plt.barh(grouped_importance.index, grouped_importance.values)
plt.gca().invert_yaxis()
plt.xlabel('Importance Score')
plt.title('Feature Importances')
plt.show()