from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

# Load your data
data = pd.read_csv('bus-data.csv')

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Min Gap'], data['Min Delay'], alpha=0.5, color='blue')
plt.xlabel('Min Gap (minutes)')
plt.ylabel('Min Delay (minutes)')
plt.title('Min Delay vs. Min Gap')
plt.grid(True)

# Save and display the plot
plt.savefig('min_delay_vs_min_gap.png')
plt.show()

# Define the target column
target_column = 'Min Delay'

# Drop non-useful columns
X = data.drop(columns=[target_column, 'Vehicle'])  # Drop 'Vehicle' since it's not useful
y = data[target_column]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Encode categorical features
label_encoder = LabelEncoder()
for column in categorical_columns:
    X[column] = X[column].astype(str)  # Ensure it's string before encoding
    X[column] = label_encoder.fit_transform(X[column])

# Convert 'Min Gap' to numeric (fixes dtype issue)
X['Min Gap'] = pd.to_numeric(X['Min Gap'], errors='coerce')

# Convert 'Date' to Unix timestamp
X['Date'] = pd.to_datetime(X['Date'], errors='coerce').astype(int) / 10**9

# Fill any remaining NaN values with 0
X = X.fillna(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost with regularization to reduce 'Min Gap' dominance
xgb_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    random_state=42,
    reg_lambda=1,      # L2 regularization (shrinks large weights)
    reg_alpha=0,        # L1 regularization (sparsity)
    colsample_bytree=0.7  # Forces the model to use 70% of features per tree
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
score = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model square rooted MSE: {score}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted (XGBoost Regression)')
plt.legend()
plt.savefig('Bus_XGBoost_Actual_vs_Predicted.png')

# Feature importance analysis
importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
# Sort feature importance values in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='red')
plt.ylabel('Importance')
plt.title('Bus Feature Importances from XGBoost Model')

# Rotate x-axis labels diagonally
plt.xticks(rotation=45, ha='right')  # Rotate by 45 degrees and align text to the right

plt.tight_layout()
plt.savefig('Bus_XGBoost_Importance.png')
plt.show()

print(data[['Min Gap', 'Min Delay']].corr())

# ==========================
# TESTING BLOCK (User Input)
# ==========================

def convert_time_to_minutes(time_value):
    """Convert 'hh:mm' format to minutes since midnight."""
    if isinstance(time_value, (int, float)):
        return time_value  # Already numeric
    try:
        hh, mm = map(int, str(time_value).split(':'))
        return hh * 60 + mm
    except ValueError:
        return 0  # Default for invalid times

def preprocess_user_input(user_input, label_encoder, categorical_columns, X_train):
    """Preprocess user input to match model training format."""
    user_input_df = pd.DataFrame([user_input])

    # Convert categorical features
    for column in categorical_columns:
        if column in user_input_df:
            user_input_df[column] = user_input_df[column].astype(str)
            known_categories = set(label_encoder.classes_)
            user_input_df[column] = user_input_df[column].apply(
                lambda x: label_encoder.transform([x])[0] if x in known_categories else -1
            )  # Handle unknown values

    # Convert 'Min Gap' to numeric
    if 'Min Gap' in user_input_df.columns:
        user_input_df['Min Gap'] = pd.to_numeric(user_input_df['Min Gap'], errors='coerce')

    # Convert 'Date' to Unix timestamp
    if 'Date' in user_input_df.columns:
        user_input_df['Date'] = pd.to_datetime(user_input_df['Date'], errors='coerce').astype(int) / 10**9

    # Convert 'Time' to minutes since midnight
    if 'Time' in user_input_df.columns:
        user_input_df['Time'] = user_input_df['Time'].apply(convert_time_to_minutes)

    # Ensure the input matches the training feature order
    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

    return user_input_df

# Sample user input
user_input = {
    'Date': ['06-Jan-24'],
    'Route': ['89'],
    'Time': ['18:00'],
    'Day': ['Sunday'],
    'Incident': ['Security'],
    'Min Gap': '3',
    'Direction': ['']
}

# Process user input
user_input_df = preprocess_user_input(user_input, label_encoder, categorical_columns, X_train)

# Make a prediction
predicted_delay = xgb_model.predict(user_input_df)
print(f"Predicted Delay: {predicted_delay[0]:.2f} minutes")
