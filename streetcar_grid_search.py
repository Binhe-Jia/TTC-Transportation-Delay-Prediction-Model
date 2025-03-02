import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_excel('C:/Users/Binhe Jia/Desktop/UofT 2021/2025 Winter/SDSS Competition/ttc-streetcar-delay-data-2024.xlsx')

# Define features (X) and target (y)
X = data.drop(columns=['Min Delay', 'Vehicle'])  # Drop target and unnecessary columns
y = data['Min Delay']

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)

# Convert 'Date' to Unix timestamp if needed
if 'Date' in X.columns:
    X['Date'] = pd.to_datetime(X['Date'], errors='coerce').astype(int) / 10**9

# Fill missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],  # Learning rate
    'reg_lambda': [0, 0.1, 1, 10, 20]  # L2 regularization (lambda)
}

# Initialize model
xgb = XGBRegressor(n_estimators=150, random_state=42)

# Perform Grid Search with 5-fold CV
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters
best_learning_rate = grid_search.best_params_['learning_rate']
best_lambda = grid_search.best_params_['reg_lambda']

print(f"Best Learning Rate: {best_learning_rate}")
print(f"Best L2 Regularization (lambda): {best_lambda}")

# Train model with best parameters
best_xgb = XGBRegressor(n_estimators=150, learning_rate=best_learning_rate, reg_lambda=best_lambda, random_state=42)
best_xgb.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Best Model RMSE: {rmse}")

# Extract results for plotting
results = pd.DataFrame(grid_search.cv_results_)
learning_rates = results['param_learning_rate'].astype(float)
lambdas = results['param_reg_lambda'].astype(float)
scores = -results['mean_test_score']  # Convert negative RMSE to positive

# Create 3D Surface Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(learning_rates, lambdas, scores, c=scores, cmap='viridis')

ax.set_xlabel('Learning Rate')
ax.set_ylabel('L2 Regularization (Lambda)')
ax.set_zlabel('RMSE')
ax.set_title('Hyperparameter Tuning for XGBoost (Streetcar)')

plt.savefig('xgboost_hyperparameter_tuning.png')
plt.show()
