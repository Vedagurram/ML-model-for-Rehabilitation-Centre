import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the preprocessed dataset
file_path = 'final_synthetic_data.xlsx'
df = pd.read_excel(file_path)
print("Loaded dataset:")
print(df.head())

# Use the first three columns as input features
X = df.iloc[:, :3]
print("\nInput features (X):")
print(X.head())

# Use the remaining columns as output targets
y = df.iloc[:, 3:]
print("\nOutput targets (y):")
print(y.head())

# Split the data into training and testing sets (90-95% train, 5-10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print("\nTraining input (X_train):")
print(X_train.head())
print("\nTesting input (X_test):")
print(X_test.head())

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nNormalized training input (X_train_scaled):")
print(pd.DataFrame(X_train_scaled, columns=X.columns))
print("\nNormalized testing input (X_test_scaled):")
print(pd.DataFrame(X_test_scaled, columns=X.columns))

# Define the base models
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42)),
    ('svr', SVR())
]

# Define the stacking regressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=XGBRegressor(random_state=42)
)

# Wrap the stacking regressor with MultiOutputRegressor
multi_output_model = MultiOutputRegressor(stacking_regressor)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'estimator__rf__n_estimators': [100, 200],
    'estimator__rf__max_depth': [10, 20, None],
    'estimator__xgb__n_estimators': [100, 200],
    'estimator__xgb__max_depth': [3, 5],
    'estimator__svr__C': [0.1, 1, 10],
    'estimator__svr__kernel': ['linear', 'rbf'],
    'estimator__final_estimator__n_estimators': [100, 200],
    'estimator__final_estimator__max_depth': [3, 5]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(multi_output_model, param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)

print("Starting randomized search...")
random_search.fit(X_train_scaled, y_train)

# Best model after hyperparameter tuning
best_model = random_search.best_estimator_
print("Best model found:")
print(best_model)

# Predict on the test data
y_pred = best_model.predict(X_test_scaled)

# Rounding predictions for integer columns
integer_columns = ['Lower_Limb_Inversion', 'Upper_Limb_Adduction.1', 'Upper_Limb_Flexion', 'Treatments_SYREBO_for_functional_training', 'Treatments_TYMO_for_balance_training.']
y_pred = pd.DataFrame(y_pred, columns=y.columns)
y_pred[integer_columns] = y_pred[integer_columns].round()

print("\nPredictions on test data (y_pred):")
print(y_pred.head())

# Evaluate the model using Mean Squared Error for each output
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f'\nMean Squared Error for each output: {mse}')

# Calculate the overall accuracy using R2 score
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
average_r2_score = np.mean(r2_scores)
print(f'\nAverage R2 Score (Accuracy): {average_r2_score * 100:.2f}%')

# Create a DataFrame to compare actual vs predicted values for each output column
comparison_df = pd.DataFrame(y_test.values, columns=[f'Actual_{col}' for col in y.columns])
predicted_df = pd.DataFrame(y_pred.values, columns=[f'Predicted_{col}' for col in y.columns])
comparison_df = pd.concat([comparison_df, predicted_df], axis=1)
print("\nComparison of actual vs predicted values:")
print(comparison_df.head())

# Save the comparison results to an Excel file
comparison_df.to_excel('predicted_vs_actual.xlsx', index=False)
print(f"\nComparison results saved to 'predicted_vs_actual.xlsx'.")

# Visualize the results
for column in y.columns[:5]:  # Plotting the first 5 targets for simplicity
    plt.figure(figsize=(10, 5))
    plt.scatter(comparison_df[f'Actual_{column}'], comparison_df[f'Predicted_{column}'], alpha=0.5)
    plt.plot([comparison_df[f'Actual_{column}'].min(), comparison_df[f'Actual_{column}'].max()],
             [comparison_df[f'Actual_{column}'].min(), comparison_df[f'Actual_{column}'].max()],
             'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted for {column}')
    plt.show()
