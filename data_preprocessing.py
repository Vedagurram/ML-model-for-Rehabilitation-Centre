import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer

# Load the dataset with the first row as header
file_path = 'kunal_left_hemipherisis.xlsx'
data = pd.read_excel(file_path)

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

#print(data)
print(data.columns.tolist())

# Reference data for regression
reference_data_tug = {
    'Days_of_Treatment': [0, 15, 32, 48, 68],
    'Tug_Test': [73, 58.34, 39.4, 38, 38]
}
reference_df_tug = pd.DataFrame(reference_data_tug)

reference_data_10m = {
    'Days_of_Treatment': [0, 2, 15, 32, 48, 68],
    '10M_Test': [65, 65, 65, 57.2, 38, 38]
}
reference_df_10m = pd.DataFrame(reference_data_10m)

# Fill missing values for Tug_Test using linear regression
def fill_missing_values_linear(df, reference_df, target_column):
    missing = df[target_column].isnull()
    X_train = reference_df[['Days_of_Treatment']]
    y_train = reference_df[target_column]
    X_pred = df.loc[missing, ['Days_of_Treatment']]
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted_values = regressor.predict(X_pred)
    df.loc[missing, target_column] = predicted_values
    return df

data = fill_missing_values_linear(data, reference_df_tug, 'Tug_Test')
data = fill_missing_values_linear(data, reference_df_10m, '10M_Test')

Tug_test_values = data['Tug_Test']
print(Tug_test_values)
M10_test_values = data['10M_Test']
print(M10_test_values)

# Reference data for KNN
reference_data_ml = {
    'Days_of_Treatment': [0, 2, 15, 32, 48, 68],
    'Medio_Lateral_Length_M3': [4, 4, 4, 8, 6, 6],
    'Medio_Lateral_Length_M4': [5, 9, 6, 8, 7, 5],
    'Anterior_Posterior_M3': [3, 4, 4, 4, 2, 3],
    'Anterior_Posterior_M4': [5, 9, 5, 3, 5, 4]
}
reference_df_ml = pd.DataFrame(reference_data_ml)

# Additional features for prediction
additional_features = ['Days_of_Treatment', 'Tug_Test', '10M_Test']

# Ensure the additional features exist in the data
missing_features = [feature for feature in additional_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in the data: {missing_features}")

# Target columns to impute
target_columns = ['Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4', 'Anterior_Posterior_M3', 'Anterior_Posterior_M4']

for column in target_columns:
    # Merge reference data with actual data for training
    reference_df_full_ml = pd.merge(reference_df_ml, data[additional_features + [column]], on='Days_of_Treatment', how='left', suffixes=('', '_data'))

    # Drop rows with missing values in the training data
    training_data_ml = reference_df_full_ml.dropna(subset=additional_features + [column])

    if training_data_ml.empty:
        print(f"No training data available for column: {column}")
        continue

    X_train_ml = training_data_ml[additional_features]
    y_train_ml = training_data_ml[column]

    # Extract rows where the current column is missing
    missing = data[column].isnull()
    X_pred_ml = data.loc[missing, additional_features]

    # Standardize the features
    scaler_ml = StandardScaler()
    X_train_scaled_ml = scaler_ml.fit_transform(X_train_ml)
    X_pred_scaled_ml = scaler_ml.transform(X_pred_ml)

    # Train the k-NN classifier
    knn_ml = KNeighborsClassifier(n_neighbors=3)
    knn_ml.fit(X_train_scaled_ml, y_train_ml)

    # Predict missing values
    predicted_values = knn_ml.predict(X_pred_scaled_ml)

    # Fill missing values in the original DataFrame
    data.loc[missing, column] = predicted_values

MedioLateral_M3_test_values = data['Medio_Lateral_Length_M3']
print(MedioLateral_M3_test_values)
MedioLateral_M4_test_values = data['Medio_Lateral_Length_M4']
print(MedioLateral_M4_test_values)
Anterior_Posterior_M3_test_values = data['Anterior_Posterior_M3']
print(Anterior_Posterior_M3_test_values)
Anterior_Posterior_M4_test_values = data['Anterior_Posterior_M4']
print(Anterior_Posterior_M4_test_values)

# Reference data for BBS_Score and BBG
reference_data_bbs = {
    'Days_of_Treatment': [0, 2, 15, 32, 48],
    'BBS_Score': [21, 46, 33, 34, 39]
}
reference_df_bbs = pd.DataFrame(reference_data_bbs)

reference_data_bbg = {
    'Days_of_Treatment': [0, 2, 15, 32, 48],
    'BBG': [21, 21, 33, 34, 39]
}
reference_df_bbg = pd.DataFrame(reference_data_bbg)

# Additional features for prediction
additional_features_bbs = ['Days_of_Treatment', 'Tug_Test', '10M_Test', 'Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4', 'Anterior_Posterior_M3', 'Anterior_Posterior_M4']

# Function to fill missing values using KNN classifier
def fill_missing_values_knn(df, reference_df, target_column, additional_features):
    # Merge reference data with actual data for training
    reference_df_full = pd.merge(reference_df, df[additional_features + [target_column]], on='Days_of_Treatment', how='left', suffixes=('', '_data'))
    
    # Drop rows with missing values in the training data
    training_data = reference_df_full.dropna(subset=additional_features + [target_column])
    
    if training_data.empty:
        print(f"No training data available for column: {target_column}")
        return df

    X_train = training_data[additional_features]
    y_train = training_data[target_column]

    # Extract rows where the current column is missing
    missing = df[target_column].isnull()
    X_pred = df.loc[missing, additional_features]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Train the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)

    # Predict missing values
    predicted_values = knn.predict(X_pred_scaled)

    # Fill missing values in the original DataFrame
    df.loc[missing, target_column] = predicted_values
    
    return df

# Fill missing values for BBS_Score and BBG
data = fill_missing_values_knn(data, reference_df_bbs, 'BBS_Score', additional_features_bbs)
data = fill_missing_values_knn(data, reference_df_bbg, 'BBG', additional_features_bbs)

BBS_Score_values = data['BBS_Score']
print(BBS_Score_values)
BBG_values = data['BBG']
print(BBG_values)

# Function to fill missing values using KNN regressor
def fill_missing_values_knn_regressor(df, reference_df, target_column, additional_features):
    # Check if all additional features exist in the DataFrame
    missing_features = [feature for feature in additional_features if feature not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in the data: {missing_features}")

    # Merge reference data with actual data for training
    reference_df_full = pd.merge(reference_df, df[additional_features + [target_column]], on='Days_of_Treatment', how='left', suffixes=('', '_data'))
    
    # Drop rows with missing values in the training data
    training_data = reference_df_full.dropna(subset=additional_features + [target_column])
    
    if training_data.empty:
        print(f"No training data available for column: {target_column}")
        return df

    X_train = training_data[additional_features]
    y_train = training_data[target_column]

    # Extract rows where the current column is missing
    missing = df[target_column].isnull()
    X_pred = df.loc[missing, additional_features]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Train the k-NN regressor
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)

    # Predict missing values
    predicted_values = knn.predict(X_pred_scaled)

    # Fill missing values in the original DataFrame
    df.loc[missing, target_column] = predicted_values
    
    return df

# Ensure the required columns exist in the data
required_columns = [
    'Days_of_Treatment', 'Tug_Test', '10M_Test',
    'Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4',
    'Anterior_Posterior_M3', 'Anterior_Posterior_M4',
    'BBS_Score', 'BBG'
]

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the data: {missing_columns}")

# Reference data for columns O to U
reference_data_velocity = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Velocity': [0.6, 0.7, 0.9, 0.6]
}
reference_data_cadence = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Cadence': [47.8, 50, 62.6, 46.4]
}
reference_data_stride = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Stride_Length': [45, 49, 47, 45]
}
reference_data_stance_rt = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Stance_Phase_RT': [78.2, 76, 86.7, 76.3]
}
reference_data_stance_lt = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Stance_Phase_LT': [78, 78, 86.9, 83.6]
}
reference_data_swing_rt = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Swing_Phase_RT': [21.8, 26, 13.3, 23.7]
}
reference_data_swing_lt = {
    'Days_of_Treatment': [0, 15, 48, 72],
    'Swing_Phase_LT': [22, 25, 13.1, 16.4]
}

# Convert reference data to DataFrames
reference_df_velocity = pd.DataFrame(reference_data_velocity)
reference_df_cadence = pd.DataFrame(reference_data_cadence)
reference_df_stride = pd.DataFrame(reference_data_stride)
reference_df_stance_rt = pd.DataFrame(reference_data_stance_rt)
reference_df_stance_lt = pd.DataFrame(reference_data_stance_lt)
reference_df_swing_rt = pd.DataFrame(reference_data_swing_rt)
reference_df_swing_lt = pd.DataFrame(reference_data_swing_lt)

# List of reference DataFrames and corresponding columns
reference_data_list = [
    (reference_df_velocity, 'Velocity'),
    (reference_df_cadence, 'Cadence'),
    (reference_df_stride, 'Stride_Length'),
    (reference_df_stance_rt, 'Stance_Phase_RT'),
    (reference_df_stance_lt, 'Stance_Phase_LT'),
    (reference_df_swing_rt, 'Swing_Phase_RT'),
    (reference_df_swing_lt, 'Swing_Phase_LT')
]

# Additional features for prediction for columns O to U
additional_features_OU = ['Days_of_Treatment', 'Tug_Test', '10M_Test', 'Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4', 'Anterior_Posterior_M3', 'Anterior_Posterior_M4', 'BBS_Score', 'BBG']

# Fill missing values for each target column using the corresponding reference data
for reference_df, target_column in reference_data_list:
    if target_column in data.columns:
        data = fill_missing_values_knn_regressor(data, reference_df, target_column, additional_features_OU)
    else:
        print(f"Column '{target_column}' is not present in the data.")

velocity_values = data['Velocity']
print(velocity_values)
cadence_values = data['Cadence']
print(cadence_values)
stride_length_values = data['Stride_Length']
print(stride_length_values)
stance_phase_RT_values = data['Stance_Phase_RT']
print(stance_phase_RT_values)
stance_phase_LT_values = data['Stance_Phase_LT']
print(stance_phase_LT_values)
swing_phase_RT_values = data['Swing_Phase_RT']
print(swing_phase_RT_values)
swing_phase_LT_values = data['Swing_Phase_LT']
print(swing_phase_LT_values)

# Reference data for categorical columns
reference_data_categorical = {
    'Days_of_Treatment': [0, 15, 32, 48],
    'Pectoral': [1, 0, 0, 0],
    'Elbow_Flexor': [1, 1, 1, 1],
    'Wrist_Flexor': [1, 1, 1, 1],
    'Finger_Flexor': [2, 2, 1, 1],
    'Knee_Flexor': [1, 1, 1, 1],
    'Plantar_Flexor': [1, 1, 1, 1]
}
reference_df_categorical = pd.DataFrame(reference_data_categorical)

# Ensure the additional features exist in the data
additional_features = ['Days_of_Treatment', 'Tug_Test', '10M_Test', 
                       'Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4', 
                       'Anterior_Posterior_M3', 'Anterior_Posterior_M4', 
                       'BBS_Score', 'BBG']

missing_features = [feature for feature in additional_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in the data: {missing_features}")

# Target columns to impute
target_columns_categorical = ['Pectoral', 'Elbow_Flexor', 'Wrist_Flexor', 'Finger_Flexor', 'Knee_Flexor', 'Plantar_Flexor']

# Function to fill missing values using KNN classifier and One Hot Encoder
def fill_missing_values_knn_with_ohe(df, reference_df, target_columns, additional_features):
    for target_column in target_columns:
        if target_column not in df.columns:
            print(f"Column '{target_column}' is not present in the data. Creating it with NaN values.")
            df[target_column] = np.nan

        # Merge reference data with actual data for training
        reference_df_full = pd.merge(reference_df, df[additional_features + [target_column]], on='Days_of_Treatment', how='left', suffixes=('', '_data'))

        # Drop rows with missing values in the training data
        training_data = reference_df_full.dropna(subset=additional_features + [target_column])
        if training_data.empty:
            print(f"No training data available for column: {target_column}")
            continue

        X_train = training_data[additional_features]
        y_train = training_data[target_column]

        # Extract rows where the current column is missing
        missing = df[target_column].isnull()
        X_pred = df.loc[missing, additional_features]

        if X_pred.empty:
            print(f"No missing values to impute for column: {target_column}")
            continue

        # Handle numerical and categorical features separately
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, additional_features)
            ],
            remainder='passthrough'
        )

        # Train the k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('knn', knn)])

        pipeline.fit(X_train, y_train)

        # Predict missing values
        predicted_values = pipeline.predict(X_pred)

        # Fill missing values in the original DataFrame
        df.loc[missing, target_column] = predicted_values

    return df

# Fill missing values for categorical columns using the reference data
data = fill_missing_values_knn_with_ohe(data, reference_df_categorical, target_columns_categorical, additional_features)

# Print the updated categorical columns
for column in target_columns_categorical:
    if column in data.columns:
        print(f"{column} values:")
        print(data[column])

# Target columns from AB to BE
target_columns = data.columns[27:57]  # Adjust indices based on your specific DataFrame

# Function to fill missing values using KNN classifier and One Hot Encoder
def fill_missing_values_knn_with_ohe(df, target_columns, additional_features):
    for target_column in target_columns:
        if target_column not in df.columns:
            print(f"Column '{target_column}' is not present in the data. Creating it with NaN values.")
            df[target_column] = np.nan

        # Drop rows with missing values in the additional features
        training_data = df.dropna(subset=additional_features + [target_column])
        if training_data.empty:
            print(f"No training data available for column: {target_column}")
            continue

        X_train = training_data[additional_features]
        y_train = training_data[target_column]

        # Extract rows where the current column is missing
        missing = df[target_column].isnull()
        X_pred = df.loc[missing, additional_features]

        if X_pred.empty:
            print(f"No missing values to impute for column: {target_column}")
            continue

        # Define numerical and categorical feature transformers
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessor for additional features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, selector(dtype_include=np.number)),
                ('cat', cat_transformer, selector(dtype_include=object))
            ]
        )

        # Train the k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('knn', knn)])

        pipeline.fit(X_train, y_train)

        # Predict missing values
        predicted_values = pipeline.predict(X_pred)

        # Fill missing values in the original DataFrame
        df.loc[missing, target_column] = predicted_values

    return df

# Fill missing values for categorical columns using the reference data
data = fill_missing_values_knn_with_ohe(data, target_columns, additional_features)

# Print the updated categorical columns
for column in target_columns:
    if column in data.columns:
        print(f"{column} values:")
        print(data[column])

required_columns = [
    'Days_of_Treatment', 'Tug_Test', '10M_Test',
    'Medio_Lateral_Length_M3', 'Medio_Lateral_Length_M4',
    'Anterior_Posterior_M3', 'Anterior_Posterior_M4',
    'BBS_Score', 'BBG'
]

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the data: {missing_columns}")

# Function to fill missing values using KNN regressor
def fill_missing_values_knn_regressor(df, target_columns, additional_features):
    for target_column in target_columns:
        if target_column not in df.columns:
            print(f"Column '{target_column}' is not present in the data. Creating it with NaN values.")
            df[target_column] = np.nan

        # Drop rows with missing values in the additional features
        training_data = df.dropna(subset=additional_features + [target_column])
        if training_data.empty:
            print(f"No training data available for column: {target_column}")
            continue

        X_train = training_data[additional_features]
        y_train = training_data[target_column]

        # Extract rows where the current column is missing
        missing = df[target_column].isnull()
        X_pred = df.loc[missing, additional_features]

        if X_pred.empty:
            print(f"No missing values to impute for column: {target_column}")
            continue

        # Define numerical and categorical feature transformers
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocessor for additional features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, selector(dtype_include=np.number))
            ]
        )

        # Scale features
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_pred_scaled = preprocessor.transform(X_pred)

        # Adjust n_neighbors dynamically
        n_neighbors = min(3, len(X_train_scaled))

        # Train the k-NN regressor
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_train_scaled, y_train)

        # Predict missing values
        predicted_values = knn.predict(X_pred_scaled)

        # Fill missing values in the original DataFrame
        df.loc[missing, target_column] = predicted_values

    return df

# Define the target columns BF to BK (adjust indices based on your specific DataFrame)
target_columns_BF_BK = data.columns[57:63]  # Adjust these indices based on your DataFrame
additional_features = required_columns

# Fill missing values for the target columns using the KNN regressor
data = fill_missing_values_knn_regressor(data, target_columns_BF_BK, additional_features)

print("\nUpdated values for columns 57 to 63:")
print(data.iloc[:, 57:64])

# Save the updated data to a new Excel file
output_file_path = 'C:/Users/cadfem.SL-L14-21/Desktop/data/veda_Lefthemipherasis data/updated_kunal_left_hemipherisis_bf_bk.xlsx'
data.to_excel(output_file_path, index=False)

# Print the updated DataFrame
print("\nUpdated DataFrame:")
print(data)


