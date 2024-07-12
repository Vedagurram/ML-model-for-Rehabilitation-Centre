import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset with the first row as header
file_path = 'combined.xlsx'
df = pd.read_excel(file_path, header=0)

# Perform one-hot encoding on the 'Treatments' and 'Gender' columns
ohe = OneHotEncoder(drop='first')

# Fit and transform the data
encoded_data = ohe.fit_transform(df[['Treatments', 'Gender']]).toarray()
# Get the feature names for the one-hot encoded columns
ohe_feature_names = ohe.get_feature_names_out(['Treatments', 'Gender'])

# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=ohe_feature_names, index=df.index)

# Drop the original 'Treatments' and 'Gender' columns
df = df.drop(columns=['Treatments', 'Gender'])
# Combine the original data with the one-hot encoded data
df_combined = pd.concat([df, encoded_df], axis=1)

# Save the combined DataFrame to a new Excel file
output_file_path = 'combined_encoded.xlsx'
df_combined.to_excel(output_file_path, index=False)
print(f"The combined data with one-hot encoding has been saved to {output_file_path}.")
