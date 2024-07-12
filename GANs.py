import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
import time

# Load the dataset
file_path = 'synthetic_data.xlsx'
original_df = pd.read_excel(file_path)

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(original_df.values)

# Generator model
def build_generator(latent_dim, n_features):
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation='sigmoid')
    ])
    return model

# Discriminator model
def build_discriminator(n_features):
    model = models.Sequential([
        layers.Dense(2048, activation='relu', input_dim=n_features),
        layers.Dropout(0.4),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
def build_gan(generator, discriminator):
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Hyperparameters
latent_dim = 100
n_features = normalized_data.shape[1]

# Build and compile the discriminator
discriminator = build_discriminator(n_features)
discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.00005, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, n_features)

# Build and compile the GAN
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(optimizer=optimizers.Adam(learning_rate=0.00005, beta_1=0.5), loss='binary_crossentropy')

# Training parameters
epochs = 200  # Increase epochs for better learning
batch_size = 15  # Suitable for a small dataset
sample_interval = 20  # Adjusted to print progress more frequently

# Labels for real and fake data
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Training loop with timing
start_time = time.time()

for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, normalized_data.shape[0], batch_size)
    real_data = normalized_data[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_data = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_data, real)
    d_loss_fake = discriminator.train_on_batch(gen_data, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real)

    # Print progress
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time for {epochs} epochs: {elapsed_time:.2f} seconds")

# Generate synthetic data
num_samples = original_df.shape[0]  # Generating synthetic data equal to the number of original samples (210 rows)
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)

# Inverse transform to get the original scale
synthetic_data_original_scale = scaler.inverse_transform(synthetic_data)

# Convert to DataFrame
synthetic_data_df = pd.DataFrame(synthetic_data_original_scale, columns=original_df.columns)

# Post-process categorical and discrete columns
synthetic_data_df['Gender_M'] = synthetic_data_df['Gender_M'].round().astype(int)
synthetic_data_df['Gender_M'] = synthetic_data_df['Gender_M'].clip(0, 1)

# Post-process columns that should contain whole numbers
whole_number_columns = ['Days_of_Treatment', 'Age', 'Lower_Limb_Inversion', 'Upper_Limb_Adduction.1', 'Anterior_Posterior_M3', 'Lower_Limb_Extension', 'Upper_Limb_Abduction', 'Upper_Limb_Flexion', 'Upper_Limb_Flexion.1', 'Lower_Limb_Flexion.1', 'Upper_Limb_Internal_rotation', 'Upper_Limb_Flexion.3', 'Lower_Limb_Extension.1', 'Upper_Limb_Extension', 'Lower_Limb_Adduction.1', 'Lower_Limb_Internal_rotation', 'Lower_Limb_Dosi_Flexion', 'Upper_Limb_Extension.2', 'Medio_Lateral_Length_M3', 'Lower_Limb_Flexion.2', 'Anterior_Posterior_M4', 'Upper_Limb_Flexion.2', 'Lower_Limb_Abduction', 'Lower_Limb_Extension.2', 'Upper_Limb_Abduction.1', 'Lower_Limb_Adduction', 'Upper_Limb_Adduction', 'Upper_Limb_External_Rotation', 'Lower_Limb_Planter_Flexion', 'Lower_Limb_Flexion', 'Lower_Limb_Abduction.1', 'Upper_Limb_Extension.1', 'BBS_Score', 'Lower_Limb_Eversion', 'Lower_Limb_External_Rotation', 'Medio_Lateral_Length_M4', 'Upper_Limb_Extension.3', 'Treatments_HOH_for_reducing_hand_spasticity', 'Treatments_Luna_EMG_for_left_lower_limb_and_upper_limb_strengthening', 'Treatments_Postural_training_for_posture_correction', 'Treatments_SYREBO_for_functional_training', 'Treatments_TYMO_for_balance_training.', 'Gender_M']  
for column in whole_number_columns:
    synthetic_data_df[column] = synthetic_data_df[column].round().astype(int)

# Ensure columns are within the original data's min-max range
for column in whole_number_columns:
    synthetic_data_df[column] = synthetic_data_df[column].clip(lower=original_df[column].min(), upper=original_df[column].max())

# Display the first few rows of synthetic data
print("Synthetic Data:")
print(synthetic_data_df)

# Save the synthetic data to a new Excel file
output_file_path = 'synthetic_data_generated_final.xlsx'
synthetic_data_df.to_excel(output_file_path, index=False)

print(f"Synthetic data has been saved to '{output_file_path}'.")
