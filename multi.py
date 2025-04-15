import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
import pandas as pd
from skimage.transform import resize

# Load GCM data (zos)
gcm_data = xr.open_dataset('zos_Omon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc')
latitudes = gcm_data['latitude']
longitudes = gcm_data['longitude']

# Filter GCM data
gcm_zos = gcm_data['zos'].where((latitudes >= -15) & (latitudes <= 10) & (longitudes >= 90) & (longitudes <= 145),
                                 drop=True).sel(time=slice('1994-01', '2014-12')).values

# Load SSH and SST data
reanalysis_ssh = xr.open_dataset('cmems_mod_glo_phy_my_0.083deg_P1M-m_SSH.nc')
reanalysis_sst = xr.open_dataset('cmems_mod_glo_phy_my_0.083deg_P1M-m_SST.nc')

ssh_data = reanalysis_ssh['zos'].sel(latitude=slice(-15, 10), longitude=slice(90, 145), time=slice('1994-01', '2014-12')).values
sst_data = reanalysis_sst['thetao'].sel(latitude=slice(-15, 10), longitude=slice(90, 145), time=slice('1994-01', '2014-12')).values.squeeze()

# Downsample to match target resolution
ssh_data_downsampled = ssh_data[:, ::6, ::6]  # Adjust as necessary
sst_data_downsampled = sst_data[:, ::6, ::6]

# Scale data
scaler = MinMaxScaler()
ssh_data_scaled = scaler.fit_transform(ssh_data_downsampled.reshape(-1, ssh_data_downsampled.shape[-1])).reshape(ssh_data_downsampled.shape)
sst_data_scaled = scaler.fit_transform(sst_data_downsampled.reshape(-1, sst_data_downsampled.shape[-1])).reshape(sst_data_downsampled.shape)
gcm_zos_scaled = scaler.fit_transform(gcm_zos.reshape(-1, gcm_zos.shape[-1])).reshape(gcm_zos.shape)

# Create combined inputs
combined_inputs = np.stack([ssh_data_scaled, sst_data_scaled], axis=-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_inputs, gcm_zos_scaled, test_size=0.2, random_state=42)

# Adjust input shape
downscale_input_shape = (ssh_data_scaled.shape[1], ssh_data_scaled.shape[2], 2)
target_shape = (11, 11)

# Downsample target data
ssh_data_scaled_downsampled = ssh_data_scaled[:, :target_shape[0], :target_shape[1]]

def build_downscale_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(target_shape[0] * target_shape[1], activation='linear'))
    model.add(layers.Reshape(target_shape))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create and fit model
downscale_model = build_downscale_model(downscale_input_shape)
history = downscale_model.fit(X_train, ssh_data_scaled_downsampled, epochs=50, batch_size=32, validation_split=0.2)

# Prepare y_train and y_test
y_train_scaled = ssh_data_scaled_downsampled[:X_train.shape[0]]
y_test_scaled = ssh_data_scaled_downsampled[X_train.shape[0]:]

# Evaluate model
test_loss, test_mae = downscale_model.evaluate(X_test, y_test_scaled)

# Load and preprocess future data
gcm_future = xr.open_dataset('zos_Omon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.nc')
gcm_future_filtered = gcm_future['zos'].where((latitudes >= -15) & (latitudes <= 10) & (longitudes >= 90) & (longitudes <= 145),
                                              drop=True).sel(time=slice('2030-01', '2050-12')).values

# Scale the future data
future_ssh_scaled = scaler.transform(gcm_future_filtered.reshape(-1, gcm_future_filtered.shape[-1])).reshape(gcm_future_filtered.shape)

# Resize the SSH data to match the model's expected input size
future_ssh_resized = resize(future_ssh_scaled, (128, 128), mode='reflect', anti_aliasing=True)
print(f"Shape of future_ssh_resized: {future_ssh_resized.shape}")  # Print shape

# Resize the SST data to the same size
sst_data_scaled_resized = resize(sst_data_scaled, (128, 128), mode='reflect', anti_aliasing=True)
print(f"Shape of sst_data_scaled_resized: {sst_data_scaled_resized.shape}")  # Print shape

# Resize future_ssh_resized and sst_data_scaled_resized to match model input
desired_height = 51  # Set this based on your model's expected shape
desired_width = 110  # Set this based on your model's expected shape

# Resize the arrays
future_ssh_resized = resize(future_ssh_resized, (desired_height, desired_width), mode='reflect', anti_aliasing=True)
sst_data_scaled_resized = resize(sst_data_scaled_resized, (desired_height, desired_width), mode='reflect', anti_aliasing=True)

# Stack them to create the combined input
future_combined_inputs = np.stack([future_ssh_resized, sst_data_scaled_resized], axis=-1)  # Shape should now be (51, 110, 2)

# Add a batch dimension
future_combined_inputs = np.expand_dims(future_combined_inputs, axis=0)  # Add batch dimension

print("Shape of future_combined_inputs before prediction:", future_combined_inputs.shape)

# Now predict with the downscaling model
predicted_downscaled_ssh = downscale_model.predict(future_combined_inputs)

# Create xarray DataArray for predictions
predicted_downscaled_ssh_array = xr.DataArray(
    predicted_downscaled_ssh,
    dims=["time", "lat", "lon"],
    coords={
        "time": pd.date_range(start='2030-01', periods=future_combined_inputs.shape[0], freq='M'),  # Ensure correct length
        "lat": np.linspace(-15, 10, 128),  # Ensure the number of lat points matches
        "lon": np.linspace(90, 145, 128)   # Ensure the number of lon points matches
    }
)

# Save predictions to NetCDF
predicted_downscaled_ssh_array.to_netcdf('predicted_downscaled_SSH_2030_2050.nc')

print("Predicted downscaled SSH saved to NetCDF.")
