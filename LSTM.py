import xarray as xr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to load and mask dataset
def load_and_mask_dataset(file_path, var_name, lat_range, lon_range, time_range):
    """Load and mask the dataset for a specific region and time."""
    data = xr.open_dataset(file_path)
    sliced_data = data[var_name].where(
        (data['latitude'] >= lat_range[0]) & (data['latitude'] <= lat_range[1]) &
        (data['longitude'] >= lon_range[0]) & (data['longitude'] <= lon_range[1]),
        drop=True
    ).sel(time=slice(*time_range))
    return sliced_data

# Define file paths
gcm_data = 'zos_Omon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc'
ssh_data = 'cmems_mod_glo_phy_my_0.083deg_P1M-m_SSH.nc'
future_data = 'zos_Omon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.nc'

# Define variable names
gcm_var = 'zos'  # Sea Surface Height above Geoid from GCM
ssh_var = 'zos'  # Sea Surface Height above Geoid from SSH dataset

# Define the region for Indonesia
lat_range = (-15, 10)
lon_range = (90, 145)
time_range1 = ('1994-01', '2014-12')
time_range2 = ('2030-01', '2050-12')

# Load and mask datasets
gcm_sliced = load_and_mask_dataset(gcm_data, gcm_var, lat_range, lon_range, time_range1)
ssh_sliced = load_and_mask_dataset(ssh_data, ssh_var, lat_range, lon_range, time_range1)
future_sliced = load_and_mask_dataset(future_data, gcm_var, lat_range, lon_range, time_range2)

# # Save to netCDF
# gcm_sliced.to_netcdf('sliced_historical_1994-2014.nc')
# ssh_sliced.to_netcdf('sliced_reanalysis_1994-2014.nc')
# future_sliced.to_netcdf('sliced_ssp245_2030-2050.nc')

# Prepare data for training
X = np.nan_to_num(gcm_sliced.values)  # Input data from GCM (resampled)
y = np.nan_to_num(ssh_sliced.values)  # Target data from SSH

# Ensure that X and y have the same number of samples
min_samples = min(X.shape[0], y.shape[0])
X = X[:min_samples]
y = y[:min_samples]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale GCM data
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
# Scale SSH data
y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Flatten spatial dimensions (latitude, longitude) and treat them as features for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], -1, 1))  # (samples, timesteps, 1 feature)
X_test_lstm = X_test.reshape((X_test.shape[0], -1, 1))

y_train_lstm = y_train.reshape((y_train.shape[0], -1))  # Flatten y to match output of LSTM
y_test_lstm = y_test.reshape((y_test.shape[0], -1))

# Define a simple CNN model
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False))  # LSTM layer
model_lstm.add(Dense(64, activation='relu'))  # Dense layer for feature extraction
model_lstm.add(Dense(y_train_lstm.shape[1], activation='linear'))  # Output layer

model_lstm.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

model_lstm.summary()

# Ensure that future GCM data has the correct shape
X_future = np.nan_to_num(future_sliced.values)  # Shape: (1032, 55, 55)

# Ensure that future GCM data has the correct shape for LSTM
X_future_lstm = X_future.reshape(X_future.shape[0], -1, 1)  # Reshape to (samples, timesteps, features)

# Predict future SSH data using the LSTM model
predictions_lstm = model_lstm.predict(X_future_lstm)

# Reshape predictions back to the original SSH grid shape
predicted_ssh_lstm = predictions_lstm.reshape(predictions_lstm.shape[0], y_train.shape[1], y_train.shape[2])

# Rescale the predictions back to the original scale
predicted_ssh_rescaled = scaler_y.inverse_transform(predicted_ssh_lstm.reshape(-1, predicted_ssh_lstm.shape[-1]))
predicted_ssh_reshaped = predicted_ssh_rescaled.reshape(predictions_lstm.shape[0], y_train.shape[1], y_train.shape[2])

# Create a time coordinate for 252 months (21 years)
time_steps_future = pd.date_range(start='2030-01', periods=252, freq='ME')  # Monthly intervals

# Create a list of month names
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Create the time labels based on the number of time steps
time_labels1 = [month_names[i % 12] + ' ' + str(2030 + (i // 12)) for i in range(len(time_steps_future))]

# Convert predictions to xarray DataArray and save
predicted_ssh_xr_lstm = xr.DataArray(
    predicted_ssh_reshaped,
    dims=['time', 'latitude', 'longitude'],
    coords={
        'time': time_labels1,
        'latitude': np.linspace(-15, 10, 301),  # Latitude range
        'longitude': np.linspace(90, 145, 660)  # Longitude range
    }
)

# Add attributes and save to netCDF
predicted_ssh_xr_lstm.attrs['description'] = 'LSTM-predicted sea surface height from 2030 to 2050'
predicted_ssh_xr_lstm.attrs['units'] = 'meters'  # Or the appropriate units for your data
predicted_ssh_xr_lstm['time'].attrs['description'] = 'Monthly time steps from January 2030 to December 2050'
# Rename the variable for consistency
predicted_ssh_xr_lstm.name = 'zos'  # Rename to 'zos'
# Add attributes to the 'zos' variable
predicted_ssh_xr_lstm.attrs['standard_name'] = 'sea_surface_height_above_geoid'
# save
predicted_ssh_xr_lstm.to_netcdf('predicted_ssh_lstm_ssp245_2030-2050.nc')

# Ensure that future GCM data has the correct shape
X_corrected = np.nan_to_num(gcm_sliced.values)  # Shape: (1980, 55, 55)

# Normalize the future GCM data
X_corrected_scaled = scaler_X.transform(X_corrected.reshape(-1, X_corrected.shape[-1]))  # Reshape to 2D for scaling
X_corrected_scaled = X_corrected_scaled.reshape(X_corrected.shape[0], 55, 55)  # Reshape back to original shape

# Ensure that X_corrected_scaled has the correct shape for LSTM
X_corrected_lstm = X_corrected_scaled.reshape(X_corrected_scaled.shape[0], -1, 1)  # Reshape to (samples, timesteps, features)

# Predict future SSH data
corrections = model_lstm.predict(X_corrected_lstm)

# The corrected_ssh is flattened; reshape it according to corrections
corrected_ssh = scaler_y.inverse_transform(corrections.reshape(-1, corrections.shape[-1]))

# Reshape corrected_ssh to match the original data dimensions
corrected_ssh_reshaped = corrected_ssh.reshape(corrections.shape[0], corrections.shape[1], corrections.shape[2])

# Create time labels based on the number of time steps in corrected SSH
time_steps_correct = pd.date_range(start='1994-01', periods=corrected_ssh_reshaped.shape[0], freq='ME')  # Monthly intervals

# Create the time labels based on the number of time steps
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

time_labels2 = [month_names[i % 12] + ' ' + str(1994 + (i // 12)) for i in range(corrected_ssh_reshaped.shape[0])]

# Convert to xarray DataArray
corrected_ssh_xr = xr.DataArray(
    corrected_ssh_reshaped,
    dims=['time', 'latitude', 'longitude'],
    coords={
        'time': time_labels2,  # Custom time labels
        'latitude': np.linspace(-15, 10, 301),  # Latitude range from -15 to 10
        'longitude': np.linspace(90, 145, 660)   # Longitude range from 90 to 145
    }
)

# Add attributes to the DataArray
corrected_ssh_xr.attrs['description'] = 'Corrected sea surface height from 1993 to 2014'
corrected_ssh_xr.attrs['units'] = 'meters'  # Or the appropriate units for your data
corrected_ssh_xr['time'].attrs['description'] = 'Monthly time steps from January 1993 to December 2014'
corrected_ssh_xr['time'].attrs['calendar'] = 'proleptic_gregorian'

# Rename the variable for consistency
corrected_ssh_xr.name = 'zos'  # Rename to 'zos'

# Add attributes to the 'zos' variable
corrected_ssh_xr.attrs['standard_name'] = 'sea_surface_height_above_geoid'
corrected_ssh_xr.attrs['long_name'] = 'Sea Surface Height Above Geoid'

# Save to netCDF
corrected_ssh_xr.to_netcdf('corrected_ssh_lstm_1994-2014.nc')