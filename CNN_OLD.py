import xarray as xr
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape

# Function to load and mask dataset
# Function to load and mask dataset
def load_and_mask_dataset(data, var_name, lat_range, lon_range, time_range):
    # 1. Pastikan variabel tersedia
    if var_name not in data.variables:
        raise ValueError(f"Variable '{var_name}' tidak ditemukan dalam dataset. Variabel yang tersedia: {list(data.variables.keys())}")

    # 2. Pastikan dimensi waktu ada
    if 'time' not in data.dims:
        raise ValueError("Dimensi 'time' tidak ditemukan dalam dataset. Mungkin dataset sudah di-slice sebelumnya.")

    # 3. Pastikan ada data waktu dalam dataset
    time_values = data['time'].values
    if len(time_values) == 0:
        raise ValueError("Dataset tidak memiliki data waktu.")

    # 4. Pastikan rentang waktu valid
    min_time, max_time = time_values.min(), time_values.max()
    hist1, hist2 = time_range

    if hist1 < min_time or hist2 > max_time:
        raise ValueError(f"Rentang waktu {time_range} berada di luar cakupan dataset ({min_time} - {max_time})")

    # 5. Pilih data berdasarkan waktu (hanya jika waktu masih dalam cakupan dataset)
    sliced_data = data[var_name].sel(time=slice(hist1, hist2))

    # 6. Cek apakah hasil slicing kosong
    if sliced_data.size == 0:
        raise ValueError(f"Data hasil slicing kosong. Cek kembali rentang waktu {time_range}.")

    # 7. Deteksi nama variabel latitude dan longitude dalam dataset
    lat_names = ["lat", "latitude", "j", "Y"]
    lon_names = ["lon", "longitude", "i", "X"]

    detected_lat = next((lat for lat in lat_names if lat in data.dims), None)
    detected_lon = next((lon for lon in lon_names if lon in data.dims), None)

    if detected_lat is None or detected_lon is None:
        raise ValueError("Dimensi latitude dan longitude tidak ditemukan dalam dataset.")

    print(f"✅ Menggunakan '{detected_lat}' sebagai latitude dan '{detected_lon}' sebagai longitude.")

    # 8. Filter berdasarkan lat/lon
    masked_data = sliced_data.where(
        (data[detected_lat] >= lat_range[0]) & (data[detected_lat] <= lat_range[1]) &
        (data[detected_lon] >= lon_range[0]) & (data[detected_lon] <= lon_range[1]),
        drop=True
    )

    return masked_data

# Define variable names
gcm_name = 'mpi'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'  
ssh_var = 'zos'  
hist1 = np.datetime64('1995-01-01')
hist2 = np.datetime64('2014-12-01')
fut1 = np.datetime64('2031-01-01')
fut2 = np.datetime64('2050-12-31')

# Define file paths
gcm_data = f'{indir}MPI_historical_1993_2014.nc'
ssh_data = 'data/cmems_mod_glo_phy_my_0.083deg_P1M-m_1993_2014.nc'
future_data = {
    "ssp245": f'{indir}MPI_ssp245_2015_2100.nc',
    "ssp370": f'{indir}MPI_ssp370_2015_2100.nc',
    "ssp585": f'{indir}MPI_ssp585_2015_2100.nc'
}

# Define the region for Indonesia
lat_range = (-15, 10)
lon_range = (90, 145)

# Load dataset GCM historical
gcm_ds = xr.open_dataset(gcm_data, engine="netcdf4")
gcm_min_time = gcm_ds.time.min().values
gcm_max_time = gcm_ds.time.max().values

if (hist1 >= gcm_min_time) and (hist2 <= gcm_max_time):
    gcm_sliced = load_and_mask_dataset(gcm_ds, gcm_var, lat_range, lon_range, (hist1, hist2))
    print("✅ Data historis berhasil di-slice!")
else:
    print(f"⚠️ Rentang waktu yang diminta ({hist1} - {hist2}) tidak ada dalam dataset! "
          f"Rentang dataset: {gcm_min_time} - {gcm_max_time}")
    gcm_sliced = None  

# Load dataset SSH
ssh_ds = xr.open_dataset(ssh_data, engine="netcdf4")
ssh_sliced = load_and_mask_dataset(ssh_ds, ssh_var, lat_range, lon_range, (hist1, hist2))

# Cek rentang waktu pada setiap skenario SSP
future_sliced = {}
for scen, fpath in future_data.items():
    fut_ds = xr.open_dataset(fpath, engine="netcdf4")
    fut_min_time = fut_ds.time.min().values
    fut_max_time = fut_ds.time.max().values

    if (fut1 >= fut_min_time) and (fut2 <= fut_max_time):
        future_sliced[scen] = load_and_mask_dataset(fut_ds, gcm_var, lat_range, lon_range, (fut1, fut2))
        print(f"✅ Data {scen} berhasil di-slice!")
    else:
        print(f"⚠️ Rentang waktu yang diminta ({fut1} - {fut2}) tidak ada dalam dataset {scen}! "
              f"Rentang dataset: {fut_min_time} - {fut_max_time}")

# Save to NetCDF
outdir = f"data/hasil/{gcm_name}/"
os.makedirs(outdir, exist_ok=True)

if gcm_sliced is not None:
    gcm_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_historical_{hist1}_{hist2}.nc")
if ssh_sliced is not None:
    ssh_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_reanalysis_{hist1}_{hist2}.nc")

for scen, ds in future_sliced.items():
    ds.to_netcdf(f"{outdir}{gcm_name}_sliced_{scen}_{fut1}_{fut2}.nc")

# Prepare data for training
X = np.nan_to_num(gcm_sliced.values)  
y = np.nan_to_num(ssh_sliced.values)  

# Ensure that X and y have the same number of samples
min_samples = min(X.shape[0], y.shape[0])
X = X[:min_samples]
y = y[:min_samples]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1] * y_train.shape[2], activation='linear'),
    Reshape((y_train.shape[1], y_train.shape[2]))
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.summary()

# # Normalize future GCM data
# X_future = np.nan_to_num(future_sliced.values)  
# X_future_scaled = scaler_X.transform(X_future.reshape(-1, X_future.shape[-1]))  
# X_future_scaled = X_future_scaled.reshape(X_future.shape[0], 55, 55, 1)  

# Predict future SSH data
predictions = {}

for scen, future_ds in future_sliced.items():
    X_future = np.nan_to_num(future_ds.values)
    X_future_scaled = scaler_X.transform(X_future.reshape(-1, X_future.shape[-1]))
    X_future_scaled = X_future_scaled.reshape(X_future.shape[0], 55, 55, 1)

    # Predict future SSH
    predictions[scen] = model.predict(X_future_scaled)

    # Rescale predictions
    predicted_ssh = scaler_y.inverse_transform(predictions[scen].reshape(-1, predictions[scen].shape[-1]))
    predicted_ssh_reshaped = predicted_ssh.reshape(predictions[scen].shape[0], predictions[scen].shape[1], predictions[scen].shape[2])
    predicted_ssh_reshaped = np.squeeze(predicted_ssh_reshaped) # Hilangkan dimensi terakhir

    # Buat koordinat waktu untuk 240 bulan / 20 tahun
    time_steps_future = pd.date_range(start='2031-01', periods=240, freq='ME')

    # Konversi prediksi SSH ke DataArray
    predicted_ssh = xr.DataArray(
        predicted_ssh_reshaped,  # Pastikan ini memiliki dimensi [time, lat, lon]
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': time_steps_future,  # Gunakan pandas datetime index
            'latitude': np.linspace(-15, 10, 301),
            'longitude': np.linspace(90, 145, 660)
        }
    )

    # Menghitung rata-rata tahunan
    predicted_ssh_annual = predicted_ssh.groupby("time.year").mean(dim="time")

    # Konversi ke Dataset dengan variabel 'zos'
    predicted_ssh_annual = predicted_ssh_annual.to_dataset(name="zos")

    # Pastikan dimensi j dan i bertipe int agar sesuai dengan corrected_ssh_annual
    if "j" in predicted_ssh_annual.coords:
        predicted_ssh_annual = predicted_ssh_annual.assign_coords(j=("j", predicted_ssh_annual["j"].values.astype(int)))
    if "i" in predicted_ssh_annual.coords:
        predicted_ssh_annual = predicted_ssh_annual.assign_coords(i=("i", predicted_ssh_annual["i"].values.astype(int)))

    # Tambahkan atribut ke variabel 'zos' (bukan ke Dataset)
    predicted_ssh_annual['zos'].attrs['description'] = 'Predicted sea surface height from 2031 to 2050'
    predicted_ssh_annual['zos'].attrs['units'] = 'meters'
    predicted_ssh_annual['zos'].attrs['standard_name'] = 'sea_surface_height'
    predicted_ssh_annual['zos'].attrs['long_name'] = 'Sea Surface Height Above Geoid'

    # Tambahkan atribut koordinat ke 'zos'
    predicted_ssh_annual["zos"].attrs["coordinates"] = "latitude longitude"

    # Simpan sebagai NetCDF
    pred_name = f"{outdir}predicted_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
    predicted_ssh_annual.to_netcdf(pred_name)

print("Predictions saved for all scenarios.")

### **PENGOLAHAN DATA CORRECTED SSH (1995-2014)**

# Simpan koordinat sebelum konversi
coords = gcm_sliced.coords
dims = gcm_sliced.dims

# Konversi dan normalisasi
X_corrected = np.nan_to_num(gcm_sliced.values)
X_corrected_scaled = scaler_X.transform(X_corrected.reshape(-1, X_corrected.shape[-1]))  
X_corrected_scaled = X_corrected_scaled.reshape(X_corrected.shape[0], 55, 55, 1)  

# Prediksi dengan model
corrections = model.predict(X_corrected_scaled)

# Rescale ke bentuk aslinya
corrected_ssh = scaler_y.inverse_transform(corrections.reshape(-1, corrections.shape[-1]))
corrected_ssh_reshaped = corrected_ssh.reshape(corrections.shape[0], corrections.shape[1], corrections.shape[2])

# Buat koordinat waktu untuk 240 bulan / 20 tahun
time_steps_cor = pd.date_range(start='1995-01', periods=240, freq='ME')

# Konversi prediksi SSH ke DataArray
corrected_ssh = xr.DataArray(
    corrected_ssh_reshaped,  # Pastikan ini memiliki dimensi [time, lat, lon]
    dims=['time', 'latitude', 'longitude'],
    coords={
        'time': time_steps_cor,  # Gunakan pandas datetime index
        'latitude': np.linspace(-15, 10, 301),
        'longitude': np.linspace(90, 145, 660)
    }
)
# Konversi ke rata-rata tahunan
corrected_ssh_annual = corrected_ssh.groupby("time.year").mean(dim="time")

# Ubah DataArray menjadi Dataset dengan nama 'zos'
corrected_ssh_annual = corrected_ssh_annual.to_dataset(name='zos')

# Pastikan dimensi j dan i bertipe int agar sesuai dengan format sebelumnya
if "j" in corrected_ssh_annual.coords:
    corrected_ssh_annual = corrected_ssh_annual.assign_coords(j=("j", corrected_ssh_annual["j"].values.astype(int)))
if "i" in corrected_ssh_annual.coords:
    corrected_ssh_annual = corrected_ssh_annual.assign_coords(i=("i", corrected_ssh_annual["i"].values.astype(int)))

# Tambahkan atribut ke variabel 'zos' (bukan ke Dataset)
corrected_ssh_annual['zos'].attrs['description'] = 'Corrected sea surface height from 1995 to 2014'
corrected_ssh_annual['zos'].attrs['units'] = 'meters'
corrected_ssh_annual['zos'].attrs['standard_name'] = 'sea_surface_height'
corrected_ssh_annual['zos'].attrs['long_name'] = 'Sea Surface Height Above Geoid'

# Simpan ke NetCDF
cor_name = f"{outdir}corrected_ssh_annual_{gcm_name}_1995-2014.nc"
corrected_ssh_annual.to_netcdf(cor_name)

# GABUNG & SIMPAN HASIL SEMUA SCEN DALAM 1 NC

# # Dictionary untuk menyimpan hasil setiap skenario
# all_predictions = {}

# for scen, future_ds in future_sliced.items():
#     X_future = np.nan_to_num(future_ds.values)
#     X_future_scaled = scaler_X.transform(X_future.reshape(-1, X_future.shape[-1]))
#     X_future_scaled = X_future_scaled.reshape(X_future.shape[0], 55, 55, 1)

#     # Predict future SSH
#     predictions[scen] = model.predict(X_future_scaled)

#     # Rescale predictions
#     predicted_ssh = scaler_y.inverse_transform(predictions[scen].reshape(-1, predictions[scen].shape[-1]))

#     # Reshape ke bentuk grid
#     predicted_ssh_reshaped = predicted_ssh.reshape(X_future.shape[0], 55, 55)
#     predicted_ssh_reshaped = np.squeeze(predicted_ssh_reshaped)  

#     # Buat koordinat waktu
#     time_steps_future = pd.date_range(start='2031-01', periods=240, freq='ME')

#     # Konversi prediksi SSH ke DataArray
#     predicted_ssh = xr.DataArray(
#         predicted_ssh_reshaped,
#         dims=['time', 'latitude', 'longitude'],
#         coords={
#             'time': time_steps_future,
#             'latitude': np.linspace(-15, 10, 301),
#             'longitude': np.linspace(90, 145, 660)
#         }
#     )

#     # Menghitung rata-rata tahunan
#     predicted_ssh_annual = predicted_ssh.groupby("time.year").mean(dim="time")

#     # Simpan ke dictionary
#     all_predictions[scen] = predicted_ssh_annual

# # Gabungkan semua skenario menjadi satu Dataset Xarray
# combined_dataset = xr.Dataset(all_predictions)

# # Simpan ke NetCDF jika perlu
# combined_dataset.to_netcdf("predicted_ssh_annual.nc")

