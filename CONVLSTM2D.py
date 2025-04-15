#### COBA NEW VERSION CODE (CONVLSTM2D) #### --> uji coba, hasil kurang bagus

import xarray as xr
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Conv2D, ConvLSTM2D, TimeDistributed, ZeroPadding2D, Cropping2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================
# 1. Fungsi untuk Load Dataset
# ==========================
def load_and_mask_dataset(data, var_name, lat_range, lon_range, time_range):
    """Load, slice, and mask dataset based on the given variable and spatial-temporal range."""
    if var_name not in data.variables:
        raise ValueError(f"Variable '{var_name}' tidak ditemukan dalam dataset.")
    if 'time' not in data.dims:
        raise ValueError("Dimensi 'time' tidak ditemukan dalam dataset.")
    if data[var_name].size == 0:
        raise ValueError(f"Dataset tidak memiliki data untuk variabel '{var_name}'.")
    
    # Konversi waktu ke numpy.datetime64
    time_values = data['time'].values
    min_time, max_time = np.datetime64(time_values.min(), 'D'), np.datetime64(time_values.max(), 'D')
    start_time, end_time = np.datetime64(time_range[0], 'D'), np.datetime64(time_range[1], 'D')
    if start_time < min_time or end_time > max_time:
        raise ValueError(f"Rentang waktu {time_range} berada di luar cakupan dataset ({min_time} - {max_time})")
    
    # Pilih waktu dengan metode "nearest"
    start_time = data.time.sel(time=start_time, method="nearest").values
    end_time = data.time.sel(time=end_time, method="nearest").values
    
    # Lakukan slicing
    sliced_data = data[var_name].sel(time=slice(start_time, end_time))
    if sliced_data.size == 0:
        raise ValueError(f"Data hasil slicing kosong untuk rentang waktu {time_range}.")
    
    # Deteksi nama variabel lat/lon
    lat_names = ["lat", "latitude", "j", "y"]
    lon_names = ["lon", "longitude", "i", "x"]
    detected_lat = next((lat for lat in lat_names if lat in data.dims), None)
    detected_lon = next((lon for lon in lon_names if lon in data.dims), None)
    if detected_lat is None or detected_lon is None:
        raise ValueError("Dimensi latitude dan longitude tidak ditemukan dalam dataset.")
    
    print(f"✓ Menggunakan '{detected_lat}' sebagai latitude dan '{detected_lon}' sebagai longitude.")
    
    # Filter berdasarkan lat/lon
    masked_data = sliced_data.where(
        (data[detected_lat] >= lat_range[0]) & (data[detected_lat] <= lat_range[1]) &
        (data[detected_lon] >= lon_range[0]) & (data[detected_lon] <= lon_range[1]),
        drop=False  # Jangan hapus time step, hanya isi dengan NaN
    ).dropna(dim="time", how="all")  # Hapus time step kosong akibat masking
    return masked_data

# ==========================
# 2. Parameter Input
# ==========================
gcm_name = 'miroc6'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'
ssh_var = 'zos'

# Rentang waktu
hist1, hist2 = '1995-01-16', '2014-12-16'  # GCM Historical
fut1, fut2 = '2021-01-16', '2040-12-16'   # GCM Future
ssh1, ssh2 = '1995-01-01', '2014-12-01'   # SSH

# File paths
gcm_data = f'{indir}{gcm_name}_historical_1993_2014.nc'
ssh_data = 'data/cmems_mod_glo_phy_my_0.083deg_P1M-m_1993_2014.nc'
future_data = {
    "ssp245": f'{indir}{gcm_name}_ssp245_2015_2100.nc',
    "ssp370": f'{indir}{gcm_name}_ssp370_2015_2100.nc',
    "ssp585": f'{indir}{gcm_name}_ssp585_2015_2100.nc'
}

# Define region (Indonesia)
lat_range = (-15, 10)
lon_range = (90, 145)

# ==========================
# 3. Load Data
# ==========================
print("📚 Membuka dataset GCM historical...")
gcm_ds = xr.open_dataset(gcm_data, engine="netcdf4", decode_times=True)
gcm_sliced = load_and_mask_dataset(gcm_ds, gcm_var, lat_range, lon_range, (hist1, hist2))

print("📚 Membuka dataset SSH...")
ssh_ds = xr.open_dataset(ssh_data, engine="netcdf4", decode_times=True)
ssh_sliced = load_and_mask_dataset(ssh_ds, ssh_var, lat_range, lon_range, (ssh1, ssh2))

print("📚 Membuka dataset GCM Future...")
future_sliced = {}
for scen, fpath in future_data.items():
    print(f"📂 Membuka dataset {scen}...")
    fut_ds = xr.open_dataset(fpath, engine="netcdf4", decode_times=True)

    fut_min_time = np.datetime64(fut_ds.time.min().values, 'D')
    fut_max_time = np.datetime64(fut_ds.time.max().values, 'D')

    if (np.datetime64(fut1, 'D') >= fut_min_time) and (np.datetime64(fut2, 'D') <= fut_max_time):
        future_sliced[scen] = load_and_mask_dataset(fut_ds, gcm_var, lat_range, lon_range, (fut1, fut2))
        print(f"✅ Data {scen} berhasil di-slice!")
    else:
        print(f"⚠️ Rentang waktu {fut1} - {fut2} tidak tersedia dalam dataset {scen}! "
              f"Rentang dataset: {fut_min_time} - {fut_max_time}")
    
# Save to NetCDF
outdir = f"data/hasil/{gcm_name}/{fut1}_{fut2}/"
os.makedirs(outdir, exist_ok=True)

if gcm_sliced is not None:
    gcm_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_historical_{hist1}_{hist2}.nc")
if ssh_sliced is not None:
    ssh_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_reanalysis_{hist1}_{hist2}.nc")

for scen, ds in future_sliced.items():
    ds.to_netcdf(f"{outdir}{gcm_name}_sliced_{scen}_{fut1}_{fut2}.nc")

# ==========================
# 4. Preprocessing Data
# ==========================

# Pastikan jumlah sampel X dan y sama
min_samples = min(gcm_sliced.shape[0], ssh_sliced.shape[0])
gcm_sliced = gcm_sliced[:min_samples]
ssh_sliced = ssh_sliced[:min_samples]

# Konversi NaN ke angka (gunakan interpolasi jika memungkinkan)
X = np.nan_to_num(gcm_sliced.values)  # Ganti NaN dengan 0
y = np.nan_to_num(ssh_sliced.values)  # Ganti NaN dengan 0

# Future data (misalnya skenario iklim)
X_future = {scen: np.nan_to_num(ds.values) for scen, ds in future_sliced.items()}

# Normalisasi menggunakan MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

# Reshape data untuk normalisasi (flatten -> normalize -> reshape back)
X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

# Normalisasi future data dengan scaler yang sudah di-fit
X_future_scaled = {
    scen: scaler_X.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    for scen, data in X_future.items()
}

# Split data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Tambahkan dimensi channel untuk CNN (batch, height, width, channels)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
y_train = y_train[..., np.newaxis]
y_test = y_test[..., np.newaxis]

# Tambahkan dimensi time_steps untuk ConvLSTM2D (batch, time_steps, height, width, channels)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

# Pastikan future data juga memiliki 5 dimensi (batch, time_steps, height, width, channels)
X_future_scaled = {
    scen: np.expand_dims(data, axis=1) for scen, data in X_future_scaled.items()
}

# Cek shape data untuk memastikan semuanya sesuai
print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
for scen, data in X_future_scaled.items():
    print(f"Shape X_future_scaled[{scen}]:", data.shape)

# Debugging tambahan
assert X_train.shape == (X_train.shape[0], 1, X_train.shape[2], X_train.shape[3], 1), "Shape X_train tidak sesuai!"
assert y_train.shape == (y_train.shape[0], 1, y_train.shape[2], y_train.shape[3], 1), "Shape y_train tidak sesuai!"

# ==========================
# 5. Build U-Net Model (dengan ConvLSTM2D)
# ==========================
def build_unet_conv_lstm(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    pool1 = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(conv1)
    
    conv2 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(pool1)
    
    # Decoder
    up1 = TimeDistributed(UpSampling2D((2, 2)))(conv2)
    conv1_padded = TimeDistributed(ZeroPadding2D(((0, 0), (1, 0))))(conv1)  # Adjust padding
    concat1 = Concatenate()([up1, conv1_padded])
    
    conv3 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(concat1)
    up2 = TimeDistributed(UpSampling2D((8, 12)))(conv3)  # Upsample to match output size
    
    outputs = TimeDistributed(Cropping2D(((9, 10), (6, 6))))(up2)  # Crop to match output size
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(outputs)  # Final output layer
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

input_shape = X_train.shape[1:]  # (time_steps, height, width, channels)
model = build_unet_conv_lstm(input_shape)

# Cek arsitektur model
model.summary()

# save model
model.save(f"model/model_convLSTM2D_{gcm_name}.keras")

# ==========================
# 6. Train Model
# ==========================
# Pastikan tidak ada kesalahan dimensi sebelum training
print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=3,  # Bisa ditingkatkan
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# ==========================
# 7. Prediksi Future Data
# ==========================

def prepare_future_data(future_ds, scaler_X):
    print(f"New Shape X_future after interpolation: {future_ds.sizes}")  # Harus (240, 40, 55)
    
    # Konversi DataArray ke numpy array
    X_future = future_ds.values  # Ambil nilai array langsung
    
    # Reshape agar sesuai dengan model
    X_future_reshaped = X_future.reshape(X_future.shape[0], -1)  # Flatten ke (240, 40*55)

    # Fit scaler_X dengan data sebelumnya
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  
    scaler_X.fit(X_train_reshaped)
    
    # Normalisasi
    X_future_scaled = scaler_X.transform(X_future_reshaped)
    X_future_scaled = X_future_scaled.reshape(X_future.shape)  # Kembali ke bentuk awal
    
    return X_future_scaled[..., np.newaxis]  # Tambahkan dimensi channel terakhir

# Load model dan scaler
model = load_model(f"model/model_convLSTM2D_{gcm_name}.keras")

predictions = {}

for scen, future_ds in future_data.items():
    print(f"Processing scenario: {scen}")
    
    future_ds_sliced = load_and_mask_dataset(
        xr.open_dataset(future_ds), gcm_var, lat_range, lon_range, (fut1, fut2))

    X_future_scaled = prepare_future_data(future_ds_sliced, scaler_X)
    print("Shape of X_future_scaled:", X_future_scaled.shape)  # (240, 40, 55, 1)

    # Pastikan shape sesuai sebelum dimasukkan ke model
    batch_size, height, width, channels = X_future_scaled.shape  # (240, 40, 55, 1)
    X_future_scaled = X_future_scaled.reshape(batch_size, 1, height, width, channels)  # (240, 1, 40, 55, 1)

    print("✅ Final shape before prediction:", X_future_scaled.shape)  # Harus (240, 1, 40, 55, 1)

    # Gunakan ini untuk prediksi
    pred_ssh_scaled = model.predict(X_future_scaled)
    print(f"Original pred_ssh_scaled shape: {pred_ssh_scaled.shape}")  # (240, 1, 301, 660, 32)

    # Flatten prediksi sebelum inverse transform
    pred_ssh_reshaped = pred_ssh_scaled.reshape(pred_ssh_scaled.shape[0], -1)

    print("y_train shape:", y_train.shape)  # Harus sama dengan output model
    y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
    print("y_train_reshaped shape:", y_train_reshaped.shape)  # Pastikan fitur cocok

    scaler_y.fit(y_train_reshaped)  # Fit ulang jika perlu

    # Pastikan jumlah fitur sama dengan y_train_reshaped
    num_features_y = np.prod(y_train.shape[1:])  # Total elemen per sample
    pred_ssh_reshaped = pred_ssh_scaled.reshape(pred_ssh_scaled.shape[0], num_features_y)

    print("Reshaped pred_ssh shape:", pred_ssh_reshaped.shape)
    print("Expected features from y_train:", num_features_y)

    # Inverse transform ke skala asli
    pred_ssh_rescaled = scaler_y.inverse_transform(pred_ssh_reshaped)
    pred_ssh_rescaled = pred_ssh_rescaled.reshape(pred_ssh_scaled.shape)

    print("Final pred_ssh_rescaled shape:", pred_ssh_rescaled.shape)  # Harus (240, 1, 301, 660, 32)

    # Hapus dimensi batch (axis=1) dan channel (axis=-1)
    predicted_ssh_squeezed = pred_ssh_rescaled.squeeze(axis=(1, -1))  # (240, 301, 660)

    print(f"Predicted SSH Shape (After Squeeze): {predicted_ssh_squeezed.shape}")  # (240, 301, 660)

    # Ambil nilai latitude dan longitude dari dataset asli
    lat_values = future_ds_sliced.latitude.values  # Jika sudah 1D, langsung pakai ini
    lon_values = future_ds_sliced.longitude.values  # Sama dengan lat

    # Ambil dimensi dari hasil prediksi secara dinamis
    num_lat = predicted_ssh_squeezed.shape[1]  # 301
    num_lon = predicted_ssh_squeezed.shape[2]  # 660

    # Buat koordinat baru berdasarkan jumlah lat/lon dari prediksi
    new_lat = np.linspace(lat_values.min(), lat_values.max(), num_lat)
    new_lon = np.linspace(lon_values.min(), lon_values.max(), num_lon)

    # Simpan ke DataArray dengan koordinat baru
    predicted_ssh_da = xr.DataArray(
        predicted_ssh_squeezed,  # (240, num_lat, num_lon)
        coords={
            "time": future_ds_sliced.time.values,  # (240,)
            "latitude": new_lat,  # (num_lat,)
            "longitude": new_lon  # (num_lon,)
        },
        dims=["time", "latitude", "longitude"]
    )

    # Cek hasilnya
    print(predicted_ssh_da)

    predictions[scen] = predicted_ssh_da

    # Menghitung rata-rata tahunan
    predicted_ssh_annual = predicted_ssh_da.groupby("time.year").mean(dim="time")

    # Konversi ke Dataset dengan variabel 'zos'
    predicted_ssh_annual = predicted_ssh_annual.to_dataset(name="zos")

    # Pastikan koordinat 'j' dan 'i' bertipe int (jika ada)
    if "j" in predicted_ssh_annual.coords:
        predicted_ssh_annual = predicted_ssh_annual.assign_coords(j=("j", predicted_ssh_annual["j"].values.astype(int)))
    if "i" in predicted_ssh_annual.coords:
        predicted_ssh_annual = predicted_ssh_annual.assign_coords(i=("i", predicted_ssh_annual["i"].values.astype(int)))

    # Tambahkan atribut ke variabel 'zos'
    predicted_ssh_annual['zos'].attrs.update({
        'description': f'Predicted sea surface height from {fut1} to {fut2}',
        'units': 'meters',
        'standard_name': 'sea_surface_height',
        'long_name': 'Sea Surface Height Above Geoid',
        'coordinates': 'latitude longitude'
    })

    # Simpan ke NetCDF
    output_file = f"{outdir}predicted_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
    print(f"Saving NetCDF to: {output_file}")
    predicted_ssh_annual.to_netcdf(output_file)

# ==========================
# 8. Koreksi GCM Data
# ==========================

# # Persiapkan data GCM yang akan dikoreksi
# X_gcm_scaled = prepare_future_data(gcm_sliced, scaler_X)

# # Prediksi koreksi model
# corrected_gcm = model.predict(X_gcm_scaled)

# # Inverse transform
# corrected_gcm_rescaled = scaler_y.inverse_transform(
#     corrected_gcm.reshape(corrected_gcm.shape[0], -1)  # Harus 2D sebelum inverse_transform
# ).reshape(corrected_gcm.shape)  # Kembali ke bentuk asli

# # Simpan hasil koreksi ke NetCDF
# corrected_gcm_da = xr.DataArray(
#     corrected_gcm_rescaled.squeeze(),  # Hilangkan dimensi tambahan jika perlu
#     coords={
#         "time": gcm_sliced.time.values,
#         "latitude": ssh_sliced.latitude.values,
#         "longitude": ssh_sliced.longitude.values
#     },
#     dims=["time", "latitude", "longitude"]
# )

# corrected_gcm_da.to_netcdf(f"{outdir}corrected_gcm_{gcm_name}_{hist1}_{hist2}.nc")

# ==========================
# 9. Visualisasi dan Evaluasi
# ==========================
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="SSH (meters)")
    plt.show()

# Bandingkan hasil prediksi dengan target
plot_heatmap(y_test[0].squeeze(), "Ground Truth")
plot_heatmap(pred_ssh_scaled[0].squeeze(), "Prediction")

# Hitung metrik evaluasi
mae = mean_absolute_error(y_test.squeeze(), pred_ssh_scaled.squeeze())
rmse = np.sqrt(mean_squared_error(y_test.squeeze(), pred_ssh_scaled.squeeze()))
r2 = r2_score(y_test.squeeze().ravel(), pred_ssh_scaled.squeeze().ravel())

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")