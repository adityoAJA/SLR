import xarray as xr
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cftime
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential, load_model
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, TimeDistributed, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Function to load and mask dataset
def load_and_mask_dataset(data, var_name, lat_range, lon_range, time_range):
    """Load, slice, and mask dataset based on the given variable and spatial-temporal range."""
    if var_name not in data.variables:
        raise ValueError(f"Variable '{var_name}' tidak ditemukan dalam dataset. "
                         f"Variabel yang tersedia: {list(data.variables.keys())}")

    if 'time' not in data.dims:
        raise ValueError("Dimensi 'time' tidak ditemukan dalam dataset.")

    if data[var_name].size == 0:
        raise ValueError(f"Dataset tidak memiliki data untuk variabel '{var_name}'.")

    # Ambil array waktu
    time_values = data['time'].values

    # Konversi waktu input ke np.datetime64 dengan resolusi harian
    start_time = np.datetime64(time_range[0], 'D')
    end_time = np.datetime64(time_range[1], 'D')

    # Konversi waktu dataset ke np.datetime64 juga
    min_time = np.datetime64(str(time_values.min()), 'D')
    max_time = np.datetime64(str(time_values.max()), 'D')

    # 3. Validasi time range
    time_type = type(data.time.values[0])

    # Konversi rentang waktu input (string) ke format yang sesuai kalender data
    if issubclass(time_type, cftime.DatetimeNoLeap) or issubclass(time_type, cftime.DatetimeGregorian):
        # Hapus jam/menit/detik
        dt1 = pd.to_datetime(time_range[0]).replace(hour=0, minute=0, second=0)
        dt2 = pd.to_datetime(time_range[1]).replace(hour=0, minute=0, second=0)
        start_time = time_type(dt1.year, dt1.month, dt1.day, 12)
        end_time = time_type(dt2.year, dt2.month, dt2.day, 12)
    else:
        # Konversi langsung ke datetime64[D]
        start_time = np.datetime64(time_range[0], 'D')
        end_time = np.datetime64(time_range[1], 'D')

    # Validasi waktu dalam rentang
    time_values = data['time'].values

    # Deteksi apakah waktu dalam bentuk cftime
    time_type = type(time_values[0])
    is_cftime = 'cftime' in str(time_type).lower()

    # Konversi waktu input ke pandas Timestamp dan ambil .date()
    dt_start = pd.to_datetime(time_range[0]).date()
    dt_end = pd.to_datetime(time_range[1]).date()

    # Konversi min/max waktu dataset dan ambil .date()
    if is_cftime:
        min_time = pd.to_datetime(str(time_values.min())).date()
        max_time = pd.to_datetime(str(time_values.max())).date()
    else:
        min_time = pd.to_datetime(time_values.min()).date()
        max_time = pd.to_datetime(time_values.max()).date()

    # Validasi rentang waktu tanpa jam/menit
    if dt_start < min_time or dt_end > max_time:
        raise ValueError(f"Waktu di luar rentang data: {dt_start} to {dt_end} vs {min_time} to {max_time}")

     # Seleksi waktu
    try:
        start_time_sel = data.time.sel(time=start_time, method="nearest").values
        end_time_sel = data.time.sel(time=end_time, method="nearest").values
    except Exception as e:
        raise ValueError(f"Gagal melakukan seleksi waktu: {e}")
    
    # print("Start time:", start_time_sel)
    # print("End time:", end_time_sel)

    # Slicing data berdasarkan waktu
    sliced_data = data[var_name].sel(time=slice(start_time_sel, end_time_sel))

    if sliced_data.size == 0:
        raise ValueError(f"Data kosong setelah slicing waktu {time_range}")
    
    # # Tampilkan data setelah slicing
    # print(f"Data setelah slicing (time): {sliced_data}")
    # print(f"Dimensi data setelah slicing: {sliced_data.shape}")
    # print(f"Isi data setelah slicing (contoh): {sliced_data.values[:5]}")
    
    # Deteksi nama variabel lat/lon
    lat_names = ["lat", "latitude", "j", "y"]
    lon_names = ["lon", "longitude", "i", "x"]

    detected_lat = next((lat for lat in lat_names if lat in data.dims), None)
    detected_lon = next((lon for lon in lon_names if lon in data.dims), None)

    if detected_lat is None or detected_lon is None:
        raise ValueError("Dimensi latitude dan longitude tidak ditemukan dalam dataset.")

    # print(f"✅ Menggunakan '{detected_lat}' sebagai latitude dan '{detected_lon}' sebagai longitude.")

    # Masking berdasarkan koordinat lat/lon
    masked_data = sliced_data.where(
        (data[detected_lat] >= lat_range[0]) & (data[detected_lat] <= lat_range[1]) & 
        (data[detected_lon] >= lon_range[0]) & (data[detected_lon] <= lon_range[1]), 
        drop=False
    ).dropna(dim="time", how="all")

    # print(f"Data setelah masking: {masked_data}")
    # print(f"Dimensi data setelah masking: {masked_data.shape}")
    # print(f"Jumlah data yang valid setelah masking: {masked_data.count()}")

    if masked_data.size == 0:
        raise ValueError(f"Data kosong setelah masking berdasarkan rentang waktu dan koordinat.")

    return masked_data

def to_np_datetime64_safe(t):
    """Konversi waktu (termasuk cftime) ke numpy.datetime64."""
    try:
        return np.datetime64(str(t), 'D')
    except Exception:
        return np.datetime64(t, 'D')

# Define variable names
gcm_name = 'canesm5'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'
ssh_var = 'zos'

# Rentang waktu (pastikan sesuai dengan dataset)
hist1, hist2 = '1995-01-16', '2014-12-16'   # GCM Historical (pertengahan bulan)
fut1, fut2 = '2081-01-16', '2100-12-16'     # GCM Future (pertengahan bulan)
ssh1, ssh2 = '1995-01-01', '2014-12-01'  # SSH (awal bulan)

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

# Load GCM Historical
print("📂 Membuka dataset GCM historical...")
gcm_ds = xr.open_dataset(gcm_data, engine="netcdf4", decode_times=True)

gcm_min_time = to_np_datetime64_safe(gcm_ds.time.min().values)
gcm_max_time = to_np_datetime64_safe(gcm_ds.time.max().values)

if (np.datetime64(hist1, 'D') >= gcm_min_time) and (np.datetime64(hist2, 'D') <= gcm_max_time):
    gcm_sliced = load_and_mask_dataset(gcm_ds, gcm_var, lat_range, lon_range, (hist1, hist2))
    print("✅ Data historis berhasil di-slice!")
else:
    print(f"⚠️ Rentang waktu yang diminta ({hist1} - {hist2}) tidak ada dalam dataset! "
          f"Rentang dataset: {gcm_min_time} - {gcm_max_time}")
    gcm_sliced = None

# Load SSH Data
print("📂 Membuka dataset SSH...")
ssh_ds = xr.open_dataset(ssh_data, engine="netcdf4", decode_times=True)
ssh_sliced = load_and_mask_dataset(ssh_ds, ssh_var, lat_range, lon_range, (ssh1, ssh2))

# Load Future Scenarios
future_sliced = {}
for scen, fpath in future_data.items():
    print(f"📂 Membuka dataset {scen}...")
    fut_ds = xr.open_dataset(fpath, engine="netcdf4", decode_times=True)

    fut_min_time = to_np_datetime64_safe(fut_ds.time.min().values)
    fut_max_time = to_np_datetime64_safe(fut_ds.time.max().values)

    if (np.datetime64(fut1, 'D') >= fut_min_time) and (np.datetime64(fut2, 'D') <= fut_max_time):
        future_sliced[scen] = load_and_mask_dataset(fut_ds, gcm_var, lat_range, lon_range, (fut1, fut2))
        print(f"✅ Data {scen} berhasil di-slice!")
    else:
        print(f"⚠️ Rentang waktu {fut1} - {fut2} tidak tersedia dalam dataset {scen}! "
              f"Rentang dataset: {fut_min_time} - {fut_max_time}")
    
# Save to NetCDF
outdir = f"data/hasil/{gcm_name}/CNN-LSTM/{fut1}_{fut2}/"
os.makedirs(outdir, exist_ok=True)

if gcm_sliced is not None:
    gcm_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_historical_{hist1}_{hist2}.nc")
if ssh_sliced is not None:
    ssh_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_reanalysis_{hist1}_{hist2}.nc")

for scen, ds in future_sliced.items():
    ds.to_netcdf(f"{outdir}{gcm_name}_sliced_{scen}_{fut1}_{fut2}.nc")

# Pastikan jumlah sample X dan y sama otomatis
min_samples = min(gcm_sliced.shape[0], ssh_sliced.shape[0])

# Ambil hanya min_samples pertama jika jumlahnya berbeda
if gcm_sliced.shape[0] != ssh_sliced.shape[0]:
    gcm_sliced = gcm_sliced[:min_samples]
    ssh_sliced = ssh_sliced[:min_samples]

# Set seed untuk hasil yang konsisten
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Konversi NaN menjadi angka agar tidak error
X = np.nan_to_num(gcm_sliced.values)
y = np.nan_to_num(ssh_sliced.values)  # Tetap dalam ukuran asli (301x660)

# Normalisasi X
scaler_X = MinMaxScaler()
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten dulu
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)  # Kembali ke shape aslinya

# Normalisasi y
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_reshaped = y.reshape(y.shape[0], -1)  # Flatten
y_scaled = scaler_y.fit_transform(y_reshaped).reshape(y.shape)  # Kembali ke ukuran (301x660)

def create_sequences(X, y, timesteps):
    X_seq = []
    y_seq = []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i])   # ambil window
        y_seq.append(y[i])                 # prediksi next timestep
    return np.array(X_seq), np.array(y_seq)

# Buat sequence untuk hybrid CNN-LSTM
timesteps = 12  # atau sesuaikan
X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)

# Tambahkan channel dimensi terakhir: (samples, timesteps, h, w, 1)
X_seq = X_seq[..., np.newaxis]
y_seq = y_seq[..., np.newaxis]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# # CNN encoder untuk ekstraksi fitur spatial
# def cnn_encoder(input_shape): # (16,32,64,128)-> klo gcm hires, (32,64,128,256)-> klo gcm low res
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='linear', padding='same', input_shape=input_shape),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),

#         Conv2D(64, (3, 3), activation='linear', padding='same'),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),

#         Conv2D(128, (3, 3), activation='linear', padding='same'),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),

#         Conv2D(256, (3, 3), activation='linear', padding='same'),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),

#         GlobalAveragePooling2D()
#     ])
#     return model

# def build_hybrid_model(input_shape, output_shape, timesteps):
#     height, width = output_shape[:2]

#     # CNN encoder
#     cnn = cnn_encoder(input_shape=(input_shape[1], input_shape[2], input_shape[3]))  # Sesuaikan CNN input shape

#     # Input shape: (timesteps, height, width, channels)
#     input_layer = Input(shape=(timesteps, input_shape[1], input_shape[2], input_shape[3]))  # Pastikan input shape sesuai
    
#     # Apply CNN to each timestep
#     x = TimeDistributed(cnn)(input_layer)
    
#     # Temporal modeling
#     x = LSTM(64, return_sequences=False)(x)  # Pastikan return_sequences=False karena output hanya untuk timestep terakhir
#     x = Dropout(0.3)(x)
    
#     # Output layer: reshape ke citra
#     x = Dense(height * width, activation='linear')(x)
#     output_layer = Reshape((height, width, 1))(x)

#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
#     return model

# # Buat model dengan input dari X_train dan output sesuai y_train
# model = build_hybrid_model(X_train.shape[1:], y_train.shape[1:], timesteps=12)  # Pastikan timesteps sesuai
# model.summary()

# # Training
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# history = model.fit(X_train, y_train,
#           validation_data=(X_test, y_test),
#           epochs=50,
#           batch_size=8,
#           callbacks=[early_stopping, reduce_lr])

# # Evaluasi model
# loss, mae = model.evaluate(X_test, y_test)
# print(f"✅ Evaluasi Model CNN-LSTM selesai! MSE: {loss:.4f}, MAE: {mae:.4f}")

# # Save model
# model.save(f"model/model_cnn_lstm_{gcm_name}.keras")
# print("💾 Model berhasil disimpan!")

model = load_model(f"model/model_cnn_lstm_{gcm_name}.keras") # --> buka pagar ini jika sudah ada model yg dilatih, dan pagar build model diatas
print("💾 Model berhasil diload!")

# Function to reshape corrected data
def reshape_cor_data(X_cor_scaled, X_cor):
    """
    Menyesuaikan shape berdasarkan resolusi dataset yang digunakan untuk corrected data.
    """
    num_samples = X_cor.shape[0]  # Pastikan dimensi pertama adalah waktu
    height, width = X_cor.shape[1], X_cor.shape[2]  # Ambil ukuran spasial

    # Hitung ukuran yang diharapkan
    expected_size = num_samples * height * width
    actual_size = X_cor_scaled.size

    # Debugging: Print bentuk array sebelum reshape
    print(f"Shape X_cor: {X_cor.shape}")
    print(f"Shape X_cor_scaled sebelum reshape: {X_cor_scaled.shape}")
    print(f"Expected size: {expected_size}, Actual size: {actual_size}")

    if expected_size != actual_size:
        raise ValueError(f"Mismatch: Expected {expected_size} elements, but got {actual_size}")

    # Reshape ke bentuk yang benar
    return X_cor_scaled.reshape(num_samples, height, width, 1)

# Function untuk menyiapkan data future_ds tanpa mereshape
def prepare_data(future_ds):
    # Konversi NaN menjadi 0 agar bisa diproses oleh model
    X_future = np.nan_to_num(future_ds.values)

    # Pastikan memiliki shape (samples, height, width, 1) agar cocok dengan model
    X_future = X_future[..., np.newaxis]  # Menambahkan dimensi channel terakhir

    # Normalisasi data menggunakan scaler yang sudah dipakai untuk pelatihan
    num_features = X_future.shape[1] * X_future.shape[2]
    
    if num_features != scaler_X.n_features_in_:
        print(f"⚠️ Jumlah fitur berbeda! Menggunakan scaler baru.")
        scaler_X_new = MinMaxScaler()
        X_future_scaled = scaler_X_new.fit_transform(X_future.reshape(X_future.shape[0], -1)).reshape(X_future.shape)
    else:
        X_future_scaled = scaler_X.transform(X_future.reshape(X_future.shape[0], -1)).reshape(X_future.shape)

    return X_future_scaled

# Pastikan semua dataset memiliki dimensi "time", "latitude", dan "longitude"
def standardize_dims(ds):
    rename_dict = {}
    
    # Pastikan 'time' digunakan sebagai dimensi waktu
    if "year" in ds.dims:
        rename_dict["year"] = "time"
    
    # Pastikan 'latitude' digunakan sebagai dimensi spatial Y
    for lat_var in ["lat", "y", "j"]:
        if lat_var in ds.dims:
            rename_dict[lat_var] = "latitude"
    
    # Pastikan 'longitude' digunakan sebagai dimensi spatial X
    for lon_var in ["lon", "x", "i"]:
        if lon_var in ds.dims:
            rename_dict[lon_var] = "longitude"
    
    # Lakukan rename jika diperlukan
    ds = ds.rename(rename_dict)

    return ds

# Prediksi future SSH dengan input resolusi rendah (future_ds)
predictions = {}

for scen, future_ds in future_sliced.items():
    print(f"\nProcessing scenario: {scen}")

    # Persiapkan data future_ds
    X_future_scaled = prepare_data(future_ds)

    # Ambil koordinat target dari dataset referensi
    lat_target = ssh_sliced['latitude'].values
    lon_target = ssh_sliced['longitude'].values

    # tambahkan newaxis
    X_future_scaled_expanded = np.expand_dims(X_future_scaled, axis=1)

    # Kita perlu memperbanyak dimensi waktu untuk mendapatkan bentuk yang sesuai (samples, timesteps, 40, 55, 1)
    X_future_scaled_expanded = np.repeat(X_future_scaled_expanded, timesteps, axis=1)

    # Prediksi SSH dengan model CNN
    pred_ssh_scaled = model.predict(X_future_scaled_expanded)

    # Hapus dimensi keempat (channel) agar sesuai dengan (time, height, width)
    pred_ssh_scaled = pred_ssh_scaled.squeeze(axis=-1)  

    # Tentukan koordinat latitude & longitude dari dataset referensi
    lat_name = [dim for dim in future_ds.coords if "lat" in dim.lower()]
    lon_name = [dim for dim in future_ds.coords if "lon" in dim.lower()]

    if not lat_name or not lon_name:
        print(f"❌ Latitude/Longitude tidak ditemukan dalam dataset {scen}!")
        print(f"Tersedia koordinat: {list(future_ds.coords.keys())}")
        raise ValueError(f"Koordinat latitude/lon tidak ditemukan dalam dataset {scen}")

    lat_name = lat_name[0]
    lon_name = lon_name[0]

    if len(lat_target) != pred_ssh_scaled.shape[1] or len(lon_target) != pred_ssh_scaled.shape[2]:
        print(f"⚠️ Dimensi latitude/lon tidak cocok! Interpolasi ulang diperlukan.")

    lat_interp = interp1d(np.linspace(0, 1, len(lat_target)), lat_target, kind="linear")(np.linspace(0, 1, pred_ssh_scaled.shape[1]))
    lon_interp = interp1d(np.linspace(0, 1, len(lon_target)), lon_target, kind="linear")(np.linspace(0, 1, pred_ssh_scaled.shape[2]))

    lat_target = lat_interp
    lon_target = lon_interp

    # Gunakan lat_target dan lon_target langsung sebagai dimensi
    predicted_ssh_da = xr.DataArray(
        pred_ssh_scaled,
        coords={"time": future_ds.time.values, "latitude": lat_target, "longitude": lon_target},
        dims=["time", "latitude", "longitude"]
    )

    # Simpan hasil prediksi dalam dictionary
    predictions[scen] = predicted_ssh_da

    # Menghitung rata-rata tahunan
    predicted_ssh_annual = predicted_ssh_da.groupby("time.year").mean(dim="time")

    # Standarisasi dimensi predicted SSH
    predicted_ssh_annual = standardize_dims(predicted_ssh_annual)

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
    # Set units dan calendar ke time
    predicted_ssh_annual["time"].attrs["units"] = f"days since {fut1}-01-01"
    predicted_ssh_annual["time"].attrs["calendar"] = "standard"

    # Simpan sebagai NetCDF
    pred_name = f"{outdir}predicted_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
    predicted_ssh_annual.to_netcdf(pred_name)

print("Predictions saved for all scenarios.")

# CORRECTED GCM ##

# Pastikan gcm_sliced sudah memiliki bentuk yang sesuai
X_gcm_scaled = prepare_data(gcm_sliced)

# tambahkan newaxis
X_gcm_scaled_expanded = np.expand_dims(X_gcm_scaled, axis=1)

# Kita perlu memperbanyak dimensi waktu untuk mendapatkan bentuk yang sesuai (samples, timesteps, 40, 55, 1)
X_gcm_scaled_expanded = np.repeat(X_gcm_scaled_expanded, timesteps, axis=1)

# Prediksi dengan model
corrected_gcm = model.predict(X_gcm_scaled_expanded)

# Pastikan output memiliki shape yang sesuai (time, lat, lon)
corrected_gcm = corrected_gcm.squeeze(axis=-1)  # Hilangkan dimensi terakhir jika ada

# Ambil koordinat target dari dataset referensi
lat_target = ssh_sliced['latitude'].values
lon_target = ssh_sliced['longitude'].values

# Tentukan koordinat latitude & longitude dari dataset referensi
lat_name = [dim for dim in gcm_sliced.coords if "lat" in dim.lower()]
lon_name = [dim for dim in gcm_sliced.coords if "lon" in dim.lower()]

if not lat_name or not lon_name:
    print(f"❌ Latitude/Longitude tidak ditemukan dalam dataset {scen}!")
    print(f"Tersedia koordinat: {list(gcm_sliced.coords.keys())}")
    raise ValueError(f"Koordinat latitude/lon tidak ditemukan dalam dataset {scen}")

lat_name = lat_name[0]
lon_name = lon_name[0]

if len(lat_target) != corrected_gcm.shape[1] or len(lon_target) != corrected_gcm.shape[2]:
    print(f"⚠️ Dimensi latitude/lon tidak cocok! Interpolasi ulang diperlukan.")

lat_interp = interp1d(np.linspace(0, 1, len(lat_target)), lat_target, kind="linear")(np.linspace(0, 1, corrected_gcm.shape[1]))
lon_interp = interp1d(np.linspace(0, 1, len(lon_target)), lon_target, kind="linear")(np.linspace(0, 1, corrected_gcm.shape[2]))

lat_target = lat_interp
lon_target = lon_interp

# Simpan hasil koreksi ke NetCDF
corrected_gcm_da = xr.DataArray(
    corrected_gcm,
    coords={"time": gcm_sliced.time.values, "latitude": lat_target, "longitude": lon_target},
    dims=["time", "latitude", "longitude"]
)

# Menghitung rata-rata tahunan
corrected_gcm_annual = corrected_gcm_da.groupby("time.year").mean(dim="time")

# Standarisasi dimensi corrected GCM
corrected_gcm_annual = standardize_dims(corrected_gcm_annual)

# Konversi ke Dataset dengan variabel 'zos'
corrected_gcm_annual = corrected_gcm_annual.to_dataset(name="zos")

# Pastikan koordinat 'j' dan 'i' bertipe int (jika ada)
if "j" in corrected_gcm_annual.coords:
    corrected_gcm_annual = corrected_gcm_annual.assign_coords(j=("j", corrected_gcm_annual["j"].values.astype(int)))
if "i" in corrected_gcm_annual.coords:
    corrected_gcm_annual = corrected_gcm_annual.assign_coords(i=("i", corrected_gcm_annual["i"].values.astype(int)))

# Tambahkan atribut ke variabel 'zos'
corrected_gcm_annual['zos'].attrs.update({
    'description': f'Corrected sea surface height from {hist1} to {hist2}',
    'units': 'meters',
    'standard_name': 'sea_surface_height',
    'long_name': 'Sea Surface Height Above Geoid',
    'coordinates': 'latitude longitude'
})

# Simpan sebagai NetCDF
out_file = f"{outdir}corrected_gcm_{gcm_name}_{hist1}_{hist2}.nc"
corrected_gcm_da.to_netcdf(out_file)
print(f"✅ Hasil koreksi disimpan: {out_file}")

# ========== LOSS CURVE ==========
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['loss'], linestyle='--', alpha=0.7, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("📉 Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== MAE CURVE ==========
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_mae'], label='Val MAE')
plt.plot(history.history['mae'], linestyle='--', alpha=0.7, label='Train MAE')
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("📏 MAE Curve Comparison per Model Version")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()