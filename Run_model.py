import xarray as xr
import numpy as np
import pandas as pd
import os
import cftime
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape
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

    # Seleksi waktu dengan nearest (untuk mendapatkan indeks waktu di dataset)
    try:
        start_time_sel = data.time.sel(time=start_time, method="nearest").values
        end_time_sel = data.time.sel(time=end_time, method="nearest").values
    except Exception as e:
        raise ValueError(f"Gagal melakukan seleksi waktu: {e}")

    # Slicing data
    sliced_data = data[var_name].sel(time=slice(start_time_sel, end_time_sel))

    if sliced_data.size == 0:
        raise ValueError(f"Data kosong setelah slicing waktu {time_range}")

    # Deteksi nama variabel lat/lon
    lat_names = ["lat", "latitude", "j", "y"]
    lon_names = ["lon", "longitude", "i", "x"]
    detected_lat = next((lat for lat in lat_names if lat in data.dims), None)
    detected_lon = next((lon for lon in lon_names if lon in data.dims), None)

    if detected_lat is None or detected_lon is None:
        raise ValueError("Dimensi latitude dan longitude tidak ditemukan dalam dataset.")

    print(f"✅ Menggunakan '{detected_lat}' sebagai latitude dan '{detected_lon}' sebagai longitude.")

    # Masking berdasarkan lat/lon
    masked_data = sliced_data.where(
        (data[detected_lat] >= lat_range[0]) & (data[detected_lat] <= lat_range[1]) &
        (data[detected_lon] >= lon_range[0]) & (data[detected_lon] <= lon_range[1]),
        drop=False  # Jangan hapus time step, hanya isi dengan NaN
    ).dropna(dim="time", how="all")  # Hapus time step kosong akibat masking

    return masked_data

def to_np_datetime64_safe(t):
    """Konversi waktu (termasuk cftime) ke numpy.datetime64."""
    try:
        return np.datetime64(str(t), 'D')
    except Exception:
        return np.datetime64(t, 'D')

# Define variable names
gcm_name = 'mpi'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'
ssh_var = 'zos'

# Rentang waktu (pastikan sesuai dengan dataset)
hist1, hist2 = '1995-01-16', '2014-12-16'   # GCM Historical (pertengahan bulan)
fut1, fut2 = '2021-01-16', '2040-12-16'     # GCM Future (pertengahan bulan)
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
outdir = f"data/hasil/{gcm_name}/{fut1}_{fut2}/"
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
y = np.nan_to_num(ssh_sliced.values)

# Normalisasi X
scaler_X = MinMaxScaler()
X_reshaped = X.reshape(X.shape[0], -1)
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)

# Normalisasi y
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_reshaped = y.reshape(y.shape[0], -1)
y_scaled = scaler_y.fit_transform(y_reshaped).reshape(y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Tambahkan dimensi channel untuk CNN
X_train = X_train[..., np.newaxis]  # shape: (samples, height, width, 1)
X_test = X_test[..., np.newaxis]
y_train = y_train[..., np.newaxis]  # shape: (samples, height, width, 1)
y_test = y_test[..., np.newaxis]

# Cek shape akhir
print("X_train shape:", X_train.shape)  # (samples, 40, 55, 1)
print("y_train shape:", y_train.shape)  # (samples, 301, 660, 1)

print(f"GCM sliced time: {gcm_sliced.time.shape}")  # Harusnya (240,)
print(f"SSH sliced time: {ssh_sliced.time.shape}")  # Harusnya (240,)
for scen in future_sliced:
    print(f"Future {scen} sliced time: {future_sliced[scen].time.shape}")  # Harusnya (240,)

print(f"Hist range: {hist1} - {hist2}")  # Harusnya 1995-01-01 sampai 2014-12-01
print(f"Min Time GCM: {gcm_ds.time.min().values}, Max Time GCM: {gcm_ds.time.max().values}")

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

# build model
def build_high_res_cnn(input_shape, output_shape):
    height, width = output_shape[:2]  # Ambil dimensi output

    model = Sequential([
        Conv2D(32, (3, 3), activation='linear', padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(height * width, activation='linear'),
        Reshape((height, width, 1))
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

# **Buat model dengan input dari X_train dan output sesuai y_train**
model = build_high_res_cnn(X_train.shape[1:], y_train.shape[1:])
model.summary()

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr],
        verbose=1)

# Evaluasi model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Evaluasi selesai! MSE: {loss:.4f}, MAE: {mae:.4f}")

# save model
model.save(f"model/model_FCNN_{gcm_name}.keras")
print("💾 Model berhasil disimpan!")