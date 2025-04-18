import xarray as xr
import numpy as np
import os
import tensorflow as tf
import cftime
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, TimeDistributed, LSTM, UpSampling2D, Cropping2D
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

    # Konversi waktu ke numpy.datetime64
    time_values = data['time'].values
    min_time, max_time = np.datetime64(time_values.min(), 'D'), np.datetime64(time_values.max(), 'D')

    # Pastikan rentang waktu dalam dataset
    start_time, end_time = np.datetime64(time_range[0], 'D'), np.datetime64(time_range[1], 'D')
    if start_time < min_time or end_time > max_time:
        raise ValueError(f"Rentang waktu {time_range} berada di luar cakupan dataset ({min_time} - {max_time})")

    # Pilih waktu dengan metode "nearest" untuk data bulanan
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

    print(f"✅ Menggunakan '{detected_lat}' sebagai latitude dan '{detected_lon}' sebagai longitude.")

    # Filter berdasarkan lat/lon
    masked_data = sliced_data.where(
        (data[detected_lat] >= lat_range[0]) & (data[detected_lat] <= lat_range[1]) &
        (data[detected_lon] >= lon_range[0]) & (data[detected_lon] <= lon_range[1]),
        drop=False  # Jangan hapus time step, hanya isi dengan NaN
    ).dropna(dim="time", how="all")  # Hapus time step kosong akibat masking

    return masked_data

# Define variable names
gcm_name = 'miroc6'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'
ssh_var = 'zos'

# Rentang waktu (pastikan sesuai dengan dataset)
hist1, hist2 = '1995-01-16', '2014-12-16'   # GCM Historical (pertengahan bulan)
fut1, fut2 = '2015-01-16', '2100-12-16'     # GCM Future (pertengahan bulan)
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

gcm_min_time = np.datetime64(gcm_ds.time.min().values, 'D')
gcm_max_time = np.datetime64(gcm_ds.time.max().values, 'D')

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

    fut_min_time = np.datetime64(fut_ds.time.min().values, 'D')
    fut_max_time = np.datetime64(fut_ds.time.max().values, 'D')

    if (np.datetime64(fut1, 'D') >= fut_min_time) and (np.datetime64(fut2, 'D') <= fut_max_time):
        future_sliced[scen] = load_and_mask_dataset(fut_ds, gcm_var, lat_range, lon_range, (fut1, fut2))
        print(f"✅ Data {scen} berhasil di-slice!")
    else:
        print(f"⚠️ Rentang waktu {fut1} - {fut2} tidak tersedia dalam dataset {scen}! "
              f"Rentang dataset: {fut_min_time} - {fut_max_time}")
    
# Save to NetCDF
outdir = f"data/hasil/{gcm_name}/CNN-LSTM-AUTOREG/{fut1}_{fut2}/"
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

# CNN encoder untuk ekstraksi fitur spasial
def cnn_encoder(input_shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='linear', padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        GlobalAveragePooling2D()
    ])
    return model

# Hybrid model: CNN encoder + LSTM
def build_autoregressive_model(input_shape, output_shape, timesteps):
    height, width = output_shape[:2]  # output dari CNN-LSTM sebelum upscaling

    cnn = cnn_encoder(input_shape=(input_shape[1], input_shape[2], input_shape[3]))

    input_layer = Input(shape=(timesteps, input_shape[1], input_shape[2], input_shape[3]))

    x = TimeDistributed(cnn)(input_layer)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # Output low-res
    x = Dense(height * width, activation='linear')(x)
    x = Reshape((height, width, 1))(x)

    # Tambahkan upscaler
    x = UpSampling2D(size=(2, 2))(x)  # 160x220 → 320x440
    x = Conv2D(32, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    # Akhirnya crop/conv ke 301x660    
    x = tf.image.resize(x, size=(height, width), method="bilinear")
    x = Conv2D(1, (3, 3), padding='same', activation='linear')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

# Buat model dengan input dari X_train dan output sesuai y_train
model = build_autoregressive_model(X_train.shape[1:], y_train.shape[1:], timesteps=12)
model.summary()

# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=8,
          callbacks=[early_stopping, reduce_lr])

# Evaluasi model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Evaluasi Model CNN-LSTM-AUTOREG selesai! MSE: {loss:.4f}, MAE: {mae:.4f}")

# Save model
model.save(f"model/model_cnn_lstm_autoreg_{gcm_name}.keras")
print("💾 Model berhasil disimpan!")

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

def autoregressive_forecast(model, initial_input, total_months):
    current_input = initial_input
    predictions = []

    for i in range(total_months):
        # Ambil 12 timestep terakhir dari current_input
        input_for_prediction = current_input[:, -12:, :, :, :]

        # Prediksi 1 timestep ke depan
        pred_exp = model.predict(input_for_prediction, verbose=0)

        # TIDAK PERLU resize, model sudah output resolusi akhir (301, 660)
        predictions.append(pred_exp[0])  # shape: (301, 660, 1)

        # Perkecil prediksi agar bisa digabung ke current_input
        pred_downscaled = tf.image.resize(pred_exp, size=(current_input.shape[2], current_input.shape[3]), method="bilinear").numpy()

        print(f"Step {i} | Pred mean: {pred_downscaled.mean():.4f}, std: {pred_downscaled.std():.4f}")

        # Siapkan untuk iterasi berikutnya (pakai input low-res)
        pred_downscaled = np.expand_dims(pred_downscaled, axis=1)  # shape: (1, 1, 40, 55, 1)
        current_input = np.concatenate([current_input[:, 1:], pred_downscaled], axis=1)

    # Gabungkan semua prediksi: (total_months, 301, 660, 1)
    predictions = np.stack(predictions, axis=0)
    return predictions

# Prediksi future SSH dengan input resolusi rendah (future_ds)
predictions = {}

for scen, future_ds in future_sliced.items():
    print(f"\nProcessing scenario: {scen}")

    # initial input 12 bulan sebelumnya
    initial_input = X_train[-1:]  # Bentuknya (1, 12, height, width, channels)

    # Prediksi SSH dengan model CNN
    pred_ssh_scaled = autoregressive_forecast(model, initial_input, total_months=1032)

    # Periksa dimensi pred_ssh_scaled
    print(f"Shape of pred_ssh_scaled: {pred_ssh_scaled.shape}")

    # Cek apakah dimensi pertama berukuran 1 dan hanya hapus jika ukuran 1
    if pred_ssh_scaled.shape[0] == 1:
        pred_ssh_scaled = pred_ssh_scaled.squeeze(axis=0)

    # Hapus dimensi keempat (channel) jika perlu
    if pred_ssh_scaled.shape[-1] == 1:
        pred_ssh_scaled = pred_ssh_scaled.squeeze(axis=-1)

    # Ambil koordinat target dari dataset referensi
    lat_target = ssh_sliced['latitude'].values
    lon_target = ssh_sliced['longitude'].values

    # Tentukan koordinat latitude & longitude dari dataset referensi
    lat_name = [dim for dim in future_ds.coords if "lat" in dim.lower()]
    lon_name = [dim for dim in future_ds.coords if "lon" in dim.lower()]

    if not lat_name or not lon_name:
        print(f"❌ Latitude/Longitude tidak ditemukan dalam dataset {scen}!")
        print(f"Tersedia koordinat: {list(future_ds.coords.keys())}")
        raise ValueError(f"Koordinat latitude/lon tidak ditemukan dalam dataset {scen}!")

    lat_name = lat_name[0]
    lon_name = lon_name[0]

    # Memeriksa dimensi latitude dan longitude untuk interpolasi
    if len(lat_target) != pred_ssh_scaled.shape[1] or len(lon_target) != pred_ssh_scaled.shape[2]:
        print(f"⚠️ Dimensi latitude/lon tidak cocok! Interpolasi ulang diperlukan.")

        # Melakukan interpolasi dengan memastikan dimensi latitude dan longitude sesuai
        lat_interp = interp1d(np.linspace(0, 1, len(lat_target)), lat_target, kind="linear")(np.linspace(0, 1, pred_ssh_scaled.shape[1]))
        lon_interp = interp1d(np.linspace(0, 1, len(lon_target)), lon_target, kind="linear")(np.linspace(0, 1, pred_ssh_scaled.shape[2]))

        # Pastikan hasil interpolasi adalah array dengan dimensi yang sesuai
        lat_target = lat_interp
        lon_target = lon_interp
    
    print(f"Shape of pred_ssh_scaled: {pred_ssh_scaled.shape}")
    print(f"Shape of lat_target: {lat_target.shape}")
    print(f"Shape of lon_target: {lon_target.shape}")

    forecast_time = future_ds.time.values[:pred_ssh_scaled.shape[0]]

    # Gunakan lat_target dan lon_target langsung sebagai dimensi
    predicted_ssh_da = xr.DataArray(
        pred_ssh_scaled,
        coords={"time": forecast_time, "latitude": lat_target, "longitude": lon_target},
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
initial_input = prepare_data(gcm_sliced.isel(time=slice(0, 12)))  # (12, lat, lon)

# Tambahkan batch dan channel dimensi
initial_input = np.expand_dims(initial_input, axis=0)  # (1, 12, lat, lon)
initial_input = np.expand_dims(initial_input, axis=-1)  # (1, 12, lat, lon, 1)

# Forecast sepanjang periode
corrected_gcm = autoregressive_forecast(model, initial_input, total_months=240)

print("Shape corrected_gcm :", corrected_gcm.shape)

# Cek apakah dimensi pertama berukuran 1 dan hanya hapus jika ukuran 1
if corrected_gcm.shape[0] == 1:
    corrected_gcm = corrected_gcm.squeeze(axis=0)

# Hapus dimensi keempat (channel) jika perlu
if corrected_gcm.shape[-1] == 1:
    corrected_gcm = corrected_gcm.squeeze(axis=-1)

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

print(f"time: {gcm_sliced.time.shape} | lat: {len(lat_target)} | lon: {len(lon_target)}")
print("Final shape corrected_gcm:", corrected_gcm.shape)

corrected_time = gcm_sliced.time.values[:corrected_gcm.shape[0]]

# Simpan hasil koreksi ke NetCDF
corrected_gcm_da = xr.DataArray(
    corrected_gcm,
    coords={"time": corrected_time, "latitude": lat_target, "longitude": lon_target},
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