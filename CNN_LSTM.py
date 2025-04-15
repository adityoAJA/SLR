# =================
# 0. import library
# =================
import xarray as xr
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Reshape, TimeDistributed, GlobalAveragePooling2D, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 1. load and mask dataset
# ========================

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

# ======================
# 2. Parameter Input
# ======================

# Define variable names
gcm_name = 'miroc6'
indir = f'/SLR/data/{gcm_name}/'
gcm_var = 'zos'
ssh_var = 'zos'

# Rentang waktu (pastikan sesuai dengan dataset)
hist1, hist2 = '1995-01-16', '2014-12-16'   # GCM Historical (pertengahan bulan)
fut1, fut2 = '2015-01-16', '2034-12-16'     # GCM Future (pertengahan bulan)
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
outdir = f"data/hasil/{gcm_name}/{fut1}_{fut2}/"
os.makedirs(outdir, exist_ok=True)

if gcm_sliced is not None:
    gcm_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_historical_{hist1}_{hist2}.nc")
if ssh_sliced is not None:
    ssh_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_reanalysis_{hist1}_{hist2}.nc")

for scen, ds in future_sliced.items():
    ds.to_netcdf(f"{outdir}{gcm_name}_sliced_{scen}_{fut1}_{fut2}.nc")

# ======================
# 3. Pre-Processing Data
# ======================

# Ambil bagian yang beririsan 240 time step
print("gcm_sliced:", gcm_sliced.shape)
min_samples = min(gcm_sliced.shape[0], ssh_sliced.shape[0])
gcm_sliced = gcm_sliced[:min_samples]
ssh_sliced = ssh_sliced[:min_samples]

# Konversi ke numpy dan isi NaN
X = np.nan_to_num(gcm_sliced.values)  # (240, 40, 55)
y = np.nan_to_num(ssh_sliced.values)  # (240, 301, 660)

# Normalisasi
scaler_X = MinMaxScaler()
X_reshaped = X.reshape(X.shape[0], -1)
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_reshaped = y.reshape(y.shape[0], -1)
y_scaled = scaler_y.fit_transform(y_reshaped).reshape(y.shape)

# Tambahkan dimensi channel
X_scaled = X_scaled[..., np.newaxis]  # (240, 40, 55, 1)
y_scaled = y_scaled[..., np.newaxis]  # (240, 301, 660, 1)

# Tambahkan batch dimension → (1, 240, H, W, 1)
X_input = np.expand_dims(X_scaled, axis=0)
y_output = np.expand_dims(y_scaled, axis=0)

# Konversi ke float16
X_input = X_input.astype('float16')
y_output = y_output.astype('float16')

# Cek shape akhir
print("X_train shape sebelum sliding:", X_input.shape)  # (samples, 40, 55, 1)
print("y_train shape sebelum sliding:", y_output.shape)  # (samples, 301, 660, 1)

# ==========================
# 4. Build CNN+LSTM Model
# ==========================

def build_model(input_shape, output_shape):
    height, width = output_shape[:2]  # Ambil dimensi output

    model = Sequential([
        TimeDistributed(Conv2D(4, (3, 3), activation='relu', padding='same'), input_shape=input_shape),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(8, return_sequences=False),
        Dense(np.prod(output_shape), activation='linear'),
        Reshape(output_shape)
    ])

    return model  # <--- tambahkan ini

# ==========================
# 5. Train Model
# ==========================

input_shape = X_input.shape[1:]      # (240, 40, 55, 1)
output_shape = y_output.shape[1:]    # (240, 301, 660, 1)

model = build_model(input_shape, output_shape)
model.summary()

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# save model
model.save(f"model/model_conv2D_{gcm_name}.keras")

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(
        X_input, y_output, epochs=10, batch_size=1,
        callbacks=[early_stopping]
    )

# ========================
# 6. Prediction
# ========================

# Fungsi menyiapkan data masa depan
def prepare_future_data(future_ds, scaler_X):
    X_future = future_ds.values  # <- INI HARUS ADA SEBELUM DIPAKAI

    if X_future.ndim != 3:
        raise ValueError("Expected future_ds to have shape (time, lat, lon)")

    X_future = X_future[..., np.newaxis]  # (time, lat, lon, 1)

    num_features = X_future.shape[1] * X_future.shape[2]
    if num_features != scaler_X.n_features_in_:
        print("⚠️ Jumlah fitur berbeda! Menggunakan scaler baru.")
        scaler_X_new = MinMaxScaler()
        X_scaled = scaler_X_new.fit_transform(
            X_future.reshape(X_future.shape[0], -1)
        ).reshape(X_future.shape)
    else:
        X_scaled = scaler_X.transform(
            X_future.reshape(X_future.shape[0], -1)
        ).reshape(X_future.shape)
    
    return X_scaled  # shape: (time, lat, lon, 1)

# menyiapkan data koreksi
def prepare_cor_data(gcm_sliced, scaler):
    if len(gcm_sliced) == 0:
        print("No data")
        return None

    # Reshape ke 2D: (time, lat * lon) = (240, 2200)
    X_reshaped = gcm_sliced.values.reshape(gcm_sliced.shape[0], -1)

    # Transform dengan scaler
    X_scaled = scaler.transform(X_reshaped)

    # Kembalikan ke bentuk (time, lat, lon, 1)
    X_scaled_reshaped = X_scaled.reshape((gcm_sliced.shape[0], gcm_sliced.shape[1], gcm_sliced.shape[2], 1))

    return X_scaled_reshaped

# fungsi utilitas
def standardize_dims(ds):
    rename_dict = {}
    if "year" in ds.dims:
        rename_dict["year"] = "time"
    for lat_var in ["lat", "y", "j"]:
        if lat_var in ds.dims:
            rename_dict[lat_var] = "latitude"
    for lon_var in ["lon", "x", "i"]:
        if lon_var in ds.dims:
            rename_dict[lon_var] = "longitude"
    return ds.rename(rename_dict)

# Fungsi prediksi full-sequence
def predict_fullsequence(model, X_scaled, scaler_y):
    y_pred = model.predict(X_scaled[np.newaxis, ...])  # shape: (1, 240, lat, lon, 1)
    y_pred = y_pred[0]  # remove batch dim -> (240, lat, lon, 1)

    y_pred_reshaped = y_pred.reshape(y_pred.shape[0], -1)
    y_rescaled = scaler_y.inverse_transform(y_pred_reshaped)
    return y_rescaled.reshape(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2])

# Fungsi konversi ke array
def to_dataarray(pred, future_ds, ref_ds):
    lat_target = ref_ds["latitude"].values
    lon_target = ref_ds["longitude"].values

    lat_interp = interp1d(np.linspace(0, 1, len(lat_target)), lat_target)(np.linspace(0, 1, pred.shape[1]))
    lon_interp = interp1d(np.linspace(0, 1, len(lon_target)), lon_target)(np.linspace(0, 1, pred.shape[2]))

    return xr.DataArray(
        pred,
        coords={"time": future_ds.time.values, "latitude": lat_interp, "longitude": lon_interp},
        dims=["time", "latitude", "longitude"]
    )

# Fungsi Utama Prediksi
def run_predictions(future_sliced, model, scaler_X, scaler_y, ssh_sliced, gcm_name, fut1, fut2, outdir):
    predictions = {}

    for scen, future_ds in future_sliced.items():
        print(f"\n🔸 Processing scenario: {scen}")
        
        X_scaled = prepare_future_data(future_ds, scaler_X)
        pred_scaled = predict_fullsequence(model, X_scaled, scaler_y)
        pred_da = to_dataarray(pred_scaled, future_ds, ssh_sliced)

        predictions[scen] = pred_da

        # pred_annual = pred_da.groupby("time.year").mean("time")
        pred_annual = standardize_dims(pred_da).to_dataset(name="zos")

        # Tambahkan atribut NetCDF
        pred_annual['zos'].attrs.update({
            'description': f'Predicted sea surface height from {fut1} to {fut2}',
            'units': 'meters',
            'standard_name': 'sea_surface_height',
            'long_name': 'Sea Surface Height Above Geoid',
            'coordinates': 'latitude longitude'
        })

        # Simpan hasil prediksi
        file_out = f"{outdir}predicted_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
        pred_annual.to_netcdf(file_out)
        print(f"✅ Disimpan: {file_out}")

    print("✅ Semua skenario selesai diprediksi.")
    return predictions

# Fungsi Prediksi jangka panjang
def run_longterm_prediction(model, X_init_scaled, scaler_y, lat_target, lon_target, fut1, outdir, gcm_name):
    future_steps = 4
    all_predictions = []
    X_step = X_init_scaled.copy()

    for step in range(future_steps):
        print(f"🔁 Step {step+1}")
        pred_step = predict_fullsequence(model, X_step, scaler_y)
        all_predictions.append(pred_step)
        X_step = np.expand_dims(pred_step, axis=-1)

    final_pred = np.concatenate(all_predictions, axis=0)
    time_coords = pd.date_range(start=f"{fut1}-01", periods=final_pred.shape[0], freq="MS")

    final_da = xr.DataArray(
        final_pred,
        coords={"time": time_coords, "latitude": lat_target, "longitude": lon_target},
        dims=["time", "latitude", "longitude"]
    )

    final_annual = final_da.groupby("time.year").mean("time").to_dataset(name="zos")

    file_out = f"{outdir}predicted_ssh_{gcm_name}_all.nc"
    final_annual.to_netcdf(file_out)
    print(f"✅ Prediksi jangka panjang disimpan: {file_out}")
    return final_annual

# Jalankan prediksi
predicted = run_predictions(
    future_sliced=future_sliced,
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    ssh_sliced=ssh_sliced,
    gcm_name=gcm_name,
    fut1=fut1,
    fut2=fut2,
    outdir=outdir
)

# Jalankan prediksi jangka panjang

# ===================
# 7. Koreksi Data GCM
# ===================

# Function to reshape corrected data
def reshape_cor_data(corrected_output):
    # Remove batch & channel dim
    corrected_output = np.squeeze(corrected_output)  # (240, 301, 660)
    return corrected_output

# Pastikan gcm_sliced sudah memiliki bentuk yang sesuai
X_gcm_scaled = prepare_cor_data(gcm_sliced, scaler_X)
X_gcm_scaled = np.expand_dims(X_gcm_scaled, axis=0)  # (1, 240, 40, 55, 1)

# Prediksi dengan model
corrected_output = model.predict(X_gcm_scaled)  # (1, 240, 301, 660, 1)
corrected_gcm = reshape_cor_data(corrected_output)  # (240, 301, 660)

# Pastikan output memiliki shape yang sesuai (time, lat, lon)
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

# Simpan hasil koreksi ke NetCDF
corrected_gcm_da = xr.DataArray(
    corrected_gcm,
    coords={"time": gcm_sliced.time.values, "latitude": lat_target, "longitude": lon_target},
    dims=["time", "latitude", "longitude"]
)

# Menghitung rata-rata tahunan
# corrected_gcm_annual = corrected_gcm_da.groupby("time.year").mean(dim="time")

# Standarisasi dimensi corrected GCM
corrected_gcm_annual = standardize_dims(corrected_gcm_da)

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

# # # ========================
# # # 8. Validasi Manual
# # # ========================

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_predictions(true_data, pred_data_1, pred_data_2=None, timesteps=[0, 50, 100, 150, 200], vmin=None, vmax=None, title_1='Model 1', title_2='Model 2'):
#     """
#     true_data     : shape (T, H, W)
#     pred_data_1   : shape (T, H, W)
#     pred_data_2   : shape (T, H, W), optional
#     timesteps     : list of int
#     vmin, vmax    : color range for consistent comparison
#     """
#     num_cols = 3 if pred_data_2 is not None else 2
#     fig, axes = plt.subplots(len(timesteps), num_cols, figsize=(4*num_cols, 3*len(timesteps)))
    
#     for i, t in enumerate(timesteps):
#         if pred_data_2 is not None:
#             axs = axes[i]
#         else:
#             axs = axes[i] if len(timesteps) > 1 else axes
            
#         # Observasi
#         im0 = axs[0].imshow(true_data[t], cmap='viridis', vmin=vmin, vmax=vmax)
#         axs[0].set_title(f"Ground Truth\nTimestep {t}")
#         axs[0].axis('off')
        
#         # Prediksi Model 1
#         im1 = axs[1].imshow(pred_data_1[t], cmap='viridis', vmin=vmin, vmax=vmax)
#         axs[1].set_title(f"{title_1}\nTimestep {t}")
#         axs[1].axis('off')

#         # Prediksi Model 2 (jika ada)
#         if pred_data_2 is not None:
#             im2 = axs[2].imshow(pred_data_2[t], cmap='viridis', vmin=vmin, vmax=vmax)
#             axs[2].set_title(f"{title_2}\nTimestep {t}")
#             axs[2].axis('off')

#     plt.tight_layout()
#     plt.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6)
#     plt.show()

# # X_test: shape (N, T, H, W, 1)
# # y_true: shape (N, T, H, W, 1)

# # Ambil sample pertama
# y_true_sample = y_true[0, ..., 0]
# y_pred_model1 = model1.predict(X_test[0:1])[0, ..., 0]
# y_pred_model2 = model2.predict(X_test[0:1])[0, ..., 0]  # Jika ada

# plot_predictions(
#     true_data=y_true_sample,
#     pred_data_1=y_pred_model1,
#     pred_data_2=y_pred_model2,
#     timesteps=[0, 50, 100, 150, 200],
#     vmin=np.min(y_true_sample),
#     vmax=np.max(y_true_sample),
#     title_1="CNN+LSTM",
#     title_2="ConvLSTM"
# )
