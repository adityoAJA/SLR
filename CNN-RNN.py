import numpy as np
import pandas as pd
import os
import cv2
import cftime
import matplotlib.pyplot as plt
import xarray as xr
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, LSTM, Dense, Reshape, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, TimeDistributed, Flatten, Dense, Reshape, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

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

    # ========================
    # 3. Validasi time range
    # ========================
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

# ======================
# 2. Parameter Input
# ======================

# Define variable names
gcm_name = 'canesm5'
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
panjang_seq = 24
panjang_pred = 1008

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
outdir = f"data/hasil/{gcm_name}/{fut1}_{panjang_seq}_{panjang_pred}/"
os.makedirs(outdir, exist_ok=True)

if gcm_sliced is not None:
    gcm_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_historical_{hist1}_{hist2}.nc")
if ssh_sliced is not None:
    ssh_sliced.to_netcdf(f"{outdir}{gcm_name}_sliced_reanalysis_{hist1}_{hist2}.nc")

for scen, ds in future_sliced.items():
    ds.to_netcdf(f"{outdir}{gcm_name}_sliced_{scen}_{fut1}_{panjang_seq}_{panjang_pred}.nc")

# Auto-detect shape
n_time = gcm_sliced.shape[0]
low_res_shape = gcm_sliced.shape[1:]   # misal (40, 55)
high_res_shape = ssh_sliced.shape[1:]  # misal (301, 660)

print(f"Input (GCM): {low_res_shape}, Output (OBS): {high_res_shape}")

# ===================== #
# 3. Data Preparation   #
# ===================== #

def create_sequences(x_data, y_data, seq_length):
    x_seq, y_seq = [], []
    for i in range(len(x_data) - seq_length):
        x_seq.append(x_data[i:i + seq_length])
        y_seq.append(y_data[i + seq_length])
    return np.array(x_seq), np.array(y_seq)

def reshape_input_for_model(future_input, sequence_length):
    sequences = []
    for i in range(len(future_input) - sequence_length + 1):
        seq = future_input[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# ===================== #
# 4. Model Architecture #
# ===================== #

def cnn_rnn_v1(input_shape, output_shape):
    inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Dropout(0.3)(x)

    x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(x)

    # ✅ Dense intermediate untuk stabilisasi
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    flat_output_size = output_shape[0] * output_shape[1]
    x = Dense(flat_output_size, activation='linear')(x)  # Bisa diganti 'relu' kalau target >= 0
    outputs = Reshape((*output_shape, 1))(x)

    return Model(inputs, outputs)

def cnn_rnn_v2(input_shape, output_shape):
    inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Dropout(0.4)(x)

    x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(x)

    # ✅ Dense intermediate stabilisasi
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    flat_output_size = output_shape[0] * output_shape[1]
    x = Dense(flat_output_size, activation='linear')(x)
    outputs = Reshape((*output_shape, 1))(x)

    return Model(inputs, outputs)

def cnn_rnn_v3(input_shape, output_shape):
    inputs = Input(shape=input_shape)

    # CNN TimeDistributed
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)  # Hasil: (batch, time, features)

    # Temporal modeling
    x = LSTM(64, return_sequences=False)(x)

    # Dense & reshape output
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    flat_output_size = output_shape[0] * output_shape[1]
    x = Dense(flat_output_size)(x)
    outputs = Reshape((*output_shape, 1))(x)

    return Model(inputs, outputs)

def masked_mse(mask):
    mask = mask.astype(np.float32)

    def loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        mask_tensor = tf.convert_to_tensor(mask)  # (H, W, 1)
        mask_tensor = tf.expand_dims(mask_tensor, axis=0)  # (1, H, W, 1)
        mask_tensor = tf.tile(mask_tensor, [batch_size, 1, 1, 1])  # (B, H, W, 1)

        masked_diff = (y_true - y_pred) * mask_tensor
        squared_error = tf.square(masked_diff)
        sum_squared_error = tf.reduce_sum(squared_error)
        num_valid = tf.reduce_sum(mask_tensor)  # total valid pixels * batch size
        return sum_squared_error / (num_valid + K.epsilon())

    return loss

# =============================== #
# 5. Prediction Utility Functions #
# =============================== #

def predict_future_sequence_autoregressive(
    future_input_seq, model, seq_length, num_preds,
    scaler_y=None, inverse_transform=False):

    if future_input_seq.shape[0] != seq_length:
        raise ValueError(f"Expected input with shape ({seq_length}, H, W, 1), "
                         f"but got {future_input_seq.shape}")

    current_seq = future_input_seq.copy()
    predictions_scaled = []

    for i in range(num_preds):
        input_batch = np.expand_dims(current_seq, axis=0)  # (1, seq_len, H, W, 1)
        pred = model.predict(input_batch, verbose=0)       # (1, H_out, W_out, 1)

        # Simpan hasil asli dari model (tanpa resize) untuk evaluasi akhir
        predictions_scaled.append(pred[0])  # (H_out, W_out, 1)

        # Resize hasil prediksi agar cocok dimasukkan lagi ke current_seq
        pred_resized = cv2.resize(pred[0, :, :, 0], (current_seq.shape[2], current_seq.shape[1]))  # (W, H)
        pred_resized = pred_resized[..., np.newaxis]  # (H, W, 1)
        pred_resized = pred_resized[np.newaxis, ...]  # (1, H, W, 1)

        # Rolling window update
        current_seq = np.concatenate([current_seq[1:], pred_resized], axis=0)

    predictions_scaled = np.stack(predictions_scaled, axis=0)  # (T, H_out, W_out, 1)

    if inverse_transform and scaler_y is not None:
        num_preds, H, W, _ = predictions_scaled.shape
        predictions_reshaped = predictions_scaled.reshape(num_preds, -1)
        predictions_inversed = scaler_y.inverse_transform(predictions_reshaped).reshape(num_preds, H, W)
        return predictions_inversed

    return predictions_scaled.squeeze(-1)  # (T, H, W)

def run_predictions_for_all_scenarios(
    future_sliced, scaler_X, scaler_y, model, panjang_seq, panjang_pred,
    selected_version, output_dir="predicted", max_sequences=10):

    os.makedirs(output_dir, exist_ok=True)
    all_predictions = {}

    for scenario, future_ds in future_sliced.items():
        print(f"\n🔮 Predicting future SSH for: {scenario}")

        future_input = future_ds.values
        if future_input.ndim == 3:
            future_input = future_input[..., np.newaxis]

        # Ganti NaN
        if np.isnan(future_input).any():
            print("⚠️ Found NaN in raw input! Replacing with 0...")
            future_input = np.nan_to_num(future_input, nan=0.0)

        # Scaling
        future_input_flat = future_input.reshape(future_input.shape[0], -1)
        future_input_scaled = scaler_X.transform(future_input_flat)
        future_input_scaled = future_input_scaled.reshape(future_input.shape)

        # Bikin sequence
        future_input_seq = reshape_input_for_model(future_input_scaled, sequence_length=panjang_seq)
        
        last_seq = future_input_seq[-1]
        pred_ssh = predict_future_sequence_autoregressive(
            last_seq,
            model=model,
            seq_length=panjang_seq,
            num_preds=panjang_pred,
            scaler_y=scaler_y,
            inverse_transform=True
        )

        # Ambil nama koordinat dari future_ds
        lat_name = [dim for dim in future_ds.coords if "lat" in dim.lower()]
        lon_name = [dim for dim in future_ds.coords if "lon" in dim.lower()]
        if not lat_name or not lon_name:
            raise ValueError(f"Latitude/Longitude tidak ditemukan dalam dataset {scenario}!\nTersedia: {list(future_ds.coords.keys())}")

        lat_target = ssh_sliced['latitude'].values
        lon_target = ssh_sliced['longitude'].values

        # Kalau 2D, ubah ke 1D
        if lat_target.ndim == 2:
            lat_target = lat_target[:, 0]
        if lon_target.ndim == 2:
            lon_target = lon_target[0, :]

        # Hitung tanggal prediksi awal
        time_start = pd.to_datetime(future_ds.time.values[-1])  # ambil waktu terakhir
        time_range = pd.date_range(start=time_start + pd.DateOffset(months=1), periods=panjang_pred, freq="MS")

        # Gabungkan semua prediksi secara time-series
        preds = pred_ssh.reshape(-1, len(lat_target), len(lon_target))

        print("lat shape:", lat_target.shape)
        print("lon shape:", lon_target.shape)
        print(f"⏳ Scenario: {scenario}, Pred shape: {preds.shape}")

        # Buat DataArray
        pred_da = xr.DataArray(
            preds,
            coords={
                "time": time_range,
                "latitude": lat_target,
                "longitude": lon_target,
            },
            dims=["time", "latitude", "longitude"],
            name="predicted_ssh"
        )

        # Simpan ke NetCDF
        pred_ds = xr.Dataset({"predicted_ssh": pred_da})
        out_path = os.path.join(
            output_dir,
            f"predicted_ssh_{selected_version}_{scenario}_seq{panjang_seq}_pred{panjang_pred}.nc"
        )
        pred_ds.to_netcdf(out_path)
        print(f"💾 Saved to: {out_path}")

        # Simpan ke dictionary hasil akhir
        all_predictions[scenario] = preds

    return all_predictions

# ===================== #
# 6. Main Training Flow #
# ===================== #

def main(selected_version="v1"):
    # === 3. Build dan compile model ===
    model_versions = {
        "v1": cnn_rnn_v1,
        "v2": cnn_rnn_v2,
        "v3": cnn_rnn_v3,
    }

    # === 1. Inisialisasi scaler dan data ===
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # === FLATTEN ===
    X = gcm_sliced.values.reshape(n_time, -1)
    X_scaled = scaler_X.fit_transform(X).reshape((n_time, *low_res_shape, 1))

    y = ssh_sliced.values.reshape(n_time, -1)
    y_reshaped = y.reshape(n_time, *high_res_shape, 1)

    # === SEQUENCE ===
    X_train_seq, y_train_seq = create_sequences(X_scaled, y_reshaped, seq_length=panjang_seq)

    # === FIT SCALER ONLY ON y_train ===
    split_idx = int(len(X_train_seq) * 0.9)
    y_train_only = y_train_seq[:split_idx].reshape(-1, high_res_shape[0] * high_res_shape[1])
    y_train_only = np.nan_to_num(y_train_only, nan=0.0)
    scaler_y.fit(y_train_only)

    print("scaler_y.data_min_:", scaler_y.data_min_)
    print("scaler_y.data_max_:", scaler_y.data_max_)
    print("apakah ada NaN di scaler?", np.isnan(scaler_y.data_min_).any() or np.isnan(scaler_y.data_max_).any())


    # === TRANSFORM y_train & y_val ===
    y_train_scaled = scaler_y.transform(y_train_only).reshape(-1, *high_res_shape, 1)
    y_val_only = y_train_seq[split_idx:].reshape(-1, high_res_shape[0] * high_res_shape[1])
    y_val_scaled = scaler_y.transform(y_val_only).reshape(-1, *high_res_shape, 1)

    X_train, X_val = X_train_seq[:split_idx], X_train_seq[split_idx:]

    # === Ocean Mask ===
    ocean_mask = ~np.isnan(y_train_scaled[0])
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_val_scaled = np.nan_to_num(y_val_scaled, nan=0.0)

    print("X_train:", np.isnan(X_train).sum(), X_train.min(), X_train.max())
    print("y_train_scaled:", np.isnan(y_train_scaled).sum(), y_train_scaled.min(), y_train_scaled.max())


    # run model
    model_fn = model_versions[selected_version]
    model = model_fn(X_train.shape[1:], high_res_shape)
    model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss=masked_mse(ocean_mask), metrics=['mae'])
    model.summary()

    # === CALLBACKS ===
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # === TRAINING ===
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Simpan model
    model_path = os.path.join(outdir, f'cnn_rnn_{selected_version}_{gcm_name}.keras')
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")

    # Evaluasi model
    val_loss, val_mae = model.evaluate(X_val, y_val_scaled)
    print(f"📊 Validation ({selected_version}) — Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

    # Prediksi masa depan
    all_predictions = run_predictions_for_all_scenarios(
        future_sliced=future_sliced,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        model=model,
        panjang_seq=panjang_seq,
        panjang_pred=panjang_pred,
        output_dir=outdir,
        selected_version=selected_version
    )

    return {
        "loss": val_loss,
        "mae": val_mae,
        "model_path": model_path,
        "predictions": all_predictions,
        "history": history
    }

def main_for_all_versions():
    versions = ["v1", "v2", "v3"]
    all_performances = {}
    all_histories = {}

    for version in versions:
        print(f"\n🚀 Running model version: {version}")
        results = main(selected_version=version)
        all_performances[version] = {
            "loss": results["loss"],
            "mae": results["mae"]
        }
        all_histories[version] = results["history"]

    print("\n📊 🔚 Model Performance Summary:")
    for version, metrics in all_performances.items():
        print(f"🧠 {version}: Loss = {metrics['loss']:.4f}, MAE = {metrics['mae']:.4f}")

    # === PLOT LOSS SEMUA VERSI ===
    plt.figure(figsize=(10, 6))
    for version, history in all_histories.items():
        plt.plot(history.history['val_loss'], label=f'{version} - Val Loss')
        plt.plot(history.history['loss'], linestyle='--', alpha=0.7, label=f'{version} - Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("📉 Loss Curve Comparison per Model Version")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_performances

if __name__ == "__main__":
    performances = main_for_all_versions()
    # performances = main(selected_version="v1") # kalau mau run 1 model saja secara spesifik

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load prediksi dari 3 model versi berbeda
ds_v1 = xr.open_dataset(f"{outdir}predicted_ssh_v1_ssp245_seq{panjang_seq}_pred{panjang_pred}.nc")
ds_v2 = xr.open_dataset(f"{outdir}predicted_ssh_v2_ssp245_seq{panjang_seq}_pred{panjang_pred}.nc")
ds_v3 = xr.open_dataset(f"{outdir}predicted_ssh_v3_ssp245_seq{panjang_seq}_pred{panjang_pred}.nc")
ds_obs = xr.open_dataset(f"{outdir}{gcm_name}_sliced_reanalysis_1995-01-16_2014-12-16.nc")

# Ambil array data
ssh_v1 = ds_v1['predicted_ssh'].values
ssh_v2 = ds_v2['predicted_ssh'].values
ssh_v3 = ds_v3['predicted_ssh'].values
ssh_obs = ds_obs['zos'].values  # atau 'ssh_obs' tergantung nama variabelnya

# Time step untuk dibandingkan
time_step = 238

# Hitung error absolut (selisih)
err_v1 = np.abs(ssh_v1[time_step] - ssh_obs[time_step])
err_v2 = np.abs(ssh_v2[time_step] - ssh_obs[time_step])
err_v3 = np.abs(ssh_v3[time_step] - ssh_obs[time_step])

# Plot Error
plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.imshow(np.flipud(err_v1), cmap='Reds')
plt.title(f"Error |v1 - Observasi|")
plt.colorbar(label="Error")

plt.subplot(1, 3, 2)
plt.imshow(np.flipud(err_v2), cmap='Reds')
plt.title(f"Error |v2 - Observasi|")
plt.colorbar(label="Error")

plt.subplot(1, 3, 3)
plt.imshow(np.flipud(err_v3), cmap='Reds')
plt.title(f"Error |v3 - Observasi|")
plt.colorbar(label="Error")

plt.tight_layout()
plt.show()

# Plot perbandingan
plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.imshow(np.flipud(ssh_v1[time_step]), cmap='viridis')
plt.title(f"v1 Prediction\nLoss: 0.0287, MAE: 0.1279")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.imshow(np.flipud(ssh_v2[time_step]), cmap='viridis')
plt.title(f"v2 Prediction\nLoss: 0.0414, MAE: 0.1558")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(np.flipud(ssh_v3[time_step]), cmap='viridis')
plt.title(f"v3 Prediction\nLoss: 0.0225, MAE: 0.1147")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(np.flipud(ssh_obs[time_step]), cmap='viridis')
plt.title(f"Observasi\nReanalysis")
plt.colorbar()

plt.tight_layout()
plt.show()