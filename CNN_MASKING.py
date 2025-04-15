import xarray as xr
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, BatchNormalization, Dropout

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

# Pastikan jumlah sample X dan y sama otomatis
min_samples = min(gcm_sliced.shape[0], ssh_sliced.shape[0])

# Ambil hanya min_samples pertama jika jumlahnya berbeda
if gcm_sliced.shape[0] != ssh_sliced.shape[0]:
    gcm_sliced = gcm_sliced[:min_samples]
    ssh_sliced = ssh_sliced[:min_samples]

# Konversi NaN ke angka agar tidak mengganggu analisis
X = np.nan_to_num(gcm_sliced.values)
y = np.nan_to_num(ssh_sliced.values)

# Deteksi resolusi input secara dinamis
height, width = X.shape[1], X.shape[2]

# Normalisasi data X
scaler_X = MinMaxScaler()
X_reshaped = X.reshape(X.shape[0], -1)  # (samples, lat * lon)
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)  # Kembali ke bentuk aslinya

# Resize y agar sesuai dengan resolusi input
y_resized = np.array([resize(y[i], (height, width), anti_aliasing=True) for i in range(y.shape[0])])

# Normalisasi y agar sesuai dengan skala yang sama
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y_resized.reshape(y.shape[0], -1))  # Menyesuaikan shape (samples, height * width)

# Reshape y_train to match model's output shape (samples, height, width, 1)
y_train_reshaped = y_scaled.reshape(-1, height, width, 1)
y_test_reshaped = y_scaled.reshape(-1, height, width, 1)

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_train_reshaped, test_size=0.2, random_state=42
)

# Tambahkan channel untuk CNN (samples, height, width, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"GCM sliced time: {gcm_sliced.time.shape}")  # Harusnya (240,)
print(f"SSH sliced time: {ssh_sliced.time.shape}")  # Harusnya (240,)
for scen in future_sliced:
    print(f"Future {scen} sliced time: {future_sliced[scen].time.shape}")  # Harusnya (240,)

print(f"Hist range: {hist1} - {hist2}")  # Harusnya 1995-01-01 sampai 2014-12-01
print(f"Min Time GCM: {gcm_ds.time.min().values}, Max Time GCM: {gcm_ds.time.max().values}")

# Build CNN dengan fleksibilitas resolusi input
def build_flexible_cnn(input_shape, output_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Input fleksibel
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        GlobalAveragePooling2D(),  # Mengubah dimensi menjadi 1D
        Dense(np.prod(output_shape), activation='linear')  # Sesuaikan output dengan shape `y`
    ])
    
    # Remove last layer and add new layers
    model.pop()  # Hapus layer terakhir (Dense layer sebelumnya)
    model.add(Dense(np.prod(output_shape)))  # Tambahkan Dense layer baru
    model.add(Reshape((height, width, 1)))  # Ubah output menjadi grid spasial
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Pass X_train.shape[1:] as input_shape and y_train
model = build_flexible_cnn(X_train.shape[1:], y_train.shape[1:])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Print model summary
model.summary()

# save model
model.save(f"model/model_cnn_{gcm_name}.keras")

# Function to reshape future data before prediction
def reshape_future_data(X_future_scaled, X_future):
    """
    Menyesuaikan shape berdasarkan resolusi dataset yang digunakan.
    """
    num_samples = X_future.shape[0]
    height, width = X_future.shape[1], X_future.shape[2]  # Deteksi ukuran spasial
    
    expected_size = num_samples * height * width * 1
    actual_size = X_future_scaled.size
    
    if expected_size != actual_size:
        raise ValueError(f"Mismatch: Expected {expected_size} elements, but got {actual_size}")

    return X_future_scaled.reshape(num_samples, height, width, 1)

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

# Predict future SSH data
predictions = {}

for scen, future_ds in future_sliced.items():
    print(f"\nProcessing scenario: {scen}")

    # Konversi NaN ke 0 agar bisa diproses model
    X_future = np.nan_to_num(future_ds.values)

    # Normalisasi data
    X_future_scaled = scaler_X.transform(X_future.reshape(X_future.shape[0], -1)).reshape(X_future.shape)

    # Panggil fungsi reshape
    X_future_scaled = reshape_future_data(X_future_scaled, X_future)

    # Predict future SSH
    pred_ssh_scaled = model.predict(X_future_scaled)

    # Menentukan nama koordinat latitude & longitude secara otomatis
    lat_name = "latitude" if "latitude" in future_ds.dims else list(future_ds.dims)[1]
    lon_name = "longitude" if "longitude" in future_ds.dims else list(future_ds.dims)[2]

    if lat_name is None or lon_name is None:
        raise ValueError(f"Koordinat latitude/lon tidak ditemukan dalam dataset {scen}")

    # Bentuk asli data
    time_steps = pred_ssh_scaled.shape[0]
    y_dim, x_dim = future_ds.sizes[lat_name], future_ds.sizes[lon_name]

    # Reshape prediksi ke bentuk spasial (time, y, x)
    pred_ssh_reshaped = pred_ssh_scaled.reshape(time_steps, y_dim, x_dim)

    # Pastikan lat/lon dalam bentuk 2D jika dataset asli menggunakan grid
    if len(future_ds[lat_name].shape) == 1 and len(future_ds[lon_name].shape) == 1:
        lat_2d, lon_2d = np.meshgrid(future_ds[lat_name], future_ds[lon_name], indexing="ij")
    else:
        lat_2d, lon_2d = future_ds[lat_name], future_ds[lon_name]

    # Buat koordinat waktu untuk 240 bulan / 20 tahun
    time_steps_future = pd.date_range(start='2031-01', periods=240, freq='ME')

    # **Gunakan lat/lon dari data yang sudah dikoreksi untuk interpolasi**
    lat_target = ssh_sliced['latitude'].values
    lon_target = ssh_sliced['longitude'].values
    lat_target_grid, lon_target_grid = np.meshgrid(lon_target, lat_target)

    print("Predicted SSH shape:", pred_ssh_scaled.shape)
    print("Time coordinates shape:", ssh_sliced.time.shape)

    # Interpolasi agar hasil prediksi sesuai dengan resolusi data yang sudah dikoreksi
    predicted_ssh_da = xr.DataArray(
        pred_ssh_reshaped,
        coords={"time": time_steps_future, lat_name: lat_2d[:, 0], lon_name: lon_2d[0, :]},
        dims=["time", lat_name, lon_name]
    ).interp({lat_name: lat_target, lon_name: lon_target}, method="linear")

    # Simpan hasil prediksi
    predictions[scen] = predicted_ssh_da

    # # Menghitung rata-rata tahunan
    # predicted_ssh_annual = predicted_ssh_da.groupby("time.year").mean(dim="time")

    # Konversi ke Dataset dengan variabel 'zos'
    predicted_ssh_annual = predicted_ssh_da.to_dataset(name="zos")

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

    # Simpan sebagai NetCDF
    pred_name = f"{outdir}predicted_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
    predicted_ssh_annual.to_netcdf(pred_name)

print("Predictions saved for all scenarios.")

### **PENGOLAHAN DATA CORRECTED SSH (1995-2014)**

# deteksi nama koordinat
lat_names = ["latitude", "lat", "y", "j"]
lon_names = ["longitude", "lon", "x", "i"]

# Cari koordinat atau variabel yang mengandung latitude dan longitude
detected_lat = next((lat for lat in lat_names if lat in gcm_sliced.dims), None)
detected_lon = next((lon for lon in lon_names if lon in gcm_sliced.dims), None)

# Jika tidak ditemukan di koordinat, cari di variabel
if detected_lat is None or detected_lon is None:
    detected_lat = next((lat for lat in lat_names if lat in gcm_sliced.variables), None)
    detected_lon = next((lon for lon in lon_names if lon in gcm_sliced.variables), None)

# Jika tetap tidak ditemukan, keluar dengan error
if detected_lat is None or detected_lon is None:
    raise ValueError("Latitude atau Longitude tidak ditemukan dalam gcm_ds.")

# Konversi NaN ke 0 agar bisa diproses model
X_cor = np.nan_to_num(gcm_sliced.values)

# Simpan bentuk asli sebelum normalisasi
original_shape = X_cor.shape

# Normalisasi data
X_cor_scaled = scaler_X.transform(X_cor.reshape(original_shape[0], -1)).reshape(original_shape)

# Panggil fungsi reshape agar cocok dengan input model
X_cor_scaled = reshape_cor_data(X_cor_scaled, X_future)

# Predict future SSH
cor_ssh_scaled = model.predict(X_cor_scaled)

# Dapatkan ukuran grid dari dataset interpolasi
y_cor_dim = gcm_sliced.sizes[detected_lat]
x_cor_dim = gcm_sliced.sizes[detected_lon]
time_steps_cor = cor_ssh_scaled.shape[0]

# Verifikasi jumlah elemen cocok sebelum reshape
if cor_ssh_scaled.size != time_steps_cor * y_cor_dim * x_cor_dim:
    raise ValueError(f"Mismatch in total number of elements: {cor_ssh_scaled.size} != {time_steps_cor * y_cor_dim * x_cor_dim}")

# Reshape prediksi ke bentuk spasial (time, y, x)
cor_ssh_reshaped = cor_ssh_scaled.reshape(time_steps_cor, y_cor_dim, x_cor_dim)

# Pastikan lat/lon dalam bentuk 2D
if len(gcm_sliced[detected_lat].shape) == 1 and len(gcm_sliced[detected_lon].shape) == 1:
    lat_2d, lon_2d = np.meshgrid(gcm_sliced[detected_lat], gcm_sliced[detected_lon], indexing="ij")
else:
    lat_2d, lon_2d = gcm_sliced[detected_lat], gcm_sliced[detected_lon]

# Buat koordinat waktu untuk 240 bulan (20 tahun)
time_steps_cor = pd.date_range(start='1995-01', periods=240, freq='ME')

# **Interpolasi hasil prediksi agar sesuai dengan ssh_sliced**
lat_target = ssh_sliced['latitude'].values
lon_target = ssh_sliced['longitude'].values

# Interpolasi hasil prediksi agar resolusi sesuai dengan data yang dikoreksi
corrected_ssh_da = xr.DataArray(
    cor_ssh_reshaped,
    coords={"time": time_steps_cor, detected_lat: lat_2d[:, 0], detected_lon: lon_2d[0, :]},
    dims=["time", detected_lat, detected_lon]
).interp({detected_lat: lat_target, detected_lon: lon_target}, method="linear")

# Simpan hasil koreksi
corrections = corrected_ssh_da

# Menghitung rata-rata tahunan
# corrected_ssh_annual = corrected_ssh_da.groupby("time.year").mean(dim="time")

# Konversi ke Dataset dengan variabel 'zos'
corrected_ssh_annual = corrected_ssh_da.to_dataset(name="zos")

# Pastikan koordinat 'j' dan 'i' bertipe int (jika ada)
if "j" in corrected_ssh_annual.coords:
    corrected_ssh_annual = corrected_ssh_annual.assign_coords(j=("j", corrected_ssh_annual["j"].values.astype(int)))
if "i" in corrected_ssh_annual.coords:
    corrected_ssh_annual = corrected_ssh_annual.assign_coords(i=("i", corrected_ssh_annual["i"].values.astype(int)))

# Tambahkan atribut ke variabel 'zos'
corrected_ssh_annual['zos'].attrs.update({
    'description': f'Corrected sea surface height from {hist1} to {hist2}',
    'units': 'meters',
    'standard_name': 'sea_surface_height',
    'long_name': 'Sea Surface Height Above Geoid',
    'coordinates': 'latitude longitude'
})

# Simpan sebagai NetCDF
cor_name = f"{outdir}corrected_ssh_{gcm_name}_{scen}_{fut1}_{fut2}.nc"
corrected_ssh_annual.to_netcdf(cor_name)

print("Corrections saved.")

# MASKING DARATAN dan LAUTAN

import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import os

def standarize_coordinates(ds):
    """Menyesuaikan koordinat dataset agar menggunakan 'lat' dan 'lon'"""
    
    possible_x_dims = ["lon", "longitude", "X", "x", "i"]
    possible_y_dims = ["lat", "latitude", "Y", "y", "j"]

    x_dim = next((dim for dim in possible_x_dims if dim in ds.dims), None)
    y_dim = next((dim for dim in possible_y_dims if dim in ds.dims), None)

    if not x_dim or not y_dim:
        raise ValueError(f"Tidak ditemukan koordinat yang sesuai dalam dataset: {ds.dims}")

    if x_dim in ds and y_dim in ds:
        if ds[x_dim].ndim == 2 and ds[y_dim].ndim == 2:
            ds = ds.assign_coords({"lat": (("j", "i"), ds[y_dim].values),
                                   "lon": (("j", "i"), ds[x_dim].values)})
        else:
            ds = ds.assign_coords({"lat": ds[y_dim], "lon": ds[x_dim]})

        ds = ds.swap_dims({y_dim: "lat", x_dim: "lon"})

        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        ds = ds.rio.write_crs("EPSG:4326")
        
        print("Koordinat berhasil distandarisasi ke lat/lon!")
        return ds
    else:
        raise ValueError(f"Variabel {x_dim} dan {y_dim} tidak ditemukan dalam dataset!")

def mask_dataset(ds, shapefile_path, land_or_sea="sea"):
    """Masking dataset menggunakan shapefile Indonesia"""
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Pastikan CRS shapefile sama dengan dataset
    gdf = gdf.to_crs(ds.rio.crs)
    
    # Buat mask (True untuk wilayah darat, False untuk laut)
    mask = ds.rio.clip(gdf.geometry, gdf.crs, drop=False, all_touched=True)
    
    if land_or_sea == "land":
        print("Menjaga wilayah darat, menghapus laut...")
        return mask
    elif land_or_sea == "sea":
        print("Menjaga wilayah laut, menghapus darat...")
        return ds.where(mask.isnull(), drop=True)
    else:
        raise ValueError("Pilihan 'land_or_sea' harus 'land' atau 'sea'")

# Path folder input dan output
input_folder = outdir  # Folder penyimpanan NetCDF hasil koreksi
output_folder = f'{outdir}masked'  # Folder untuk hasil masking
shapefile_path = 'shapefile/indonesia.geojson'  # Path ke shapefile batas provinsi

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):
        file_path = os.path.join(input_folder, filename)
        print(f"\nMemproses file: {filename}")

        ds = xr.open_dataset(file_path)

        try:
            ds = standarize_coordinates(ds)
            
            # Hapus daratan, hanya simpan lautan
            ds = mask_dataset(ds, shapefile_path)

            output_file = os.path.join(output_folder, f"sea_only_{filename}")
            ds.to_netcdf(output_file)

            print(f"✅ Berhasil diproses: {output_file}")

        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")