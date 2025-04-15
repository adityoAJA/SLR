import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error

# Fungsi metrik evaluasi
def compute_metrics(obs, model):
    mask = ~np.isnan(obs) & ~np.isnan(model)
    obs = obs[mask]
    model = model[mask]
    correlation = np.corrcoef(obs, model)[0, 1]
    rmse = np.sqrt(np.mean((obs - model)**2))
    bias = np.mean(model - obs)
    return correlation, rmse, bias

def standardize_lat_lon(ds):
    # Coba cari nama dimensi latitude
    for lat_name in ['lat', 'latitude', 'y', 'j']:
        if lat_name in ds.coords:
            ds = ds.rename({lat_name: 'lat'})
            break
    for lon_name in ['lon', 'longitude', 'x', 'i']:
        if lon_name in ds.coords:
            ds = ds.rename({lon_name: 'lon'})
            break
    return ds

# Load data reanalysis
reanalysis = xr.open_dataset("cmems_mod_glo_phy_my_0.083deg_P1M-m_1993_2014.nc")
reanalysis = standardize_lat_lon(reanalysis)
zos_ref = reanalysis['zos'].sel(time=slice('1995-01', '2014-12'))
zos_ref = zos_ref.sel(lat=slice(-11, 6), lon=slice(95, 141))

# Ambil daftar file GCM
gcm_files = [
    "data/canesm5/canesm5_historical_1993_2014.nc",
    "data/ec-earth/ec-earth_historical_1993_2014.nc",
    "data/miroc6/miroc6_historical_1993_2014.nc",
    "data/mpi/mpi_historical_1993_2014.nc",
]

gcmcorr_files = [
    "hasil/corrected_gcm_canesm5_1995-01-16_2014-12-16.nc",
    "hasil/corrected_gcm_ec-earth_1995-01-16_2014-12-16.nc",
    "hasil/corrected_gcm_miroc6_1995-01-16_2014-12-16.nc",
    "hasil/corrected_gcm_mpi_1995-01-16_2014-12-16.nc",
]

results = []

# Loop GCM asli
for gcm_file in gcm_files:
    ds = xr.open_dataset(gcm_file)
    ds = standardize_lat_lon(ds)

    # Ambil hanya variabel 'zos' dan slice waktu
    zos_model = ds['zos'].sel(time=slice('1995-01', '2014-12'))

    # Ambil koordinat target dari dataset referensi
    lat_target = zos_ref['lat'].values
    lon_target = zos_ref['lon'].values

    # Tentukan koordinat latitude & longitude dari dataset referensi
    lat_name = [dim for dim in zos_model.coords if "lat" in dim.lower()]
    lon_name = [dim for dim in zos_model.coords if "lon" in dim.lower()]

    if not lat_name or not lon_name:
        print(f"❌ Latitude/Longitude tidak ditemukan dalam dataset!")
        print(f"Tersedia koordinat: {list(zos_model.coords.keys())}")
        raise ValueError(f"Koordinat latitude/lon tidak ditemukan dalam dataset")

    lat_name = lat_name[0]
    lon_name = lon_name[0]

    # Cek apakah dimensi lat_target dan lon_target sesuai dengan data yang diharapkan
    if len(lat_target) != zos_model.shape[1] or len(lon_target) != zos_model.shape[2]:
        print(f"Dimensi latitude/lon tidak cocok! Interpolasi ulang diperlukan.")

    # Interpolasi latitude dan longitude agar sesuai dengan grid target (dimensi pred_ssh_scaled)
    lat_interp = interp1d(np.linspace(0, 1, len(lat_target)), lat_target, kind="linear")(np.linspace(0, 1, zos_model.shape[1]))
    lon_interp = interp1d(np.linspace(0, 1, len(lon_target)), lon_target, kind="linear")(np.linspace(0, 1, zos_model.shape[2]))

    # Gunakan hasil interpolasi sebagai dimensi dalam DataArray
    zos_model_interp = xr.DataArray(
        zos_model.values,  # Gunakan data asli dari zos_model
        coords={"time": zos_model.time.values, "latitude": lat_interp, "longitude": lon_interp},
        dims=["time", "latitude", "longitude"]
    )

    # Potong wilayah Indonesia
    zos_model = zos_model_interp.sel(latitude=slice(-11, 6), longitude=slice(95, 141))

    # Ambil rata-rata spasial
    ref_series = zos_ref.mean(dim=['lat', 'lon'])
    model_series = zos_model_interp.mean(dim=['latitude', 'longitude'])

    # Ambil nilai numpy array
    ref_vals = ref_series.values.flatten()
    model_vals = model_series.values.flatten()

    # Hitung metrik
    corr, rmse, bias = compute_metrics(ref_vals, model_vals)

    # Contoh perhitungan dalam loop untuk setiap model:
    mae = mean_absolute_error(ref_vals, model_vals)

    results.append({
        'Model': gcm_file.split('/')[-1],
        'Source': 'GCM asli',  # Menandakan ini adalah GCM asli
        'Correlation': corr,
        'RMSE': rmse,
        'Bias': bias,
        'MAE': mae,  # tambahkan ini
    })

# Loop GCM corrected
for gcmcorr_file in gcmcorr_files:
    ds1 = xr.open_dataset(gcmcorr_file)
    ds1 = standardize_lat_lon(ds1)
    ds1 = ds1.rename({'__xarray_dataarray_variable__': 'zos'})
    zos = ds1['zos'].sel(time=slice('1995-01', '2014-12'), lat=slice(-11, 6), lon=slice(95, 141))
    model_series = zos.mean(dim=['lat', 'lon']).values.flatten()
    corr, rmse, bias = compute_metrics(ref_series, model_series)
    # Contoh perhitungan dalam loop untuk setiap model:
    mae = mean_absolute_error(ref_series, model_series)
    results.append({'Model': gcmcorr_file.split('/')[-1],
                    'Source': 'Corrected',
                    'Correlation': corr,
                    'RMSE': rmse,
                    'Bias': bias,
                    'MAE': mae,})

# DataFrame hasil
df = pd.DataFrame(results)
print(df)

# --- Visualisasi ---
metrics = ['Correlation', 'RMSE', 'Bias', 'MAE']

# Melting DataFrame hasil ke format long
df_melt = df.melt(id_vars=['Model', 'Source'], value_vars=metrics, var_name='Metric', value_name='Value')

# Ubah kolom 'Value' menjadi numerik dan hapus NaN
df_melt['Value'] = pd.to_numeric(df_melt['Value'], errors='coerce')
df_melt = df_melt.dropna(subset=['Value'])

# Pastikan 'Model' adalah string
df_melt['Model'] = df_melt['Model'].astype(str)

# Pisahkan df_melt menjadi dua DataFrame berdasarkan 'Source'
df_gcm_asli = df_melt[df_melt['Source'] == 'GCM asli']
df_gcm_corr = df_melt[df_melt['Source'] == 'Corrected']

# Fungsi untuk mengekstrak nilai numerik dari DataArray jika ada
def extract_value(val):
    if isinstance(val, xr.DataArray):
        return val.values.item()  # Mengambil nilai pertama (jika hanya satu nilai dalam DataArray)
    else:
        return val  # Jika bukan DataArray, kembalikan nilai itu sendiri

# Terapkan fungsi untuk mengekstrak nilai numerik
df_gcm_corr.loc[:, 'Value'] = df_gcm_corr['Value'].apply(extract_value)

# Sekarang kita bisa mengonversi 'Value' menjadi numerik
df_gcm_corr['Value'] = pd.to_numeric(df_gcm_corr['Value'], errors='coerce')

# --- Visualisasi 1: GCM Asli vs zos_ref ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df_gcm_asli, x='Model', y='Value', hue='Metric', palette='Set2', ci=None)
plt.title("Evaluasi GCM Asli vs zos_ref (1995-2014)")
plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("Nilai Metrik")
plt.legend(title='Metrik')
plt.tight_layout()
plt.show()

# --- Visualisasi 2: GCM Dikoreksi vs zos_ref ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df_gcm_corr, x='Model', y='Value', hue='Metric', palette='Set2', ci=None)
plt.title("Evaluasi GCM Dikoreksi vs zos_ref (1995-2014)")
plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("Nilai Metrik")
plt.legend(title='Metrik')
plt.tight_layout()
plt.show()