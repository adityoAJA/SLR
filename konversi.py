import xarray as xr
import pandas as pd

# # Rata2 Tahunan
# # Muat dataset NetCDF
# file_path = 'sliced_historical_1994-2014.nc'  # Ganti dengan path file Anda
# data = xr.open_dataset(file_path)

# # Hitung rata-rata sepanjang dimensi waktu
# mean_zos = data['zos'].mean(dim='time')

# # Menyimpan hasil rata-rata ke NetCDF
# mean_zos.to_netcdf('mean_raw_historical_19942014.nc')

# # Atau menyimpan hasil rata-rata ke CSV
# # Pertama, kita perlu mengubah DataArray menjadi DataFrame
# mean_zos_df = mean_zos.to_dataframe().reset_index()  # Reset index untuk mendapatkan format DataFrame yang lebih baik

# # Menyimpan DataFrame ke CSV
# mean_zos_df.to_csv('mean_raw_historical_19942014.csv', index=False)

# print("Rata-rata berhasil disimpan ke mean_predicted_ssh.nc dan mean_predicted_ssh.csv.")

## Bulanan

# import xarray as xr
# import pandas as pd
# import numpy as np

# # Muat dataset NetCDF
# file_path = 'sliced_ssp245_2030-2050.nc'  # Ganti dengan path file Anda
# data = xr.open_dataset(file_path)

# # Buat array bulan berdasarkan indeks waktu
# # Menghitung indeks bulan dari 0 hingga 11
# # 0-11 untuk Januari hingga Desember, diulang selama 21 tahun
# months = np.tile(np.arange(12), 21)

# # Tambahkan dimensi bulan ke data
# data = data.assign_coords(month=('time', months))

# # Hitung rata-rata untuk setiap bulan
# mean_monthly_zos = data['zos'].groupby('month').mean(dim='time')

# # Menyimpan hasil rata-rata bulanan ke NetCDF
# mean_monthly_zos.to_netcdf('mean_monthly_raw_ssh.nc')

# # Atau menyimpan hasil rata-rata bulanan ke CSV
# mean_monthly_zos_df = mean_monthly_zos.to_dataframe().reset_index()

# # Menyimpan DataFrame ke CSV
# mean_monthly_zos_df.to_csv('mean_monthly_raw_ssh.csv', index=False)

# print("Rata-rata bulanan berhasil disimpan ke mean_monthly_predicted_ssh.nc dan mean_monthly_predicted_ssh.csv.")

## Musiman

import xarray as xr
import pandas as pd
import numpy as np

# Muat dataset NetCDF
file_path = 'sliced_ssp245_2030-2050.nc'  # Ganti dengan path file Anda
data = xr.open_dataset(file_path)

# Buat array bulan berdasarkan indeks waktu
# Menghitung indeks bulan dari 0 hingga 11
months = np.tile(np.arange(12), 21)

# Tambahkan dimensi bulan ke data
data = data.assign_coords(month=('time', months))

# Tentukan indeks musim
season_indices = {
    'DJF': [11, 0, 1],  # Desember, Januari, Februari
    'MAM': [2, 3, 4],   # Maret, April, Mei
    'JJA': [5, 6, 7],   # Juni, Juli, Agustus
    'SON': [8, 9, 10]   # September, Oktober, November
}

# Hitung rata-rata musiman
mean_seasonal_zos = {}

for season, indices in season_indices.items():
    # Mengelompokkan data berdasarkan bulan dan menghitung rata-rata untuk masing-masing musim
    mean_seasonal_zos[season] = data['zos'].where(data['month'].isin(indices)).mean(dim='time')

# Simpan hasil rata-rata musiman ke NetCDF
seasonal_mean_ds = xr.Dataset({season: mean for season, mean in mean_seasonal_zos.items()})
seasonal_mean_ds.to_netcdf('mean_seasonal_raw_ssh.nc')

# Atau menyimpan hasil rata-rata musiman ke CSV
seasonal_mean_df = pd.DataFrame({season: mean.values.flatten() for season, mean in mean_seasonal_zos.items()})
seasonal_mean_df.to_csv('mean_seasonal_raw_ssh.csv', index=False)

print("Rata-rata musiman berhasil disimpan ke mean_seasonal_predicted_ssh.nc dan mean_seasonal_predicted_ssh.csv.")
