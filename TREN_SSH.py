import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress

# Initialisasi
tahun_awal = "2021"
tahun_akhir = "2100"
skenario = "ssp585"
gcm = "canesm5"

# Buka file NetCDF
file_path = f"data/hasil/{gcm}/F_CNN/merged/masked/Global_sea_merged_{gcm}_{skenario}_{tahun_awal}_{tahun_akhir}.nc"
ds = xr.open_dataset(file_path)

# Ambil variabel yang diperlukan
ssh = ds['zos']  # Ganti zos atau sla sesuai dataset
time = pd.to_datetime(ds['time'].values)

# Konversi waktu ke tahun decimal
time_decimal = time.year.astype(float)
ds["time_decimal"] = ("time", time_decimal)

# Fungsi untuk menghitung tren regresi linear dalam mm/tahun
def compute_trend(y, t):
    mask = ~np.isnan(y)  # Mask NaN values
    if np.sum(mask) > 1:  # Minimal 2 titik data untuk regresi
        slope, _, _, _, _ = linregress(t[mask], y[mask])
        return slope * 1000  # Konversi ke mm/tahun
    else:
        return np.nan

# Terapkan fungsi ke setiap grid
trend = xr.apply_ufunc(
    compute_trend, 
    ssh, 
    ds["time_decimal"],  # Gunakan waktu dalam tahun decimal
    input_core_dims=[["time"], ["time"]],  
    vectorize=True
)

# Simpan tren dalam dataset
trend_ds = xr.Dataset({"trend": trend})

# 🔥 PENGATURAN UKURAN PETA & COLORBAR 🔥
fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={"projection": ccrs.PlateCarree()})  # Atur ukuran lebih besar
ax.set_title(f"Laju Perubahan Tinggi Muka Laut {tahun_awal}-{tahun_akhir} Skenario Iklim {skenario}", fontsize=20, fontweight="bold")  # Ukuran judul lebih besar

# Mentukan batas warna dengan interval vmin, vmax, dan interval 20 mm 
levels = np.arange(-1, 1.25, 0.25)  # nilai max, min, dan rentang per bar

# BoundaryNorm buat warna menjadi diskret
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=plt.cm.coolwarm.N, extend="both") # ncolors bisa RdBu, coolwarm, viridis, dll

# Plot data dengan resolusi tinggi
c = ax.pcolormesh(
    trend["longitude"], trend["latitude"], trend,
    transform=ccrs.PlateCarree(), shading='auto', cmap='coolwarm', # cmap = coolwarm, RdBu, viridis, dll
    # vmin=-0.5, vmax=0.5,  # Sesuaikan skala warna
    norm=norm
)

# Tambahkan fitur peta
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)  # Tambahkan garis pantai
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)  # Tambahkan batas negara
ax.add_feature(cfeature.LAND, facecolor="white")  # Warna daratan

# 🔥 ATUR UKURAN COLORBAR 🔥
cbar = plt.colorbar(c, ax=ax, orientation="horizontal", fraction=0.08, pad=0.08, extend="both")
cbar.set_label("Laju Perubahan (mm/tahun)", fontsize=14, fontweight="bold")
cbar.ax.tick_params(labelsize=14)

# 🔥 HITUNG & TAMPILKAN RATA-RATA 🔥
mean_trend = trend.mean().item()  # Hitung rata-rata tren
ax.text(
    0.99, 0.97,  # Posisi relatif (x, y) dalam koordinat figure (1.0 = sudut kanan atas)
    f"Tren TML Indonesia: {mean_trend:.2f} mm/tahun",  # Format 3 desimal
    fontsize=14, fontweight="bold", color="black",
    ha="right", va="top", transform=ax.transAxes,  # Transformasi agar tetap di sudut
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")  # Tambahkan kotak latar belakang
)

# 🔥 Load gambar logo 🔥
logo1 = mpimg.imread("Logo_BMKG.png")  # Ganti dengan path logo pertama
logo2 = mpimg.imread("esgf.png")  # Ganti dengan path logo kedua

# 🔥 Tambahkan logo sebagai inset di kiri atas 🔥
# Logo 1
axin1 = ax.inset_axes([0.008, 0.88, 0.10, 0.10])  # [left, bottom, width, height] relatif ke ax
axin1.imshow(logo1)
axin1.axis("off")  # Hilangkan axis

# Logo 2
axin2 = ax.inset_axes([0.08, 0.88, 0.10, 0.10])  # [left, bottom, width, height]
axin2.imshow(logo2)
axin2.axis("off")

plt.savefig(f"SSH_trend_{gcm}_{skenario}_{tahun_awal}_{tahun_akhir}.png", dpi=300, bbox_inches="tight")  # Simpan gambar dengan resolusi tinggi
plt.show()

# Simpan NetCDF hasil
trend_ds.to_netcdf(f"SSH_trend_{gcm}_{skenario}_{tahun_awal}_{tahun_akhir}.nc")
