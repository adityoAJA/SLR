import xarray as xr
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import os

# path
gcm = "canesm5"
path = f"data/hasil/{gcm}/F_CNN/"
var = "predicted_ssh"
scen = "ssp585"

# Load masing-masing file
ds1 = xr.open_dataset(f'{path}2021-01-16_2040-12-16/{var}_{gcm}_{scen}_2021-01-16_2040-12-16.nc')
ds2 = xr.open_dataset(f'{path}2041-01-16_2060-12-16/{var}_{gcm}_{scen}_2041-01-16_2060-12-16.nc')
ds3 = xr.open_dataset(f'{path}2061-01-16_2080-12-16/{var}_{gcm}_{scen}_2061-01-16_2080-12-16.nc')
ds4 = xr.open_dataset(f'{path}2081-01-16_2100-12-16/{var}_{gcm}_{scen}_2081-01-16_2100-12-16.nc')

# Gabungkan berdasarkan dimensi waktu
ds_merged = xr.concat([ds1, ds2, ds3, ds4], dim='time')

# cek dan plot
outdir = f"{path}merged/"
os.makedirs(outdir, exist_ok=True)

# Simpan kalau perlu
ds_merged.to_netcdf(f'{outdir}merged_{gcm}_{scen}_2021_2100.nc')

# MASKING DARATAN dan LAUTAN

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
shapefile_path = 'shapefile/ne_10m_land.shp'  # Path ke shapefile batas provinsi

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

            output_file = os.path.join(output_folder, f"Global_sea_{filename}")
            ds.to_netcdf(output_file)

            print(f"✅ Berhasil diproses: {output_file}")

        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")