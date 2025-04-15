import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import os

# ==== PATH SETTING ====
gcm = "miroc6"
var = "predicted_ssh"
scen = "ssp585"
path = f"data/hasil/{gcm}/F_CNN/"
outdir = f"{path}merged/"
shapefile_path = 'shapefile/ne_10m_land.shp'
output_folder = f'{outdir}masked'

os.makedirs(outdir, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ==== GABUNG DATA ====
ds1 = xr.open_dataset(f'{path}2021-01-16_2040-12-16/{var}_{gcm}_{scen}_2021-01-16_2040-12-16.nc')
ds2 = xr.open_dataset(f'{path}2041-01-16_2060-12-16/{var}_{gcm}_{scen}_2041-01-16_2060-12-16.nc')
ds3 = xr.open_dataset(f'{path}2061-01-16_2080-12-16/{var}_{gcm}_{scen}_2061-01-16_2080-12-16.nc')
ds4 = xr.open_dataset(f'{path}2081-01-16_2100-12-16/{var}_{gcm}_{scen}_2081-01-16_2100-12-16.nc')

ds_merged = xr.concat([ds1, ds2, ds3, ds4], dim='time')
ds_merged.to_netcdf(f'{outdir}merged_{gcm}_{scen}_2021_2100.nc')


# ==== FUNGSI STANDARDISASI KOORDINAT ====
def standarize_coordinates(ds):
    possible_x_dims = ["lon", "longitude", "X", "x", "i"]
    possible_y_dims = ["lat", "latitude", "Y", "y", "j"]

    x_dim = next((dim for dim in possible_x_dims if dim in ds.dims), None)
    y_dim = next((dim for dim in possible_y_dims if dim in ds.dims), None)

    if not x_dim or not y_dim:
        raise ValueError(f"Tidak ditemukan koordinat yang sesuai dalam dataset: {ds.dims}")

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


# ==== FUNGSI MASKING ====
def mask_dataset(ds, shapefile_path, land_or_sea="sea"):
    gdf = gpd.read_file(shapefile_path).to_crs(ds.rio.crs)
    mask = ds.rio.clip(gdf.geometry, gdf.crs, drop=False, all_touched=True)

    if land_or_sea == "land":
        print("Menjaga wilayah darat, menghapus laut...")
        return mask
    elif land_or_sea == "sea":
        print("Menjaga wilayah laut, menghapus darat...")
        return ds.where(mask.isnull(), drop=True)
    else:
        raise ValueError("Pilihan 'land_or_sea' harus 'land' atau 'sea'")


# ==== PROSES MASKING ====
for filename in os.listdir(outdir):
    if filename.endswith(".nc") and "merged" in filename:
        file_path = os.path.join(outdir, filename)
        print(f"\n🔄 Memproses file: {filename}")

        try:
            ds = xr.open_dataset(file_path)
            ds = standarize_coordinates(ds)
            ds = mask_dataset(ds, shapefile_path, land_or_sea="sea")

            output_file = os.path.join(output_folder, f"Global_sea_{filename}")
            ds.to_netcdf(output_file)
            print(f"✅ Berhasil disimpan: {output_file}")

        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")
