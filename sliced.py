import xarray as xr

# Load future GCM data
future_gcm_data = xr.open_dataset('zos_Omon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.nc')  # Future GCM data (100 km)

# Variable names
gcm_var = 'zos'  # Sea Surface Height above Geoid from GCM

# Define the region for Indonesia
lat_min, lat_max = -15, 10
lon_min, lon_max = 90, 145

# Mask future GCM data similarly as done previously
future_gcm_sliced = future_gcm_data[gcm_var].where(
    (future_gcm_data['latitude'] >= lat_min) & (future_gcm_data['latitude'] <= lat_max) &
    (future_gcm_data['longitude'] >= lon_min) & (future_gcm_data['longitude'] <= lon_max),
    drop=True
)

# Slice future GCM data to include only the years 2030 to 2050
future_gcm_sliced = future_gcm_sliced.sel(time=slice('2030-01', '2050-12'))

# Save to netCDF
future_gcm_sliced.to_netcdf('sliced_ssh_2030-2050.nc')
