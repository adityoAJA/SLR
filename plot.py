import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load prediksi dari 3 model versi berbeda (1 gcm)
ds_v1 = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_1008/predicted_ssh_v1_ssp245_seq24_pred1008.nc")
ds_v2 = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_1008/predicted_ssh_v2_ssp245_seq24_pred1008.nc")
ds_v3 = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_1008/predicted_ssh_v3_ssp245_seq24_pred1008.nc")
ds_obs = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_1008/miroc6_sliced_reanalysis_1995-01-16_2014-12-16.nc")

# Ambil array data
ssh_v1 = ds_v1['predicted_ssh'].values
ssh_v2 = ds_v2['predicted_ssh'].values
ssh_v3 = ds_v3['predicted_ssh'].values
ssh_obs = ds_obs['zos'].values  # atau 'ssh_obs' tergantung nama variabelnya

# Time step untuk dibandingkan
time_step = 200

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

# Plot perbandingan antar model
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

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ambil timestep
time_step = 200

# Flatten data
obs_flat = ssh_obs[time_step].flatten()
v1_flat = ssh_v1[time_step].flatten()
v2_flat = ssh_v2[time_step].flatten()
v3_flat = ssh_v3[time_step].flatten()

# Hitung MAE & RMSE
def get_metrics(pred, obs):
    mask = ~np.isnan(obs) & ~np.isnan(pred)  # Hindari NaN
    mae = mean_absolute_error(obs[mask], pred[mask])
    rmse = np.sqrt(mean_squared_error(obs[mask], pred[mask]))
    return mae, rmse

mae_v1, rmse_v1 = get_metrics(v1_flat, obs_flat)
mae_v2, rmse_v2 = get_metrics(v2_flat, obs_flat)
mae_v3, rmse_v3 = get_metrics(v3_flat, obs_flat)

print(f"v1 — MAE: {mae_v1:.4f}, RMSE: {rmse_v1:.4f}")
print(f"v2 — MAE: {mae_v2:.4f}, RMSE: {rmse_v2:.4f}")
print(f"v3 — MAE: {mae_v3:.4f}, RMSE: {rmse_v3:.4f}")

# Error absolut (selisih per grid)
error_v1 = np.abs(ssh_v1 - ssh_obs)
error_v2 = np.abs(ssh_v2 - ssh_obs)
error_v3 = np.abs(ssh_v3 - ssh_obs)

# Rata-rata error spasial (mean over time axis)
mean_error_v1 = np.nanmean(error_v1, axis=0)
mean_error_v2 = np.nanmean(error_v2, axis=0)
mean_error_v3 = np.nanmean(error_v3, axis=0)

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.imshow(mean_error_v1, cmap='Reds', origin='lower')
plt.title("Mean Error v1")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(mean_error_v2, cmap='Reds', origin='lower')
plt.title("Mean Error v2")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(mean_error_v3, cmap='Reds', origin='lower')
plt.title("Mean Error v3")
plt.colorbar()

plt.tight_layout()
plt.show()

# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np

# # Load dataset
# ds_mpi = xr.open_dataset("data/hasil/mpi/2015-01-16_2100-12-16/predicted_ssh_v3_ssp245.nc")
# ds_miroc6 = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_240/predicted_ssh_v3_ssp245_2015-01-16_24_240.nc")
# ds_ecearth = xr.open_dataset("data/hasil/ec-earth/2015-01-16_24_240/predicted_ssh_v3_ssp245_2015-01-16_24_240.nc")
# ds_obs = xr.open_dataset("data/hasil/miroc6/2015-01-16_24_240/miroc6_sliced_reanalysis_1995-01-16_2014-12-16.nc")

# # Ambil data
# ssh_mpi = ds_mpi['predicted_ssh'].values
# ssh_miroc6 = ds_miroc6['predicted_ssh'].values
# ssh_ecearth = ds_ecearth['predicted_ssh'].values
# ssh_obs = ds_obs['zos'].values  # 'zos' = sea surface height anomaly

# # Pastikan jumlah timestep cocok
# num_timesteps = min(ssh_mpi.shape[0], ssh_miroc6.shape[0], ssh_ecearth.shape[0], ssh_obs.shape[0])

# print("Timesteps diambil:", num_timesteps)

# # Hapus data di luar timestep yang sama
# ssh_mpi = ssh_mpi[:num_timesteps]
# ssh_miroc6 = ssh_miroc6[:num_timesteps]
# ssh_ecearth = ssh_ecearth[:num_timesteps]
# ssh_obs = ssh_obs[:num_timesteps]

# # Hitung MAE dan korelasi per timestep
# def compute_metrics(pred, obs):
#     maes, corrs = [], []
#     for t in range(num_timesteps):
#         pred_t = pred[t].flatten()
#         obs_t = obs[t].flatten()

#         mask = ~np.isnan(pred_t) & ~np.isnan(obs_t)
#         if np.sum(mask) > 0:
#             mae = np.mean(np.abs(pred_t[mask] - obs_t[mask]))
#             corr = np.corrcoef(pred_t[mask], obs_t[mask])[0, 1]
#         else:
#             mae, corr = np.nan, np.nan

#         maes.append(mae)
#         corrs.append(corr)
#     return np.array(maes), np.array(corrs)

# mae_mpi, corr_mpi = compute_metrics(ssh_mpi, ssh_obs)
# mae_miroc6, corr_miroc6 = compute_metrics(ssh_miroc6, ssh_obs)
# mae_ecearth, corr_ecearth = compute_metrics(ssh_ecearth, ssh_obs)

# print("MPI shape:", ssh_mpi.shape, "NaNs:", np.isnan(ssh_mpi).sum())
# print("MIROC6 shape:", ssh_miroc6.shape, "NaNs:", np.isnan(ssh_miroc6).sum())
# print("EC-EARTH shape:", ssh_ecearth.shape, "NaNs:", np.isnan(ssh_ecearth).sum())

# # Ensemble mean
# models = [ssh_mpi, ssh_miroc6, ssh_ecearth]
# valid_models = [m for m in models if m.size > 0 and not np.all(np.isnan(m))]

# if valid_models:
#     ssh_ensemble = np.nanmean(np.stack(valid_models), axis=0)
# else:
#     print("Semua model kosong atau NaN.")
#     ssh_ensemble = np.full_like(ssh_mpi, np.nan)

# mae_ens, corr_ens = compute_metrics(ssh_ensemble, ssh_obs)

# # Plot Korelasi
# plt.figure(figsize=(12, 4))
# plt.plot(corr_mpi, label='MPI')
# plt.plot(corr_miroc6, label='MIROC6')
# plt.plot(corr_ecearth, label='EC-EARTH')
# plt.plot(corr_ens, label='Ensemble', linestyle='--')
# plt.title("Korelasi Spasial per Timestep terhadap Observasi")
# plt.xlabel("Timestep")
# plt.ylabel("Pearson Correlation")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot MAE
# plt.figure(figsize=(12, 4))
# plt.plot(mae_mpi, label='MPI')
# plt.plot(mae_miroc6, label='MIROC6')
# plt.plot(mae_ecearth, label='EC-EARTH')
# plt.plot(mae_ens, label='Ensemble', linestyle='--')
# plt.title("MAE per Timestep terhadap Observasi")
# plt.xlabel("Timestep")
# plt.ylabel("MAE")
# plt.legend()
# plt.tight_layout()
# plt.show()
