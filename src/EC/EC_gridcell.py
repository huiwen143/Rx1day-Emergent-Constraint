#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-cell emergent constraint correction for Rx1day projections.

Required input:
1. CMIP6 NetCDF file containing:
   - data_his_all(model, lat, lon): historical Rx1day
   - data_future_all(model, lat, lon): future Rx1day

2. CMIP5 NetCDF file containing:
   - data_his_all(model, lat, lon): historical Rx1day
   - data_future_all(model, lat, lon): future Rx1day

3. Observational baseline NetCDF file containing:
   - rx1day_pred(lat, lon) for the DL-corrected baseline
     or another observational baseline variable, e.g. obs_his_all(lat, lon)

All inputs are assumed to be regridded to the same resolution and aligned
with identical latitude and longitude coordinates.
"""

import xarray as xr
import numpy as np
from scipy.stats import spearmanr
import os


scenario = "ssp585"

output_dir = "/path/to/output"
output_path = os.path.join(
    output_dir,
    f"EC_corrected_rx1day_cmip56_{scenario}_50NS.nc"
)

# === Load CMIP5 and CMIP6 data ===
ds_6 = xr.open_dataset("/path/to/cmip6_rx1day.nc")
ds_5 = xr.open_dataset("/path/to/cmip5_rx1day.nc")

# === Load observational historical baseline ===
# For DL-corrected baseline, use variable: rx1day_pred
ds_obs = xr.open_dataset("/path/to/observational_baseline.nc")
data_obs = ds_obs["rx1day_pred"].values

# If using another observational product, replace the variable name above, e.g.:
# data_obs = ds_obs["obs_his_all"].values

data_future_all_6 = ds_6["data_future_all"]
data_his_all_6 = ds_6["data_his_all"]  # (model, lat, lon)

data_future_all_5 = ds_5["data_future_all"]
data_his_all_5 = ds_5["data_his_all"]  # (model, lat, lon)

# Inputs are assumed to have already been regridded and coordinate-aligned.
data_his_all_6 = data_his_all_6.values
data_his_all_5 = data_his_all_5.values
data_future_all_6 = data_future_all_6.values
data_future_all_5 = data_future_all_5.values

data_his_all = np.concatenate([data_his_all_5, data_his_all_6], axis=0)
data_future_all = np.concatenate([data_future_all_5, data_future_all_6], axis=0)

lat, lon = ds_obs["lat"].values, ds_obs["lon"].values

dp = data_future_all - data_his_all

# === Flatten data to (model, grid) ===
M, H, W = data_his_all.shape
N = H * W

X = data_his_all.reshape(M, N)      # Historical Rx1day
Y = dp.reshape(M, N)                # Future Rx1day change
obs = data_obs.reshape(N)           # Observational historical baseline

# === Initialize output arrays ===
EC = np.full(N, np.nan)
R = np.full(N, np.nan)
P = np.full(N, np.nan)
A = np.full(N, np.nan)
B = np.full(N, np.nan)

# === Apply EC correction independently at each grid cell ===
for i in range(N):
    xi = X[:, i]
    yi = Y[:, i]
    x_obs = obs[i]

    valid = ~np.isnan(xi) & ~np.isnan(yi)

    if np.sum(valid) >= 5 and not np.isnan(x_obs):
        r, p = spearmanr(xi[valid], yi[valid])
        a, b = np.polyfit(xi[valid], yi[valid], 1)

        EC[i] = a * x_obs + b
        R[i] = r
        P[i] = p
        A[i] = a
        B[i] = b


# === Restore grid structure ===
history = np.nanmean(data_his_all, axis=0)
change_uncor = np.nanmean(data_future_all, axis=0) - np.nanmean(data_his_all, axis=0)
change_cor = EC.reshape(H, W)

R = R.reshape(H, W)
P = P.reshape(H, W)
A = A.reshape(H, W)
B = B.reshape(H, W)


# === Save results to NetCDF ===
ds_out = xr.Dataset({
    "history": xr.DataArray(history, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "change_uncor": xr.DataArray(change_uncor, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "change_cor": xr.DataArray(change_cor, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "R": xr.DataArray(R, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "P": xr.DataArray(P, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "A": xr.DataArray(A, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
    "B": xr.DataArray(B, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}),
})

os.makedirs(output_dir, exist_ok=True)
ds_out.to_netcdf(output_path)

print(f"Grid-cell EC correction saved to: {output_path}")