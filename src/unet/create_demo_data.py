#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a small simulated NetCDF dataset for testing the U-Net workflow.

The generated files are not scientific input data. They only demonstrate the
required file structure and allow the training script to run on a small example.
"""

from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(12345)

    time = np.arange(4)
    lat = np.linspace(-20.0, 20.0, 16)
    lon = np.linspace(0.5, 47.5, 24)

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    base_pattern = 45.0 + 12.0 * np.cos(np.deg2rad(lat_grid))
    base_pattern += 4.0 * np.sin(np.deg2rad(lon_grid * 3.0))

    elevation = 500.0 + 300.0 * np.sin(np.deg2rad(lat_grid * 2.0))
    elevation += rng.normal(0.0, 30.0, size=elevation.shape)
    elevation = np.clip(elevation, 0.0, None).astype(np.float32)

    rx1day_imerg = []
    annual_max_tcwv = []
    rx1day_label = []

    for year_index in range(len(time)):
        year_shift = year_index * 0.8
        imerg_year = base_pattern + year_shift + rng.normal(0.0, 3.0, size=base_pattern.shape)
        tcwv_year = 35.0 + 0.25 * base_pattern + rng.normal(0.0, 1.5, size=base_pattern.shape)
        label_year = imerg_year * 0.9 + 6.0 + 0.002 * elevation
        label_year += rng.normal(0.0, 2.0, size=base_pattern.shape)

        rx1day_imerg.append(imerg_year.astype(np.float32))
        annual_max_tcwv.append(tcwv_year.astype(np.float32))
        rx1day_label.append(label_year.astype(np.float32))

    rx1day_imerg = np.stack(rx1day_imerg, axis=0)
    annual_max_tcwv = np.stack(annual_max_tcwv, axis=0)
    rx1day_label = np.stack(rx1day_label, axis=0)

    missing_mask = rng.random(size=rx1day_label.shape) < 0.15
    rx1day_label[missing_mask] = np.nan

    ds_train = xr.Dataset(
        {
            "rx1day_imerg": (("time", "lat", "lon"), rx1day_imerg),
            "annual_max_tcwv": (("time", "lat", "lon"), annual_max_tcwv),
            "elevation": (("lat", "lon"), elevation),
            "rx1day_label": (("time", "lat", "lon"), rx1day_label),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds_train["rx1day_imerg"].attrs["units"] = "mm/day"
    ds_train["annual_max_tcwv"].attrs["units"] = "kg m-2"
    ds_train["elevation"].attrs["units"] = "m"
    ds_train["rx1day_label"].attrs["units"] = "mm/day"

    land_mask = np.ones((len(lat), len(lon)), dtype=np.float32)
    land_mask[:2, :4] = np.nan
    land_mask[-2:, -4:] = np.nan

    ds_mask = xr.Dataset(
        {"land_mask_50NS": (("lat", "lon"), land_mask)},
        coords={"lat": lat, "lon": lon},
    )

    train_path = out_dir / "data_inputs_50NS_demo.nc"
    mask_path = out_dir / "land_mask_50NS_demo.nc"

    ds_train.to_netcdf(train_path)
    ds_mask.to_netcdf(mask_path)

    print(f"Created {train_path}")
    print(f"Created {mask_path}")


if __name__ == "__main__":
    main()
