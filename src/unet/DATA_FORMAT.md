# Input and Output Data Format

This document describes the NetCDF files required by `src/unet/train.py` and the files produced by the U-Net training workflow.

## Training Input NetCDF

The training input file is passed with:

```bash
--netcdf /path/to/data_inputs_50NS_2010_2019.nc
```

Required dimensions:

- `time`: annual samples or years used to compute the multi-year mean
- `lat`: latitude
- `lon`: longitude

Required variables:

- `rx1day_imerg(time, lat, lon)`: IMERG-based annual maximum daily precipitation, expected in mm/day.
- `annual_max_tcwv(time, lat, lon)`: annual maximum total column water vapor, expected in kg m-2 or the unit used consistently in preprocessing.
- `elevation(lat, lon)` or `elevation(time, lat, lon)`: surface elevation, expected in meters. If a time dimension is present, the script keeps the static elevation field after computing multi-year means.
- `rx1day_label(time, lat, lon)`: gauge-based Rx1day target values, expected in mm/day.

Missing values in `rx1day_label` are allowed. They are excluded from train, validation, and test masks. Missing elevation values are filled with zero before elevation normalization.

## Land Mask NetCDF

The land mask file is passed with:

```bash
--land_mask_nc /path/to/land_mask_50NS.nc
```

Required dimensions:

- `lat`
- `lon`

Required variable:

- `land_mask_50NS(lat, lon)`: land mask. Non-NaN values indicate land pixels; NaN values indicate ocean or excluded pixels.

The land mask must be on the same grid as the training input file.

## Coordinates

The training input and land mask must use aligned `lat` and `lon` coordinates. The example workflow assumes the 50S-50N domain used in the manuscript analysis.

Longitude may be stored as either `0-360` or `-180-180`, but all input files used in the same run must use the same convention and grid ordering.

Example full-size shape:

- `time`: number of years
- `lat`: 100
- `lon`: 360

## U-Net Output Files

The output directory is passed with:

```bash
--out_dir /path/to/outputs
```

The training script writes:

- `best_model.pt`: PyTorch state dictionary for the best validation model.
- `metrics.json`: training loss and validation metrics by epoch.
- `norm_stats.npz`: normalization statistics for the IMERG Rx1day and TCWV input channels.
- `<out_name>`: NetCDF file specified by `--out_name`.

The output NetCDF contains:

- `rx1day_pred(lat, lon)`: reconstructed bias-corrected Rx1day climatology, in mm/day.
- `test_mask(lat, lon)`: independent test pixels. Values are `1` for test pixels, `0` for non-test land pixels, and NaN for ocean pixels.
- `test_values(point, variable)`: values at independent test pixels. The `variable` coordinate is `pred`, `imerg`, and `label`.

## Regional Holdout Testing

If the following arguments are provided, the independent test mask is defined by latitude and longitude bounds:

```bash
--test_lat_min -20
--test_lat_max 50
--test_lon_min 60
--test_lon_max 150
```

If these bounds are not provided, the script creates a random independent test split using `--test_ratio`.

## Demonstration Dataset

A small demonstration dataset should follow the same variable names and dimensions but may use a much smaller grid and shorter time axis. It is intended only to verify installation and script execution, not to reproduce the full manuscript results.
