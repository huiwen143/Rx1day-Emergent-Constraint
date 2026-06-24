# Emergent-Constraint Input Data Format

`EC_gridcell.py` requires prepared NetCDF files. The full manuscript input files are not redistributed with this repository because of file size and third-party data-use terms.

## CMIP Input Files

The script uses one CMIP5 file and one CMIP6 file. Each file must contain:

- `data_his_all(model, lat, lon)`: historical Rx1day from individual models.
- `data_future_all(model, lat, lon)`: future Rx1day from the same models.

Expected units are mm/day for both variables. The future-change field is calculated in the script as:

```text
data_future_all - data_his_all
```

## Observational Baseline File

The observational baseline NetCDF must contain one gridded historical Rx1day baseline on the same grid as the CMIP files.

Default variable:

- `rx1day_pred(lat, lon)`: U-Net reconstructed historical Rx1day climatology.

Alternative variable if using another observational product:

- `obs_his_all(lat, lon)`

If using `obs_his_all`, update the variable name in `EC_gridcell.py`.

## Grid Requirements

All input files must be regridded and aligned before running the EC script:

- same `lat` coordinate
- same `lon` coordinate
- same grid shape
- same longitude convention
- same land/ocean mask treatment where applicable

## Output File

The EC script writes one NetCDF file containing:

- `history(lat, lon)`: multi-model mean historical Rx1day.
- `change_uncor(lat, lon)`: unconstrained multi-model mean future change.
- `change_cor(lat, lon)`: emergent-constrained future change.
- `R(lat, lon)`: Spearman correlation at each grid cell.
- `P(lat, lon)`: p-value of the Spearman correlation.
- `A(lat, lon)`: linear regression slope.
- `B(lat, lon)`: linear regression intercept.

Grid cells with fewer than five valid model pairs or missing observational baseline values are written as NaN.
