# Input Data Format

The training NetCDF file should contain the following variables:

- rx1day_imerg
- annual_max_tcwv
- elevation
- rx1day_label

## Dimensions

Required dimensions:

- time
- lat
- lon

## Variable descriptions

### rx1day_imerg
IMERG-based annual maximum daily precipitation.

### annual_max_tcwv
Annual maximum total column water vapor.

### elevation
Surface elevation (static variable).

### rx1day_label
Gauge-based Rx1day target values.

Missing values in `rx1day_label` are allowed and will be excluded from training.

## Land mask

The land mask NetCDF file must contain:

- land_mask_50NS

Non-NaN values indicate land pixels.

## Example shapes

- time: N years
- lat: 100
- lon: 360
