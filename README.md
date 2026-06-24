# Rx1day Emergent Constraint

Brief description.

## Repository structure

- `src/unet/`: U-Net reconstruction code, demo data, and input format documentation.
- `src/EC/`: grid-cell emergent-constraint code and input format documentation.
- `src/visualization/`: plotting scripts for prepared manuscript outputs.

## Requirements

Tested with Python 3.10 on Linux/Windows.
Dependencies: numpy, xarray, netCDF4, torch, scipy, matplotlib, cartopy.
No non-standard hardware is required for the demo; GPU is optional for full-size training.

## Installation

conda/pip commands.

Typical installation time: 10-30 minutes.

## Demo

One short command to create demo data.
One short command to run U-Net demo.
Expected output: list 3-4 files.
Expected runtime: several minutes.

## Running on user data

Prepare files according to:
- `src/unet/DATA_FORMAT.md`
- `src/EC/DATA_FORMAT.md`

Then run `src/unet/train.py` or adapt `src/unet/run_unet_region.sh`.

## Data availability

Demo data are simulated. Full manuscript datasets are not redistributed because of size and third-party data terms.

## License

MIT License.
