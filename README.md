# Rx1day Emergent Constraint

This repository contains Python code and a small simulated demo dataset for reconstructing bias-corrected historical Rx1day climatology and applying emergent-constraint analysis to future precipitation extremes.

## Overview

The code is organized into three parts: U-Net-based historical Rx1day reconstruction, grid-cell emergent-constraint analysis, and plotting scripts for prepared manuscript outputs. The included demo data are simulated and are intended only to test the code workflow and file formats. Full manuscript input datasets are not redistributed because of file size and third-party data-use terms.

## Repository Structure

- `src/unet/`: U-Net training code, demo data, and input/output format documentation.
- `src/EC/`: grid-cell emergent-constraint code and required input format documentation.
- `src/visualization/`: plotting scripts for prepared manuscript outputs.

Key files:

- `src/unet/train.py`: trains the U-Net model and writes reconstructed Rx1day outputs.
- `src/unet/create_demo_data.py`: creates the simulated demo dataset.
- `src/unet/run_unet_region.sh`: example SLURM script for a regional holdout run.
- `src/unet/DATA_FORMAT.md`: input/output format for the U-Net workflow.
- `src/EC/DATA_FORMAT.md`: input/output format for the emergent-constraint workflow.

## Requirements

Tested with Python 3.10.

Required Python packages:

- numpy
- xarray
- netCDF4
- torch
- scipy
- matplotlib
- cartopy

No non-standard hardware is required for the demo. A GPU is optional and may be useful for full-size model training.

## Installation

```bash
conda create -n rx1day-ec python=3.10
conda activate rx1day-ec
conda install -c conda-forge numpy xarray netcdf4 scipy matplotlib cartopy
pip install torch
