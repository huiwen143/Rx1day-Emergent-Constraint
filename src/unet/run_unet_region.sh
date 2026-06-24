#!/bin/bash
# Example SLURM script for training the U-Net Rx1day correction model with a
# geographically independent regional holdout test.

#SBATCH -J unet-rx1day-region
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Start time: $(date)"

# Update this path for your local conda installation.
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate rx1day-ec

# Update these paths for your input and output files.
INPUT_NC="/path/to/data_inputs_50NS_2010_2019.nc"
LAND_MASK_NC="/path/to/land_mask_50NS.nc"
OUT_DIR="/path/to/outputs_region"
OUT_NAME="predictions_region_mean.nc"

# Example regional holdout bounds. Edit these for the region being tested.
TEST_LAT_MIN=-20
TEST_LAT_MAX=50
TEST_LON_MIN=60
TEST_LON_MAX=150

mkdir -p logs

python -u src/unet/train.py \
  --netcdf "${INPUT_NC}" \
  --land_mask_nc "${LAND_MASK_NC}" \
  --out_dir "${OUT_DIR}" \
  --out_name "${OUT_NAME}" \
  --epochs 100 \
  --batch_size 1 \
  --lr 2e-4 \
  --seed 12345 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --test_lat_min "${TEST_LAT_MIN}" \
  --test_lat_max "${TEST_LAT_MAX}" \
  --test_lon_min "${TEST_LON_MIN}" \
  --test_lon_max "${TEST_LON_MAX}"

echo "End time: $(date)"
