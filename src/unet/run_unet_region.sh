#!/bin/bash
#SBATCH -J unet-rx1day-test
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/path/to/logs/%j.out
#SBATCH --error=/path/to/logs/%j.err

# Example SLURM script for training the U-Net Rx1day correction model
# with a geographically independent test region.

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/env

python -u src/unet/train.py \
    --netcdf /path/to/data_inputs_50NS_2010_2019.nc \
    --land_mask_nc /path/to/land_mask_50NS.nc \
    --out_dir /path/to/outputs_mean \
    --out_name predictions_region.nc \
    --epochs 100 \
    --batch_size 1 \
    --lr 2e-4 \
    --seed 12345 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    # Specify the latitude and longitude bounds for the independent test region.
    --test_lat_min -20 \   # Minimum latitude of test region (south boundary)
    --test_lat_max 20 \    # Maximum latitude of test region (north boundary)
    --test_lon_min 0 \     # Minimum longitude of test region (west boundary)
    --test_lon_max 60      # Maximum longitude of test region (east boundary)

echo "End time: $(date)"