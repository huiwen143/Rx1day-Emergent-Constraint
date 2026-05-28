#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet for Rx1day correction (IMERG -> station label) on 1° grid, using multi-year mean.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------
# Utility functions
# ----------------------
def set_seed(seed: int = 12345):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

# ----------------------
# Data handling
# ----------------------
@dataclass
class NormStats:
    """Normalization statistics for input channels."""
    mean: np.ndarray
    std: np.ndarray

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: str) -> "NormStats":
        data = np.load(path)
        return NormStats(mean=data["mean"], std=data["std"])

class Rx1dayMeanDataset(Dataset):
    """Dataset class for Rx1day multi-year mean correction."""

    def __init__(self, ds: xr.Dataset, land_mask_ds: xr.Dataset,
                 norm_stats: Optional[NormStats] = None,
                 compute_norm: bool = False,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int = 12345,
                 test_lat_range: tuple = None,
                 test_lon_range: tuple = None):

        # Ensure required variables exist
        assert {"rx1day_imerg", "annual_max_tcwv", "elevation", "rx1day_label"} <= set(ds.variables)

        self.ds = ds
        self.lat = ds["lat"].values
        self.lon = ds["lon"].values
        self.land_mask = ~np.isnan(land_mask_ds["land_mask_50NS"].values)
        self.ds["elevation"] = self.ds["elevation"].fillna(0)

        # Compute normalization if required
        if norm_stats is None and compute_norm:
            X = np.stack([
                ds["rx1day_imerg"].values.astype(np.float32),
                ds["annual_max_tcwv"].values.astype(np.float32),
                ds["elevation"].values.astype(np.float32)
            ], axis=0)
            mean_list, std_list = [], []
            for c in range(2):
                vals = X[c][self.land_mask]
                vals = vals[~np.isnan(vals)]
                mean_list.append(np.mean(vals))
                std_list.append(np.std(vals) + 1e-6)
            self.norm = NormStats(mean=np.array(mean_list, dtype=np.float32),
                                  std=np.array(std_list, dtype=np.float32))
        else:
            self.norm = norm_stats

        # Normalize label
        y_raw = ds["rx1day_label"].values.astype(np.float32)
        y_filled = np.where(np.isnan(y_raw), ds["rx1day_imerg"].values, y_raw)
        self.norm_y = {"mean": np.mean(y_filled[self.land_mask]),
                       "std": np.std(y_filled[self.land_mask]) + 1e-6}

        # Split into train/val/test pixels
        np.random.seed(seed)
        y_raw = ds["rx1day_label"].values
        valid_pixels = np.where(self.land_mask & ~np.isnan(y_raw))
        valid_indices = np.ravel_multi_index(valid_pixels, y_raw.shape)
        n_valid = len(valid_indices)

        # Define test mask using specified lat/lon or random
        if test_lat_range and test_lon_range:
            lat_min, lat_max = test_lat_range
            lon_min, lon_max = test_lon_range
            test_mask = np.zeros_like(self.land_mask, dtype=bool)
            lat_idx = np.where((self.lat >= lat_min) & (self.lat <= lat_max))[0]
            lon_idx = np.where((self.lon >= lon_min) & (self.lon <= lon_max))[0]
            test_mask[np.ix_(lat_idx, lon_idx)] = True
            test_mask = test_mask & self.land_mask & ~np.isnan(y_raw)
        else:
            test_mask = np.zeros_like(self.land_mask, dtype=bool)
            n_test = int(n_valid * test_ratio)
            test_indices = np.random.choice(valid_indices, n_test, replace=False)
            test_mask.flat[test_indices] = True

        train_val_mask = self.land_mask & ~np.isnan(y_raw) & ~test_mask
        train_val_indices = np.ravel_multi_index(np.where(train_val_mask), y_raw.shape)
        n_train_val = len(train_val_indices)
        n_val = int(n_train_val * val_ratio)
        n_train = n_train_val - n_val

        shuffled_indices = np.random.permutation(train_val_indices)
        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:n_train + n_val]

        self.train_mask = np.zeros_like(self.land_mask, dtype=bool)
        self.train_mask.flat[train_indices] = True
        self.val_mask = np.zeros_like(self.land_mask, dtype=bool)
        self.val_mask.flat[val_indices] = True
        self.test_mask = test_mask

    def __len__(self):
        return 1  # single mean sample

    def __getitem__(self, idx: int):
        # Prepare input tensor
        imerg = self.ds["rx1day_imerg"].values.astype(np.float32)
        tcwv = self.ds["annual_max_tcwv"].values.astype(np.float32)
        elev = self.ds["elevation"].values.astype(np.float32)
        elev_min, elev_max = np.percentile(elev[self.land_mask], [5, 95])
        elev = (np.clip(elev, elev_min, elev_max) - elev_min) / (elev_max - elev_min + 1e-6)
        X = np.stack([imerg, tcwv, elev], axis=0)

        # Normalize label
        y_raw = self.ds["rx1day_label"].values.astype(np.float32)
        y_raw = y_raw + 7
        y_filled = np.where(np.isnan(y_raw), imerg, y_raw)
        y = (y_filled - self.norm_y["mean"]) / self.norm_y["std"]
        y = y[None, ...]

        # Normalize input channels
        if self.norm is not None:
            mean = self.norm.mean[:, None, None]
            std = self.norm.std[:, None, None]
            X[:2] = (X[:2] - mean) / (std + 1e-4)

        mask = torch.tensor(self.land_mask, dtype=torch.bool)[None, ...]
        train_mask = torch.tensor(self.train_mask, dtype=torch.bool)[None, ...]
        val_mask = torch.tensor(self.val_mask, dtype=torch.bool)[None, ...]
        test_mask = torch.tensor(self.test_mask, dtype=torch.bool)[None, ...]
        return torch.from_numpy(X), torch.from_numpy(y), mask, train_mask, val_mask, test_mask, 0

# ----------------------
# Model: UNet
# ----------------------
class DoubleConv(nn.Module):
    """Basic double convolution block with batch normalization and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    """Downsampling block with max pooling followed by double conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    """Upsampling block with optional bilinear upsample and skip connection."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """Full UNet architecture for Rx1day correction."""
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512], bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3]*2//factor)
        self.up1 = Up(features[3]*2, features[3]//factor, bilinear)
        self.up2 = Up(features[3], features[2]//factor, bilinear)
        self.up3 = Up(features[2], features[1]//factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ----------------------
# Training and Evaluation
# ----------------------
@torch.no_grad()
def evaluate(model, loader, device, use_val_mask=False) -> Dict[str,float]:
    """Evaluate model performance with MAE and RMSE metrics."""
    model.eval()
    m_mae, m_rmse, n = 0.0,0.0,0
    for X,y,mask,train_m,val_m,test_m,_ in loader:
        X,y = X.to(device), y.to(device)
        eval_mask = val_m.to(device) if use_val_mask else train_m.to(device)
        pred = model(X)
        pred = pred*loader.dataset.norm_y["std"] + loader.dataset.norm_y["mean"]
        y = y*loader.dataset.norm_y["std"] + loader.dataset.norm_y["mean"]
        diff = pred-y
        masked_sum = eval_mask.sum().clamp_min(1.0)
        mae_loss = (torch.abs(diff)*eval_mask).sum()/masked_sum
        rmse_loss = torch.sqrt(((diff**2)*eval_mask).sum()/masked_sum)
        m_mae += mae_loss.item()*X.size(0)
        m_rmse += rmse_loss.item()*X.size(0)
        n += X.size(0)
    return {"mae": m_mae/n, "rmse": m_rmse/n}

def save_predictions_to_netcdf(model, loader, land_mask_ds, device, out_nc_path:str, test_matrix):
    """Save predicted Rx1day to NetCDF including test mask and values."""
    model.eval()
    lat, lon = loader.dataset.lat, loader.dataset.lon
    preds = []
    with torch.no_grad():
        for X, _, _, _, _, _, _ in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()[:,0]
            pred = pred*loader.dataset.norm_y["std"] + loader.dataset.norm_y["mean"]
            land_mask = ~np.isnan(land_mask_ds["land_mask_50NS"].values)
            for i in range(pred.shape[0]):
                pred[i][~land_mask]=np.nan
            preds.append(pred)
    preds = np.squeeze(np.array(preds))
    test_mask = loader.dataset.test_mask.astype(np.float32)
    test_mask[~loader.dataset.land_mask] = np.nan
    ds_out = xr.Dataset({
        "rx1day_pred": (("lat","lon"), preds.astype(np.float32)),
        "test_mask": (("lat","lon"), test_mask),
        "test_values": (("point","variable"), test_matrix.astype(np.float32))
    },
    coords={"lat":lat, "lon":lon,"point":np.arange(test_matrix.shape[0]),"variable":["pred","imerg","label"]})
    ds_out["rx1day_pred"].attrs["description"] = "Predicted Rx1day values after correction"
    ds_out["test_mask"].attrs["description"] = "Mask of independent test pixels (1=test, NaN=non-test/sea)"
    ds_out.to_netcdf(out_nc_path)

# ----------------------
# Argument parsing
# ----------------------
def parse_args():
    """Parse input arguments for training script."""
    p = argparse.ArgumentParser()
    p.add_argument("--netcdf", type=str, required=True)
    p.add_argument("--land_mask_nc", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--test_lat_min", type=float, default=None, help="Minimum latitude for test region")
    p.add_argument("--test_lat_max", type=float, default=None, help="Maximum latitude for test region")
    p.add_argument("--test_lon_min", type=float, default=None, help="Minimum longitude for test region")
    p.add_argument("--test_lon_max", type=float, default=None, help="Maximum longitude for test region")
    p.add_argument("--out_name", type=str, default="predictions_region.nc",
                   help="Name of output prediction file")
    return p.parse_args()

# ----------------------
# Main execution
# ----------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load input dataset and compute multi-year mean
    ds = xr.open_dataset(args.netcdf, engine="netcdf4")
    ds_mean = ds.mean(dim="time", keep_attrs=True)
    ds_mean["elevation"] = ds["elevation"]

    # Load land mask
    land_mask_ds = xr.open_dataset(args.land_mask_nc, engine="netcdf4")

    # Create dataset object
    test_lat_range = (args.test_lat_min,args.test_lat_max) if args.test_lat_min is not None and args.test_lat_max is not None else None
    test_lon_range = (args.test_lon_min,args.test_lon_max) if args.test_lon_min is not None and args.test_lon_max is not None else None
    mean_set = Rx1dayMeanDataset(ds_mean, land_mask_ds, norm_stats=None,
                                 compute_norm=True, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                                 seed=args.seed, test_lat_range=test_lat_range, test_lon_range=test_lon_range)
    norm = mean_set.norm
    norm.save(os.path.join(args.out_dir,"norm_stats.npz"))

    # DataLoader setup
    mean_loader = DataLoader(mean_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Model, optimizer, loss, scheduler
    model = UNet(in_channels=3,out_channels=1,features=[64,128,256,512],bilinear=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(beta=1.0,reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    best_val, patience, wait = float("inf"), 15, 0
    history = {"train_loss":[],"val_mae":[],"val_rmse":[]}
    test_matrix = None

    # Training loop
    for epoch in range(1,args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for step,(X,y,mask,train_m,val_m,test_m,_) in enumerate(mean_loader):
            X,y,mask = X.to(device),y.to(device),mask.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred,y)
            train_mask = train_m.to(device)&mask
            weights = torch.where(y>2,2.0,1.0)  # Amplify loss for large values
            masked_loss = (loss*train_mask*weights).sum()/(train_mask*weights).sum().clamp_min(1.0)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            epoch_loss += masked_loss.item()
        metrics
