#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a U-Net model to reconstruct bias-corrected Rx1day climatology.

This script uses multi-year mean fields from the input NetCDF file and supports
either a random independent test split or a user-defined geographic test region.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 12345) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


@dataclass
class NormStats:
    """Normalization statistics for the dynamic input channels."""

    mean: np.ndarray
    std: np.ndarray

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: str) -> "NormStats":
        data = np.load(path)
        return NormStats(mean=data["mean"], std=data["std"])


class Rx1dayMeanDataset(Dataset):
    """Single-sample dataset containing multi-year mean Rx1day predictors."""

    def __init__(
        self,
        ds: xr.Dataset,
        land_mask_ds: xr.Dataset,
        norm_stats: Optional[NormStats] = None,
        compute_norm: bool = False,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 12345,
        test_lat_range: Optional[Tuple[float, float]] = None,
        test_lon_range: Optional[Tuple[float, float]] = None,
    ):
        required = {"rx1day_imerg", "annual_max_tcwv", "elevation", "rx1day_label"}
        missing = required - set(ds.variables)
        if missing:
            raise ValueError(f"Missing required variables in input dataset: {sorted(missing)}")
        if "land_mask_50NS" not in land_mask_ds.variables:
            raise ValueError("Missing required variable in land mask dataset: land_mask_50NS")

        self.ds = ds
        self.lat = ds["lat"].values
        self.lon = ds["lon"].values
        self.land_mask = ~np.isnan(land_mask_ds["land_mask_50NS"].values)
        self.ds["elevation"] = self.ds["elevation"].fillna(0)

        if norm_stats is None and compute_norm:
            x_values = np.stack(
                [
                    ds["rx1day_imerg"].values.astype(np.float32),
                    ds["annual_max_tcwv"].values.astype(np.float32),
                    ds["elevation"].values.astype(np.float32),
                ],
                axis=0,
            )
            mean_list = []
            std_list = []
            for channel in range(2):
                values = x_values[channel][self.land_mask]
                values = values[~np.isnan(values)]
                mean_list.append(np.mean(values))
                std_list.append(np.std(values) + 1e-6)
            self.norm = NormStats(
                mean=np.array(mean_list, dtype=np.float32),
                std=np.array(std_list, dtype=np.float32),
            )
        else:
            self.norm = norm_stats

        y_raw = ds["rx1day_label"].values.astype(np.float32)
        y_filled = np.where(np.isnan(y_raw), ds["rx1day_imerg"].values, y_raw)
        self.norm_y = {
            "mean": np.mean(y_filled[self.land_mask]),
            "std": np.std(y_filled[self.land_mask]) + 1e-6,
        }

        rng = np.random.default_rng(seed)
        valid_pixels = np.where(self.land_mask & ~np.isnan(y_raw))
        valid_indices = np.ravel_multi_index(valid_pixels, y_raw.shape)
        n_valid = len(valid_indices)
        if n_valid == 0:
            raise ValueError("No valid land pixels with rx1day_label were found.")

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
            test_indices = rng.choice(valid_indices, n_test, replace=False)
            test_mask.flat[test_indices] = True

        train_val_mask = self.land_mask & ~np.isnan(y_raw) & ~test_mask
        train_val_indices = np.ravel_multi_index(np.where(train_val_mask), y_raw.shape)
        if len(train_val_indices) == 0:
            raise ValueError("No train/validation pixels remain after applying the test mask.")

        n_val = int(len(train_val_indices) * val_ratio)
        shuffled_indices = rng.permutation(train_val_indices)
        train_indices = shuffled_indices[: len(train_val_indices) - n_val]
        val_indices = shuffled_indices[len(train_val_indices) - n_val :]

        self.train_mask = np.zeros_like(self.land_mask, dtype=bool)
        self.train_mask.flat[train_indices] = True
        self.val_mask = np.zeros_like(self.land_mask, dtype=bool)
        self.val_mask.flat[val_indices] = True
        self.test_mask = test_mask

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        imerg = self.ds["rx1day_imerg"].values.astype(np.float32)
        tcwv = self.ds["annual_max_tcwv"].values.astype(np.float32)
        elev = self.ds["elevation"].values.astype(np.float32)

        elev_min, elev_max = np.percentile(elev[self.land_mask], [5, 95])
        elev = np.clip(elev, elev_min, elev_max)
        elev = (elev - elev_min) / (elev_max - elev_min + 1e-6)

        x_values = np.stack([imerg, tcwv, elev], axis=0)

        y_raw = self.ds["rx1day_label"].values.astype(np.float32)
        y_filled = np.where(np.isnan(y_raw), imerg, y_raw)
        y_values = (y_filled - self.norm_y["mean"]) / self.norm_y["std"]
        y_values = y_values[None, ...]

        if self.norm is not None:
            mean = self.norm.mean[:, None, None]
            std = self.norm.std[:, None, None]
            x_values[:2] = (x_values[:2] - mean) / (std + 1e-4)

        mask = torch.tensor(self.land_mask, dtype=torch.bool)[None, ...]
        train_mask = torch.tensor(self.train_mask, dtype=torch.bool)[None, ...]
        val_mask = torch.tensor(self.val_mask, dtype=torch.bool)[None, ...]
        test_mask = torch.tensor(self.test_mask, dtype=torch.bool)[None, ...]
        return (
            torch.from_numpy(x_values),
            torch.from_numpy(y_values),
            mask,
            train_mask,
            val_mask,
            test_mask,
            0,
        )


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU activation."""

    def __init__(self, in_ch: int, out_ch: int):
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

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downsampling block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """U-Net architecture for Rx1day correction."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[list] = None,
        bilinear: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
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


@torch.no_grad()
def evaluate(model, loader, device, mask_name: str = "val") -> Dict[str, float]:
    """Evaluate MAE and RMSE on train, validation, or test pixels."""
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    n_batches = 0

    for x_values, y_values, mask, train_m, val_m, test_m, _ in loader:
        x_values = x_values.to(device)
        y_values = y_values.to(device)
        mask = mask.to(device)
        masks = {
            "train": train_m.to(device),
            "val": val_m.to(device),
            "test": test_m.to(device),
        }
        eval_mask = masks[mask_name] & mask

        pred = model(x_values)
        pred = pred * loader.dataset.norm_y["std"] + loader.dataset.norm_y["mean"]
        y_values = y_values * loader.dataset.norm_y["std"] + loader.dataset.norm_y["mean"]
        diff = pred - y_values
        masked_sum = eval_mask.sum().clamp_min(1.0)

        mae = (torch.abs(diff) * eval_mask).sum() / masked_sum
        rmse = torch.sqrt(((diff**2) * eval_mask).sum() / masked_sum)
        total_mae += mae.item()
        total_rmse += rmse.item()
        n_batches += 1

    return {"mae": total_mae / n_batches, "rmse": total_rmse / n_batches}


@torch.no_grad()
def collect_test_values(model, loader, device) -> np.ndarray:
    """Collect predicted, IMERG, and label values at independent test pixels."""
    model.eval()
    dataset = loader.dataset

    for x_values, y_values, mask, _, _, test_m, _ in loader:
        x_values = x_values.to(device)
        y_values = y_values.to(device)
        mask = mask.to(device)
        test_mask = test_m.to(device) & mask

        pred = model(x_values)
        pred = pred * dataset.norm_y["std"] + dataset.norm_y["mean"]
        y_values = y_values * dataset.norm_y["std"] + dataset.norm_y["mean"]
        imerg = x_values[:, 0:1] * dataset.norm.std[0] + dataset.norm.mean[0]

        pred_test = pred[test_mask]
        imerg_test = imerg[test_mask]
        label_test = y_values[test_mask]
        if len(label_test) == 0:
            raise ValueError("No independent test pixels were found.")

        return np.stack(
            [
                pred_test.detach().cpu().numpy(),
                imerg_test.detach().cpu().numpy(),
                label_test.detach().cpu().numpy(),
            ],
            axis=1,
        )

    raise ValueError("The data loader did not return any batches.")


@torch.no_grad()
def save_predictions_to_netcdf(
    model,
    loader,
    land_mask_ds,
    device,
    out_nc_path: str,
    test_values: np.ndarray,
) -> None:
    """Save reconstructed Rx1day, test mask, and test values to NetCDF."""
    model.eval()
    dataset = loader.dataset
    preds = []

    for x_values, _, _, _, _, _, _ in loader:
        x_values = x_values.to(device)
        pred = model(x_values).cpu().numpy()[:, 0]
        pred = pred * dataset.norm_y["std"] + dataset.norm_y["mean"]
        land_mask = ~np.isnan(land_mask_ds["land_mask_50NS"].values)
        for batch_index in range(pred.shape[0]):
            pred[batch_index][~land_mask] = np.nan
        preds.append(pred)

    preds = np.squeeze(np.array(preds))
    test_mask = dataset.test_mask.astype(np.float32)
    test_mask[~dataset.land_mask] = np.nan

    ds_out = xr.Dataset(
        {
            "rx1day_pred": (("lat", "lon"), preds.astype(np.float32)),
            "test_mask": (("lat", "lon"), test_mask),
            "test_values": (("point", "variable"), test_values.astype(np.float32)),
        },
        coords={
            "lat": dataset.lat,
            "lon": dataset.lon,
            "point": np.arange(test_values.shape[0]),
            "variable": ["pred", "imerg", "label"],
        },
    )
    ds_out["rx1day_pred"].attrs["description"] = "Predicted Rx1day values after correction"
    ds_out["test_mask"].attrs["description"] = "Independent test pixels: 1=test, 0=non-test, NaN=ocean"
    ds_out["test_values"].attrs["description"] = "Predicted, IMERG, and label values at independent test pixels"
    ds_out.to_netcdf(out_nc_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for Rx1day climatology reconstruction."
    )
    parser.add_argument("--netcdf", type=str, required=True, help="Input training NetCDF file.")
    parser.add_argument("--land_mask_nc", type=str, required=True, help="Land mask NetCDF file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--out_name", type=str, default="predictions_region_mean.nc")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--test_lat_min", type=float, default=None)
    parser.add_argument("--test_lat_max", type=float, default=None)
    parser.add_argument("--test_lon_min", type=float, default=None)
    parser.add_argument("--test_lon_max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = xr.open_dataset(args.netcdf, engine="netcdf4")
    ds_mean = ds.mean(dim="time", keep_attrs=True)
    ds_mean["elevation"] = ds["elevation"]

    land_mask_ds = xr.open_dataset(args.land_mask_nc, engine="netcdf4")

    test_lat_range = (
        (args.test_lat_min, args.test_lat_max)
        if args.test_lat_min is not None and args.test_lat_max is not None
        else None
    )
    test_lon_range = (
        (args.test_lon_min, args.test_lon_max)
        if args.test_lon_min is not None and args.test_lon_max is not None
        else None
    )

    mean_set = Rx1dayMeanDataset(
        ds_mean,
        land_mask_ds,
        norm_stats=None,
        compute_norm=True,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_lat_range=test_lat_range,
        test_lon_range=test_lon_range,
    )
    mean_set.norm.save(os.path.join(args.out_dir, "norm_stats.npz"))

    mean_loader = DataLoader(
        mean_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512], bilinear=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    best_val = float("inf")
    patience = 15
    wait = 0
    history = {"train_loss": [], "val_mae": [], "val_rmse": []}
    best_model_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x_values, y_values, mask, train_m, _, _, _ in mean_loader:
            x_values = x_values.to(device)
            y_values = y_values.to(device)
            mask = mask.to(device)
            train_mask = train_m.to(device) & mask

            optimizer.zero_grad()
            pred = model(x_values)
            loss = criterion(pred, y_values)
            weights = torch.where(y_values > 2, 2.0, 1.0)
            masked_loss = (loss * train_mask * weights).sum() / (
                train_mask * weights
            ).sum().clamp_min(1.0)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += masked_loss.item()

        metrics = evaluate(model, mean_loader, device, mask_name="val")
        val_loss = metrics["rmse"]
        history["train_loss"].append(epoch_loss)
        history["val_mae"].append(metrics["mae"])
        history["val_rmse"].append(metrics["rmse"])
        print(
            f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f} "
            f"| val_mae={metrics['mae']:.4f} | val_rmse={metrics['rmse']:.4f}"
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, mean_loader, device, mask_name="test")
    print(f"[Final test] MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}")

    test_values = collect_test_values(model, mean_loader, device)
    out_nc = os.path.join(args.out_dir, args.out_name)
    save_predictions_to_netcdf(model, mean_loader, land_mask_ds, device, out_nc, test_values)
    print(f"Saved predictions to: {out_nc}")


if __name__ == "__main__":
    main()
