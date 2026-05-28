import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch


# ==============================================================================
# 1. Input and output paths
# ==============================================================================

# Global station-based input data.
# Required variables:
# - rx1day_label(time, lat, lon): gauge-based Rx1day reference
# - rx1day_imerg(time, lat, lon): IMERG Rx1day estimate
nc_input_path = "/path/to/data_inputs_50NS_2010_2019.nc"

# Land mask file.
# Required variable:
# - land_mask_50NS(lat, lon): non-NaN values indicate land pixels
mask_path = "/path/to/land_mask_50NS.nc"

# U-Net reconstructed historical Rx1day climatology.
# Required variable:
# - rx1day_pred(lat, lon): bias-corrected Rx1day climatology
pred_mean_path = "/path/to/predictions_mean.nc"

# CMIP historical Rx1day files.
# Required variable:
# - data_his_all(model, lat, lon): historical Rx1day from individual models
cmip6_path = "/path/to/rel_change_rx1day_cmip6_ssp245_50NS.nc"
cmip5_path = "/path/to/rel_change_rx1day_cmip5_ssp245_50NS.nc"

# Regional U-Net evaluation outputs from independent spatial holdout tests.
# Required variables:
# - rx1day_pred(lat, lon): U-Net corrected Rx1day
# - test_mask(lat, lon): independent test pixels, where 1 indicates test locations
regions = [
    ("Asia", "/path/to/predictions_region_Asia.nc"),
    ("Europe", "/path/to/predictions_region_Europe.nc"),
    ("Africa", "/path/to/predictions_region_Africa.nc"),
    ("North America", "/path/to/predictions_region_NorthAmerica.nc"),
    ("Australia", "/path/to/predictions_region_Australia.nc"),
]

# Output figure path.
output_fig = "/path/to/figures/unet_evaluation.png"


# ==============================================================================
# 2. Plotting configuration
# ==============================================================================

fontsize_1 = 11
fontsize_2 = 9

color_land = "#F0F0F0"
color_all = "#8B96AD"
color_link = "#800020"
bar_colors = ["#9EAAD2", "#EB6C71", "#8DC1E0"]

color_cmip6_l = "#9B3030"
color_cmip5_l = "#4993CB"
alpha_fill = 0.2

colors_list = [
    "#b40426", "#f4987a", "#dbe9f6",
    "#8db0fe", "#3b4cc0", "#253494", "#081d58"
]
cmap_rx1day = LinearSegmentedColormap.from_list(
    "rx1day_colormap", colors_list, N=256
)
bounds_rx1day = [10, 20, 40, 60, 80, 100, 120]
norm_rx1day = BoundaryNorm(bounds_rx1day, ncolors=cmap_rx1day.N, extend="both")

region_ylim = {
    "Asia": (25, 130),
    "Europe": (30, 85),
    "Africa": (40, 90),
    "North America": (25, 90),
    "Australia": (25, 95),
}

region_centers = {
    "North America": [-100, 40],
    "Europe": [22, 45],
    "Asia": [105, 35],
    "Africa": [30, 0],
    "Australia": [135, -25],
}


# ==============================================================================
# 3. Load and preprocess data
# ==============================================================================

# Load station-based labels and IMERG estimates.
ds_input = xr.open_dataset(nc_input_path)
rx1day_input = ds_input["rx1day_label"].values
rx1day_imerg_full = ds_input["rx1day_imerg"].values

lat_v = ds_input["lat"].values
lon_v = ds_input["lon"].values
lat_grid, lon_grid = np.meshgrid(lat_v, lon_v, indexing="ij")

# Identify grid cells with valid gauge-based labels.
all_mask = ~np.isnan(np.nanmean(rx1day_input, axis=0))
all_lats = lat_grid[all_mask].astype(np.float64)
all_lons = lon_grid[all_mask].astype(np.float64)

# Load land mask and U-Net reconstructed Rx1day climatology.
ds_mask = xr.open_dataset(mask_path)
mask_land = ds_mask["land_mask_50NS"]

ds_predictions_mean = xr.open_dataset(pred_mean_path)
data_predictions_rx1day = ds_predictions_mean["rx1day_pred"]
Lon, Lat = np.meshgrid(
    ds_predictions_mean["lon"].values,
    ds_predictions_mean["lat"].values,
)


def process_cmip(path):
    """Load CMIP historical Rx1day and compute the multi-model mean."""
    ds = xr.open_dataset(path)
    return ds["data_his_all"].mean(dim="model", skipna=True).where(~np.isnan(mask_land))


# Compute zonal-mean historical biases relative to the U-Net reconstruction.
data_cmip6_rx1day = process_cmip(cmip6_path)
data_cmip5_rx1day = process_cmip(cmip5_path)

lat_values = ds_predictions_mean["lat"].values
lat_mean_obs = np.nanmean(data_predictions_rx1day, axis=1)

lat_dif_cmip6 = np.nanmean(data_cmip6_rx1day, axis=1) - lat_mean_obs
lat_std_cmip6 = np.nanstd(data_cmip6_rx1day - data_predictions_rx1day, axis=1)

lat_dif_cmip5 = np.nanmean(data_cmip5_rx1day, axis=1) - lat_mean_obs
lat_std_cmip5 = np.nanstd(data_cmip5_rx1day - data_predictions_rx1day, axis=1)


# ==============================================================================
# 4. Define plotting functions
# ==============================================================================

def draw_panel(region_name, pos):
    """Draw a regional bar plot comparing OBS, IMERG, and U-Net estimates."""
    idx = [r[0] for r in regions].index(region_name)

    with xr.open_dataset(regions[idx][1]) as ds_region:
        mask = ds_region["test_mask"].values
        pt = np.where(mask == 1)

        pred = ds_region["rx1day_pred"].values[pt]
        imerg = np.nanmean(rx1day_imerg_full[:, pt[0], pt[1]], axis=0)
        obs = np.nanmean(rx1day_input[:, pt[0], pt[1]], axis=0)

        products = np.column_stack([obs, imerg, pred])
        mean_v = np.nanmean(products, axis=0)
        std_v = np.nanstd(products, axis=0)

    ax_bar = fig.add_axes(pos)
    ax_bar.bar(
        range(3),
        mean_v,
        yerr=std_v,
        width=0.5,
        capsize=3,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    for i, v in enumerate(mean_v):
        ax_bar.text(
            i,
            v + std_v[i] + 2,
            f"{v:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax_bar.set_xticks(range(3))
    ax_bar.set_xticklabels(["OBS", "IMERG", "UNet"], fontsize=fontsize_2)
    ax_bar.set_title(
        region_name,
        fontsize=fontsize_2,
        color="red",
        fontweight="bold",
        pad=5,
    )

    label_map = {
        "North America": "a",
        "Europe": "b",
        "Asia": "c",
        "Africa": "d",
        "Australia": "e",
    }
    ax_bar.text(
        -0.15,
        1.17,
        label_map.get(region_name, ""),
        transform=ax_bar.transAxes,
        fontsize=fontsize_1,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax_bar.set_ylim(region_ylim[region_name])
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    start_y = -0.2 if pos[1] > 0.5 else 1.2
    con = ConnectionPatch(
        xyA=(0.5, start_y),
        xyB=region_centers[region_name],
        coordsA="axes fraction",
        coordsB="data",
        axesA=ax_bar,
        axesB=ax_main,
        color=color_link,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    fig.add_artist(con)


# ==============================================================================
# 5. Create figure
# ==============================================================================

fig = plt.figure(figsize=(10, 8))
offset_yy = 0.07

# Main map showing valid gauge-label grid cells.
ax_main = fig.add_axes(
    [0.1, 0.42 + offset_yy, 0.78, 0.28],
    projection=ccrs.PlateCarree(),
)
ax_main.add_feature(cfeature.LAND, facecolor=color_land, edgecolor="none")
ax_main.coastlines("110m", linewidth=0.5)
ax_main.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
ax_main.scatter(
    all_lons,
    all_lats,
    s=16,
    color=color_all,
    alpha=1,
    transform=ccrs.PlateCarree(),
)
ax_main.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
ax_main.spines["geo"].set_visible(False)

# Regional evaluation panels.
offset_x = 0.05
offset_y = offset_yy
scale_h = 0.03
scale_w = 0.02

draw_panel("North America", [0.17 + offset_x, 0.75 + offset_y, 0.13 + scale_w, 0.11 + scale_h])
draw_panel("Europe", [0.42 + offset_x, 0.75 + offset_y, 0.13 + scale_w, 0.11 + scale_h])
draw_panel("Asia", [0.68 + offset_x, 0.75 + offset_y, 0.13 + scale_w, 0.11 + scale_h])
draw_panel("Africa", [0.33 + offset_x, 0.29 + offset_y, 0.13 + scale_w, 0.11 + scale_h])
draw_panel("Australia", [0.58 + offset_x, 0.29 + offset_y, 0.13 + scale_w, 0.11 + scale_h])

# Bottom map showing reconstructed historical Rx1day.
bottom_y, bottom_h = 0.14, 0.14

ax_f = fig.add_axes(
    [0.16, bottom_y, 0.55, bottom_h],
    projection=ccrs.PlateCarree(),
)
im0 = ax_f.pcolormesh(
    Lon,
    Lat,
    data_predictions_rx1day,
    cmap=cmap_rx1day,
    norm=norm_rx1day,
    shading="auto",
)
ax_f.set_title("U-Net-derived historical Rx1day", fontsize=fontsize_1, fontweight="bold")
ax_f.coastlines(linewidth=0.5)
ax_f.add_feature(cfeature.BORDERS, linewidth=0.3)
ax_f.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
ax_f.set_yticks([-50, -25, 0, 25, 50], crs=ccrs.PlateCarree())
ax_f.set_ylabel("Latitude (°N)", fontsize=fontsize_2)
ax_f.set_xlabel("Longitude (°E)", fontsize=fontsize_2)
ax_f.set_ylim(-50, 50)
ax_f.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
ax_f.text(-0.06, 1.1, "f", transform=ax_f.transAxes, fontsize=fontsize_1, fontweight="bold")

# Bottom panel showing zonal-mean historical model bias.
ax_g = fig.add_axes([0.68, bottom_y, 0.195, bottom_h], sharey=ax_f)

ax_g.plot(lat_dif_cmip6, lat_values, color=color_cmip6_l, linewidth=1.2, label="CMIP6")
ax_g.fill_betweenx(
    lat_values,
    lat_dif_cmip6 - lat_std_cmip6,
    lat_dif_cmip6 + lat_std_cmip6,
    color=color_cmip6_l,
    alpha=alpha_fill,
)

ax_g.plot(lat_dif_cmip5, lat_values, color=color_cmip5_l, linewidth=1.2, label="CMIP5")
ax_g.fill_betweenx(
    lat_values,
    lat_dif_cmip5 - lat_std_cmip5,
    lat_dif_cmip5 + lat_std_cmip5,
    color=color_cmip5_l,
    alpha=alpha_fill,
)

ax_g.spines["top"].set_visible(False)
ax_g.spines["right"].set_visible(False)
ax_g.axvline(0, color="k", linestyle="--", linewidth=1)
ax_g.set_title("Historical bias", fontsize=fontsize_1)
ax_g.set_xlabel("Bias (mm/day)", fontsize=fontsize_2)
plt.setp(ax_g.get_yticklabels(), visible=False)
ax_g.set_xlim(-80, 80)
ax_g.set_ylim(-50, 50)
ax_g.grid(True, linestyle="--", alpha=0.5)
ax_g.legend(loc=(0.7, 0.05), fontsize=9, frameon=False)
ax_g.text(-0.05, 1.1, "g", transform=ax_g.transAxes, fontsize=fontsize_1, fontweight="bold")

# Colorbar for reconstructed Rx1day map.
cbar_ax = fig.add_axes([0.295, bottom_y - 0.07, 0.28, 0.012])
cb = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
cb.set_label("Rx1day (mm/day)", fontsize=fontsize_2)
cb.ax.tick_params(labelsize=fontsize_2)

os.makedirs(os.path.dirname(output_fig), exist_ok=True)
plt.savefig(output_fig, dpi=300, bbox_inches="tight")
plt.show()
