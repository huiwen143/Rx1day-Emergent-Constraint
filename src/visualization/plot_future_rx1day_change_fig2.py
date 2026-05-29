import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import xarray as xr
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================================================================
# 1. Input and output paths
# ======================================================================

# CMIP5/CMIP6 future projection files under SSP2-4.5 and SSP5-8.5 scenarios
rel_245_cmip5 = "/path/to/rel_change_rx1day_cmip5_ssp245_50NS.nc"
rel_245_cmip6 = "/path/to/rel_change_rx1day_cmip6_ssp245_50NS.nc"
rel_585_cmip5 = "/path/to/rel_change_rx1day_cmip5_ssp585_50NS.nc"
rel_585_cmip6 = "/path/to/rel_change_rx1day_cmip6_ssp585_50NS.nc"

# Land mask file: NaN for ocean, non-NaN for land
mask_path = "/path/to/land_mask_50NS.mat"

# Output figure path
output_fig = "/path/to/figures/fig2_future_rx1day_change.png"

# ==============================================================================
# 2. Load CMIP data and compute absolute change
# ======================================================================

ds245_cmip5 = xr.open_dataset(rel_245_cmip5)
ds245_cmip6 = xr.open_dataset(rel_245_cmip6)
ds585_cmip5 = xr.open_dataset(rel_585_cmip5)
ds585_cmip6 = xr.open_dataset(rel_585_cmip6)
mask = loadmat(mask_path)['land_mask']

# SSP2-4.5
data_his_245_cmip5 = np.nanmean(ds245_cmip5['data_his_all'].values, axis=0)
data_his_245_cmip6 = np.nanmean(ds245_cmip6['data_his_all'].values, axis=0)

data_future_245_cmip5 = np.nanmean(ds245_cmip5['data_future_all'].values, axis=0)
data_future_245_cmip6 = np.nanmean(ds245_cmip6['data_future_all'].values, axis=0)

abs_change_245_cmip5 = np.where(np.isnan(mask), np.nan, data_future_245_cmip5 - data_his_245_cmip5)
abs_change_245_cmip6 = np.where(np.isnan(mask), np.nan, data_future_245_cmip6 - data_his_245_cmip6)

# SSP5-8.5
data_his_585_cmip5 = np.nanmean(ds585_cmip5['data_his_all'].values, axis=0)
data_his_585_cmip6 = np.nanmean(ds585_cmip6['data_his_all'].values, axis=0)

data_future_585_cmip5 = np.nanmean(ds585_cmip5['data_future_all'].values, axis=0)
data_future_585_cmip6 = np.nanmean(ds585_cmip6['data_future_all'].values, axis=0)

abs_change_585_cmip5 = np.where(np.isnan(mask), np.nan, data_future_585_cmip5 - data_his_585_cmip5)
abs_change_585_cmip6 = np.where(np.isnan(mask), np.nan, data_future_585_cmip6 - data_his_585_cmip6)

# Close datasets
ds245_cmip5.close()
ds245_cmip6.close()
ds585_cmip5.close()
ds585_cmip6.close()

# ==============================================================================
# 3. Compute zonal-mean statistics
# ======================================================================

lon = np.arange(0.5, 360.5)
lat = np.linspace(-50, 50, 100)
Lon, Lat = np.meshgrid(lon, lat)

lat_mean_245_cmip5 = np.nanmean(abs_change_245_cmip5, axis=1)
lat_mean_245_cmip6 = np.nanmean(abs_change_245_cmip6, axis=1)
lat_std_245_cmip5 = np.nanstd(abs_change_245_cmip5, axis=1)
lat_std_245_cmip6 = np.nanstd(abs_change_245_cmip6, axis=1)

lat_mean_585_cmip5 = np.nanmean(abs_change_585_cmip5, axis=1)
lat_mean_585_cmip6 = np.nanmean(abs_change_585_cmip6, axis=1)
lat_std_585_cmip5 = np.nanstd(abs_change_585_cmip5, axis=1)
lat_std_585_cmip6 = np.nanstd(abs_change_585_cmip6, axis=1)

# ==============================================================================
# 4. Plotting parameters
# ======================================================================

colors = ["#8B0000", "#CC3333", "#FF6666", "#F2F2F2", "#6699FF", "#3366CC", "#00008B"]
cmap_dif = LinearSegmentedColormap.from_list("rx1day_change_colormap", colors, N=256)
bounds_rx1day = [-40, -20, -10, -5, 0, 5, 10, 20, 40]
norm_rx1day = BoundaryNorm(bounds_rx1day, ncolors=cmap_dif.N, extend="both")

color_cmip6_l = "#9B3030"
color_cmip5_l = "#4993CB"

fontsize1 = 12
fontsize2 = 10

scenarios_info = [
    {"id": 245, "label": "SSP2-4.5", "rcp": "RCP4.5", "tags": ["a", "b"]},
    {"id": 585, "label": "SSP5-8.5", "rcp": "RCP8.5", "tags": ["c", "d"]},
]

# ==============================================================================
# 5. Create figure
# ======================================================================

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], wspace=0.03, hspace=0.3)

im_handle = None
axes_maps = []
axes_lines = []

for row_idx, info in enumerate(scenarios_info):
    scenario = info["id"]

    lat_std_cmip5 = eval(f"lat_std_{scenario}_cmip5")
    lat_std_cmip6 = eval(f"lat_std_{scenario}_cmip6")
    lat_mean_cmip5 = eval(f"lat_mean_{scenario}_cmip5")
    lat_mean_cmip6 = eval(f"lat_mean_{scenario}_cmip6")
    map_data = eval(f"abs_change_{scenario}_cmip6")

    # Left panel: CMIP6 map
    ax_map = fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree())
    axes_maps.append(ax_map)
    im = ax_map.pcolormesh(Lon, Lat, map_data, cmap=cmap_dif, norm=norm_rx1day, shading='auto')
    if row_idx == 1: im_handle = im

    ax_map.set_title(f"Future Rx1day change {info['label']}", fontsize=fontsize1)
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax_map.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax_map.set_yticks([-50, -25, 0, 25, 50], crs=ccrs.PlateCarree())
    ax_map.set_ylim(-50, 50)
    ax_map.set_ylabel("Latitude (°N)", fontsize=fontsize1)
    ax_map.set_xlabel("Longitude (°E)", fontsize=fontsize1)
    ax_map.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax_map.text(-180, 55, info['tags'][0], fontsize=fontsize1, fontweight='bold')

    # Right panel: zonal mean
    ax_zonal = fig.add_subplot(gs[row_idx, 1], sharey=ax_map)
    axes_lines.append(ax_zonal)
    ax_zonal.plot(lat_mean_cmip6, lat, color=color_cmip6_l, linewidth=1.2, label="CMIP6")
    ax_zonal.fill_betweenx(lat, lat_mean_cmip6 - lat_std_cmip6, lat_mean_cmip6 + lat_std_cmip6, color=color_cmip6_l, alpha=0.3)
    ax_zonal.plot(lat_mean_cmip5, lat, color=color_cmip5_l, linewidth=1.5, label="CMIP5")
    ax_zonal.fill_betweenx(lat, lat_mean_cmip5 - lat_std_cmip5, lat_mean_cmip5 + lat_std_cmip5, color=color_cmip5_l, alpha=0.3)

    ax_zonal.spines['top'].set_visible(False)
    ax_zonal.spines['right'].set_visible(False)
    ax_zonal.axvline(0, color='k', linestyle='--', linewidth=1)
    ax_zonal.set_xlim(-10, 50)
    ax_zonal.grid(True, linestyle='--', alpha=0.5)
    plt.setp(ax_zonal.get_yticklabels(), visible=False)
    ax_zonal.set_xlabel("Rx1day change (mm)", fontsize=fontsize1)
    ax_zonal.legend(loc="lower right", fontsize=fontsize2, frameon=False)
    ax_zonal.text(0.6, 0.95, f"{info['label']}/\n{info['rcp']}", transform=ax_zonal.transAxes, ha='left', va='top', fontsize=fontsize2)
    ax_zonal.text(0, 1.05, info['tags'][1], transform=ax_zonal.transAxes, fontsize=fontsize1, fontweight='bold')

# Colorbar
cbar_ax = fig.add_axes([0.21, 0.02, 0.4, 0.02])
cb = plt.colorbar(im_handle, cax=cbar_ax, orientation='horizontal', ticks=bounds_rx1day, spacing='uniform')
cb.set_label('Rx1day change (mm)', fontsize=fontsize1)
cb.ax.tick_params(labelsize=fontsize2)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Align zonal panels with map panels
fig.canvas.draw()
for ax_m, ax_l in zip(axes_maps, axes_lines):
    pos_map = ax_m.get_position()
    pos_line = ax_l.get_position()
    ax_l.set_position([pos_line.x0, pos_map.y0, pos_line.width, pos_map.height])

os.makedirs(os.path.dirname(output_fig), exist_ok=True)
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()
