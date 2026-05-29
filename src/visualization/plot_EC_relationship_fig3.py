import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy import stats
from scipy.stats import spearmanr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ==============================================================================
# 1. Input paths
# ==============================================================================

csv_file_245_cmip5 = '/path/to/CMIP5/R_cmip5_rx1day_ssp245_50NS_land.csv'
csv_file_245_cmip6 = '/path/to/CMIP6/R_cmip6_rx1day_ssp245_50NS_land.csv'
csv_file_585_cmip5 = '/path/to/CMIP5/R_cmip5_rx1day_ssp585_50NS_land.csv'
csv_file_585_cmip6 = '/path/to/CMIP6/R_cmip6_rx1day_ssp585_50NS_land.csv'

obs_path = '/path/to/DL/predictions_mean.nc'


# ==============================================================================
# 2. Load historical-future relationship data
# ==============================================================================

ds_obs = xr.open_dataset(obs_path)
data_obs = ds_obs['rx1day_pred'].values

R_all_245_cmip5 = np.loadtxt(csv_file_245_cmip5, delimiter=',', skiprows=1)
R_all_245_cmip6 = np.loadtxt(csv_file_245_cmip6, delimiter=',', skiprows=1)
R_all_585_cmip5 = np.loadtxt(csv_file_585_cmip5, delimiter=',', skiprows=1)
R_all_585_cmip6 = np.loadtxt(csv_file_585_cmip6, delimiter=',', skiprows=1)

his_245_cmip5 = R_all_245_cmip5[:, 2]
change_245_cmip5 = R_all_245_cmip5[:, 4]

his_245_cmip6 = R_all_245_cmip6[:, 2]
change_245_cmip6 = R_all_245_cmip6[:, 4]

his_585_cmip5 = R_all_585_cmip5[:, 2]
change_585_cmip5 = R_all_585_cmip5[:, 4]

his_585_cmip6 = R_all_585_cmip6[:, 2]
change_585_cmip6 = R_all_585_cmip6[:, 4]


# ==============================================================================
# 3. Compute correlations and historical baselines
# ==============================================================================

R_245_cmip5, _ = spearmanr(his_245_cmip5, change_245_cmip5, nan_policy='omit')
R_245_cmip6, _ = spearmanr(his_245_cmip6, change_245_cmip6, nan_policy='omit')
R_585_cmip5, _ = spearmanr(his_585_cmip5, change_585_cmip5, nan_policy='omit')
R_585_cmip6, _ = spearmanr(his_585_cmip6, change_585_cmip6, nan_policy='omit')

mean_his_245_cmip5 = np.nanmean(his_245_cmip5)
mean_his_245_cmip6 = np.nanmean(his_245_cmip6)
mean_his_585_cmip5 = np.nanmean(his_585_cmip5)
mean_his_585_cmip6 = np.nanmean(his_585_cmip6)

mean_his_unet = np.nanmean(data_obs)


# ==============================================================================
# 4. Linear regression and confidence intervals
# ==============================================================================

def linear_fit_ci(x, y, x_pred, alpha=0.05):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x_pred + intercept

    y_hat = slope * x + intercept
    residuals = y - y_hat
    dof = n - 2
    s_err = np.sqrt(np.sum(residuals ** 2) / dof)

    t_val = stats.t.ppf(1 - alpha / 2, dof)
    x_mean = np.mean(x)
    conf = t_val * s_err * np.sqrt(
        1 / n + (x_pred - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    )

    lower = y_fit - conf
    upper = y_fit + conf

    return y_fit, lower, upper, r_value, slope, intercept


fit_245_cmip5, ci_245_lo5, ci_245_hi5, R5_245, k5_245, b5_245 = linear_fit_ci(
    his_245_cmip5, change_245_cmip5, his_245_cmip5
)
fit_245_cmip6, ci_245_lo6, ci_245_hi6, R6_245, k6_245, b6_245 = linear_fit_ci(
    his_245_cmip6, change_245_cmip6, his_245_cmip6
)

fit_585_cmip5, ci_585_lo5, ci_585_hi5, R5_585, k5_585, b5_585 = linear_fit_ci(
    his_585_cmip5, change_585_cmip5, his_585_cmip5
)
fit_585_cmip6, ci_585_lo6, ci_585_hi6, R6_585, k6_585, b6_585 = linear_fit_ci(
    his_585_cmip6, change_585_cmip6, his_585_cmip6
)

idx5_245 = np.argsort(his_245_cmip5)
his_cmip5_sorted_245 = his_245_cmip5[idx5_245]
fit_cmip5_sorted_245 = fit_245_cmip5[idx5_245]
ci_lo5_sorted_245 = ci_245_lo5[idx5_245]
ci_hi5_sorted_245 = ci_245_hi5[idx5_245]

idx6_245 = np.argsort(his_245_cmip6)
his_cmip6_sorted_245 = his_245_cmip6[idx6_245]
fit_cmip6_sorted_245 = fit_245_cmip6[idx6_245]
ci_lo6_sorted_245 = ci_245_lo6[idx6_245]
ci_hi6_sorted_245 = ci_245_hi6[idx6_245]

idx5_585 = np.argsort(his_585_cmip5)
his_cmip5_sorted_585 = his_585_cmip5[idx5_585]
fit_cmip5_sorted_585 = fit_585_cmip5[idx5_585]
ci_lo5_sorted_585 = ci_585_lo5[idx5_585]
ci_hi5_sorted_585 = ci_585_hi5[idx5_585]

idx6_585 = np.argsort(his_585_cmip6)
his_cmip6_sorted_585 = his_585_cmip6[idx6_585]
fit_cmip6_sorted_585 = fit_585_cmip6[idx6_585]
ci_lo6_sorted_585 = ci_585_lo6[idx6_585]
ci_hi6_sorted_585 = ci_585_hi6[idx6_585]


# ==============================================================================
# 5. Compute observational-product baselines
# ==============================================================================

products = ['mswep', 'imerg', 'era5', 'cpc']
baseline_dict = {}

for product in products:
    file_path = f'/path/to/obs/his_{product}/rx1day_{product}_his_all.nc'

    ds = xr.open_dataset(file_path)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

    data = ds['obs_his_all']

    if data.shape != data_obs.shape:
        data = data.interp_like(ds_obs)

    baseline_dict[product] = np.nanmean(data.where(~np.isnan(data_obs)))

mean_his_mswep = baseline_dict['mswep']
mean_his_imerg = baseline_dict['imerg']
mean_his_era5 = baseline_dict['era5']
mean_his_cpc = baseline_dict['cpc']


# ==============================================================================
# 6. Plot Figure 3
# ==============================================================================

color_cmip5 = '#4A90E2'
color_cmip6 = '#000000'

color_unet = '#3ac569'
color_mswep = '#FA00FF'
color_cpc = '#01BBDC'
color_imerg = '#6265F1'
color_era5 = '#FFC300'

fontsize_1 = 12
fontsize_2 = 10
fontsize_3 = 9

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
plt.subplots_adjust(wspace=8)

box_pos = [85, 90]
box_colors = [color_cmip5, color_cmip6]


# ==============================================================================
# Panel a: SSP2-4.5 / RCP4.5
# ==============================================================================

ax = axes[0]

mask5 = ~np.isnan(his_245_cmip5) & ~np.isnan(change_245_cmip5)
mask6 = ~np.isnan(his_245_cmip6) & ~np.isnan(change_245_cmip6)

ax.scatter(
    his_245_cmip5[mask5],
    change_245_cmip5[mask5],
    facecolors='none',
    edgecolors=color_cmip5,
    marker='s',
    s=14,
    linewidths=1.5,
)
ax.scatter(
    his_245_cmip6[mask6],
    change_245_cmip6[mask6],
    facecolors='none',
    edgecolors=color_cmip6,
    marker='s',
    s=14,
    linewidths=1.5,
)

ax.plot(his_cmip5_sorted_245, fit_cmip5_sorted_245, color=color_cmip5, linestyle='--')
ax.fill_between(his_cmip5_sorted_245, ci_lo5_sorted_245, ci_hi5_sorted_245, color=color_cmip5, alpha=0.2)

ax.plot(his_cmip6_sorted_245, fit_cmip6_sorted_245, color=color_cmip6, linestyle='--')
ax.fill_between(his_cmip6_sorted_245, ci_lo6_sorted_245, ci_hi6_sorted_245, color=color_cmip6, alpha=0.2)

ax.text(19, 24, f"$R_{{CMIP5}}$ = {R_245_cmip5:.2f}", color=color_cmip5, fontsize=fontsize_2)
ax.text(19, 22, f"$R_{{CMIP6}}$ = {R_245_cmip6:.2f}", color=color_cmip6, fontsize=fontsize_2)
ax.text(17, 28, 'a', fontsize=fontsize_1, fontweight='bold')

ax.axvline(mean_his_245_cmip5, color=color_cmip5, linestyle='--', linewidth=1.2)
ax.axvline(mean_his_245_cmip6, color=color_cmip6, linestyle='--', linewidth=1.2)
ax.axvline(mean_his_unet, color=color_unet, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_mswep, color=color_mswep, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_era5, color=color_era5, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_imerg, color=color_imerg, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_cpc, color=color_cpc, linestyle='-', linewidth=1.5)

for data, pos, color in zip([change_245_cmip5, change_245_cmip6], box_pos, box_colors):
    clean_data = data[~np.isnan(data)]
    ax.boxplot(
        clean_data,
        positions=[pos],
        widths=2,
        patch_artist=True,
        vert=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(facecolor=color, color=color, alpha=0.6),
        medianprops=dict(color='white', linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
    )
    ax.plot(pos, np.nanmean(clean_data), 'o', color='red', markersize=4, zorder=3)

ax.set_xlabel("Historical Rx1day (mm)", fontsize=fontsize_1)
ax.set_ylabel("Future Rx1day change (mm)", fontsize=fontsize_1)
ax.set_title("SSP2-4.5/RCP4.5")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_xticks(box_pos)
ax.set_xticklabels(['', ''])
ax.text(
    np.mean(box_pos),
    np.nanmax(change_245_cmip6) + 3,
    "Future\n(2080–2100)",
    ha='center',
    va='bottom',
    fontsize=fontsize_2,
)

ax.set_xlim(18, 100)
xticks = np.arange(20, 100, 10)
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])
ax.set_yticks(np.arange(0, 14, 2))
axes[1].tick_params(labelleft=True)


# ==============================================================================
# Panel b: SSP5-8.5 / RCP8.5
# ==============================================================================

ax = axes[1]

mask5 = ~np.isnan(his_585_cmip5) & ~np.isnan(change_585_cmip5)
mask6 = ~np.isnan(his_585_cmip6) & ~np.isnan(change_585_cmip6)

ax.scatter(
    his_585_cmip5[mask5],
    change_585_cmip5[mask5],
    facecolors='none',
    edgecolors=color_cmip5,
    marker='s',
    s=14,
    linewidths=1.5,
)
ax.scatter(
    his_585_cmip6[mask6],
    change_585_cmip6[mask6],
    facecolors='none',
    edgecolors=color_cmip6,
    marker='s',
    s=14,
    linewidths=1.5,
)

ax.plot(his_cmip5_sorted_585, fit_cmip5_sorted_585, color=color_cmip5, linestyle='--')
ax.fill_between(his_cmip5_sorted_585, ci_lo5_sorted_585, ci_hi5_sorted_585, color=color_cmip5, alpha=0.2)

ax.plot(his_cmip6_sorted_585, fit_cmip6_sorted_585, color=color_cmip6, linestyle='--')
ax.fill_between(his_cmip6_sorted_585, ci_lo6_sorted_585, ci_hi6_sorted_585, color=color_cmip6, alpha=0.2)

ax.text(19, 24, f"$R_{{CMIP5}}$ = {R_585_cmip5:.2f}", color=color_cmip5, fontsize=fontsize_2)
ax.text(19, 22, f"$R_{{CMIP6}}$ = {R_585_cmip6:.2f}", color=color_cmip6, fontsize=fontsize_2)
ax.text(17, 28, 'b', fontsize=fontsize_1, fontweight='bold')

ax.axvline(mean_his_585_cmip5, color=color_cmip5, linestyle='--', linewidth=1.2)
ax.axvline(mean_his_585_cmip6, color=color_cmip6, linestyle='--', linewidth=1.2)
ax.axvline(mean_his_unet, color=color_unet, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_mswep + 0.5, color=color_mswep, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_era5 + 0.5, color=color_era5, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_imerg, color=color_imerg, linestyle='-', linewidth=1.5)
ax.axvline(mean_his_cpc, color=color_cpc, linestyle='-', linewidth=1.5)

for data, pos, color in zip([change_585_cmip5, change_585_cmip6], box_pos, box_colors):
    clean_data = data[~np.isnan(data)]
    ax.boxplot(
        clean_data,
        positions=[pos],
        widths=2,
        patch_artist=True,
        vert=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(facecolor=color, color=color, alpha=0.6),
        medianprops=dict(color='white', linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
    )
    ax.plot(pos, np.nanmean(clean_data), 'o', color='red', markersize=4, zorder=3)

ax.set_xlabel("Historical Rx1day (mm)", fontsize=fontsize_1)
ax.set_ylabel("Future Rx1day change (mm)", fontsize=fontsize_1)
ax.set_title("SSP5-8.5/RCP8.5")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_xticks(box_pos)
ax.set_xticklabels(['', ''])
ax.text(
    np.mean(box_pos) + 5,
    np.nanmax(change_245_cmip6) + 10,
    "Future\n(2080–2100)",
    ha='center',
    va='bottom',
    fontsize=fontsize_2,
)

ax.set_xlim(18, 100)
xticks = np.arange(20, 100, 10)
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])
ax.set_yticks(np.arange(0, 28, 4))


# ==============================================================================
# Legend
# ==============================================================================

cmip_elements = [
    Patch(facecolor=color_cmip5, edgecolor=color_cmip5, label='CMIP5'),
    Patch(facecolor=color_cmip6, edgecolor=color_cmip6, label='CMIP6'),
]

product_elements = [
    Line2D([0], [0], color=color_unet, linestyle='-', linewidth=1.5, label='UNET'),
    Line2D([0], [0], color=color_mswep, linestyle='-', linewidth=1.5, label='MSWEP'),
    Line2D([0], [0], color=color_era5, linestyle='-', linewidth=1.5, label='ERA5'),
    Line2D([0], [0], color=color_imerg, linestyle='-', linewidth=1.5, label='IMERG'),
    Line2D([0], [0], color=color_cpc, linestyle='-', linewidth=1.5, label='CPC'),
]

leg1 = axes[0].legend(handles=cmip_elements, loc=(0.55, 0.05), fontsize=fontsize_2, frameon=False)
axes[0].add_artist(leg1)
axes[0].legend(handles=product_elements, loc='upper right', fontsize=fontsize_3, frameon=False)

axes[1].legend(handles=cmip_elements, loc=(0.55, 0.05), fontsize=fontsize_2, frameon=False)

plt.tight_layout()
plt.show()
