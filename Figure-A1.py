#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install cartopy
import cartopy



# Obj1Fig1_SO
# Seasonal SIC (%) and SIC Anomaly (%) from merged SIF NetCDF
#
# Input:
# /content/drive/MyDrive/SAM_Thesis/Data/OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
#
# Output:
# /content/drive/MyDrive/SAM_Thesis/Fig/Obj1Fig1_SO.png

# 1) Imports

import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# 2) File paths

sic_file = "/content/drive/MyDrive/SAM_Thesis/Data/OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, "Obj1Fig1_SO.png")

print("Input file:", sic_file)
print("Output file:", out_file)
print("File exists:", os.path.exists(sic_file))

if not os.path.exists(sic_file):
    raise FileNotFoundError("SIF NetCDF file not found. Check the file path.")



# 3) Settings

CLIM_START_YEAR = 2008
CLIM_END_YEAR   = 2025
TARGET_YEAR     = 2025

SEASONS = {
    "Spring": [9, 10, 11],   # SON
    "Summer": [12, 1, 2],    # DJF
    "Autumn": [3, 4, 5],     # MAM
    "Winter": [6, 7, 8],     # JJA
}

ROW_ORDER = ["Spring", "Summer", "Autumn", "Winter"]



# 4) Open NetCDF
ds = xr.open_dataset(
    sic_file,
    decode_times=True,
    chunks={"time": 12, "lat": 300, "lon": 600}
)

print(ds)

sic_var = "SIF"

if sic_var not in ds.data_vars:
    raise ValueError(f"Variable '{sic_var}' not found. Available variables: {list(ds.data_vars)}")

print("SIC/SIF variable used:", sic_var)



# 5) Prepare coordinate and SIC data
# Convert longitude from 0–360 to -180–180 if needed
if float(ds["lon"].max()) > 180:
    ds = ds.assign_coords(
        lon=(((ds["lon"] + 180) % 360) - 180)
    ).sortby("lon")

# Sort coordinates
ds = ds.sortby("lat")
ds = ds.sortby("lon")
ds = ds.sortby("time")

# Keep Southern Ocean only
ds = ds.where(ds["lat"] <= -60, drop=True)

# Convert SIF to SIC percent
sic = ds[sic_var].astype("float32")

sic_max = float(sic.max(skipna=True).compute().values)

if sic_max <= 1.5:
    sic_pct = sic * 100.0
    print("SIF is stored as 0–1 fraction. Converted to SIC percent.")
else:
    sic_pct = sic
    print("SIC already appears to be in percent.")

sic_pct = sic_pct.clip(min=0, max=100)



# 6) Seasonal-year functions
def get_season_year(time_da, months):
    """
    DJF rule:
    December belongs to the following year.
    Example:
    Dec 2024 + Jan 2025 + Feb 2025 = Summer 2025
    """
    year = time_da.dt.year

    if 12 in months and 1 in months:
        year = xr.where(time_da.dt.month == 12, year + 1, year)

    return year.rename("season_year")


def seasonal_mean_by_year(da, months, min_months=2):
    """
    Calculate seasonal mean for each season-year.
    min_months=2 allows DJF 2008 using Jan-Feb if Dec 2007 is absent.
    """
    da_season = da.where(da["time"].dt.month.isin(months), drop=True)

    if da_season.sizes.get("time", 0) == 0:
        return None

    sy = get_season_year(da_season["time"], months)
    da_season = da_season.assign_coords(season_year=sy)

    mean_da = da_season.groupby("season_year").mean("time", skipna=True)

    years = sy.values
    unique_years = np.unique(years)

    valid_years = []

    for y in unique_years:
        n_months = np.sum(years == y)
        if n_months >= min_months:
            valid_years.append(int(y))

    if len(valid_years) == 0:
        return None

    mean_da = mean_da.sel(season_year=valid_years)

    return mean_da



# 7) Calculate seasonal SIC and anomaly

maps_2025 = {}
maps_clim = {}
maps_anom = {}

for season_name, months in SEASONS.items():

    print("\nProcessing:", season_name)

    seasonal_yearly = seasonal_mean_by_year(sic_pct, months, min_months=2)

    if seasonal_yearly is None:
        raise ValueError(f"No valid seasonal data found for {season_name}")

    available_years = seasonal_yearly["season_year"].values.astype(int)
    print(f"{season_name} available season-years:", available_years)

    # 2008–2025 climatology
    clim = seasonal_yearly.sel(
        season_year=slice(CLIM_START_YEAR, CLIM_END_YEAR)
    ).mean("season_year", skipna=True)

    # 2025 seasonal mean
    if TARGET_YEAR not in available_years:
        raise ValueError(f"{TARGET_YEAR} not available for {season_name}")

    target = seasonal_yearly.sel(season_year=TARGET_YEAR)

    # anomaly = 2025 - 2008–2025 climatology
    anomaly = target - clim

    maps_clim[season_name] = clim.compute()
    maps_2025[season_name] = target.compute()
    maps_anom[season_name] = anomaly.compute()

    gc.collect()

print("\nAll seasonal maps calculated successfully.")



# 8) Meshgrid

lon_vals = ds["lon"].values
lat_vals = ds["lat"].values

lon2, lat2 = np.meshgrid(lon_vals, lat_vals)



# 9) Polar axis helper

def polar_ax(fig, nrows, ncols, idx):

    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, idx, projection=proj)

    # circular boundary
    theta = np.linspace(0, 2 * np.pi, 240)
    center = [0.5, 0.5]
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    # land and coastline
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.82",
        edgecolor="black",
        linewidth=0.4,
        zorder=3
    )

    ax.coastlines(linewidth=0.6, zorder=4)

    # gridlines
    lon_grid = [-180, -140, -100, -60, -20, 20, 60, 100, 140]
    lat_grid = [-60, -70, -80]

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        linestyle=":",
        color="black",
        alpha=0.65,
        zorder=5
    )

    gl.xlocator = mticker.FixedLocator(lon_grid)
    gl.ylocator = mticker.FixedLocator(lat_grid)

    # longitude labels
    edge_lat = -57.2

    for lon in lon_grid:
        if lon < 0:
            label = f"{abs(lon)}°W"
        elif lon > 0:
            label = f"{lon}°E"
        else:
            label = "0°"

        ax.text(
            lon,
            edge_lat,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold"
        )

    # latitude labels
    for lat in [-70, -80]:
        ax.text(
            0,
            lat,
            f"{abs(lat)}°S",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold"
        )

    return ax


# 10) Plot figure
fig = plt.figure(figsize=(9, 14))

sic_vmin, sic_vmax = 0, 100
anom_vmin, anom_vmax = -20, 20

sic_ticks = [0, 20, 40, 60, 80, 100]
anom_ticks = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

sic_cmap = "turbo"
anom_cmap = "RdBu_r"

# Important for anomaly map:
# blue = negative anomaly, white = near zero, red = positive anomaly
anom_norm = TwoSlopeNorm(vmin=anom_vmin, vcenter=0, vmax=anom_vmax)

letters = list("abcdefgh")
letter_i = 0

axs = np.empty((4, 2), dtype=object)

for r, season_name in enumerate(ROW_ORDER):


    # Left column: SIC 2025

    ax1 = polar_ax(fig, 4, 2, r * 2 + 1)
    axs[r, 0] = ax1

    pcm1 = ax1.pcolormesh(
        lon2,
        lat2,
        maps_2025[season_name],
        transform=ccrs.PlateCarree(),
        cmap=sic_cmap,
        vmin=sic_vmin,
        vmax=sic_vmax,
        shading="auto",
        zorder=1
    )

    ax1.text(
        -0.06,
        0.985,
        letters[letter_i],
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold"
    )

    letter_i += 1

    cb1 = fig.colorbar(
        pcm1,
        ax=ax1,
        shrink=0.78,
        pad=0.06,
        ticks=sic_ticks
    )

    cb1.set_label("SIC (%)", fontsize=9, fontweight="bold")
    cb1.ax.tick_params(labelsize=8)


    # Right column: SIC anomaly

    ax2 = polar_ax(fig, 4, 2, r * 2 + 2)
    axs[r, 1] = ax2

    pcm2 = ax2.pcolormesh(
        lon2,
        lat2,
        maps_anom[season_name],
        transform=ccrs.PlateCarree(),
        cmap=anom_cmap,
        norm=anom_norm,
        shading="auto",
        zorder=1
    )

    ax2.text(
        -0.06,
        0.985,
        letters[letter_i],
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold"
    )

    letter_i += 1

    cb2 = fig.colorbar(
        pcm2,
        ax=ax2,
        shrink=0.78,
        pad=0.06,
        ticks=anom_ticks
    )

    cb2.set_label("SIC Anomaly (%)", fontsize=9, fontweight="bold")
    cb2.ax.tick_params(labelsize=8)



# 11) Layout labels

plt.tight_layout(rect=[0.06, 0.03, 0.98, 0.95])

# Column headers
column_titles = ["SIC (%)", "SIC Anomaly (%)"]

for j, title in enumerate(column_titles):

    pos = axs[0, j].get_position()
    x_center = 0.5 * (pos.x0 + pos.x1)
    y_top = pos.y1

    fig.text(
        x_center,
        y_top + 0.02,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )

# Row labels
for r, season_name in enumerate(ROW_ORDER):

    posL = axs[r, 0].get_position()
    posR = axs[r, 1].get_position()

    y_center = 0.5 * (min(posL.y0, posR.y0) + max(posL.y1, posR.y1))
    x_left = min(posL.x0, posR.x0)

    fig.text(
        x_left - 0.04,
        y_center,
        season_name,
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold"
    )


# 12) Save output

plt.savefig(out_file, dpi=1080, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(out_file)

ds.close()
