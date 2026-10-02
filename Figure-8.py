# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install cartopy
import cartopy



# Fig. 5 Dynamic Atmospheric Forcing of SIC
#
# Background = 2025 SIC anomaly
# Arrows     = 2025 seasonal mean 10 m wind vectors
# Contours   = 2025 seasonal mean SLP
#
# Output:
# /content/drive/MyDrive/SAM_Thesis/Fig/Fig5_SO_fixed_v5.png



# 1) Imports

import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature



# 2) File paths

sic_file  = "/content/drive/MyDrive/SAM_Thesis/Data/OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"
wind_file = "/content/drive/MyDrive/SAM_Thesis/Data/Wind_U_V_era.nc"
slp_file  = "/content/drive/MyDrive/SAM_Thesis/Data/2mT_MSLP_era.nc"

OUT_PATH = "/content/drive/MyDrive/SAM_Thesis/Fig/Fig5_SO_fixed_v5.png"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

for f in [sic_file, wind_file, slp_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

print("All input files found.")



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

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)"]

# SIC anomaly color setting
ANOM_VMIN = -30
ANOM_VMAX =  30
ANOM_TICKS = [-30, -20, -10, 0, 10, 20, 30]
ANOM_NORM = TwoSlopeNorm(vmin=ANOM_VMIN, vcenter=0, vmax=ANOM_VMAX)
ANOM_CMAP = "RdBu_r"



# MANUAL EDIT AREA


# Wind vector visibility
TARGET_ARROWS_LON = 31
TARGET_ARROWS_LAT = 15

# Longer arrows  -> decrease QUIVER_SCALE
# Shorter arrows -> increase QUIVER_SCALE
QUIVER_SCALE = 179
QUIVER_WIDTH = 0.0025
QUIVER_HEADWIDTH = 4.2
QUIVER_HEADLENGTH = 5.0
QUIVER_HEADAXISLENGTH = 4.5

# SLP contour visibility
SLP_CONTOUR_INTERVAL = 10

# IMPORTANT:
# 1 = label every contour line
# 2 = label every second contour line
SLP_LABEL_EVERY = 1

SLP_LINEWIDTH = 0.80

# Season/panel label positions
SEASON_Y = 1.045
PANEL_Y  = 1.035

# Figure and spacing
FIGSIZE = (12.2, 8.7)

SUBPLOT_LEFT   = 0.045
SUBPLOT_RIGHT  = 0.845
SUBPLOT_TOP    = 0.945
SUBPLOT_BOTTOM = 0.055
SUBPLOT_WSPACE = 0.035
SUBPLOT_HSPACE = 0.20

# Colorbar position
CBAR_LEFT   = 0.875
CBAR_BOTTOM = 0.24
CBAR_WIDTH  = 0.030
CBAR_HEIGHT = 0.52



plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.dpi": 300
})



# 4) Helper functions

def standardize_coords(ds):
    rename_dict = {}

    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"

    ds = ds.rename(rename_dict)

    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(
            lon=(((ds["lon"] + 180) % 360) - 180)
        )

    if "lat" in ds.coords:
        ds = ds.sortby("lat")

    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def find_var(ds, possible_names):
    for name in possible_names:
        if name in ds.data_vars:
            return name

    raise ValueError(
        f"None of {possible_names} found. Available variables: {list(ds.data_vars)}"
    )


def get_season_year(time_da, months):
    year = time_da.dt.year

    if 12 in months and 1 in months:
        year = xr.where(time_da.dt.month == 12, year + 1, year)

    return year.rename("season_year")


def seasonal_mean_by_year(da, months, min_months=2):
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


def seasonal_target_and_clim(da, months):
    yearly = seasonal_mean_by_year(da, months, min_months=2)

    if yearly is None:
        raise ValueError("No valid seasonal data found.")

    available_years = yearly["season_year"].values.astype(int)

    if TARGET_YEAR not in available_years:
        raise ValueError(
            f"{TARGET_YEAR} not available. Available years: {available_years}"
        )

    clim = yearly.sel(
        season_year=slice(CLIM_START_YEAR, CLIM_END_YEAR)
    ).mean("season_year", skipna=True)

    target = yearly.sel(season_year=TARGET_YEAR)

    return target, clim


def subset_so(ds):
    return ds.where(ds["lat"] <= -60, drop=True)


def get_lonlat_mesh(ds):
    lon_vals = ds["lon"].values
    lat_vals = ds["lat"].values

    return np.meshgrid(lon_vals, lat_vals)


def thin_field_for_quiver(u_da, v_da, target_lon=36, target_lat=18):
    nlon = u_da.sizes["lon"]
    nlat = u_da.sizes["lat"]

    step_lon = max(1, nlon // target_lon)
    step_lat = max(1, nlat // target_lat)

    u_sub = u_da.isel(
        lat=slice(None, None, step_lat),
        lon=slice(None, None, step_lon)
    )

    v_sub = v_da.isel(
        lat=slice(None, None, step_lat),
        lon=slice(None, None, step_lon)
    )

    return u_sub, v_sub


def polar_ax(fig, nrows, ncols, idx):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, idx, projection=proj)

    theta = np.linspace(0, 2 * np.pi, 300)
    center = [0.5, 0.5]
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    # Gridlines
    lon_grid = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    lat_grid = [-60, -70, -80]

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.42,
        linestyle=":",
        color="black",
        alpha=0.45,
        zorder=27
    )

    gl.xlocator = mticker.FixedLocator(lon_grid)
    gl.ylocator = mticker.FixedLocator(lat_grid)

    # Longitude labels
    edge_lat = -57.8

    label_lons = [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]

    for lon in label_lons:
        if lon < 0:
            label = f"{abs(lon)}°W"
        elif lon == 180:
            label = "180°"
        else:
            label = f"{lon}°E"

        ax.text(
            lon,
            edge_lat,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            zorder=30
        )

    return ax



# NEW LAND FUNCTIONS

def add_land_background(ax):
    """
    Draw land before contours and wind vectors.
    This allows wind vectors and SLP contours to appear over land also.
    """
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.82",
        edgecolor="black",
        linewidth=0.50,
        zorder=5
    )

    ax.coastlines(
        linewidth=0.65,
        zorder=6
    )


def add_coastline_on_top(ax):
    """
    Redraw coastline on top only, so Antarctica boundary remains clear.
    This does not hide wind vectors.
    """
    ax.coastlines(
        linewidth=0.70,
        color="black",
        zorder=45
    )



# 5) Open datasets with low-RAM chunking

ds_sic = xr.open_dataset(
    sic_file,
    decode_times=True,
    chunks={"time": 12, "lat": 300, "lon": 600}
)

ds_wind = xr.open_dataset(
    wind_file,
    decode_times=True,
    chunks={"time": 12}
)

ds_slp = xr.open_dataset(
    slp_file,
    decode_times=True,
    chunks={"time": 12}
)

ds_sic  = standardize_coords(ds_sic)
ds_wind = standardize_coords(ds_wind)
ds_slp  = standardize_coords(ds_slp)

ds_sic  = subset_so(ds_sic)
ds_wind = subset_so(ds_wind)
ds_slp  = subset_so(ds_slp)

print("SIC dataset:")
print(ds_sic)

print("Wind dataset:")
print(ds_wind)

print("SLP dataset:")
print(ds_slp)



# 6) Detect variables

sic_var = find_var(
    ds_sic,
    ["SIF", "sic", "SIC", "sea_ice_fraction", "siconc"]
)

u_var = find_var(
    ds_wind,
    ["u10", "u", "U10", "U", "eastward_wind"]
)

v_var = find_var(
    ds_wind,
    ["v10", "v", "V10", "V", "northward_wind"]
)

slp_var = find_var(
    ds_slp,
    ["msl", "MSL", "mslp", "MSLP", "slp", "SLP"]
)

print("SIC variable:", sic_var)
print("U wind variable:", u_var)
print("V wind variable:", v_var)
print("SLP variable:", slp_var)



# 7) Prepare variables

sic = ds_sic[sic_var].astype("float32")

sic_max = float(sic.max(skipna=True).compute().values)

if sic_max <= 1.5:
    sic_pct = sic * 100.0
    print("SIC/SIF converted from fraction to percent.")
else:
    sic_pct = sic
    print("SIC already appears to be in percent.")

sic_pct = sic_pct.clip(min=0, max=100)

u10 = ds_wind[u_var].astype("float32")
v10 = ds_wind[v_var].astype("float32")

slp = ds_slp[slp_var].astype("float32")

# Convert SLP Pa to hPa if needed
slp_test = float(slp.mean(skipna=True).compute().values)

if slp_test > 2000:
    slp = slp / 100.0
    print("SLP converted from Pa to hPa.")
else:
    print("SLP already appears to be hPa.")



# 8) SIC mesh

lon2_sic, lat2_sic = get_lonlat_mesh(ds_sic)



# 9) Plot figure

fig = plt.figure(figsize=FIGSIZE)

last_pcm = None

for i, season_name in enumerate(SEASON_ORDER):

    print("\nProcessing:", season_name)

    months = SEASONS[season_name]


    # Background: 2025 SIC anomaly

    sic_target, sic_clim = seasonal_target_and_clim(sic_pct, months)
    sic_anom = (sic_target - sic_clim).compute()

    # Arrows: 2025 wind

    u_target, _ = seasonal_target_and_clim(u10, months)
    v_target, _ = seasonal_target_and_clim(v10, months)

    u_plot = u_target.compute()
    v_plot = v_target.compute()

    u_sub, v_sub = thin_field_for_quiver(
        u_plot,
        v_plot,
        target_lon=TARGET_ARROWS_LON,
        target_lat=TARGET_ARROWS_LAT
    )

    # Contours: 2025 SLP
    slp_target, _ = seasonal_target_and_clim(slp, months)
    slp_plot = slp_target.compute()


    # Axis
    ax = polar_ax(fig, 2, 2, i + 1)

    # Draw land first so wind vectors are NOT hidden over land
    add_land_background(ax)

    # SIC anomaly background
    pcm = ax.pcolormesh(
        lon2_sic,
        lat2_sic,
        sic_anom,
        transform=ccrs.PlateCarree(),
        cmap=ANOM_CMAP,
        norm=ANOM_NORM,
        shading="auto",
        zorder=1
    )

    last_pcm = pcm


    # SLP contours
    slp_min = float(np.nanmin(slp_plot.values))
    slp_max = float(np.nanmax(slp_plot.values))

    slp_levels = np.arange(
        np.floor(slp_min / SLP_CONTOUR_INTERVAL) * SLP_CONTOUR_INTERVAL,
        np.ceil(slp_max / SLP_CONTOUR_INTERVAL) * SLP_CONTOUR_INTERVAL + SLP_CONTOUR_INTERVAL,
        SLP_CONTOUR_INTERVAL
    )

    cs = ax.contour(
        slp_plot["lon"].values,
        slp_plot["lat"].values,
        slp_plot.values,
        levels=slp_levels,
        colors="black",
        linewidths=SLP_LINEWIDTH,
        alpha=0.95,
        transform=ccrs.PlateCarree(),
        zorder=18
    )

    # SLP contour labels
    try:
        labels = ax.clabel(
            cs,
            cs.levels[::SLP_LABEL_EVERY],
            inline=True,
            inline_spacing=4,
            fontsize=8,
            fmt="%d",
            colors="black"
        )

        for txt in labels:
            txt.set_fontweight("bold")
            txt.set_zorder(50)
            txt.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground="white")
            ])

    except Exception:
        pass


    # Wind arrows
    ax.quiver(
        u_sub["lon"].values,
        u_sub["lat"].values,
        u_sub.values,
        v_sub.values,
        transform=ccrs.PlateCarree(),
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH,
        headwidth=QUIVER_HEADWIDTH,
        headlength=QUIVER_HEADLENGTH,
        headaxislength=QUIVER_HEADAXISLENGTH,
        pivot="middle",
        color="black",
        alpha=0.95,
        zorder=22
    )

    # Redraw only coastline on top
    add_coastline_on_top(ax)

    # Panel letter
    ax.text(
        -0.10,
        PANEL_Y,
        PANEL_LETTERS[i],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        zorder=60
    )

    # Season label
    ax.text(
        0.50,
        SEASON_Y,
        season_name,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="normal",
        zorder=60
    )

    # RAM cleanup
    del sic_target, sic_clim, sic_anom
    del u_target, v_target, u_plot, v_plot, u_sub, v_sub
    del slp_target, slp_plot
    gc.collect()



# 10) Shared colorbar
cax = fig.add_axes([
    CBAR_LEFT,
    CBAR_BOTTOM,
    CBAR_WIDTH,
    CBAR_HEIGHT
])

cb = fig.colorbar(
    last_pcm,
    cax=cax,
    extend="both",
    ticks=ANOM_TICKS
)

cb.set_label(
    "SIC anomaly (%)",
    fontsize=18,
    fontweight="bold",
    labelpad=12
)

cb.ax.tick_params(labelsize=14)

for t in cb.ax.get_yticklabels():
    t.set_fontweight("bold")



# 11) Layout and save

plt.subplots_adjust(
    left=SUBPLOT_LEFT,
    right=SUBPLOT_RIGHT,
    top=SUBPLOT_TOP,
    bottom=SUBPLOT_BOTTOM,
    wspace=SUBPLOT_WSPACE,
    hspace=SUBPLOT_HSPACE
)

plt.savefig(OUT_PATH, dpi=1080, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(OUT_PATH)

ds_sic.close()
ds_wind.close()
ds_slp.close()
