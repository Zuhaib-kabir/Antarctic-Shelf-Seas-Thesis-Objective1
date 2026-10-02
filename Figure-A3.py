# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')



# Obj1Fig7_SO
# Seasonal Oceanic and Cryospheric Controls on Antarctic SIC
# Improved version:
#   ✔ Black contours
#   ✔ Season-balanced colorbars
#   ✔ Cleaner contour density
#   ✔ Better scientific colormaps
#   ✔ Cleaner labels
#   ✔ Better visual balance


# 0) Install
!pip -q install h5netcdf netCDF4 xarray dask cartopy


# 1) Imports
import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature



# 2) File paths
mld_file = "/content/drive/MyDrive/SAM_Thesis/Data/MLD_monthly_2008_2025_SO.nc"
snowfall_file = "/content/drive/MyDrive/SAM_Thesis/Data/P_Snow_era.nc"
sst_file = "/content/drive/MyDrive/SAM_Thesis/Data/OSTIA_SST_monthly_2008_2025_SO.nc"
sit_file = "/content/drive/MyDrive/SAM_Thesis/Data/GLORYS_SIT_sithick_monthly_2008_2025_SO.nc"

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, "Obj1Fig7_SO.png")

print("Output:", out_file)



# 3) Settings
CLIM_START_YEAR = 2008
CLIM_END_YEAR   = 2025

SEASONS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}

ROW_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


# SST COLORBAR
SST_CBAR = {

    "Spring": {
        "vmin": -2.0,
        "vmax": 2.8,
        "ticks": [-2, -1, 0, 1, 2]
    },

    "Summer": {
        "vmin": -2.0,
        "vmax": 4.5,
        "ticks": [-2, -1, 0, 1, 2, 3, 4]
    },

    "Autumn": {
        "vmin": -2.0,
        "vmax": 4.0,
        "ticks": [-2, -1, 0, 1, 2, 3, 4]
    },

    "Winter": {
        "vmin": -2.0,
        "vmax": 2.8,
        "ticks": [-2, -1, 0, 1, 2]
    },
}



# SNOWFALL COLORBAR
SNOW_CBAR = {

    "Spring": {
        "vmin": 0.0,
        "vmax": 3.5,
        "ticks": [0, 1, 2, 3]
    },

    "Summer": {
        "vmin": 0.0,
        "vmax": 3.0,
        "ticks": [0, 1, 2, 3]
    },

    "Autumn": {
        "vmin": 0.0,
        "vmax": 3.5,
        "ticks": [0, 1, 2, 3]
    },

    "Winter": {
        "vmin": 0.0,
        "vmax": 3.5,
        "ticks": [0, 1, 2, 3]
    },
}



# MLD CONTOURS
MLD_CONTOURS = {

    "Spring": {
        "levels": [50, 100, 150, 200],
        "label_levels": [100, 200],
        "linewidth": 0.7,
        "alpha": 0.9,
    },

    "Summer": {
        "levels": [15, 25, 35, 50],
        "label_levels": [25, 50],
        "linewidth": 0.7,
        "alpha": 0.9,
    },

    "Autumn": {
        "levels": [25, 50, 75, 100],
        "label_levels": [50, 100],
        "linewidth": 0.7,
        "alpha": 0.9,
    },

    "Winter": {
        "levels": [50, 100, 150, 200],
        "label_levels": [100, 200],
        "linewidth": 0.7,
        "alpha": 0.9,
    },
}



# SIT CONTOURS
SIT_CONTOURS = {

    "Spring": {
        "levels": [0.5, 1.0, 1.5, 2.0],
        "label_levels": [0.5, 1.0, 1.5, 2.0],
        "linewidth": 0.7,
        "alpha": 0.95,
    },

    "Summer": {
        "levels": [0.5, 1.0, 1.5],
        "label_levels": [0.5, 1.0, 1.5],
        "linewidth": 0.7,
        "alpha": 0.95,
    },

    "Autumn": {
        "levels": [0.3, 0.7, 1.0, 1.5],
        "label_levels": [0.3, 0.7, 1.0, 1.5],
        "linewidth": 0.7,
        "alpha": 0.95,
    },

    "Winter": {
        "levels": [0.5, 1.0, 1.5, 2.0],
        "label_levels": [0.5, 1.0, 1.5, 2.0],
        "linewidth": 0.7,
        "alpha": 0.95,
    },
}



# COLORMAPS
sst_cmap = "RdYlBu_r"
snow_cmap = "GnBu"



# 4) Helper Functions
def open_dataset_safe(path, name):

    print("\nOpening:", name)

    if name == "Snowfall":

        engines = ["h5netcdf", "netcdf4", "scipy"]

        for eng in engines:

            try:
                ds = xr.open_dataset(
                    path,
                    engine=eng,
                    decode_times=True,
                    chunks="auto"
                )

                print(f"Opened with {eng}")
                return ds

            except Exception as e:
                print("Failed:", eng, e)

        raise RuntimeError(f"Could not open {name}")

    else:

        ds = xr.open_dataset(
            path,
            decode_times=True,
            chunks="auto"
        )

        print("Opened successfully")
        return ds


def standardize_coords(ds):

    rename_dict = {}

    for name in list(ds.coords) + list(ds.dims):

        low = name.lower()

        if low in ["valid_time", "time_counter"]:
            rename_dict[name] = "time"

        elif low in ["latitude", "nav_lat"]:
            rename_dict[name] = "lat"

        elif low in ["longitude", "nav_lon"]:
            rename_dict[name] = "lon"

    rename_dict = {
        k: v for k, v in rename_dict.items()
        if k in ds.coords or k in ds.dims
    }

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def prepare_ds(ds):

    ds = standardize_coords(ds)

    if float(ds["lon"].max()) > 180:

        ds = ds.assign_coords(
            lon=((ds["lon"] + 180) % 360) - 180
        ).sortby("lon")

    ds = ds.sortby("lon")
    ds = ds.sortby("lat")
    ds = ds.sortby("time")

    ds = ds.where(ds["lat"] <= -60, drop=True)

    return ds


def seasonal_climatology(da, months):

    da = da.sel(
        time=slice(
            f"{CLIM_START_YEAR}-01-01",
            f"{CLIM_END_YEAR}-12-31"
        )
    )

    da = da.where(
        da["time"].dt.month.isin(months),
        drop=True
    )

    return da.mean("time", skipna=True)


def mesh_from_da(da):

    lon2, lat2 = np.meshgrid(
        da["lon"].values,
        da["lat"].values
    )

    return lon2, lat2


def polar_ax(fig, nrows, ncols, idx):

    proj = ccrs.SouthPolarStereo()

    ax = fig.add_subplot(
        nrows,
        ncols,
        idx,
        projection=proj
    )

    theta = np.linspace(0, 2*np.pi, 240)

    center = [0.5, 0.5]
    radius = 0.5

    verts = np.vstack([
        np.sin(theta),
        np.cos(theta)
    ]).T

    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_extent(
        [-180, 180, -90, -60],
        ccrs.PlateCarree()
    )

    # Grid
    lon_grid = [-180, -140, -100, -60, -20, 20, 60, 140]
    lat_grid = [-60, -70, -80]

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        linestyle=":",
        color="black",
        alpha=0.30,
        zorder=2
    )

    gl.xlocator = mticker.FixedLocator(lon_grid)
    gl.ylocator = mticker.FixedLocator(lat_grid)

    # Land
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.86",
        edgecolor="black",
        linewidth=0.5,
        zorder=5
    )

    ax.coastlines(
        linewidth=0.55,
        zorder=6
    )

    # Longitude labels
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
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10
        )

    # Latitude labels
    for lat in [-70, -80]:

        ax.text(
            0,
            lat,
            f"{abs(lat)}°S",
            transform=ccrs.PlateCarree(),
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10
        )

    return ax



# 5) Open Datasets
ds_sst  = prepare_ds(open_dataset_safe(sst_file, "SST"))
ds_mld  = prepare_ds(open_dataset_safe(mld_file, "MLD"))
ds_snow = prepare_ds(open_dataset_safe(snowfall_file, "Snowfall"))
ds_sit  = prepare_ds(open_dataset_safe(sit_file, "SIT"))



# 6) Variables
sst  = ds_sst["SST"].astype("float32")
mld  = ds_mld["MLD"].astype("float32")
snow = ds_snow["sf"].astype("float32")
sit  = ds_sit["SIT"].astype("float32")



# 7) Unit Conversion
if float(sst.mean().compute()) > 100:
    sst = sst - 273.15

snow = snow * 1000.0

snow.attrs["units"] = "mm w.e. day⁻¹"



# 8) Cleaning
sst  = sst.where((sst > -5) & (sst < 30))
mld  = mld.where((mld >= 0) & (mld < 2000))
snow = snow.where((snow >= 0) & (snow < 1000))
sit  = sit.where((sit >= 0) & (sit < 20))



# 9) Seasonal Maps
maps_sst  = {}
maps_mld  = {}
maps_snow = {}
maps_sit  = {}

for season_name, months in SEASONS.items():

    print("Processing:", season_name)

    maps_sst[season_name]  = seasonal_climatology(sst, months).compute()
    maps_mld[season_name]  = seasonal_climatology(mld, months).compute()
    maps_snow[season_name] = seasonal_climatology(snow, months).compute()
    maps_sit[season_name]  = seasonal_climatology(sit, months).compute()

    gc.collect()


# 10) Mesh
lon_sst, lat_sst   = mesh_from_da(maps_sst["Spring"])
lon_mld, lat_mld   = mesh_from_da(maps_mld["Spring"])

lon_snow, lat_snow = mesh_from_da(maps_snow["Spring"])
lon_sit, lat_sit   = mesh_from_da(maps_sit["Spring"])



# 11) Plot
fig = plt.figure(figsize=(9, 14))

letters = list("abcdefgh")
letter_i = 0

axs = np.empty((4, 2), dtype=object)

for r, season_name in enumerate(ROW_ORDER):


    # LEFT PANEL
    ax1 = polar_ax(fig, 4, 2, r*2 + 1)

    axs[r, 0] = ax1

    sst_cfg = SST_CBAR[season_name]
    mld_cfg = MLD_CONTOURS[season_name]

    pcm1 = ax1.pcolormesh(
        lon_sst,
        lat_sst,
        maps_sst[season_name],
        transform=ccrs.PlateCarree(),
        cmap=sst_cmap,
        vmin=sst_cfg["vmin"],
        vmax=sst_cfg["vmax"],
        shading="auto",
        zorder=1
    )

    cs1 = ax1.contour(
        lon_mld,
        lat_mld,
        maps_mld[season_name],
        levels=mld_cfg["levels"],
        colors="black",
        linewidths=mld_cfg["linewidth"],
        alpha=mld_cfg["alpha"],
        transform=ccrs.PlateCarree(),
        zorder=8
    )

    ax1.clabel(
        cs1,
        levels=mld_cfg["label_levels"],
        inline=True,
        inline_spacing=3,
        fontsize=5.5,
        fmt="%d",
        colors="black"
    )

    # subplot letter
    ax1.text(
        -0.06,
        0.985,
        letters[letter_i],
        transform=ax1.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top"
    )

    letter_i += 1

    cb1 = fig.colorbar(
        pcm1,
        ax=ax1,
        shrink=0.68,
        pad=0.03,
        ticks=sst_cfg["ticks"],
        extend="max"
    )

    cb1.set_label(
        "SST (°C)",
        fontsize=9,
        fontweight="bold"
    )

    cb1.ax.tick_params(labelsize=8)



    # RIGHT PANEL
    ax2 = polar_ax(fig, 4, 2, r*2 + 2)

    axs[r, 1] = ax2

    snow_cfg = SNOW_CBAR[season_name]
    sit_cfg = SIT_CONTOURS[season_name]

    pcm2 = ax2.pcolormesh(
        lon_snow,
        lat_snow,
        maps_snow[season_name],
        transform=ccrs.PlateCarree(),
        cmap=snow_cmap,
        vmin=snow_cfg["vmin"],
        vmax=snow_cfg["vmax"],
        shading="auto",
        zorder=1
    )

    cs2 = ax2.contour(
        lon_sit,
        lat_sit,
        maps_sit[season_name],
        levels=sit_cfg["levels"],
        colors="black",
        linewidths=sit_cfg["linewidth"],
        alpha=sit_cfg["alpha"],
        transform=ccrs.PlateCarree(),
        zorder=8
    )

    ax2.clabel(
        cs2,
        levels=sit_cfg["label_levels"],
        inline=True,
        inline_spacing=3,
        fontsize=5.5,
        fmt="%.1f",
        colors="black"
    )

    # subplot letter
    ax2.text(
        -0.06,
        0.985,
        letters[letter_i],
        transform=ax2.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top"
    )

    letter_i += 1

    cb2 = fig.colorbar(
        pcm2,
        ax=ax2,
        shrink=0.68,
        pad=0.03,
        ticks=snow_cfg["ticks"],
        extend="max"
    )

    cb2.set_label(
        "Snowfall (mm w.e. day$^{-1}$)",
        fontsize=9,
        fontweight="bold"
    )

    cb2.ax.tick_params(labelsize=8)



# 12) Layout
plt.tight_layout(rect=[0.06, 0.03, 0.98, 0.95])

column_titles = [
    "SST + MLD Contours",
    "Snowfall + SIT Contours"
]

for j, title in enumerate(column_titles):

    pos = axs[0, j].get_position()

    x_center = 0.5 * (pos.x0 + pos.x1)

    fig.text(
        x_center,
        pos.y1 + 0.02,
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

    y_center = 0.5 * (
        min(posL.y0, posR.y0) +
        max(posL.y1, posR.y1)
    )

    fig.text(
        posL.x0 - 0.04,
        y_center,
        season_name,
        rotation=90,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center"
    )



# 13) Save
plt.savefig(
    out_file,
    dpi=1080,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(out_file)



# 14) Close
ds_sst.close()
ds_mld.close()
ds_snow.close()
ds_sit.close()

print("\nDone.")
