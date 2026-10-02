# 1) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# FIGURE 3 — ANTARCTIC SIC TRENDS

# Panels:
#   (a) Spatial Sen's slope (% decade^-1)
#       + trend-free-prewhitened Mann-Kendall significance
#       + Benjamini-Hochberg FDR correction
#
#   (b) Sea-wise annual Sen's slope (% decade^-1)
#       + 95% CI
#       + trend-free-prewhitened Mann-Kendall tau/significance
#
#   (c) Seasonal Sen's slope (% decade^-1)
#       + trend-free-prewhitened Mann-Kendall significance
#
# IMPORTANT CORRECTIONS:
#   ✔ Spatial panel now uses Sen/Theil-Sen, NOT OLS
#   ✔ Autocorrelation handled using trend-free prewhitening
#   ✔ Spatial significance corrected using FDR
#   ✔ Incomplete DJF 2008 removed
#   ✔ Incomplete DJF 2026 removed
#   ✔ Annual means require complete 12-month years
#   ✔ Seasonal means require complete 3-month seasons
#   ✔ 95% Sen-slope CI added to panel (b)
#   ✔ Exact statistics saved as CSV
#   ✔ 1080-dpi output

# 2. INSTALL — GOOGLE COLAB
!pip -q install cartopy scipy xarray dask netCDF4 h5netcdf

# 3. IMPORTS
import os
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.gridspec import GridSpec

from scipy.stats import (
    theilslopes,
    kendalltau
)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter



# 4. PATHS
NC_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"
)

CSV_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SIC_seawise_monthly_2008_2025_LOW_RAM.csv"
)

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(
    OUT_DIR,
    "Fig3_SIC_Trends_Q1_Corrected.png"
)

SUPP_DIR = os.path.join(
    OUT_DIR,
    "Fig3_Trend_Supplementary"
)
os.makedirs(SUPP_DIR, exist_ok=True)


# 5. ANALYSIS SETTINGS
START_YEAR = 2008
END_YEAR = 2025

sea_order = [
    "WED", "KHV", "RLS",
    "LAZ", "COS", "COO",
    "DAV", "MAW", "DUR",
    "SOM", "ROS", "AMU",
    "BEL"
]


# Spatial resolution
COARSEN_FACTOR = 6

# Trend plotting limits
SPATIAL_TREND_LIMIT = 15
SEA_TREND_LIMIT = 20


# Minimum valid observations
MIN_ANNUAL_YEARS = 10
MIN_SEASONAL_YEARS = 8

# Require complete annual/seasonal data
MIN_MONTHS_ANNUAL = 12
MIN_MONTHS_SEASON = 3


# Statistical significance
ALPHA = 0.05

# FDR for spatial significance
FDR_ALPHA = 0.05

# Figure resolution
SAVE_DPI = 1080


# 6. HELPER FUNCTIONS — DATA
def find_sic_variable(ds):

    possible = [
        "SIF",
        "sic",
        "SIC",
        "sea_ice_fraction",
        "ice_conc",
        "sea_ice_concentration",
        "siconc",
    ]

    for v in possible:
        if v in ds.data_vars:
            return v

    for v in ds.data_vars:
        if len(ds[v].dims) >= 3:
            return v

    raise ValueError(
        "Could not identify SIC variable."
    )


def ensure_lat_lon_names(ds):

    rename_dict = {}

    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def ensure_lon_180(ds):

    if "lon" not in ds.coords:
        return ds

    if float(ds["lon"].max()) > 180:

        ds = ds.assign_coords(
            lon=(
                (ds["lon"] + 180) % 360
            ) - 180
        )

        ds = ds.sortby("lon")

    return ds


def convert_sic_to_percent(da):

    sample = float(
        da.isel(
            time=slice(
                0,
                min(12, da.sizes["time"])
            )
        )
        .max(skipna=True)
        .compute()
    )

    if sample <= 1.5:

        print(
            "SIC stored as fraction. "
            "Converting to percent."
        )

        da = da * 100.0

    else:

        print(
            "SIC already appears to be percent."
        )

    return da.clip(
        min=0,
        max=100
    )



# 7. SEN SLOPE + TREND-FREE PREWHITENING

def sen_slope_with_ci(
    years,
    values,
    min_n=8
):
    """
    Calculate Theil-Sen slope and its 95% CI.

    Returns
    -------
    slope_decade
    lower_decade
    upper_decade
    """

    years = np.asarray(
        years,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    mask = (
        np.isfinite(years) &
        np.isfinite(values)
    )

    x = years[mask]
    y = values[mask]

    if len(y) < min_n:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    try:

        slope, intercept, low, high = (
            theilslopes(
                y,
                x,
                alpha=0.95
            )
        )

        return (
            slope * 10.0,
            low * 10.0,
            high * 10.0
        )

    except Exception:

        return (
            np.nan,
            np.nan,
            np.nan
        )


def tfpw_mann_kendall(
    years,
    values,
    min_n=8
):
    """
    Trend-free prewhitened Mann-Kendall-style test.

    Procedure
    ---------
    1. Estimate Sen slope.
    2. Remove the trend.
    3. Estimate lag-1 autocorrelation of detrended residuals.
    4. Prewhiten residuals.
    5. Add trend back.
    6. Calculate Kendall tau and p-value against time.

    This preserves the monotonic trend while reducing inflation
    of significance caused by lag-1 serial dependence.

    Returns
    -------
    tau
    p
    lag1_r
    """

    years = np.asarray(
        years,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    mask = (
        np.isfinite(years) &
        np.isfinite(values)
    )

    x = years[mask]
    y = values[mask]

    if len(y) < min_n:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    # Sen slope in original units per year

    try:

        slope = theilslopes(
            y,
            x
        )[0]

    except Exception:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    # Numerical stability
    x0 = x[0]
    x_centered = x - x0


    # Remove monotonic trend

    detrended = (
        y -
        slope * x_centered
    )


    # Lag-1 correlation

    if len(detrended) < 4:

        tau, p = kendalltau(
            x,
            y
        )

        return (
            tau,
            p,
            np.nan
        )

    y0 = detrended[:-1]
    y1 = detrended[1:]

    valid_pair = (
        np.isfinite(y0) &
        np.isfinite(y1)
    )

    if valid_pair.sum() < 3:

        tau, p = kendalltau(
            x,
            y
        )

        return (
            tau,
            p,
            np.nan
        )

    try:

        lag1_r = np.corrcoef(
            y0[valid_pair],
            y1[valid_pair]
        )[0, 1]

    except Exception:

        lag1_r = np.nan


    # If autocorrelation cannot be estimated,
    # fall back to ordinary Kendall test
                                                              
    if not np.isfinite(lag1_r):

        tau, p = kendalltau(
            x,
            y
        )

        return (
            tau,
            p,
            np.nan
        )

    # Prevent pathological values
    lag1_r = np.clip(
        lag1_r,
        -0.99,
        0.99
    )


    # Trend-free prewhitening
    prewhitened = (
        detrended[1:] -
        lag1_r * detrended[:-1]
    )

    x_pw = x[1:]
    x_pw_centered = (
        x_pw - x0
    )

    # Add Sen trend back
    tfpw_series = (
        prewhitened +
        slope * x_pw_centered
    )


    # Kendall/MK significance

    tau, p = kendalltau(
        x_pw,
        tfpw_series
    )

    return (
        tau,
        p,
        lag1_r
    )



# 8. FDR CORRECTION

def benjamini_hochberg_fdr(
    p_array,
    alpha=0.05
):
    """
    Benjamini-Hochberg false discovery rate correction.

    Returns
    -------
    significance_mask
    p_cutoff
    """

    p = np.asarray(
        p_array,
        dtype=float
    )

    flat = p.ravel()

    valid_idx = np.where(
        np.isfinite(flat)
    )[0]

    sig_flat = np.zeros(
        flat.shape,
        dtype=bool
    )

    if len(valid_idx) == 0:

        return (
            sig_flat.reshape(p.shape),
            np.nan
        )

    p_valid = flat[
        valid_idx
    ]

    order = np.argsort(
        p_valid
    )

    sorted_p = p_valid[
        order
    ]

    m = len(sorted_p)

    thresholds = (
        alpha *
        np.arange(1, m + 1) /
        m
    )

    passed = (
        sorted_p <= thresholds
    )

    if not np.any(passed):

        return (
            sig_flat.reshape(p.shape),
            np.nan
        )

    k_max = np.where(
        passed
    )[0].max()

    cutoff = sorted_p[
        k_max
    ]

    sig_flat[
        valid_idx
    ] = (
        p_valid <= cutoff
    )

    return (
        sig_flat.reshape(p.shape),
        cutoff
    )



# 9. SIGNIFICANCE STAR

def sig_marker(p):

    if not np.isfinite(p):
        return ""

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    return ""


# 10. COMPLETE SEASON ASSIGNMENT


def assign_season_and_year(df):

    out = df.copy()

    month = out["time"].dt.month
    year = out["time"].dt.year

    season = np.full(
        len(out),
        "",
        dtype=object
    )

    season_year = (
        year.astype(int).copy()
    )

    # DJF
    djf = month.isin(
        [12, 1, 2]
    )

    season[
        djf
    ] = "DJF"

    # December belongs to following summer
    dec = (
        month == 12
    )

    season_year.loc[
        dec
    ] = (
        year.loc[dec] + 1
    )

    # MAM
    mam = month.isin(
        [3, 4, 5]
    )

    season[
        mam
    ] = "MAM"

    # JJA
    jja = month.isin(
        [6, 7, 8]
    )

    season[
        jja
    ] = "JJA"

    # SON
    son = month.isin(
        [9, 10, 11]
    )

    season[
        son
    ] = "SON"

    out["season"] = season
    out["season_year"] = season_year

    return out



# 11. PANEL (a) — SPATIAL SEN TREND


print(
    "\n============================================================"
)
print(
    "PANEL (a): SPATIAL SEN TREND"
)
print(
    "============================================================"
)

ds = xr.open_dataset(
    NC_FILE,
    decode_times=True,
    chunks={"time": 12}
)

ds = ensure_lat_lon_names(
    ds
)

ds = ensure_lon_180(
    ds
)

sic_var = find_sic_variable(
    ds
)

print(
    "SIC variable:",
    sic_var
)

da = ds[
    sic_var
]

da = da.sel(
    time=slice(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-31"
    )
)

da = da.where(
    da["lat"] <= -60,
    drop=True
)

da = convert_sic_to_percent(
    da
)


# Coarsen before trend calculation

print(
    "Coarsening spatial grid..."
)

da_c = da.coarsen(
    lat=COARSEN_FACTOR,
    lon=COARSEN_FACTOR,
    boundary="trim"
).mean(
    skipna=True
)


# COMPLETE annual means


print(
    "Calculating complete annual means..."
)

annual_mean = da_c.resample(
    time="YS"
).mean(
    skipna=True
)

annual_count = da_c.resample(
    time="YS"
).count()

# Require all 12 months for each annual grid-cell value
annual = annual_mean.where(
    annual_count >= MIN_MONTHS_ANNUAL
)

annual = annual.load()

years_spatial = (
    annual["time"]
    .dt.year
    .values
    .astype(float)
)

lat = annual["lat"].values
lon = annual["lon"].values

arr = annual.values.astype(
    np.float32
)

n_time, n_lat, n_lon = arr.shape

print(
    "Annual array shape:",
    arr.shape
)



# Allocate outputs

spatial_sen = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)

spatial_low = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)

spatial_high = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)

spatial_tau = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)

spatial_p = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)

spatial_r1 = np.full(
    (n_lat, n_lon),
    np.nan,
    dtype=np.float32
)


# Loop over coarsened grid cells

print(
    "Calculating spatial Sen slopes + "
    "TFPW Mann-Kendall significance..."
)

total_cells = (
    n_lat * n_lon
)

counter = 0

for iy in range(
    n_lat
):

    for ix in range(
        n_lon
    ):

        values = arr[
            :,
            iy,
            ix
        ]

        valid_n = np.isfinite(
            values
        ).sum()

        if valid_n < MIN_ANNUAL_YEARS:
            continue

        (
            slope,
            low,
            high
        ) = sen_slope_with_ci(
            years_spatial,
            values,
            min_n=MIN_ANNUAL_YEARS
        )

        (
            tau,
            p,
            r1
        ) = tfpw_mann_kendall(
            years_spatial,
            values,
            min_n=MIN_ANNUAL_YEARS
        )

        spatial_sen[
            iy,
            ix
        ] = slope

        spatial_low[
            iy,
            ix
        ] = low

        spatial_high[
            iy,
            ix
        ] = high

        spatial_tau[
            iy,
            ix
        ] = tau

        spatial_p[
            iy,
            ix
        ] = p

        spatial_r1[
            iy,
            ix
        ] = r1

        counter += 1

    if (
        (iy + 1) % 10 == 0 or
        iy == n_lat - 1
    ):

        print(
            f"  completed latitude row "
            f"{iy + 1}/{n_lat}"
        )



# FDR correction across spatial map

spatial_sig_fdr, fdr_cutoff = (
    benjamini_hochberg_fdr(
        spatial_p,
        alpha=FDR_ALPHA
    )
)

print(
    "\nSpatial FDR cutoff:",
    fdr_cutoff
)

print(
    "FDR-significant grid cells:",
    int(
        spatial_sig_fdr.sum()
    )
)


# 12. SAVE SPATIAL STATISTICS


spatial_ds = xr.Dataset(
    {
        "sen_slope_pct_decade": (
            ("lat", "lon"),
            spatial_sen
        ),
        "sen_ci_lower_pct_decade": (
            ("lat", "lon"),
            spatial_low
        ),
        "sen_ci_upper_pct_decade": (
            ("lat", "lon"),
            spatial_high
        ),
        "tfpw_kendall_tau": (
            ("lat", "lon"),
            spatial_tau
        ),
        "tfpw_p_value": (
            ("lat", "lon"),
            spatial_p
        ),
        "lag1_autocorrelation": (
            ("lat", "lon"),
            spatial_r1
        ),
        "fdr_significant_0_05": (
            ("lat", "lon"),
            spatial_sig_fdr.astype(
                np.int8
            )
        ),
    },
    coords={
        "lat": lat,
        "lon": lon,
    }
)

spatial_ds.attrs[
    "period"
] = f"{START_YEAR}-{END_YEAR}"

spatial_ds.attrs[
    "trend_method"
] = (
    "Theil-Sen slope"
)

spatial_ds.attrs[
    "significance_method"
] = (
    "Trend-free prewhitening followed by "
    "Kendall trend test; spatial p-values "
    "corrected with Benjamini-Hochberg FDR."
)

spatial_ds.to_netcdf(
    os.path.join(
        SUPP_DIR,
        "Fig3_spatial_Sen_TFPW_FDR.nc"
    )
)


# Memory cleanup
del da
del da_c
del annual_mean
del annual_count
del annual

ds.close()

gc.collect()



# 13. SEA-WISE CSV

print(
    "\n============================================================"
)
print(
    "PANELS (b,c): SEA-WISE TRENDS"
)
print(
    "============================================================"
)

df = pd.read_csv(
    CSV_FILE,
    parse_dates=["time"]
)

df = df.sort_values(
    "time"
).reset_index(
    drop=True
)

df = df[
    ["time"] + sea_order
]

df = df[
    (
        df["time"].dt.year >= START_YEAR
    ) &
    (
        df["time"].dt.year <= END_YEAR
    )
].copy()

df["year"] = (
    df["time"].dt.year
)

df["month"] = (
    df["time"].dt.month
)



# 14. PANEL (b) — COMPLETE ANNUAL SEA-WISE TRENDS

annual_mean_sea = (
    df.groupby(
        "year"
    )[sea_order]
    .mean()
)

annual_count_sea = (
    df.groupby(
        "year"
    )[sea_order]
    .count()
)

# Require all 12 months
annual_sea = annual_mean_sea.where(
    annual_count_sea >= MIN_MONTHS_ANNUAL
)

years = annual_sea.index.values.astype(
    float
)

annual_rows = []

for sea in sea_order:

    vals = annual_sea[
        sea
    ].values.astype(
        float
    )

    (
        sen,
        ci_low,
        ci_high
    ) = sen_slope_with_ci(
        years,
        vals,
        min_n=MIN_ANNUAL_YEARS
    )

    (
        tau,
        p,
        r1
    ) = tfpw_mann_kendall(
        years,
        vals,
        min_n=MIN_ANNUAL_YEARS
    )

    annual_rows.append(
        {
            "Sea": sea,
            "Sen_slope_pct_decade": sen,
            "CI_2.5": ci_low,
            "CI_97.5": ci_high,
            "TFPW_Kendall_tau": tau,
            "TFPW_p": p,
            "Lag1_r": r1,
            "N_years": int(
                np.isfinite(
                    vals
                ).sum()
            ),
        }
    )

annual_stats = pd.DataFrame(
    annual_rows
)

annual_stats.to_csv(
    os.path.join(
        SUPP_DIR,
        "Fig3_seawise_annual_Sen_TFPW.csv"
    ),
    index=False
)

print(
    "\nAnnual sea-wise trend statistics:"
)

print(
    annual_stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# Arrays used for plotting
sen_values = (
    annual_stats[
        "Sen_slope_pct_decade"
    ].values
)

sen_low = (
    annual_stats[
        "CI_2.5"
    ].values
)

sen_high = (
    annual_stats[
        "CI_97.5"
    ].values
)

tau_values = (
    annual_stats[
        "TFPW_Kendall_tau"
    ].values
)

p_values = (
    annual_stats[
        "TFPW_p"
    ].values
)



# 15. PANEL (c) — COMPLETE SEASONAL TRENDS

df_season = assign_season_and_year(
    df
)

# Keep only complete target season-years
df_season = df_season[
    (
        df_season["season_year"] >= START_YEAR
    ) &
    (
        df_season["season_year"] <= END_YEAR
    )
].copy()


season_order = [
    "DJF",
    "MAM",
    "JJA",
    "SON"
]

season_labels_full = [
    "Summer",
    "Autumn",
    "Winter",
    "Spring"
]


seasonal_trend = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_low = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_high = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_tau = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_p = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_r1 = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)

seasonal_n = pd.DataFrame(
    index=season_order,
    columns=sea_order,
    dtype=float
)


for season in season_order:

    sub = df_season[
        df_season["season"] == season
    ].copy()

    seasonal_mean = (
        sub.groupby(
            "season_year"
        )[sea_order]
        .mean()
    )

    seasonal_count = (
        sub.groupby(
            "season_year"
        )[sea_order]
        .count()
    )


    # CRITICAL:
    # require exactly three valid months

    seasonal_mean = seasonal_mean.where(
        seasonal_count >= MIN_MONTHS_SEASON
    )

    yrs = (
        seasonal_mean
        .index
        .values
        .astype(float)
    )

    for sea in sea_order:

        vals = seasonal_mean[
            sea
        ].values.astype(
            float
        )

        (
            sen,
            low,
            high
        ) = sen_slope_with_ci(
            yrs,
            vals,
            min_n=MIN_SEASONAL_YEARS
        )

        (
            tau,
            p,
            r1
        ) = tfpw_mann_kendall(
            yrs,
            vals,
            min_n=MIN_SEASONAL_YEARS
        )

        seasonal_trend.loc[
            season,
            sea
        ] = sen

        seasonal_low.loc[
            season,
            sea
        ] = low

        seasonal_high.loc[
            season,
            sea
        ] = high

        seasonal_tau.loc[
            season,
            sea
        ] = tau

        seasonal_p.loc[
            season,
            sea
        ] = p

        seasonal_r1.loc[
            season,
            sea
        ] = r1

        seasonal_n.loc[
            season,
            sea
        ] = np.isfinite(
            vals
        ).sum()



# Save full seasonal statistics


seasonal_rows = []

for season in season_order:

    for sea in sea_order:

        seasonal_rows.append(
            {
                "Season": season,
                "Sea": sea,
                "Sen_slope_pct_decade":
                    seasonal_trend.loc[
                        season,
                        sea
                    ],
                "CI_2.5":
                    seasonal_low.loc[
                        season,
                        sea
                    ],
                "CI_97.5":
                    seasonal_high.loc[
                        season,
                        sea
                    ],
                "TFPW_Kendall_tau":
                    seasonal_tau.loc[
                        season,
                        sea
                    ],
                "TFPW_p":
                    seasonal_p.loc[
                        season,
                        sea
                    ],
                "Lag1_r":
                    seasonal_r1.loc[
                        season,
                        sea
                    ],
                "N_complete_seasons":
                    seasonal_n.loc[
                        season,
                        sea
                    ],
            }
        )

seasonal_stats = pd.DataFrame(
    seasonal_rows
)

seasonal_stats.to_csv(
    os.path.join(
        SUPP_DIR,
        "Fig3_seasonal_Sen_TFPW.csv"
    ),
    index=False
)

print(
    "\nSeasonal statistics saved."
)



# 16. FIGURE SETTINGS


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.9,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

fig = plt.figure(
    figsize=(15.8, 9.8),
    dpi=300
)

gs = GridSpec(
    nrows=2,
    ncols=2,
    figure=fig,
    height_ratios=[
        1.25,
        0.95
    ],
    width_ratios=[
        1.0,
        1.08
    ],
    hspace=0.26,
    wspace=0.24
)



# Trend colormap
# Negative = red
# Positive = blue

cmap_trend = mpl.colormaps[
    "RdBu"
]

norm_spatial = mpl.colors.TwoSlopeNorm(
    vmin=-SPATIAL_TREND_LIMIT,
    vcenter=0,
    vmax=SPATIAL_TREND_LIMIT
)

norm_heat = mpl.colors.TwoSlopeNorm(
    vmin=-SEA_TREND_LIMIT,
    vcenter=0,
    vmax=SEA_TREND_LIMIT
)



# 17. PANEL (a) — SPATIAL SEN SLOPE


ax_map = fig.add_subplot(
    gs[0, 0],
    projection=ccrs.SouthPolarStereo()
)

ax_map.set_extent(
    [-180, 180, -90, -60],
    crs=ccrs.PlateCarree()
)


pcm = ax_map.pcolormesh(
    lon,
    lat,
    spatial_sen,
    transform=ccrs.PlateCarree(),
    cmap=cmap_trend,
    norm=norm_spatial,
    shading="auto",
    rasterized=True,
    zorder=1
)


# FDR significance stippling


lon2, lat2 = np.meshgrid(
    lon,
    lat
)

sig = (
    spatial_sig_fdr &
    np.isfinite(
        spatial_sen
    )
)

# Thin dots slightly for readability if necessary
sig_y, sig_x = np.where(
    sig
)

if len(sig_x) > 0:

    # If many significant cells exist,
    # subsample only the plotted stippling, not the test.
    max_stipple = 8000

    if len(sig_x) > max_stipple:

        idx = np.linspace(
            0,
            len(sig_x) - 1,
            max_stipple
        ).astype(int)

        sig_y_plot = sig_y[idx]
        sig_x_plot = sig_x[idx]

    else:

        sig_y_plot = sig_y
        sig_x_plot = sig_x

    ax_map.scatter(
        lon2[
            sig_y_plot,
            sig_x_plot
        ],
        lat2[
            sig_y_plot,
            sig_x_plot
        ],
        s=2.0,
        c="black",
        alpha=0.55,
        marker=".",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=4
    )


ax_map.add_feature(
    cfeature.LAND,
    facecolor="0.84",
    edgecolor="0.35",
    linewidth=0.4,
    zorder=5
)

ax_map.coastlines(
    resolution="110m",
    linewidth=0.4,
    zorder=6
)

gl = ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.35,
    color="0.45",
    alpha=0.45,
    linestyle="--"
)

# Publication-style latitude/longitude labels
gl.left_labels = True
gl.right_labels = True
gl.top_labels = True
gl.bottom_labels = True
gl.xlabel_style = {"size": 9, "weight": "bold"}
gl.ylabel_style = {"size": 9, "weight": "bold"}
gl.xformatter = LongitudeFormatter(
    zero_direction_label=True
)
gl.yformatter = LatitudeFormatter()

# Avoid crowded longitude labels such as 100E and 140W
gl.xlocator = mpl.ticker.FixedLocator(
    [-180, -120, -90, -60, -30, 0, 30, 60, 120, 150]
)

# Circular boundary
theta = np.linspace(
    0,
    2 * np.pi,
    300
)

center = [
    0.5,
    0.5
]

radius = 0.5

verts = np.vstack(
    [
        np.sin(theta),
        np.cos(theta)
    ]
).T

circle = mpl.path.Path(
    verts * radius +
    center
)

ax_map.set_boundary(
    circle,
    transform=ax_map.transAxes
)


ax_map.text(
    -0.09,
    1.04,
    "(a)",
    transform=ax_map.transAxes,
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top",
    clip_on=False
)



# Map colorbar


fig.canvas.draw()

map_pos = ax_map.get_position()

cax_map = fig.add_axes(
    [
        map_pos.x1 + 0.010,
        map_pos.y0 + 0.23 * map_pos.height,
        0.012,
        0.55 * map_pos.height
    ]
)

cb_map = fig.colorbar(
    pcm,
    cax=cax_map,
    extend="both"
)

cb_map.set_label(
    "Sen's slope\n(% decade$^{-1}$)",
    fontsize=8,
    labelpad=7
)

cb_map.set_ticks(
    [-15, -10, -5, 0, 5, 10, 15]
)

cb_map.ax.tick_params(
    labelsize=8
)

cb_map.outline.set_linewidth(
    0.5
)


# Small significance note
ax_map.text(
    0.02,
    0.02,
    "Dots: TFPW trend significant\n"
    "after FDR (q < 0.05)",
    transform=ax_map.transAxes,
    fontsize=7.5,
    ha="left",
    va="bottom",
    bbox=dict(
        facecolor="white",
        edgecolor="0.4",
        alpha=0.85,
        boxstyle="round,pad=0.25"
    ),
    zorder=10
)



# 18. PANEL (b) — SEA-WISE ANNUAL TREND


ax_bar = fig.add_subplot(
    gs[0, 1]
)

xpos = np.arange(
    len(sea_order)
)

positive_color = "#1f5bbd"
negative_color = "#d94841"

bar_colors = np.where(
    sen_values >= 0,
    positive_color,
    negative_color
)


bars = ax_bar.bar(
    xpos,
    sen_values,
    color=bar_colors,
    width=0.62,
    edgecolor="0.25",
    linewidth=0.3,
    zorder=2
)



# 95% CI error bars


lower_err = (
    sen_values -
    sen_low
)

upper_err = (
    sen_high -
    sen_values
)

lower_err = np.where(
    np.isfinite(lower_err),
    np.maximum(
        lower_err,
        0
    ),
    0
)

upper_err = np.where(
    np.isfinite(upper_err),
    np.maximum(
        upper_err,
        0
    ),
    0
)

ax_bar.errorbar(
    xpos,
    sen_values,
    yerr=np.vstack(
        [
            lower_err,
            upper_err
        ]
    ),
    fmt="none",
    ecolor="black",
    elinewidth=0.8,
    capsize=2.5,
    capthick=0.8,
    zorder=4
)


ax_bar.axhline(
    0,
    color="0.25",
    lw=0.7,
    zorder=1
)

ax_bar.set_xticks(
    xpos
)

ax_bar.set_xticklabels(
    sea_order,
    fontsize=10,
    fontweight="bold"
)

ax_bar.set_ylabel(
    "Sen's slope (% decade$^{-1}$)",
    fontsize=10,
    labelpad=7
)

# Dynamic y-limit
finite_ci = np.concatenate(
    [
        sen_low[
            np.isfinite(sen_low)
        ],
        sen_high[
            np.isfinite(sen_high)
        ]
    ]
)

if finite_ci.size:

    y_abs = max(
        SEA_TREND_LIMIT,
        np.ceil(
            np.nanmax(
                np.abs(
                    finite_ci
                )
            ) / 5
        ) * 5
    )

else:

    y_abs = SEA_TREND_LIMIT


ax_bar.set_ylim(
    -y_abs,
    y_abs
)

ax_bar.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.35,
    zorder=0
)


# Significance stars


star_offset = (
    0.04 *
    2 *
    y_abs
)

for i, (
    sen,
    ci_lo,
    ci_hi,
    p
) in enumerate(
    zip(
        sen_values,
        sen_low,
        sen_high,
        p_values
    )
):

    marker = sig_marker(
        p
    )

    if marker == "":
        continue

    if sen >= 0:

        base = (
            ci_hi
            if np.isfinite(ci_hi)
            else sen
        )

        y_pos = (
            base +
            star_offset
        )

        va = "bottom"

    else:

        base = (
            ci_lo
            if np.isfinite(ci_lo)
            else sen
        )

        y_pos = (
            base -
            star_offset
        )

        va = "top"

    ax_bar.text(
        i,
        y_pos,
        marker,
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold"
    )


# TFPW Kendall tau secondary axis


ax_tau = ax_bar.twinx()

ax_tau.scatter(
    xpos,
    tau_values,
    color="black",
    s=20,
    zorder=5
)

ax_tau.set_ylim(
    -1,
    1
)

ax_tau.set_ylabel(
    "TFPW Kendall τ",
    fontsize=10
)

ax_tau.tick_params(
    axis="y",
    labelsize=8
)
for label in ax_tau.get_yticklabels():
    label.set_fontweight("bold")


# Legend

bar_proxy = mpl.patches.Patch(
    facecolor=negative_color,
    edgecolor="0.25",
    label="Sen's slope"
)

ci_proxy = mpl.lines.Line2D(
    [],
    [],
    color="black",
    marker="_",
    linestyle="-",
    markersize=7,
    label="95% Sen CI"
)

dot_proxy = mpl.lines.Line2D(
    [],
    [],
    color="black",
    marker="o",
    linestyle="None",
    markersize=4,
    label="TFPW Kendall τ"
)

ax_bar.legend(
    handles=[
        bar_proxy,
        ci_proxy,
        dot_proxy
    ],
    loc="upper center",
    bbox_to_anchor=(
        0.50,
        0.98
    ),
    ncol=3,
    frameon=False,
    fontsize=8
)


ax_bar.text(
    0.98,
    0.97,
    "*** p < 0.001\n"
    "** p < 0.01\n"
    "* p < 0.05",
    transform=ax_bar.transAxes,
    ha="right",
    va="top",
    fontsize=7.5
)


ax_bar.text(
    0.01,
    0.98,
    "(b)",
    transform=ax_bar.transAxes,
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top"
)



# 19. PANEL (c) — SEASONAL SEN TREND HEATMAP


ax_heat = fig.add_subplot(
    gs[1, :]
)

heat_data = (
    seasonal_trend
    .loc[
        season_order,
        sea_order
    ]
    .values
    .astype(float)
)


# Dynamic heatmap limit
finite_heat = heat_data[
    np.isfinite(
        heat_data
    )
]

if finite_heat.size:

    heat_lim = max(
        10,
        np.ceil(
            np.nanmax(
                np.abs(
                    finite_heat
                )
            ) / 5
        ) * 5
    )

else:

    heat_lim = SEA_TREND_LIMIT


heat_norm = mpl.colors.TwoSlopeNorm(
    vmin=-heat_lim,
    vcenter=0,
    vmax=heat_lim
)


im = ax_heat.imshow(
    heat_data,
    cmap=cmap_trend,
    norm=heat_norm,
    aspect="auto"
)


ax_heat.set_xticks(
    np.arange(
        len(
            sea_order
        )
    )
)

ax_heat.set_xticklabels(
    sea_order,
    fontsize=9,
    fontweight="bold"
)

ax_heat.set_yticks(
    np.arange(
        len(
            season_order
        )
    )
)

ax_heat.set_yticklabels(
    season_labels_full,
    fontsize=9,
    fontweight="bold"
)


ax_heat.xaxis.tick_top()

ax_heat.tick_params(
    top=True,
    bottom=False,
    labeltop=True,
    labelbottom=False
)



# Cell borders

ax_heat.set_xticks(
    np.arange(
        -0.5,
        len(sea_order),
        1
    ),
    minor=True
)

ax_heat.set_yticks(
    np.arange(
        -0.5,
        len(season_order),
        1
    ),
    minor=True
)

ax_heat.grid(
    which="minor",
    color="0.45",
    linestyle="-",
    linewidth=0.35
)

ax_heat.tick_params(
    which="minor",
    bottom=False,
    left=False
)



# Cell values + TFPW significance

for i, season in enumerate(
    season_order
):

    for j, sea in enumerate(
        sea_order
    ):

        val = seasonal_trend.loc[
            season,
            sea
        ]

        p = seasonal_p.loc[
            season,
            sea
        ]

        marker = sig_marker(
            p
        )

        if np.isfinite(
            val
        ):

            txt = (
                f"{val:+.1f}"
                f"{marker}"
            )

        else:

            txt = ""

        if (
            np.isfinite(val) and
            abs(val) >=
            0.50 * heat_lim
        ):

            text_color = "white"

        else:

            text_color = "black"

        ax_heat.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            fontweight="bold"
        )


ax_heat.text(
    -0.04,
    1.20,
    "(c)",
    transform=ax_heat.transAxes,
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top"
)



# Heatmap colorbar


fig.canvas.draw()

heat_pos = (
    ax_heat.get_position()
)

cax_heat = fig.add_axes(
    [
        heat_pos.x1 + 0.012,
        heat_pos.y0 + 0.16 * heat_pos.height,
        0.014,
        0.70 * heat_pos.height
    ]
)

cb_heat = fig.colorbar(
    im,
    cax=cax_heat,
    extend="both"
)

cb_heat.set_label(
    "Sen's slope\n(% decade$^{-1}$)",
    fontsize=9
)

cb_heat.ax.tick_params(
    labelsize=8
)

cb_heat.outline.set_linewidth(
    0.5
)


# Heatmap note


ax_heat.text(
    0.0,
    -0.31,
    "Positive values indicate increasing SIC; negative values indicate decreasing SIC. "
    "Asterisks indicate trend-free-prewhitened Mann–Kendall significance.\n"
    "Only complete three-month seasons were retained; incomplete DJF seasons at the beginning "
    "and end of the record were excluded.",
    transform=ax_heat.transAxes,
    ha="left",
    va="top",
    fontsize=8
)



# 20. SAVE FIGURE — 1080 DPI

plt.savefig(
    OUT_PATH,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(
    "\n============================================================"
)

print(
    "FIGURE SAVED:"
)

print(
    OUT_PATH
)

print(
    "\nSUPPLEMENTARY STATISTICS:"
)

print(
    SUPP_DIR
)

print(
    "============================================================"
)



# 21. OPTIONAL SUMMARY
print(
    "\nStrongest negative annual sea-wise Sen trends:"
)

print(
    annual_stats
    .sort_values(
        "Sen_slope_pct_decade"
    )[
        [
            "Sea",
            "Sen_slope_pct_decade",
            "CI_2.5",
            "CI_97.5",
            "TFPW_p",
            "Lag1_r"
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


print(
    "\nStrongest negative seasonal Sen trends:"
)

print(
    seasonal_stats
    .sort_values(
        "Sen_slope_pct_decade"
    )
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

gc.collect()

print(
    "\nDONE."
)
