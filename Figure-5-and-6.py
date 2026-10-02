#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip install cartopy
import cartopy

"""
Create corrected Figures 4 and 5 and one combined FDR summary table.

Figure 4 rows
---------------
1. SST
2. 10-m wind speed
3. MLD

Figure 5 rows
---------------
1. 2-m air temperature
2. OHC (0–100 m)
3. Ocean-current speed

Columns
-------
Spring (SON), Summer (DJF), Autumn (MAM), Winter (JJA)

Statistical method for new variables
------------------------------------
1. Monthly climatological anomalies.
2. Grid-cell Pearson correlation with monthly SIC anomalies.
3. Lag-1 autocorrelation-adjusted effective sample size.
4. Two-sided adjusted p-value.
5. Panel-wise Benjamini–Hochberg FDR at q < 0.05.

OHC objective analysis
----------------------
Monthly Argo OHC anomalies are mapped before correlation using local
objective analysis / optimal interpolation (OI) with a Gaussian horizontal
covariance. Mapping-error and local-observation-count limits prevent
unsupported extrapolation. No interpolation is applied to the final
correlation coefficients.

Recommended Colab installation cell:
    %pip install -q netCDF4 h5netcdf xarray dask scipy \
        cartopy matplotlib pandas bottleneck
"""


# 1. IMPORTS

from pathlib import Path
import gc
import os
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial import cKDTree
from scipy.stats import t as student_t


# 2. USER PATHS


DATA_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/Data"
)

OHC_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/Fig/Fig5_OHC_supplementary"
)
# Folder containing the 16 previously calculated panels.
EXISTING_RESULT_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/Fig/"
    "Fig4_spatial_correlation_supplementary"
)

# Cleaned output path. The accidental space after "P1_Corrected"
# has been removed.
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/SAM_Thesis/Fig/"
    "P1_Corrected/Fig4_5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Gridded monthly SIC input
# Use the monthly OSTIA sea-ice-fraction product.  Do not use the
# Antarctic_SeaIce_Drift files: those contain ice-velocity variables,
# not sea-ice concentration/fraction.
SIC_NC_FILES = [
    DATA_DIR / "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc",
]



# OHC profile data
# This must contain:
# time/month, lat, lon, and OHC_0_100m_GJ_m2.
OHC_PROFILE_CSV = (
    OHC_DIR / "Argo_profile_OHC_0_100m.csv"
)



# Ocean-current components
# The repeated 2015–2025 eastward path from the question is included only once.
U_CURRENT_FILES = [
    DATA_DIR / "EastwardSeaWaterVelocity_uo_2008_14 .nc",
    DATA_DIR / "EastwardSeaWaterVelocity_uo_2015_25 .nc",
]

V_CURRENT_FILES = [
    DATA_DIR / "NorthwardSeaWaterVelocity_uo_2008_14 .nc",
    DATA_DIR / "NorthwardSeaWaterVelocity_uo_2015_25 .nc",
]


# Output files

FIG4_PNG = OUTPUT_DIR / (
    "Fig4_SST_WindSpeed_MLD_spatial_correlations.png"
)

FIG4_PDF = OUTPUT_DIR / (
    "Fig4_SST_WindSpeed_MLD_spatial_correlations.pdf"
)

FIG5_PNG = OUTPUT_DIR / (
    "Fig5_AirTemperature_OHC_Current_spatial_correlations.png"
)

FIG5_PDF = OUTPUT_DIR / (
    "Fig5_AirTemperature_OHC_Current_spatial_correlations.pdf"
)

COMBINED_TABLE_CSV = OUTPUT_DIR / (
    "Table_Spatial_Correlation_FDR_All_6_Variables.csv"
)

NEW_PANEL_DIR = OUTPUT_DIR / "new_correlation_significance_files"
NEW_PANEL_DIR.mkdir(parents=True, exist_ok=True)

OHC_OI_MONTHLY_NC = NEW_PANEL_DIR / (
    "OHC_0_100m_monthly_anomalies_local_objective_analysis_2008_2025.nc"
)



# 3. ANALYSIS SETTINGS

START_DATE = "2008-01-01"
END_DATE = "2025-12-31"

START_YEAR = 2008
END_YEAR = 2025

SEASON_ORDER = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
]

SEASON_ABBR = {
    "Spring": "SON",
    "Summer": "DJF",
    "Autumn": "MAM",
    "Winter": "JJA",
}

SEASON_MONTHS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}

FIG4_ROWS = [
    "SST",
    "Wind Speed",
    "MLD",
]

FIG5_ROWS = [
    "Air Temperature",
    "OHC",
    "Ocean Current Speed",
]

ROW_TITLES = {
    "SST": "SST",
    "Wind Speed": "10 m wind speed",
    "MLD": "MLD",
    "Air Temperature": "2 m air temperature",
    "OHC": "OHC (0–100 m)",
    "Ocean Current Speed": "Ocean current speed",
}

FILE_DRIVER_NAMES = {
    "SST": "SST",
    "Wind Speed": "Wind_Speed",
    "MLD": "MLD",
    "Air Temperature": "Air_Temperature",
}

CORRELATION_MIN = -1.0
CORRELATION_MAX = 1.0

CORRELATION_TICKS = [
    -1.00,
    -0.75,
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

FDR_Q = 0.05

# Minimum paired monthly observations required in a grid cell.
MIN_PAIRED_MONTHS = 12

# Stippling thinning for fine-resolution SST/current grids.
FINE_GRID_STIPPLE_STRIDE = 6

# Coarse source bins preserve enough repeated Argo observations for the
# calendar-month climatology. These are observation-support bins, not the
# final objective-analysis grid.
OHC_SOURCE_LAT_STEP_DEGREES = 2.0
OHC_SOURCE_LON_STEP_DEGREES = 4.0

# Local-OI target grid. It is finer than the source bins so the mapped field
# is spatially smooth, while the mapping-error/support masks still prevent
# unsupported extrapolation.
OHC_OI_LAT_STEP_DEGREES = 1.0
OHC_OI_LON_STEP_DEGREES = 2.0

# Gaussian covariance: C(d) = exp[-(d / L)^2].
OHC_OI_LENGTH_SCALE_KM = 500.0

# Only observations within this radius can influence a target cell.
OHC_OI_SEARCH_RADIUS_KM = 1200.0

# Additional observation-error variance divided by signal variance.
# The diagonal matrix term is therefore 0.15 I.
OHC_OI_NOISE_TO_SIGNAL_VARIANCE = 0.15

# Local support requirements.
OHC_OI_MIN_OBSERVATIONS = 3
OHC_OI_MAX_NEIGHBORS = 16
OHC_OI_MAX_NORMALIZED_ERROR = 0.70

# At least this many years must contribute to a source-bin calendar-month
# climatology before that source-bin anomaly can enter objective analysis.
OHC_CLIMATOLOGY_MIN_YEARS = 3

# Small diagonal term for stable matrix solution.
OHC_OI_NUMERICAL_NUGGET = 1.0e-8

# Ocean-current depth selection:
# "surface" selects the shallowest available model level.
CURRENT_DEPTH_MODE = "surface"

FIGURE_SIZE_INCHES = (12.8, 9.8)
# a 12.8 × 9.8 inch figure rendered at 1080 DPI. The PDF is also saved.
SAVE_DPI = 1080
SHOW_IN_NOTEBOOK = True



# 4. GENERAL HELPERS

def unique_paths(paths):
    """Remove duplicated paths while preserving their original order."""

    output = []
    seen = set()

    for path in paths:
        resolved_text = str(Path(path))

        if resolved_text not in seen:
            output.append(Path(path))
            seen.add(resolved_text)

    return output


U_CURRENT_FILES = unique_paths(U_CURRENT_FILES)
V_CURRENT_FILES = unique_paths(V_CURRENT_FILES)


def validate_files(paths, description):
    """Raise a clear error when one or more required files are absent."""

    missing = [Path(path) for path in paths if not Path(path).exists()]

    if missing:
        message = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            f"Missing {description} file(s):\n{message}\n\n"
            "Check spaces and underscores in the filenames carefully."
        )


def detect_coordinate_name(data_object, candidates):
    """Return the first matching coordinate or dimension name."""

    available = list(data_object.coords) + list(data_object.dims)
    lower_lookup = {str(name).lower(): name for name in available}

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    return None


def choose_data_variable(dataset, candidates, description):
    """Select a variable using common variable-name alternatives."""

    lower_lookup = {
        str(name).lower(): name
        for name in dataset.data_vars
    }

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return dataset[lower_lookup[candidate.lower()]]

    # Try partial matching.
    for variable_name in dataset.data_vars:
        lower_name = str(variable_name).lower()

        if any(candidate.lower() in lower_name for candidate in candidates):
            return dataset[variable_name]

    raise ValueError(
        f"Could not identify {description} variable.\n"
        f"Available variables: {list(dataset.data_vars)}"
    )


def standardize_dimensions(data_array):
    """Rename common time/latitude/longitude dimensions."""

    time_name = detect_coordinate_name(
        data_array,
        ["time", "valid_time", "date", "datetime"],
    )

    lat_name = detect_coordinate_name(
        data_array,
        ["lat", "latitude", "nav_lat", "y"],
    )

    lon_name = detect_coordinate_name(
        data_array,
        ["lon", "longitude", "nav_lon", "x"],
    )

    if time_name is None:
        raise ValueError(
            f"No time coordinate found. Dimensions: {data_array.dims}"
        )

    if lat_name is None or lon_name is None:
        raise ValueError(
            "No usable 1-D latitude/longitude coordinates were found. "
            f"Dimensions: {data_array.dims}"
        )

    rename_dictionary = {}

    if time_name != "time":
        rename_dictionary[time_name] = "time"

    if lat_name != "lat":
        rename_dictionary[lat_name] = "lat"

    if lon_name != "lon":
        rename_dictionary[lon_name] = "lon"

    data_array = data_array.rename(rename_dictionary)

    if data_array["lat"].ndim != 1 or data_array["lon"].ndim != 1:
        raise ValueError(
            "This script requires 1-D latitude and longitude coordinates. "
            "If the SIC file uses projected x/y or 2-D coordinates, use the "
            "same regridded SIC file employed in the original Figure 4."
        )

    return data_array


def remove_extra_dimensions(data_array):
    """Remove singleton or ensemble-like dimensions."""

    protected = {"time", "lat", "lon"}

    for dimension in list(data_array.dims):
        if dimension in protected:
            continue

        lower_name = dimension.lower()

        if lower_name in {
            "expver",
            "ensemble",
            "member",
            "number",
            "realization",
        }:
            data_array = data_array.mean(
                dimension,
                skipna=True,
            )

        elif data_array.sizes[dimension] == 1:
            data_array = data_array.isel(
                {dimension: 0},
                drop=True,
            )

    return data_array


def normalize_longitude(data_array):
    """Convert longitude to −180° to 180° and sort."""

    normalized_longitude = (
        (data_array["lon"] + 180.0) % 360.0
    ) - 180.0

    data_array = data_array.assign_coords(
        lon=normalized_longitude
    )

    data_array = data_array.sortby("lon")

    # Average duplicated 0/360-degree coordinates, if present.
    longitude_values = np.asarray(data_array["lon"].values)

    if np.unique(longitude_values).size != longitude_values.size:
        data_array = data_array.groupby("lon").mean(
            skipna=True
        )

    return data_array


def standardize_monthly_time(data_array):
    """Convert timestamps to calendar-month starts and average duplicates."""

    converted_time = (
        pd.DatetimeIndex(pd.to_datetime(data_array["time"].values))
        .to_period("M")
        .to_timestamp()
    )

    data_array = data_array.assign_coords(
        time=converted_time
    )

    if converted_time.duplicated().any():
        data_array = data_array.groupby("time").mean(
            skipna=True
        )

    data_array = data_array.sortby("time")

    data_array = data_array.sel(
        time=slice(START_DATE, END_DATE)
    )

    return data_array


def monthly_anomaly(data_array):
    """Remove each grid cell's 2008–2025 monthly climatology."""

    climatology = data_array.groupby("time.month").mean(
        "time",
        skipna=True,
    )

    anomaly = (
        data_array.groupby("time.month")
        - climatology
    )

    return anomaly.astype("float32")


def interpolate_to_grid(data_array, target_lat, target_lon):
    """Interpolate a regular-grid DataArray to a target regular grid."""

    data_array = data_array.sortby("lat")
    data_array = data_array.sortby("lon")

    interpolated = data_array.interp(
        lat=np.sort(np.asarray(target_lat, dtype=float)),
        lon=np.sort(np.asarray(target_lon, dtype=float)),
        method="linear",
    )

    # Return to the target coordinate order.
    interpolated = interpolated.sel(
        lat=np.asarray(target_lat),
        lon=np.asarray(target_lon),
    )

    return interpolated



# 5. LOAD THE EXISTING 16 RESULT FILES

def existing_result_filename(driver_name, season_name):
    driver_token = FILE_DRIVER_NAMES[driver_name]

    return (
        f"Fig4_{driver_token}_{season_name}"
        "_correlation_significance.nc"
    )


def load_saved_panel(driver_name, season_name):
    path = (
        EXISTING_RESULT_DIR
        / existing_result_filename(driver_name, season_name)
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Saved result file not found:\n{path}"
        )

    print(
        f"Loading existing result: "
        f"{driver_name} | {season_name}"
    )

    with xr.open_dataset(path, engine="netcdf4") as dataset:

        required_variables = {
            "r",
            "significant_fdr",
        }

        missing_variables = (
            required_variables.difference(dataset.data_vars)
        )

        if missing_variables:
            raise ValueError(
                f"{path.name} is missing variables: "
                f"{sorted(missing_variables)}"
            )

        correlation = (
            dataset["r"]
            .transpose("lat", "lon")
            .astype("float32")
            .load()
        )

        significant = (
            dataset["significant_fdr"]
            .transpose("lat", "lon")
            .astype(bool)
            .load()
        )

    if correlation.shape != significant.shape:
        raise ValueError(
            f"Shape mismatch in {path.name}: "
            f"r={correlation.shape}, "
            f"significant={significant.shape}"
        )

    return {
        "r": correlation,
        "r_plot": correlation,
        "significant": significant,
        "stipple_stride": FINE_GRID_STIPPLE_STRIDE,
    }


def load_existing_panels():
    panels = {}

    for driver_name in [
        "SST",
        "Wind Speed",
        "MLD",
        "Air Temperature",
    ]:
        panels[driver_name] = {}

        for season_name in SEASON_ORDER:
            panels[driver_name][season_name] = (
                load_saved_panel(
                    driver_name,
                    season_name,
                )
            )

    return panels



# 6. FIND AND LOAD GRIDDED SIC

def discover_sic_files():
    """
    Use explicitly supplied SIC paths first.

    If none were supplied, inspect candidate NetCDF files and retain only
    datasets containing an actual SIC/concentration/fraction variable.
    Sea-ice drift and velocity products are explicitly rejected.
    """

    if SIC_NC_FILES:
        paths = unique_paths(SIC_NC_FILES)
        validate_files(paths, "SIC")
        return paths

    patterns = [
        "*SIC*.nc",
        "*sic*.nc",
        "*SeaIce*.nc",
        "*sea_ice*.nc",
        "*siconc*.nc",
        "*ice_fraction*.nc",
    ]

    discovered = []

    for pattern in patterns:
        discovered.extend(DATA_DIR.rglob(pattern))

    candidates = unique_paths(
        path for path in discovered
        if "correlation_significance" not in path.name.lower()
        and "drift" not in str(path).lower()
        and "velocity" not in path.name.lower()
    )

    sic_variable_names = {
        "sif",
        "sic",
        "siconc",
        "sea_ice_fraction",
        "sea_ice_concentration",
        "ice_fraction",
        "ice_concentration",
        "ice_conc",
    }

    valid_files = []

    for path in candidates:
        try:
            with xr.open_dataset(
                path,
                decode_timedelta=False,
                chunks=None,
            ) as dataset:
                available = {
                    str(name).lower()
                    for name in dataset.data_vars
                }

                if available.intersection(sic_variable_names):
                    valid_files.append(path)

        except Exception as error:
            warnings.warn(
                f"Skipping unreadable SIC candidate {path.name}: {error}"
            )

    if not valid_files:
        raise FileNotFoundError(
            "No gridded monthly SIC NetCDF was found automatically.\n"
            "Add its exact path or paths to SIC_NC_FILES near the "
            "beginning of this script.\n\n"
            "The sea-wise SIC CSV cannot be used for spatial maps."
        )

    print("\nValidated SIC NetCDF file(s):")

    for path in valid_files:
        print("  ", path)

    return valid_files


def load_sic_piece(path):
    """Open and standardize one gridded SIC file."""

    dataset = xr.open_dataset(
        path,
        decode_timedelta=False,
        chunks="auto",
    )

    data_array = choose_data_variable(
        dataset,
        [
            "sif",
            "sic",
            "siconc",
            "sea_ice_fraction",
            "sea_ice_concentration",
            "ice_fraction",
            "ice_concentration",
            "ice_conc",
        ],
        "SIC",
    )

    data_array = standardize_dimensions(data_array)
    data_array = remove_extra_dimensions(data_array)
    data_array = normalize_longitude(data_array)
    data_array = standardize_monthly_time(data_array)

    data_array = data_array.sel(
        lat=slice(-90, -50)
        if data_array["lat"][0] < data_array["lat"][-1]
        else slice(-50, -90)
    )

    return data_array, dataset


def load_sic_on_template(template):
    """Load monthly SIC and interpolate it to the saved-result grid."""

    sic_files = discover_sic_files()

    opened_datasets = []
    pieces = []

    try:
        for path in sic_files:
            print(f"Opening SIC file: {path.name}")

            piece, dataset = load_sic_piece(path)

            pieces.append(piece)
            opened_datasets.append(dataset)

        sic = xr.concat(
            pieces,
            dim="time",
            join="outer",
            combine_attrs="override",
        )

        sic = standardize_monthly_time(sic)

        target_lat = template["lat"].values
        target_lon = template["lon"].values

        sic = interpolate_to_grid(
            sic,
            target_lat,
            target_lon,
        )

        sic = sic.transpose(
            "time",
            "lat",
            "lon",
        )

        print("Loading SIC grid into memory...")
        sic = sic.astype("float32").load()

    finally:
        for dataset in opened_datasets:
            dataset.close()

    if sic.sizes.get("time", 0) == 0:
        raise ValueError(
            "No SIC timestamps remained after restricting to 2008–2025."
        )

    print(
        "SIC ready:",
        dict(sic.sizes),
        str(sic["time"].min().values),
        "to",
        str(sic["time"].max().values),
    )

    return sic


# 7. AUTOCORRELATION-ADJUSTED CORRELATION AND FDR

def masked_vectorized_correlation(
    first_array,
    second_array,
    valid_mask,
    minimum_count,
):
    """
    Calculate independent column-wise correlations.

    Input shape:
        time × cells
    """

    count = valid_mask.sum(axis=0).astype(float)

    first_valid = np.where(
        valid_mask,
        first_array,
        0.0,
    )

    second_valid = np.where(
        valid_mask,
        second_array,
        0.0,
    )

    sum_first = first_valid.sum(axis=0)
    sum_second = second_valid.sum(axis=0)

    sum_first_squared = (
        first_valid * first_valid
    ).sum(axis=0)

    sum_second_squared = (
        second_valid * second_valid
    ).sum(axis=0)

    sum_products = (
        first_valid * second_valid
    ).sum(axis=0)

    numerator = (
        sum_products
        - (sum_first * sum_second / np.maximum(count, 1.0))
    )

    first_variance_term = (
        sum_first_squared
        - (sum_first * sum_first / np.maximum(count, 1.0))
    )

    second_variance_term = (
        sum_second_squared
        - (sum_second * sum_second / np.maximum(count, 1.0))
    )

    denominator = np.sqrt(
        first_variance_term
        * second_variance_term
    )

    correlation = np.full(
        count.shape,
        np.nan,
        dtype=np.float64,
    )

    usable = (
        (count >= minimum_count)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    correlation[usable] = (
        numerator[usable]
        / denominator[usable]
    )

    correlation = np.clip(
        correlation,
        -1.0,
        1.0,
    )

    return correlation, count


def benjamini_hochberg_mask(p_values, q=0.05):
    """Panel-wise Benjamini–Hochberg FDR significance mask."""

    flat_p = np.asarray(
        p_values,
        dtype=float,
    ).ravel()

    finite_positions = np.where(
        np.isfinite(flat_p)
    )[0]

    rejected = np.zeros(
        flat_p.shape,
        dtype=bool,
    )

    if finite_positions.size == 0:
        return rejected.reshape(
            np.asarray(p_values).shape
        )

    finite_p = flat_p[finite_positions]

    sorting_index = np.argsort(finite_p)
    sorted_p = finite_p[sorting_index]

    number_of_tests = sorted_p.size

    thresholds = (
        q
        * np.arange(1, number_of_tests + 1)
        / number_of_tests
    )

    passed = sorted_p <= thresholds

    if np.any(passed):
        largest_accepted_p = sorted_p[
            np.where(passed)[0].max()
        ]

        rejected[finite_positions] = (
            finite_p <= largest_accepted_p
        )

    return rejected.reshape(
        np.asarray(p_values).shape
    )


def calculate_adjusted_correlation(
    sic_anomaly,
    driver_anomaly,
    season_name,
):
    """
    Calculate Pearson r, lag-1 adjusted effective N, p, and FDR.

    Both DataArrays must have dimensions:
        time, lat, lon
    """

    sic_aligned, driver_aligned = xr.align(
        sic_anomaly,
        driver_anomaly,
        join="inner",
    )

    season_months = SEASON_MONTHS[season_name]

    selected_time = sic_aligned["time.month"].isin(
        season_months
    )

    sic_season = sic_aligned.sel(
        time=selected_time
    )

    driver_season = driver_aligned.sel(
        time=selected_time
    )

    sic_values = np.asarray(
        sic_season
        .transpose("time", "lat", "lon")
        .values,
        dtype=np.float64,
    )

    driver_values = np.asarray(
        driver_season
        .transpose("time", "lat", "lon")
        .values,
        dtype=np.float64,
    )

    spatial_shape = sic_values.shape[1:]

    sic_flat = sic_values.reshape(
        sic_values.shape[0],
        -1,
    )

    driver_flat = driver_values.reshape(
        driver_values.shape[0],
        -1,
    )

    paired_valid = (
        np.isfinite(sic_flat)
        & np.isfinite(driver_flat)
    )

    correlation, paired_count = (
        masked_vectorized_correlation(
            sic_flat,
            driver_flat,
            paired_valid,
            MIN_PAIRED_MONTHS,
        )
    )

    # Lag-1 autocorrelation is calculated only where both members
    # of the paired sequence are available.
    if sic_flat.shape[0] >= 3:

        lag_valid = (
            paired_valid[:-1, :]
            & paired_valid[1:, :]
        )

        sic_lag1, sic_lag_count = (
            masked_vectorized_correlation(
                sic_flat[:-1, :],
                sic_flat[1:, :],
                lag_valid,
                3,
            )
        )

        driver_lag1, driver_lag_count = (
            masked_vectorized_correlation(
                driver_flat[:-1, :],
                driver_flat[1:, :],
                lag_valid,
                3,
            )
        )

    else:
        sic_lag1 = np.full_like(
            correlation,
            np.nan,
        )

        driver_lag1 = np.full_like(
            correlation,
            np.nan,
        )

    # A missing lag-1 estimate is treated as zero autocorrelation,
    # but only for cells that passed the paired-count requirement.
    sic_lag1_for_neff = np.where(
        np.isfinite(sic_lag1),
        sic_lag1,
        0.0,
    )

    driver_lag1_for_neff = np.where(
        np.isfinite(driver_lag1),
        driver_lag1,
        0.0,
    )

    lag_product = (
        sic_lag1_for_neff
        * driver_lag1_for_neff
    )

    denominator = 1.0 + lag_product

    denominator = np.where(
        np.abs(denominator) < 1e-8,
        np.nan,
        denominator,
    )

    effective_n = (
        paired_count
        * (1.0 - lag_product)
        / denominator
    )

    effective_n = np.minimum(
        effective_n,
        paired_count,
    )

    effective_n = np.maximum(
        effective_n,
        3.0,
    )

    effective_n[~np.isfinite(correlation)] = np.nan

    one_minus_r_squared = np.maximum(
        1.0 - correlation**2,
        np.finfo(float).eps,
    )

    t_statistic = (
        correlation
        * np.sqrt(
            (effective_n - 2.0)
            / one_minus_r_squared
        )
    )

    degrees_of_freedom = (
        effective_n - 2.0
    )

    p_adjusted = (
        2.0
        * student_t.sf(
            np.abs(t_statistic),
            degrees_of_freedom,
        )
    )

    p_adjusted[
        ~np.isfinite(correlation)
    ] = np.nan

    significant_fdr = benjamini_hochberg_mask(
        p_adjusted,
        q=FDR_Q,
    )

    coordinates = {
        "lat": sic_season["lat"].values,
        "lon": sic_season["lon"].values,
    }

    correlation_da = xr.DataArray(
        correlation.reshape(spatial_shape).astype("float32"),
        coords=coordinates,
        dims=("lat", "lon"),
        name="r",
    )

    paired_count_da = xr.DataArray(
        paired_count.reshape(spatial_shape).astype("float32"),
        coords=coordinates,
        dims=("lat", "lon"),
        name="paired_n",
    )

    effective_n_da = xr.DataArray(
        effective_n.reshape(spatial_shape).astype("float32"),
        coords=coordinates,
        dims=("lat", "lon"),
        name="n_eff",
    )

    p_adjusted_da = xr.DataArray(
        p_adjusted.reshape(spatial_shape).astype("float32"),
        coords=coordinates,
        dims=("lat", "lon"),
        name="p_adjusted_autocorrelation",
    )

    significant_da = xr.DataArray(
        significant_fdr.reshape(spatial_shape),
        coords=coordinates,
        dims=("lat", "lon"),
        name="significant_fdr",
    )

    return {
        "r": correlation_da,
        "paired_n": paired_count_da,
        "n_eff": effective_n_da,
        "p_adjusted": p_adjusted_da,
        "significant": significant_da,
    }


def summarize_new_panel(
    driver_name,
    season_name,
    result,
    grid_note,
):
    """Create one combined-table row."""

    tested = np.isfinite(
        result["p_adjusted"].values
    )

    significant = (
        result["significant"].values
        & tested
    )

    cells_tested = int(tested.sum())
    significant_cells = int(significant.sum())

    if cells_tested > 0:
        significant_percent = (
            100.0
            * significant_cells
            / cells_tested
        )

        median_n_eff = float(
            np.nanmedian(
                result["n_eff"].values[tested]
            )
        )

    else:
        significant_percent = np.nan
        median_n_eff = np.nan

    return {
        "Driver": driver_name,
        "Season": season_name,
        "Cells_tested": cells_tested,
        "Median_N_eff": median_n_eff,
        "FDR_significant_cells": significant_cells,
        "FDR_significant_percent": significant_percent,
        "FDR_q": FDR_Q,
        "Minimum_paired_months": MIN_PAIRED_MONTHS,
        "Grid_note": grid_note,
    }


def save_new_panel(
    driver_name,
    season_name,
    result,
):
    """Save new statistical results before any display interpolation."""

    driver_token = (
        driver_name
        .replace(" ", "_")
        .replace("–", "_")
    )

    output_path = (
        NEW_PANEL_DIR
        / (
            f"{driver_token}_{season_name}"
            "_correlation_significance.nc"
        )
    )

    output_dataset = xr.Dataset(
        {
            "r": result["r"],
            "paired_n": result["paired_n"],
            "n_eff": result["n_eff"],
            "p_adjusted_autocorrelation":
                result["p_adjusted"],
            "significant_fdr":
                result["significant"],
        }
    )

    output_dataset.attrs.update(
        {
            "analysis_period":
                f"{START_YEAR}-{END_YEAR}",
            "season":
                season_name,
            "season_months":
                ",".join(
                    map(
                        str,
                        SEASON_MONTHS[season_name],
                    )
                ),
            "fdr_method":
                "Panel-wise Benjamini-Hochberg",
            "fdr_q":
                FDR_Q,
            "minimum_paired_months":
                MIN_PAIRED_MONTHS,
            "note":
                (
                    "Saved r and significance fields are "
                    "uninterpolated statistical results."
                ),
        }
    )

    encoding = {
        variable_name: {
            "zlib": True,
            "complevel": 4,
        }
        for variable_name in output_dataset.data_vars
    }

    output_dataset.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=encoding,
    )

    print("Saved new panel:", output_path.name)



# 8. LOAD AND PROCESS OCEAN CURRENT

def select_current_depth(data_array):
    """Select the shallowest current level."""

    depth_candidates = [
        dimension
        for dimension in data_array.dims
        if dimension.lower() in {
            "depth",
            "depthu",
            "depthv",
            "lev",
            "level",
            "z",
        }
    ]

    if not depth_candidates:
        print(
            "No current-depth dimension found; treating the data "
            "as surface current."
        )

        return data_array

    depth_dimension = depth_candidates[0]

    if CURRENT_DEPTH_MODE != "surface":
        raise ValueError(
            "Only CURRENT_DEPTH_MODE='surface' is implemented "
            "in this script."
        )

    depth_values = np.asarray(
        data_array[depth_dimension].values,
        dtype=float,
    )

    shallowest_index = int(
        np.nanargmin(np.abs(depth_values))
    )

    selected_depth = float(
        depth_values[shallowest_index]
    )

    print(
        f"Selecting shallowest current level: "
        f"{selected_depth:g}"
    )

    return data_array.isel(
        {depth_dimension: shallowest_index},
        drop=True,
    )


def load_current_component_piece(
    path,
    variable_candidates,
    description,
):
    """Load one u or v current file lazily."""

    dataset = xr.open_dataset(
        path,
        decode_timedelta=False,
        chunks="auto",
    )

    data_array = choose_data_variable(
        dataset,
        variable_candidates,
        description,
    )

    data_array = standardize_dimensions(data_array)
    data_array = select_current_depth(data_array)
    data_array = remove_extra_dimensions(data_array)
    data_array = normalize_longitude(data_array)
    data_array = standardize_monthly_time(data_array)

    data_array = data_array.sel(
        lat=slice(-90, -50)
        if data_array["lat"][0] < data_array["lat"][-1]
        else slice(-50, -90)
    )

    return data_array, dataset


def load_current_component(
    paths,
    variable_candidates,
    description,
    template,
):
    """Combine a component across the two time-period files."""

    validate_files(paths, description)

    opened_datasets = []
    pieces = []

    try:
        for path in paths:
            print(
                f"Opening {description}: {path.name}"
            )

            piece, dataset = load_current_component_piece(
                path,
                variable_candidates,
                description,
            )

            pieces.append(piece)
            opened_datasets.append(dataset)

        combined = xr.concat(
            pieces,
            dim="time",
            join="outer",
            combine_attrs="override",
        )

        combined = standardize_monthly_time(combined)

        combined = interpolate_to_grid(
            combined,
            template["lat"].values,
            template["lon"].values,
        )

        combined = combined.transpose(
            "time",
            "lat",
            "lon",
        )

        combined = combined.astype("float32").load()

    finally:
        for dataset in opened_datasets:
            dataset.close()

    return combined


def calculate_current_speed(template):
    """Calculate surface vector-current speed from u and v."""

    u_current = load_current_component(
        U_CURRENT_FILES,
        [
            "uo",
            "eastward_sea_water_velocity",
            "eastward_velocity",
            "u",
        ],
        "eastward-current",
        template,
    )

    v_current = load_current_component(
        V_CURRENT_FILES,
        [
            "vo",
            "northward_sea_water_velocity",
            "northward_velocity",
            "v",
        ],
        "northward-current",
        template,
    )

    u_current, v_current = xr.align(
        u_current,
        v_current,
        join="inner",
    )

    if u_current.sizes.get("time", 0) == 0:
        raise ValueError(
            "No matching monthly timestamps were found between "
            "the eastward and northward current files."
        )

    speed = np.hypot(
        u_current,
        v_current,
    )

    speed.name = "ocean_current_speed"

    speed.attrs["long_name"] = (
        "Surface ocean-current speed"
    )

    if (
        u_current.attrs.get("units")
        == v_current.attrs.get("units")
    ):
        speed.attrs["units"] = (
            u_current.attrs.get("units", "")
        )

    print(
        "Ocean-current speed ready:",
        dict(speed.sizes),
    )

    return speed.astype("float32")


# 9. LOAD AND GRID OHC PROFILES


def find_ohc_profile_csv():
    """Use the configured path or locate the profile file automatically."""

    if OHC_PROFILE_CSV.exists():
        return OHC_PROFILE_CSV

    candidates = list(
        DATA_DIR.rglob(
            "Argo_profile_OHC_0_100m.csv"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Argo_profile_OHC_0_100m.csv was not found.\n"
            "Extract it from your OHC results ZIP and place it in:\n"
            f"{DATA_DIR}"
        )

    raise ValueError(
        "Multiple OHC profile CSV files were found. "
        "Set OHC_PROFILE_CSV to the exact intended file:\n"
        + "\n".join(str(path) for path in candidates)
    )


def normalize_longitude_values(longitude):
    """Normalize NumPy/Pandas longitude values."""

    return (
        (np.asarray(longitude, dtype=float) + 180.0)
        % 360.0
    ) - 180.0


def spherical_xyz(longitude, latitude):
    """Convert longitude/latitude to Cartesian coordinates on a unit sphere."""

    longitude_radians = np.deg2rad(np.asarray(longitude, dtype=float))
    latitude_radians = np.deg2rad(np.asarray(latitude, dtype=float))

    return np.column_stack(
        [
            np.cos(latitude_radians) * np.cos(longitude_radians),
            np.cos(latitude_radians) * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ]
    )


def great_circle_distance_matrix_km(first_xyz, second_xyz):
    """Great-circle distances between two collections of unit vectors."""

    cosine_angle = np.clip(
        np.asarray(first_xyz) @ np.asarray(second_xyz).T,
        -1.0,
        1.0,
    )

    return 6371.0 * np.arccos(cosine_angle)


def read_and_bin_ohc_profile_anomalies():
    """
    Read Argo OHC profiles and create source-bin monthly anomalies.

    Multiple profiles in the same month and spatial bin are averaged first.
    A source-bin calendar-month climatology must contain at least
    OHC_CLIMATOLOGY_MIN_YEARS observations before its anomalies are accepted.
    """

    profile_path = find_ohc_profile_csv()
    print("Reading OHC profiles:", profile_path)

    profiles = pd.read_csv(profile_path)

    required = {"lat", "lon", "OHC_0_100m_GJ_m2"}
    missing = required.difference(profiles.columns)

    if missing:
        raise ValueError(
            f"OHC profile CSV is missing columns: {sorted(missing)}"
        )

    if "month" in profiles.columns:
        profiles["month"] = (
            pd.to_datetime(profiles["month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    elif "time" in profiles.columns:
        profiles["month"] = (
            pd.to_datetime(profiles["time"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    else:
        raise ValueError(
            "OHC profile CSV requires a 'month' or 'time' column."
        )

    profiles["lat"] = pd.to_numeric(profiles["lat"], errors="coerce")
    profiles["lon"] = pd.to_numeric(profiles["lon"], errors="coerce")
    profiles["ohc"] = pd.to_numeric(
        profiles["OHC_0_100m_GJ_m2"],
        errors="coerce",
    )

    profiles = profiles.dropna(
        subset=["month", "lat", "lon", "ohc"]
    ).copy()

    profiles = profiles.loc[
        (profiles["month"] >= START_DATE)
        & (profiles["month"] <= END_DATE)
        & (profiles["lat"] >= -90.0)
        & (profiles["lat"] <= -60.0)
    ].copy()

    if profiles.empty:
        raise ValueError("No usable OHC profiles remained after filtering.")

    profiles["lon"] = normalize_longitude_values(profiles["lon"])

    profiles["lat_bin"] = (
        -90.0
        + (
            np.floor(
                (profiles["lat"] + 90.0)
                / OHC_SOURCE_LAT_STEP_DEGREES
            )
            + 0.5
        )
        * OHC_SOURCE_LAT_STEP_DEGREES
    )

    profiles["lon_bin"] = (
        -180.0
        + (
            np.floor(
                (profiles["lon"] + 180.0)
                / OHC_SOURCE_LON_STEP_DEGREES
            )
            + 0.5
        )
        * OHC_SOURCE_LON_STEP_DEGREES
    )

    grouped = (
        profiles.groupby(
            ["month", "lat_bin", "lon_bin"],
            as_index=False,
        )
        .agg(
            ohc=("ohc", "mean"),
            profile_count=("ohc", "size"),
        )
    )

    complete_time = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-01",
        freq="MS",
    )

    complete_lat = np.arange(
        -90.0 + OHC_SOURCE_LAT_STEP_DEGREES / 2.0,
        -60.0,
        OHC_SOURCE_LAT_STEP_DEGREES,
    )

    complete_lon = np.arange(
        -180.0 + OHC_SOURCE_LON_STEP_DEGREES / 2.0,
        180.0,
        OHC_SOURCE_LON_STEP_DEGREES,
    )

    raw_grid = (
        grouped.set_index(["month", "lat_bin", "lon_bin"])["ohc"]
        .to_xarray()
        .rename(
            {
                "month": "time",
                "lat_bin": "lat",
                "lon_bin": "lon",
            }
        )
        .reindex(
            time=complete_time,
            lat=complete_lat,
            lon=complete_lon,
        )
        .transpose("time", "lat", "lon")
        .astype("float32")
    )

    climatology = raw_grid.groupby("time.month").mean(
        "time",
        skipna=True,
    )

    climatology_count = raw_grid.groupby("time.month").count("time")

    anomaly_pieces = []

    for timestamp in raw_grid["time"].values:
        month_number = int(pd.Timestamp(timestamp).month)

        anomaly_field = (
            raw_grid.sel(time=timestamp)
            - climatology.sel(month=month_number)
        )

        anomaly_field = anomaly_field.where(
            climatology_count.sel(month=month_number)
            >= OHC_CLIMATOLOGY_MIN_YEARS
        )

        anomaly_pieces.append(
            anomaly_field.expand_dims(time=[timestamp])
        )

    source_anomaly = xr.concat(anomaly_pieces, dim="time").transpose(
        "time",
        "lat",
        "lon",
    )

    source_anomaly.name = "source_OHC_anomaly"
    source_anomaly.attrs.update(
        {
            "long_name": "Source-bin monthly Argo OHC anomaly, 0–100 m",
            "units": "GJ m-2",
            "climatology_period": f"{START_YEAR}-{END_YEAR}",
            "minimum_climatology_years": OHC_CLIMATOLOGY_MIN_YEARS,
        }
    )

    print(
        "Binned OHC anomaly source grid ready:",
        dict(source_anomaly.sizes),
    )

    return source_anomaly.astype("float32")


def local_objective_analysis_one_month(
    source_field,
    target_lon_2d,
    target_lat_2d,
    target_xyz,
):
    """
    Map one monthly OHC-anomaly field using local optimal interpolation.

    The normalized covariance model is
        C(d) = exp[-(d/L)^2]

    and the weights solve
        (Coo + epsilon I) w = Cgo,

    where epsilon is the observation-error/signal-variance ratio.
    The normalized mapping error is
        E = 1 - Cgo.T w.
    """

    source_lon_2d, source_lat_2d = np.meshgrid(
        source_field["lon"].values,
        source_field["lat"].values,
    )

    source_values_2d = np.asarray(source_field.values, dtype=float)
    valid = np.isfinite(source_values_2d)

    target_shape = target_lon_2d.shape
    mapped = np.full(target_shape, np.nan, dtype=np.float32)
    mapping_error = np.full(target_shape, np.nan, dtype=np.float32)
    local_count = np.zeros(target_shape, dtype=np.int16)

    if int(valid.sum()) < OHC_OI_MIN_OBSERVATIONS:
        return mapped, mapping_error, local_count

    observation_lon = source_lon_2d[valid]
    observation_lat = source_lat_2d[valid]
    observation_values = source_values_2d[valid]
    observation_xyz = spherical_xyz(observation_lon, observation_lat)

    observation_tree = cKDTree(observation_xyz)

    chord_radius = 2.0 * np.sin(
        OHC_OI_SEARCH_RADIUS_KM / (2.0 * 6371.0)
    )

    number_of_neighbors = min(
        OHC_OI_MAX_NEIGHBORS,
        observation_values.size,
    )

    chord_distance, neighbor_index = observation_tree.query(
        target_xyz,
        k=number_of_neighbors,
        distance_upper_bound=chord_radius,
        workers=-1,
    )

    if number_of_neighbors == 1:
        chord_distance = chord_distance[:, None]
        neighbor_index = neighbor_index[:, None]

    mapped_flat = mapped.ravel()
    error_flat = mapping_error.ravel()
    count_flat = local_count.ravel()

    # Nearby target cells commonly use the same observation neighborhood.
    # Cache the inverse covariance matrix for each unique neighborhood so the
    # finer 1° × 2° target grid does not repeatedly factorize the same matrix.
    covariance_inverse_cache = {}

    for target_index in range(target_xyz.shape[0]):
        indices = neighbor_index[target_index]

        usable = (
            np.isfinite(chord_distance[target_index])
            & (indices < observation_values.size)
        )

        indices = np.sort(indices[usable])

        if indices.size < OHC_OI_MIN_OBSERVATIONS:
            continue

        neighborhood_key = tuple(int(index) for index in indices)

        if neighborhood_key in covariance_inverse_cache:
            (
                local_xyz,
                local_values,
                system_inverse,
            ) = covariance_inverse_cache[neighborhood_key]

        else:
            local_xyz = observation_xyz[indices]
            local_values = observation_values[indices]

            observation_distance = great_circle_distance_matrix_km(
                local_xyz,
                local_xyz,
            )

            observation_covariance = np.exp(
                -(
                    observation_distance
                    / OHC_OI_LENGTH_SCALE_KM
                )
                ** 2
            )

            system_matrix = observation_covariance.copy()
            system_matrix.flat[:: system_matrix.shape[0] + 1] += (
                OHC_OI_NOISE_TO_SIGNAL_VARIANCE
                + OHC_OI_NUMERICAL_NUGGET
            )

            try:
                system_inverse = np.linalg.inv(system_matrix)
            except np.linalg.LinAlgError:
                system_inverse = np.linalg.pinv(
                    system_matrix,
                    hermitian=True,
                )

            covariance_inverse_cache[neighborhood_key] = (
                local_xyz,
                local_values,
                system_inverse,
            )

        grid_distance = great_circle_distance_matrix_km(
            target_xyz[target_index : target_index + 1],
            local_xyz,
        ).ravel()

        grid_observation_covariance = np.exp(
            -(
                grid_distance
                / OHC_OI_LENGTH_SCALE_KM
            )
            ** 2
        )

        weights = system_inverse @ grid_observation_covariance

        normalized_error = float(
            1.0
            - np.dot(
                grid_observation_covariance,
                weights,
            )
        )

        normalized_error = float(
            np.clip(normalized_error, 0.0, 1.0)
        )

        if normalized_error > OHC_OI_MAX_NORMALIZED_ERROR:
            continue

        mapped_flat[target_index] = float(
            np.dot(weights, local_values)
        )

        error_flat[target_index] = normalized_error
        count_flat[target_index] = int(indices.size)

    return mapped, mapping_error, local_count


def build_monthly_ohc_objective_analysis():
    """Map all monthly OHC anomalies and save OI diagnostics."""

    source_anomaly = read_and_bin_ohc_profile_anomalies()

    target_lat = np.arange(
        -90.0 + OHC_OI_LAT_STEP_DEGREES / 2.0,
        -60.0,
        OHC_OI_LAT_STEP_DEGREES,
        dtype=float,
    )

    target_lon = np.arange(
        -180.0 + OHC_OI_LON_STEP_DEGREES / 2.0,
        180.0,
        OHC_OI_LON_STEP_DEGREES,
        dtype=float,
    )

    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    target_xyz = spherical_xyz(
        target_lon_2d.ravel(),
        target_lat_2d.ravel(),
    )

    number_of_times = source_anomaly.sizes["time"]
    output_shape = (
        number_of_times,
        target_lat.size,
        target_lon.size,
    )

    mapped_values = np.full(output_shape, np.nan, dtype=np.float32)
    mapping_errors = np.full(output_shape, np.nan, dtype=np.float32)
    local_counts = np.zeros(output_shape, dtype=np.int16)

    for time_index, timestamp in enumerate(source_anomaly["time"].values):
        if time_index % 12 == 0:
            print(
                f"Local OI progress: {time_index + 1}/{number_of_times} "
                f"({pd.Timestamp(timestamp):%Y-%m})"
            )

        mapped, error, count = local_objective_analysis_one_month(
            source_anomaly.isel(time=time_index),
            target_lon_2d,
            target_lat_2d,
            target_xyz,
        )

        mapped_values[time_index] = mapped
        mapping_errors[time_index] = error
        local_counts[time_index] = count

    coordinates = {
        "time": source_anomaly["time"].values,
        "lat": target_lat,
        "lon": target_lon,
    }

    mapped_anomaly = xr.DataArray(
        mapped_values,
        coords=coordinates,
        dims=("time", "lat", "lon"),
        name="OHC_anomaly_OI",
        attrs={
            "long_name": "Local-OI monthly Argo OHC anomaly, 0–100 m",
            "units": "GJ m-2",
        },
    )

    # Remove any small residual calendar-month mean caused by changing
    # observation coverage between years, while retaining the OI support mask.
    original_support = np.isfinite(mapped_anomaly)
    mapped_anomaly = monthly_anomaly(mapped_anomaly).where(original_support)
    mapped_anomaly.name = "OHC_anomaly_OI"

    mapping_error = xr.DataArray(
        mapping_errors,
        coords=coordinates,
        dims=("time", "lat", "lon"),
        name="normalized_mapping_error",
    )

    local_observation_count = xr.DataArray(
        local_counts,
        coords=coordinates,
        dims=("time", "lat", "lon"),
        name="local_observation_count",
    )

    oi_dataset = xr.Dataset(
        {
            "OHC_anomaly_OI": mapped_anomaly.astype("float32"),
            "normalized_mapping_error": mapping_error.astype("float32"),
            "local_observation_count": local_observation_count,
        }
    )

    oi_dataset.attrs.update(
        {
            "method": "Local objective analysis / optimal interpolation",
            "covariance_model": "C(d) = exp[-(d/L)^2]",
            "length_scale_km": OHC_OI_LENGTH_SCALE_KM,
            "search_radius_km": OHC_OI_SEARCH_RADIUS_KM,
            "noise_to_signal_variance": OHC_OI_NOISE_TO_SIGNAL_VARIANCE,
            "minimum_local_observations": OHC_OI_MIN_OBSERVATIONS,
            "maximum_local_neighbors": OHC_OI_MAX_NEIGHBORS,
            "maximum_normalized_mapping_error": OHC_OI_MAX_NORMALIZED_ERROR,
            "minimum_climatology_years": OHC_CLIMATOLOGY_MIN_YEARS,
            "source_latitude_bin_degrees":
                OHC_SOURCE_LAT_STEP_DEGREES,
            "source_longitude_bin_degrees":
                OHC_SOURCE_LON_STEP_DEGREES,
            "OI_target_latitude_step_degrees":
                OHC_OI_LAT_STEP_DEGREES,
            "OI_target_longitude_step_degrees":
                OHC_OI_LON_STEP_DEGREES,
            "analysis_period": f"{START_YEAR}-{END_YEAR}",
        }
    )

    encoding = {
        variable_name: {"zlib": True, "complevel": 4}
        for variable_name in oi_dataset.data_vars
    }

    oi_dataset.to_netcdf(
        OHC_OI_MONTHLY_NC,
        engine="netcdf4",
        encoding=encoding,
    )

    valid_fraction = float(
        np.isfinite(mapped_anomaly.values).mean() * 100.0
    )

    print(f"Local OI valid space-time coverage: {valid_fraction:.2f}%")
    print("Saved monthly OI field and diagnostics:", OHC_OI_MONTHLY_NC)

    return mapped_anomaly.astype("float32")


# 11. CALCULATE THE TWO NEW VARIABLES

def calculate_new_panels(
    sic_monthly,
    current_speed,
    ohc_oi_anomaly,
):
    panels = {
        "OHC": {},
        "Ocean Current Speed": {},
    }

    summary_rows = []

    sic_anomaly_native = monthly_anomaly(
        sic_monthly
    )

    current_anomaly = monthly_anomaly(
        current_speed
    )

    # SIC on the objective-analysis OHC grid.
    sic_on_ohc_grid = interpolate_to_grid(
        sic_monthly,
        ohc_oi_anomaly["lat"].values,
        ohc_oi_anomaly["lon"].values,
    )

    sic_on_ohc_grid = sic_on_ohc_grid.transpose(
        "time",
        "lat",
        "lon",
    )

    sic_ohc_anomaly = monthly_anomaly(
        sic_on_ohc_grid
    )

    for season_name in SEASON_ORDER:


        # Ocean-current speed

        print(
            f"Calculating Ocean Current Speed | "
            f"{season_name}"
        )

        current_result = calculate_adjusted_correlation(
            sic_anomaly_native,
            current_anomaly,
            season_name,
        )

        save_new_panel(
            "Ocean Current Speed",
            season_name,
            current_result,
        )

        panels["Ocean Current Speed"][season_name] = {
            "r": current_result["r"],
            "r_plot": current_result["r"],
            "significant":
                current_result["significant"],
            "stipple_stride":
                FINE_GRID_STIPPLE_STRIDE,
        }

        summary_rows.append(
            summarize_new_panel(
                "Ocean Current Speed",
                season_name,
                current_result,
                (
                    "Surface current speed derived as "
                    "sqrt(uo^2 + vo^2) on the SIC grid."
                ),
            )
        )


        # OHC

        print(
            f"Calculating OHC (0–100 m) | "
            f"{season_name}"
        )

        ohc_result = calculate_adjusted_correlation(
            sic_ohc_anomaly,
            ohc_oi_anomaly,
            season_name,
        )

        save_new_panel(
            "OHC",
            season_name,
            ohc_result,
        )

        panels["OHC"][season_name] = {
            "r": ohc_result["r"],
            # This is the correlation calculated from monthly local-OI
            # OHC anomalies. No post-correlation interpolation is used.
            "r_plot": ohc_result["r"],
            "significant":
                ohc_result["significant"],
            # Never thin the already-coarse OI grid.
            "stipple_stride": 1,
        }

        summary_rows.append(
            summarize_new_panel(
                "OHC",
                season_name,
                ohc_result,
                (
                    "Monthly OHC anomalies mapped before correlation by "
                    "local objective analysis; "
                    f"source bins={OHC_SOURCE_LAT_STEP_DEGREES:g}°×"
                    f"{OHC_SOURCE_LON_STEP_DEGREES:g}°, "
                    f"OI target grid={OHC_OI_LAT_STEP_DEGREES:g}°×"
                    f"{OHC_OI_LON_STEP_DEGREES:g}°, "
                    f"L={OHC_OI_LENGTH_SCALE_KM:g} km, "
                    f"search radius={OHC_OI_SEARCH_RADIUS_KM:g} km, "
                    f"minimum observations={OHC_OI_MIN_OBSERVATIONS}, "
                    f"normalized mapping error <= "
                    f"{OHC_OI_MAX_NORMALIZED_ERROR:.2f}."
                ),
            )
        )

        gc.collect()

    return panels, pd.DataFrame(summary_rows)



# 12. COMBINED SIX-VARIABLE SUMMARY TABLE

def build_combined_summary(new_summary):
    existing_summary_path = (
        EXISTING_RESULT_DIR
        / "Fig4_FDR_field_significance_summary.csv"
    )

    if not existing_summary_path.exists():
        raise FileNotFoundError(
            "Existing FDR summary table not found:\n"
            f"{existing_summary_path}"
        )

    existing_summary = pd.read_csv(
        existing_summary_path
    )

    existing_drivers = [
        "SST",
        "Wind Speed",
        "MLD",
        "Air Temperature",
    ]

    existing_summary = existing_summary.loc[
        existing_summary["Driver"].isin(
            existing_drivers
        )
    ].copy()

    if "FDR_q" not in existing_summary.columns:
        existing_summary["FDR_q"] = FDR_Q

    if (
        "Minimum_paired_months"
        not in existing_summary.columns
    ):
        existing_summary[
            "Minimum_paired_months"
        ] = np.nan

    if "Grid_note" not in existing_summary.columns:
        existing_summary["Grid_note"] = (
            "Previously saved original Figure 4 result grid."
        )

    combined = pd.concat(
        [
            existing_summary,
            new_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    driver_order = [
        "SST",
        "Wind Speed",
        "MLD",
        "Air Temperature",
        "OHC",
        "Ocean Current Speed",
    ]

    combined["_driver_order"] = pd.Categorical(
        combined["Driver"],
        categories=driver_order,
        ordered=True,
    )

    combined["_season_order"] = pd.Categorical(
        combined["Season"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(
        [
            "_driver_order",
            "_season_order",
        ]
    ).drop(
        columns=[
            "_driver_order",
            "_season_order",
        ]
    )

    preferred_columns = [
        "Driver",
        "Season",
        "Cells_tested",
        "Median_N_eff",
        "FDR_significant_cells",
        "FDR_significant_percent",
        "FDR_q",
        "Minimum_paired_months",
        "Grid_note",
    ]

    remaining_columns = [
        column
        for column in combined.columns
        if column not in preferred_columns
    ]

    combined = combined[
        preferred_columns
        + remaining_columns
    ]

    combined.to_csv(
        COMBINED_TABLE_CSV,
        index=False,
        float_format="%.6f",
    )

    print("\nCombined six-variable FDR summary:")
    print(
        combined[
            [
                "Driver",
                "Season",
                "Cells_tested",
                "Median_N_eff",
                "FDR_significant_cells",
                "FDR_significant_percent",
            ]
        ].to_string(index=False)
    )

    print(
        "\nCombined table saved:\n",
        COMBINED_TABLE_CSV,
    )

    return combined



# 13. MAP HELPERS

def coordinate_extent(data_array):
    """Return raster pixel-edge extent and image origin."""

    longitude = np.asarray(
        data_array["lon"].values,
        dtype=float,
    )

    latitude = np.asarray(
        data_array["lat"].values,
        dtype=float,
    )

    if longitude.size < 2 or latitude.size < 2:
        raise ValueError(
            "Each map needs at least two latitude and longitude cells."
        )

    longitude_spacing = float(
        np.nanmedian(
            np.abs(
                np.diff(longitude)
            )
        )
    )

    latitude_spacing = float(
        np.nanmedian(
            np.abs(
                np.diff(latitude)
            )
        )
    )

    longitude_left = float(
        longitude[0]
        - longitude_spacing / 2.0
    )

    longitude_right = float(
        longitude[-1]
        + longitude_spacing / 2.0
    )

    first_latitude_edge = float(
        latitude[0]
        - latitude_spacing / 2.0
    )

    last_latitude_edge = float(
        latitude[-1]
        + latitude_spacing / 2.0
    )

    if latitude[0] < latitude[-1]:
        origin = "lower"

        latitude_bottom = first_latitude_edge
        latitude_top = last_latitude_edge

    else:
        origin = "upper"

        latitude_bottom = last_latitude_edge
        latitude_top = first_latitude_edge

    return [
        longitude_left,
        longitude_right,
        latitude_bottom,
        latitude_top,
    ], origin


def create_polar_axis(figure, grid_spec_position):
    """Create one circular South Polar Stereo axis."""

    axis = figure.add_subplot(
        grid_spec_position,
        projection=ccrs.SouthPolarStereo(),
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        360,
    )

    vertices = (
        np.vstack(
            [
                np.sin(theta),
                np.cos(theta),
            ]
        ).T
        * 0.5
        + [0.5, 0.5]
    )

    axis.set_boundary(
        mpath.Path(vertices),
        transform=axis.transAxes,
    )

    axis.set_extent(
        [-180, 180, -90, -60],
        crs=ccrs.PlateCarree(),
    )

    axis.set_facecolor("white")

    axis.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="0.82",
        edgecolor="black",
        linewidth=0.50,
        zorder=5,
    )

    axis.coastlines(
        resolution="110m",
        linewidth=0.70,
        zorder=6,
    )

    longitude_grid = list(
        range(-180, 180, 30)
    )

    latitude_grid = [
        -60,
        -70,
        -80,
    ]

    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        linestyle=":",
        color="black",
        alpha=0.65,
        zorder=7,
    )

    gridlines.xlocator = mticker.FixedLocator(
        longitude_grid
    )

    gridlines.ylocator = mticker.FixedLocator(
        latitude_grid
    )

    # Slightly outside the circular map.
    longitude_label_latitude = -57.4

    for longitude in longitude_grid:

        if longitude == -180:
            label = "180°"

        elif longitude < 0:
            label = f"{abs(longitude)}°W"

        elif longitude > 0:
            label = f"{longitude}°E"

        else:
            label = "0°"

        axis.text(
            longitude,
            longitude_label_latitude,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            clip_on=False,
            zorder=9,
        )

    for latitude in [-70, -80]:
        axis.text(
            0,
            latitude,
            f"{abs(latitude)}°S",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            zorder=9,
        )

    return axis


def add_correlation_raster(
    axis,
    correlation,
    colormap,
):
    """Draw correlation shading as one memory-efficient raster."""

    extent, origin = coordinate_extent(
        correlation
    )

    values = np.ma.masked_invalid(
        correlation.values
    )

    image = axis.imshow(
        values,
        origin=origin,
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=colormap,
        vmin=CORRELATION_MIN,
        vmax=CORRELATION_MAX,
        interpolation="bilinear",
        zorder=1,
    )

    return image


def add_fdr_stippling(
    axis,
    statistical_correlation,
    significant,
    stride,
):
    """
    Add significance only at statistically tested grid cells.

    For OHC, statistical_correlation is the original coarse field,
    not the interpolated display field.
    """

    significant_values = np.asarray(
        significant.values,
        dtype=bool,
    )

    valid_statistical_cells = np.isfinite(
        statistical_correlation.values
    )

    significant_values = (
        significant_values
        & valid_statistical_cells
    )

    display_mask = np.zeros_like(
        significant_values,
        dtype=bool,
    )

    display_mask[
        ::stride,
        ::stride,
    ] = significant_values[
        ::stride,
        ::stride,
    ]

    row_index, column_index = np.where(
        display_mask
    )

    if column_index.size == 0:
        return

    longitude = (
        statistical_correlation["lon"]
        .values[column_index]
    )

    latitude = (
        statistical_correlation["lat"]
        .values[row_index]
    )

    marker_size = (
        2.2 if stride == 1 else 1.4
    )

    axis.scatter(
        longitude,
        latitude,
        s=marker_size,
        c="black",
        marker=".",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )



# 14. CREATE EACH 3 × 4 FIGURE

def plot_three_by_four_figure(
    all_panels,
    row_order,
    png_path,
    pdf_path,
    figure_number,
):
    """
    Create one three-row by four-column figure with manually reserved
    title, row-label, colorbar, and footnote areas.
    """

    colormap = plt.get_cmap(
        "RdBu_r"
    ).copy()

    colormap.set_bad(
        color="white",
        alpha=0.0,
    )

    figure = plt.figure(
        figsize=FIGURE_SIZE_INCHES,
        facecolor="white",
    )

    grid_spec = figure.add_gridspec(
        nrows=3,
        ncols=4,
        left=0.105,
        right=0.835,
        bottom=0.125,
        top=0.885,
        wspace=0.34,
        hspace=0.34,
    )

    axes = np.empty(
        (3, 4),
        dtype=object,
    )

    panel_letters = list(
        "abcdefghijkl"
    )

    last_image = None
    panel_number = 0

    for row, driver_name in enumerate(row_order):

        for column, season_name in enumerate(SEASON_ORDER):

            axis = create_polar_axis(
                figure,
                grid_spec[row, column],
            )

            axes[row, column] = axis

            panel = (
                all_panels[driver_name][season_name]
            )

            last_image = add_correlation_raster(
                axis,
                panel["r_plot"],
                colormap,
            )

            add_fdr_stippling(
                axis,
                panel["r"],
                panel["significant"],
                panel["stipple_stride"],
            )

            axis.text(
                -0.13,
                1.105,
                panel_letters[panel_number],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                clip_on=False,
                zorder=10,
            )

            panel_number += 1

    # Column headings.
    for column, season_name in enumerate(SEASON_ORDER):

        position = axes[0, column].get_position()

        figure.text(
            0.5 * (position.x0 + position.x1),
            position.y1 + 0.044,
            (
                f"{season_name} "
                f"({SEASON_ABBR[season_name]})"
            ),
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    # Row headings.
    for row, driver_name in enumerate(row_order):

        first_position = (
            axes[row, 0].get_position()
        )

        last_position = (
            axes[row, -1].get_position()
        )

        y_center = 0.5 * (
            min(
                first_position.y0,
                last_position.y0,
            )
            + max(
                first_position.y1,
                last_position.y1,
            )
        )

        figure.text(
            first_position.x0 - 0.057,
            y_center,
            ROW_TITLES[driver_name],
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Shared colorbar in a completely separate reserved area.
    colorbar_axis = figure.add_axes(
        [
            0.875,
            0.205,
            0.018,
            0.575,
        ]
    )

    colorbar = figure.colorbar(
        last_image,
        cax=colorbar_axis,
        ticks=CORRELATION_TICKS,
        extend="both",
    )

    colorbar.set_label(
        "Pearson correlation coefficient (r)",
        fontsize=12.5,
        fontweight="bold",
        labelpad=10,
    )

    colorbar.ax.tick_params(
        labelsize=9.2,
        width=1.0,
        pad=4,
    )

    # Footnotes have their own reserved bottom area.
    if figure_number == 4:

        figure.text(
            0.5,
            0.030,
            (
                "Black stippling: autocorrelation-adjusted p values "
                "significant after panel-wise Benjamini–Hochberg "
                "FDR (q < 0.05)."
            ),
            ha="center",
            va="bottom",
            fontsize=11.2,
            fontweight="bold",
        )

    else:

        figure.text(
            0.5,
            0.039,
            (
                "Black stippling: autocorrelation-adjusted p values "
                "significant after panel-wise Benjamini–Hochberg "
                "FDR (q < 0.05)."
            ),
            ha="center",
            va="bottom",
            fontsize=10.9,
            fontweight="bold",
        )

        figure.text(
            0.5,
            0.016,
            (
                "Monthly OHC anomalies were mapped before correlation "
                "using local objective analysis; cells failing local-support "
                "or mapping-error criteria were excluded."
            ),
            ha="center",
            va="bottom",
            fontsize=10.3,
        )

    print(
        f"\nSaving {SAVE_DPI}-DPI PNG:\n{png_path}"
    )

    figure.savefig(
        png_path,
        dpi=SAVE_DPI,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    print(f"Saving PDF:\n{pdf_path}")

    figure.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    if SHOW_IN_NOTEBOOK:
        plt.show()

    plt.close(figure)

    print(
        f"Figure {figure_number} completed successfully."
    )



# 15. MAIN

def main():

    print("=" * 72)
    print("Loading the 16 existing correlation panels")
    print("=" * 72)

    existing_panels = load_existing_panels()

    # Use the existing SST/Spring result grid as the common
    # high-resolution display and SIC target grid.
    display_template = (
        existing_panels["SST"]["Spring"]["r"]
    )

    print("\n" + "=" * 72)
    print("Loading gridded monthly SIC")
    print("=" * 72)

    sic_monthly = load_sic_on_template(
        display_template
    )

    print("\nCalculating SIC monthly anomalies...")
    # Anomalies are recalculated inside calculate_new_panels,
    # but this message marks the beginning of the new analysis.

    print("\n" + "=" * 72)
    print("Loading and deriving surface ocean-current speed")
    print("=" * 72)

    current_speed = calculate_current_speed(
        display_template
    )

    # Ensure current and SIC share exactly matching dates.
    sic_monthly, current_speed = xr.align(
        sic_monthly,
        current_speed,
        join="inner",
    )

    if sic_monthly.sizes["time"] < MIN_PAIRED_MONTHS:
        raise ValueError(
            "Too few common SIC/current months remained after alignment."
        )

    print("\n" + "=" * 72)
    print("Local objective analysis of OHC (0–100 m)")
    print("=" * 72)

    ohc_oi_anomaly = build_monthly_ohc_objective_analysis()

    print("\n" + "=" * 72)
    print("Calculating new correlations, adjusted p values, and FDR")
    print("=" * 72)

    new_panels, new_summary = calculate_new_panels(
        sic_monthly,
        current_speed,
        ohc_oi_anomaly,
    )

    all_panels = {
        **existing_panels,
        **new_panels,
    }

    print("\n" + "=" * 72)
    print("Creating one combined six-variable table")
    print("=" * 72)

    build_combined_summary(
        new_summary
    )

    del current_speed
    del ohc_oi_anomaly
    gc.collect()

    print("\n" + "=" * 72)
    print("Creating corrected Figure 4")
    print("=" * 72)

    plot_three_by_four_figure(
        all_panels=all_panels,
        row_order=FIG4_ROWS,
        png_path=FIG4_PNG,
        pdf_path=FIG4_PDF,
        figure_number=4,
    )

    print("\n" + "=" * 72)
    print("Creating corrected Figure 5")
    print("=" * 72)

    plot_three_by_four_figure(
        all_panels=all_panels,
        row_order=FIG5_ROWS,
        png_path=FIG5_PNG,
        pdf_path=FIG5_PDF,
        figure_number=5,
    )

    print("\n" + "=" * 72)
    print("ALL OUTPUTS COMPLETED")
    print("=" * 72)

    print("\nFigure 4 PNG:")
    print(FIG4_PNG)

    print("\nFigure 4 PDF:")
    print(FIG4_PDF)

    print("\nFigure 5 PNG:")
    print(FIG5_PNG)

    print("\nFigure 5 PDF:")
    print(FIG5_PDF)

    print("\nCombined table:")
    print(COMBINED_TABLE_CSV)

    print("\nNew statistical NetCDF files:")
    print(NEW_PANEL_DIR)


if __name__ == "__main__":
    main()
