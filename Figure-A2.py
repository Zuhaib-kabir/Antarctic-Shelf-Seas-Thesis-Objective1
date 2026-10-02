#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

%pip install -q xarray "dask[array]" netCDF4 h5netcdf cartopy scipy



"""
Seasonal Antarctic surface-energy climatology, 2008–2025.

This script recreates the 4-row × 3-column figure containing:

    Column 1: 2 m air temperature
    Column 2: turbulent surface heat flux (latent + sensible; positive downward)
    Column 3: downward surface radiation (shortwave + longwave)

Southern Hemisphere seasons are used:

    Spring = SON, Summer = DJF, Autumn = MAM, Winter = JJA

Important methods
-----------------
1. Only complete three-month seasons are retained.
2. December is assigned to the following DJF season-year.
3. Monthly means are weighted by the number of days in each month when each
   seasonal mean is formed.
4. ERA5 accumulated energy variables in J m-2 are converted to W m-2 by
   dividing by 86,400 s. This is correct for ERA5 *monthly averaged*
   accumulated fields, which represent a mean daily accumulation.
5. ERA5 turbulent heat-flux signs are already positive downward. No sign
   reversal is applied unless TURBULENT_SOURCE_POSITIVE is changed to
   "upward" below.
6. Each subplot has its own explicit colorbar limits and ticks, matching the
   supplied reference figure.

Colab installation, if required:
    %pip install -q xarray "dask[array]" netCDF4 h5netcdf cartopy scipy
"""

from pathlib import Path
import gc
import string
import warnings

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

warnings.filterwarnings("ignore")



# 1. FILE PATHS


DATA_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Data")
OUTPUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Fig/P1_Corrected")

AIR_TEMPERATURE_FILE = DATA_DIR / "2mT_MSLP_era.nc"
TURBULENT_HEAT_FLUX_FILE = DATA_DIR / "heatflux_latent_sensible_era.nc"
DOWNWARD_RADIATION_FILE = DATA_DIR / "SW_LW_era.nc"

OUT_PNG = (
    OUTPUT_DIR
    / "Seasonal_Antarctic_Surface_Energy_Climatology_2008_2025_1080dpi.png"
)
OUT_PDF = (
    OUTPUT_DIR
    / "Seasonal_Antarctic_Surface_Energy_Climatology_2008_2025.pdf"
)
OUT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "Seasonal_Antarctic_Surface_Energy_Climatology_summary.csv"
)
OUT_SOURCE_CSV = (
    OUTPUT_DIR
    / "Seasonal_Antarctic_Surface_Energy_input_variables.csv"
)



# 2. ANALYSIS SETTINGS


START_YEAR = 2008
END_YEAR = 2025
NORTHERN_MAP_LIMIT = -60.0
SAVE_DPI = 1080
FIGURE_SIZE = (11.8, 14.2)

# ERA5 monthly-averaged accumulated surface fields contain the mean daily
# accumulation in J m-2. Divide by one day to obtain W m-2.
ERA5_ACCUMULATION_SECONDS = 86_400.0

# Native ERA5 sensible and latent heat fluxes use positive downward.
# Change to "upward" only if a different source explicitly uses that sign.
TURBULENT_SOURCE_POSITIVE = "downward"

SEASONS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Autumn": [3, 4, 5],
    "Winter": [6, 7, 8],
}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

VARIABLE_ORDER = ["Air Temperature", "Turbulent Heat Flux", "Radiation"]

# Every panel is deliberately specified separately. These are the limits and
# exact tick intervals visible in the supplied reference figure.
PANEL_COLOR_SETTINGS = {
    ("Spring", "Air Temperature"): {
        "vmin": -32.0,
        "vmax": 12.0,
        "ticks": [-30, -20, -10, 0, 10],
        "cmap": "RdYlBu_r",
        "unit": "°C",
    },
    ("Spring", "Turbulent Heat Flux"): {
        "vmin": -100.0,
        "vmax": 60.0,
        "ticks": [-100, -80, -60, -40, -20, 0, 20, 40, 60],
        "cmap": "RdBu_r",
        "unit": "W m$^{-2}$",
    },
    ("Spring", "Radiation"): {
        "vmin": 290.0,
        "vmax": 460.0,
        "ticks": [300, 350, 400, 450],
        "cmap": "YlOrRd",
        "unit": "W m$^{-2}$",
    },
    ("Summer", "Air Temperature"): {
        "vmin": -12.0,
        "vmax": 12.0,
        "ticks": [-10, -5, 0, 5, 10],
        "cmap": "RdYlBu_r",
        "unit": "°C",
    },
    ("Summer", "Turbulent Heat Flux"): {
        "vmin": -100.0,
        "vmax": 60.0,
        "ticks": [-100, -80, -60, -40, -20, 0, 20, 40, 60],
        "cmap": "RdBu_r",
        "unit": "W m$^{-2}$",
    },
    ("Summer", "Radiation"): {
        "vmin": 440.0,
        "vmax": 560.0,
        "ticks": [450, 475, 500, 525, 550],
        "cmap": "YlOrRd",
        "unit": "W m$^{-2}$",
    },
    ("Autumn", "Air Temperature"): {
        "vmin": -32.0,
        "vmax": 12.0,
        "ticks": [-30, -20, -10, 0, 10],
        "cmap": "RdYlBu_r",
        "unit": "°C",
    },
    ("Autumn", "Turbulent Heat Flux"): {
        "vmin": -100.0,
        "vmax": 60.0,
        "ticks": [-100, -80, -60, -40, -20, 0, 20, 40, 60],
        "cmap": "RdBu_r",
        "unit": "W m$^{-2}$",
    },
    ("Autumn", "Radiation"): {
        "vmin": 140.0,
        "vmax": 360.0,
        "ticks": [150, 200, 250, 300, 350],
        "cmap": "YlOrRd",
        "unit": "W m$^{-2}$",
    },
    ("Winter", "Air Temperature"): {
        "vmin": -32.0,
        "vmax": 12.0,
        "ticks": [-30, -20, -10, 0, 10],
        "cmap": "RdYlBu_r",
        "unit": "°C",
    },
    ("Winter", "Turbulent Heat Flux"): {
        "vmin": -100.0,
        "vmax": 60.0,
        "ticks": [-100, -80, -60, -40, -20, 0, 20, 40, 60],
        "cmap": "RdBu_r",
        "unit": "W m$^{-2}$",
    },
    ("Winter", "Radiation"): {
        "vmin": 140.0,
        "vmax": 360.0,
        "ticks": [150, 200, 250, 300, 350],
        "cmap": "YlOrRd",
        "unit": "W m$^{-2}$",
    },
}


# 3. DATASET AND VARIABLE HELPERS


def open_dataset_robust(path):
    """Open a NetCDF file lazily, with a non-Dask fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    try:
        dataset = xr.open_dataset(
            path,
            decode_times=True,
            decode_timedelta=False,
            chunks="auto",
        )
    except (ImportError, ValueError) as error:
        message = str(error).lower()
        if "dask" not in message and "chunk manager" not in message:
            raise
        print(f"WARNING: Dask unavailable; opening {path.name} without chunks.")
        dataset = xr.open_dataset(
            path,
            decode_times=True,
            decode_timedelta=False,
            chunks=None,
        )

    return standardize_dataset(dataset)


def standardize_dataset(dataset):
    """Standardize coordinate names, longitude convention and ordering."""
    aliases = {
        "valid_time": "time",
        "time_counter": "time",
        "date": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    present = set(dataset.coords) | set(dataset.dims)
    rename = {
        old: new
        for old, new in aliases.items()
        if old in present and new not in present
    }
    if rename:
        dataset = dataset.rename(rename)

    required = {"time", "lat", "lon"}
    missing = required.difference(dataset.coords)
    if missing:
        raise ValueError(
            f"Missing coordinate(s) {sorted(missing)}. "
            f"Found coordinates: {list(dataset.coords)}"
        )

    if dataset["lat"].ndim != 1 or dataset["lon"].ndim != 1:
        raise ValueError(
            "This code expects one-dimensional latitude and longitude "
            "coordinates, as used by the supplied ERA files."
        )

    # Remove harmless auxiliary scalar coordinates where possible.
    dataset = dataset.drop_vars(["number"], errors="ignore")

    longitude = ((dataset["lon"].astype(float) + 180.0) % 360.0) - 180.0
    dataset = dataset.assign_coords(lon=longitude)
    dataset = dataset.sortby("lon").sortby("lat").sortby("time")

    # Drop duplicated longitudes, including a duplicated -180/180 seam.
    _, unique_lon_indices = np.unique(
        np.asarray(dataset["lon"].values),
        return_index=True,
    )
    dataset = dataset.isel(lon=np.sort(unique_lon_indices))

    dataset = dataset.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    dataset = dataset.where(dataset["lat"] <= NORTHERN_MAP_LIMIT, drop=True)

    if dataset.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No data remain inside {START_YEAR}–{END_YEAR}: "
            f"{list(dataset.data_vars)}"
        )

    return dataset


def choose_variable(dataset, candidates, description, required=True):
    """Select a variable using exact then case-insensitive candidate names."""
    lower_lookup = {
        str(variable).lower(): variable for variable in dataset.data_vars
    }
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return dataset[candidate], candidate
        match = lower_lookup.get(candidate.lower())
        if match is not None:
            return dataset[match], match

    if required:
        raise ValueError(
            f"Could not identify {description}. Tried {candidates}.\n"
            f"Available variables: {list(dataset.data_vars)}"
        )
    return None, None


def collapse_auxiliary_dimensions(data_array):
    """Merge common ERA auxiliary dimensions without altering time/space."""
    for dimension in list(data_array.dims):
        lower = str(dimension).lower()
        if lower in {"expver", "number", "member", "ensemble"}:
            data_array = data_array.mean(dimension, skipna=True)

    extra = set(data_array.dims).difference({"time", "lat", "lon"})
    if extra:
        raise ValueError(
            f"Unexpected dimensions {sorted(extra)} in {data_array.name}. "
            "Select the required level before using this script."
        )
    return data_array.transpose("time", "lat", "lon")


def monthly_unique(data_array):
    """Convert timestamps to month-start and merge duplicate monthly records."""
    month_starts = (
        pd.DatetimeIndex(pd.to_datetime(data_array["time"].values))
        .to_period("M")
        .to_timestamp()
    )
    data_array = data_array.assign_coords(time=month_starts).sortby("time")
    if pd.DatetimeIndex(month_starts).duplicated().any():
        data_array = data_array.groupby("time").mean("time", skipna=True)
    return data_array


def clean_field(data_array):
    """Keep finite, non-fill data and standardize monthly ordering."""
    data_array = collapse_auxiliary_dimensions(data_array)
    data_array = monthly_unique(data_array)
    data_array = data_array.where(np.isfinite(data_array))
    data_array = data_array.where(np.abs(data_array) < 1.0e20)
    return data_array.astype("float32")


def representative_median(data_array, maximum_points_per_dimension=32):
    """
    Return a Dask-safe representative median from a regular subsample.

    Calling ``data_array.median()`` without explicit axes sends ``axis=None``
    to ``dask.array.nanmedian``, which Dask does not implement. An exact global
    median would also require placing the entire array in one chunk. For unit
    detection and range diagnostics, a spatially and temporally distributed
    sample is sufficient and far more memory efficient.
    """
    indexers = {}
    for dimension, size in data_array.sizes.items():
        number = min(int(size), int(maximum_points_per_dimension))
        indices = np.linspace(0, size - 1, number, dtype=int)
        indexers[dimension] = np.unique(indices)

    sampled = data_array.isel(indexers).compute()
    values = np.asarray(sampled.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(
            f"No finite values found while sampling {data_array.name!r}."
        )
    return float(np.median(finite_values))


def exact_global_extreme(data_array, operation):
    """Calculate a Dask-safe exact global minimum or maximum."""
    dimensions = list(data_array.dims)
    if operation == "minimum":
        reduced = data_array.min(dim=dimensions, skipna=True)
    elif operation == "maximum":
        reduced = data_array.max(dim=dimensions, skipna=True)
    else:
        raise ValueError("operation must be 'minimum' or 'maximum'")
    return float(reduced.compute().item())


def convert_temperature_to_celsius(data_array):
    """Convert Kelvin to degrees Celsius, retaining Celsius input unchanged."""
    units = str(data_array.attrs.get("units", "")).lower().replace(" ", "")
    sample = representative_median(data_array)

    if "kelvin" in units or units in {"k", "degk"} or sample > 100.0:
        converted = data_array - 273.15
        action = "converted from Kelvin to °C"
    else:
        converted = data_array
        action = "already interpreted as °C"

    converted.attrs = dict(data_array.attrs)
    converted.attrs["units"] = "degree_Celsius"
    converted.attrs["conversion"] = action
    converted.name = "air_temperature_2m"
    print(f"2 m air temperature: {action}.")
    return converted.astype("float32")


def energy_or_flux_to_wm2(data_array, label):
    """Convert W m-2 or ERA5 mean-daily J m-2 fields to W m-2."""
    original_units = str(data_array.attrs.get("units", "")).strip()
    compact = (
        original_units.lower()
        .replace(" ", "")
        .replace("**", "^")
        .replace("−", "-")
    )

    is_watt_flux = (
        compact.startswith("w")
        and ("m-2" in compact or "m^-2" in compact or "/m2" in compact)
    )
    is_energy_area = (
        compact.startswith("j")
        and ("m-2" in compact or "m^-2" in compact or "/m2" in compact)
    )

    if is_watt_flux:
        converted = data_array
        action = "already W m-2"
    elif is_energy_area:
        converted = data_array / ERA5_ACCUMULATION_SECONDS
        action = (
            "ERA5 monthly mean daily accumulation divided by "
            f"{ERA5_ACCUMULATION_SECONDS:g} s"
        )
    else:
        sample = representative_median(np.abs(data_array))
        if sample > 2_000.0:
            converted = data_array / ERA5_ACCUMULATION_SECONDS
            action = (
                "units missing/ambiguous; magnitude indicates J m-2, "
                f"divided by {ERA5_ACCUMULATION_SECONDS:g} s"
            )
        else:
            converted = data_array
            action = "units missing/ambiguous; magnitude interpreted as W m-2"

    converted.attrs = dict(data_array.attrs)
    converted.attrs["original_units"] = original_units or "not supplied"
    converted.attrs["units"] = "W m-2"
    converted.attrs["conversion"] = action
    print(f"{label}: {action}.")
    return converted.astype("float32")


def align_like(data_array, reference, label):
    """Interpolate to a reference grid only when coordinates differ."""
    same_lat = np.array_equal(data_array["lat"], reference["lat"])
    same_lon = np.array_equal(data_array["lon"], reference["lon"])
    same_time = np.array_equal(data_array["time"], reference["time"])

    if not same_lat or not same_lon:
        print(f"{label}: interpolating to companion-variable grid.")
        data_array = data_array.interp(
            lat=reference["lat"],
            lon=reference["lon"],
            method="linear",
        )
    if not same_time:
        data_array, _ = xr.align(data_array, reference, join="inner")
    return data_array


# =============================================================================
# 4. LOAD AND DERIVE THE THREE MONTHLY FIELDS
# =============================================================================

def load_air_temperature():
    dataset = open_dataset_robust(AIR_TEMPERATURE_FILE)
    field, variable_name = choose_variable(
        dataset,
        ["t2m", "2t", "T2M", "air_temperature_2m"],
        "2 m air-temperature variable",
    )
    field = convert_temperature_to_celsius(clean_field(field))
    return field, dataset, variable_name


def load_turbulent_heat_flux():
    dataset = open_dataset_robust(TURBULENT_HEAT_FLUX_FILE)

    latent, latent_name = choose_variable(
        dataset,
        [
            "slhf",
            "mslhf",
            "surface_latent_heat_flux",
            "mean_surface_latent_heat_flux",
            "latent_heat_flux",
            "lhf",
        ],
        "surface latent-heat-flux variable",
        required=False,
    )
    sensible, sensible_name = choose_variable(
        dataset,
        [
            "sshf",
            "msshf",
            "surface_sensible_heat_flux",
            "mean_surface_sensible_heat_flux",
            "sensible_heat_flux",
            "shf",
        ],
        "surface sensible-heat-flux variable",
        required=False,
    )

    if latent is not None and sensible is not None:
        latent = energy_or_flux_to_wm2(clean_field(latent), "Latent heat flux")
        sensible = energy_or_flux_to_wm2(
            clean_field(sensible), "Sensible heat flux"
        )
        sensible = align_like(sensible, latent, "Sensible heat flux")
        latent, sensible = xr.align(latent, sensible, join="inner")
        turbulent = latent + sensible
        source_names = f"{latent_name} + {sensible_name}"
    else:
        turbulent, combined_name = choose_variable(
            dataset,
            [
                "turbulent_surface_heat_flux",
                "surface_turbulent_heat_flux",
                "turbulent_heat_flux",
                "slhf_plus_sshf",
            ],
            "combined turbulent surface heat-flux variable",
        )
        turbulent = energy_or_flux_to_wm2(
            clean_field(turbulent), "Combined turbulent heat flux"
        )
        source_names = combined_name

    if TURBULENT_SOURCE_POSITIVE.lower() == "upward":
        turbulent = -turbulent
        print("Turbulent flux: sign reversed from positive upward to downward.")
    elif TURBULENT_SOURCE_POSITIVE.lower() != "downward":
        raise ValueError(
            "TURBULENT_SOURCE_POSITIVE must be 'downward' or 'upward'."
        )

    turbulent.name = "turbulent_surface_heat_flux_positive_downward"
    turbulent.attrs["units"] = "W m-2"
    turbulent.attrs["positive"] = "downward"
    return turbulent.astype("float32"), dataset, source_names


def load_downward_radiation():
    dataset = open_dataset_robust(DOWNWARD_RADIATION_FILE)

    shortwave, shortwave_name = choose_variable(
        dataset,
        [
            "ssrd",
            "msdwswrf",
            "surface_solar_radiation_downwards",
            "surface_downwelling_shortwave_flux_in_air",
            "downward_shortwave_radiation",
            "sw_down",
        ],
        "downward shortwave-radiation variable",
        required=False,
    )
    longwave, longwave_name = choose_variable(
        dataset,
        [
            "strd",
            "msdwlwrf",
            "surface_thermal_radiation_downwards",
            "surface_downwelling_longwave_flux_in_air",
            "downward_longwave_radiation",
            "lw_down",
        ],
        "downward longwave-radiation variable",
        required=False,
    )

    if shortwave is not None and longwave is not None:
        shortwave = energy_or_flux_to_wm2(
            clean_field(shortwave), "Downward shortwave radiation"
        )
        longwave = energy_or_flux_to_wm2(
            clean_field(longwave), "Downward longwave radiation"
        )
        longwave = align_like(longwave, shortwave, "Longwave radiation")
        shortwave, longwave = xr.align(shortwave, longwave, join="inner")
        radiation = shortwave + longwave
        source_names = f"{shortwave_name} + {longwave_name}"
    else:
        radiation, combined_name = choose_variable(
            dataset,
            [
                "downward_surface_radiation",
                "total_downward_surface_radiation",
                "sw_lw_down",
                "ssrd_plus_strd",
            ],
            "combined downward surface-radiation variable",
        )
        radiation = energy_or_flux_to_wm2(
            clean_field(radiation), "Combined downward surface radiation"
        )
        source_names = combined_name

    radiation = radiation.where(radiation >= 0.0)
    radiation.name = "downward_surface_radiation_sw_plus_lw"
    radiation.attrs["units"] = "W m-2"
    radiation.attrs["positive"] = "downward"
    return radiation.astype("float32"), dataset, source_names



# 5. SEASONAL CLIMATOLOGIES

def seasonal_year_coordinate(time, season_name):
    """Assign December to the next year for DJF; use calendar year otherwise."""
    years = time.dt.year.astype("int32")
    if season_name == "Summer":
        return xr.where(time.dt.month == 12, years + 1, years)
    return years


def complete_season_climatology(data_array, season_name):
    """Return the mean of complete, day-weighted seasonal means."""
    months = SEASONS[season_name]
    selected = data_array.where(data_array["time"].dt.month.isin(months), drop=True)
    season_year = seasonal_year_coordinate(selected["time"], season_name)
    selected = selected.assign_coords(season_year=("time", season_year.values))

    table = pd.DataFrame(
        {
            "time": pd.to_datetime(selected["time"].values),
            "season_year": np.asarray(season_year.values, dtype=int),
        }
    )
    table["month"] = table["time"].dt.month
    counts = table.groupby("season_year")["month"].nunique()
    valid_years = counts[counts == 3].index.to_numpy(dtype=int)
    valid_years = valid_years[
        (valid_years >= START_YEAR) & (valid_years <= END_YEAR)
    ]
    if valid_years.size == 0:
        raise ValueError(f"No complete {season_name} seasons were found.")

    selected = selected.where(selected["season_year"].isin(valid_years), drop=True)
    days = selected["time"].dt.days_in_month.astype("float32")
    numerator = (selected * days).groupby("season_year").sum(
        "time", skipna=True, min_count=1
    )
    denominator = days.where(np.isfinite(selected)).groupby("season_year").sum(
        "time", skipna=True, min_count=1
    )
    yearly_seasonal_mean = numerator / denominator
    climatology = yearly_seasonal_mean.mean("season_year", skipna=True).compute()
    climatology.attrs = dict(data_array.attrs)
    climatology.attrs["season"] = season_name
    climatology.attrs["season_years"] = ",".join(map(str, valid_years))
    climatology.attrs["number_of_complete_seasons"] = int(valid_years.size)

    print(
        f"{season_name}: {valid_years.size} complete seasons "
        f"({valid_years.min()}–{valid_years.max()})"
    )
    return climatology, valid_years


def validate_physical_range(field, label, low, high):
    """Catch gross unit/sign errors without clipping scientifically valid data."""
    median = representative_median(field)
    minimum = exact_global_extreme(field, "minimum")
    maximum = exact_global_extreme(field, "maximum")
    print(
        f"{label}: min={minimum:.2f}, median={median:.2f}, max={maximum:.2f}"
    )
    if not (low <= median <= high):
        raise ValueError(
            f"Implausible median for {label}: {median:.3f}. "
            "Check the selected variable, units and accumulation period."
        )


def area_weighted_mean(data_array):
    weights = np.cos(np.deg2rad(data_array["lat"]))
    return float(
        data_array.weighted(weights).mean(("lat", "lon"), skipna=True).item()
    )



# 6. MAP HELPERS

def make_polar_axis(figure, subplot_spec):
    axis = figure.add_subplot(
        subplot_spec,
        projection=ccrs.SouthPolarStereo(central_longitude=0.0),
    )
    axis.set_extent(
        [-180.0, 180.0, -90.0, NORTHERN_MAP_LIMIT],
        crs=ccrs.PlateCarree(),
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.vstack([np.sin(theta), np.cos(theta)]).T
    axis.set_boundary(
        mpath.Path(circle * 0.5 + np.array([0.5, 0.5])),
        transform=axis.transAxes,
    )

    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180, 181, 30),
        ylocs=[-80, -70, -60],
        linewidth=0.45,
        color="0.45",
        alpha=0.42,
        linestyle=":",
        zorder=5,
    )
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gridlines.ylocator = mticker.FixedLocator([-80, -70, -60])

    # Longitude labels are placed just outside the circular boundary.
    longitude_labels = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    for longitude in longitude_labels:
        if longitude < 0:
            text = f"{abs(longitude)}°W"
        elif longitude == 0:
            text = "0°"
        elif longitude == 180:
            text = "180°W"
        else:
            text = f"{longitude}°E"
        axis.text(
            longitude,
            -58.85,
            text,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )

    for latitude in [-70, -80]:
        axis.text(
            5.0,
            latitude,
            f"{abs(latitude)}°S",
            transform=ccrs.PlateCarree(),
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="0.25",
            zorder=20,
        )

    return axis


def add_land_and_coastline(axis):
    axis.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="0.84",
        edgecolor="0.25",
        linewidth=0.55,
        zorder=8,
    )
    axis.coastlines(
        resolution="50m",
        color="0.20",
        linewidth=0.65,
        zorder=9,
    )


def cyclic_values(data_array):
    """Add a cyclic longitude column to eliminate the map seam."""
    values, longitude = add_cyclic_point(
        np.asarray(data_array.values),
        coord=np.asarray(data_array["lon"].values),
        axis=-1,
    )
    return longitude, np.asarray(data_array["lat"].values), values



# 7. MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Loading monthly ERA surface-energy fields")
    print("=" * 72)

    air, air_dataset, air_variable = load_air_temperature()
    turbulent, turbulent_dataset, turbulent_variables = load_turbulent_heat_flux()
    radiation, radiation_dataset, radiation_variables = load_downward_radiation()

    validate_physical_range(air, "2 m air temperature (°C)", -70.0, 15.0)
    validate_physical_range(
        turbulent,
        "turbulent heat flux, positive downward (W m-2)",
        -250.0,
        150.0,
    )
    validate_physical_range(
        radiation,
        "downward SW + LW radiation (W m-2)",
        50.0,
        800.0,
    )

    monthly_fields = {
        "Air Temperature": air,
        "Turbulent Heat Flux": turbulent,
        "Radiation": radiation,
    }

    seasonal_fields = {season: {} for season in SEASON_ORDER}
    summary_rows = []

    print("\n" + "=" * 72)
    print("Calculating complete seasonal climatologies")
    print("=" * 72)

    for variable_name, monthly_field in monthly_fields.items():
        print(f"\n{variable_name}")
        for season_name in SEASON_ORDER:
            climatology, years = complete_season_climatology(
                monthly_field,
                season_name,
            )
            seasonal_fields[season_name][variable_name] = climatology
            finite_values = np.asarray(climatology.values, dtype=float)
            panel_number = (
                SEASON_ORDER.index(season_name) * len(VARIABLE_ORDER)
                + VARIABLE_ORDER.index(variable_name)
            )
            summary_rows.append(
                {
                    "Panel": f"({string.ascii_lowercase[panel_number]})",
                    "Season": season_name,
                    "Variable": variable_name,
                    "Complete_seasons": len(years),
                    "First_season_year": int(years.min()),
                    "Last_season_year": int(years.max()),
                    "Spatial_minimum": float(np.nanmin(finite_values)),
                    "Spatial_maximum": float(np.nanmax(finite_values)),
                    "Area_weighted_mean": area_weighted_mean(climatology),
                    "Units": climatology.attrs.get("units", ""),
                }
            )

    source_table = pd.DataFrame(
        [
            {
                "Derived_field": "2 m air temperature",
                "File": str(AIR_TEMPERATURE_FILE),
                "Source_variable_or_sum": air_variable,
                "Output_units": "°C",
            },
            {
                "Derived_field": "Turbulent surface heat flux",
                "File": str(TURBULENT_HEAT_FLUX_FILE),
                "Source_variable_or_sum": turbulent_variables,
                "Output_units": "W m-2; positive downward",
            },
            {
                "Derived_field": "Downward surface radiation",
                "File": str(DOWNWARD_RADIATION_FILE),
                "Source_variable_or_sum": radiation_variables,
                "Output_units": "W m-2; SW down + LW down",
            },
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titleweight": "bold",
        }
    )

    # This size retains publication-scale text while keeping a 1080-DPI PNG
    # within a practical Colab memory footprint.
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer_grid = GridSpec(
        nrows=4,
        ncols=3,
        figure=figure,
        left=0.075,
        right=0.965,
        bottom=0.030,
        top=0.940,
        wspace=0.300,
        hspace=0.165,
    )

    column_titles = [
        "2 m air temperature",
        "Turbulent surface\nheat flux\n(positive downward)",
        "Downward surface\nradiation\n(SW↓ + LW↓)",
    ]

    axes_by_row = []
    panel_index = 0

    for row, season_name in enumerate(SEASON_ORDER):
        row_axes = []
        for column, variable_name in enumerate(VARIABLE_ORDER):
            # Each map and its colorbar have an isolated sub-grid. This keeps
            # all 12 colorbar labels clear of neighboring panels.
            panel_grid = outer_grid[row, column].subgridspec(
                1,
                2,
                width_ratios=[1.0, 0.044],
                wspace=0.075,
            )
            axis = make_polar_axis(figure, panel_grid[0, 0])
            colorbar_axis = figure.add_subplot(panel_grid[0, 1])
            row_axes.append(axis)

            field = seasonal_fields[season_name][variable_name]
            longitude, latitude, values = cyclic_values(field)
            settings = PANEL_COLOR_SETTINGS[(season_name, variable_name)]
            norm = Normalize(
                vmin=settings["vmin"],
                vmax=settings["vmax"],
                clip=True,
            )

            mesh = axis.pcolormesh(
                longitude,
                latitude,
                values,
                transform=ccrs.PlateCarree(),
                cmap=settings["cmap"],
                norm=norm,
                shading="auto",
                rasterized=True,
                zorder=1,
            )
            add_land_and_coastline(axis)

            colorbar = figure.colorbar(
                mesh,
                cax=colorbar_axis,
                orientation="vertical",
                ticks=settings["ticks"],
                extend="neither",
            )
            colorbar.set_label(
                settings["unit"],
                fontsize=8.4,
                fontweight="bold",
                labelpad=4,
            )
            colorbar.ax.tick_params(
                axis="y",
                labelsize=7.4,
                width=0.7,
                length=3.0,
                pad=2.0,
            )
            colorbar.outline.set_linewidth(0.75)
            colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            for tick_label in colorbar.ax.get_yticklabels():
                tick_label.set_fontweight("bold")

            if row == 0:
                axis.set_title(
                    column_titles[column],
                    fontsize=12.5,
                    fontweight="bold",
                    pad=22,
                    linespacing=1.08,
                )

            panel_letter = f"({string.ascii_lowercase[panel_index]})"
            axis.text(
                -0.075,
                1.025,
                panel_letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                clip_on=False,
                zorder=30,
            )

            panel_index += 1

        axes_by_row.append(row_axes)

    # Place row labels from the actual map positions after layout creation.
    figure.canvas.draw()
    for row, season_name in enumerate(SEASON_ORDER):
        position = axes_by_row[row][0].get_position()
        figure.text(
            0.026,
            0.5 * (position.y0 + position.y1),
            season_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

    summary_table = pd.DataFrame(summary_rows)
    panel_order = [f"({letter})" for letter in string.ascii_lowercase[:12]]
    summary_table["Panel"] = pd.Categorical(
        summary_table["Panel"], categories=panel_order, ordered=True
    )
    summary_table = summary_table.sort_values("Panel").reset_index(drop=True)
    summary_table["Panel"] = summary_table["Panel"].astype(str)
    summary_table.to_csv(OUT_SUMMARY_CSV, index=False)
    source_table.to_csv(OUT_SOURCE_CSV, index=False)

    print("\nSaving figure and supporting CSV files...")
    figure.savefig(
        OUT_PNG,
        dpi=SAVE_DPI,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    figure.savefig(
        OUT_PDF,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.show()
    plt.close(figure)

    print("\nSaved outputs:")
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)
    print("Summary CSV:", OUT_SUMMARY_CSV)
    print("Source-variable CSV:", OUT_SOURCE_CSV)

    air_dataset.close()
    turbulent_dataset.close()
    radiation_dataset.close()
    gc.collect()


if __name__ == "__main__":
    main()
