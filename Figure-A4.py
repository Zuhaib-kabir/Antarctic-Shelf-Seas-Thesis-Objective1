#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

%pip install -q xarray "dask[array]" netCDF4 h5netcdf cartopy scipy


"""
Seasonal mean Southern Ocean surface-current speed and streamlines, 2008–2025.

Panels
------
(a) Spring (SON)
(b) Summer (DJF)
(c) Autumn (MAM)
(d) Winter (JJA)

Method
------
1. Load both eastward-current files and concatenate 2008–2014 with 2015–2025.
2. Load both northward-current files and concatenate 2008–2014 with 2015–2025.
3. Select the shallowest available depth level in each component.
4. Align northward velocity to the eastward-velocity grid when required.
5. Calculate monthly grid-cell speed as sqrt(uo**2 + vo**2).
6. Calculate complete-season means by year, then average those seasonal means.
7. Plot mean current speed as shading and the seasonal mean vector field as
   continuous streamlines with directional arrowheads.

Important distinction
---------------------
The shading is mean[sqrt(uo**2 + vo**2)], not
sqrt(mean(uo)**2 + mean(vo)**2). The former correctly represents mean current
speed, while the streamlines represent the direction of the mean current
vector. Streamline density is a visualization setting and does not represent
current magnitude.

Colab installation, if required:
    %pip install -q xarray "dask[array]" netCDF4 h5netcdf cartopy scipy
"""

from pathlib import Path
import gc
import warnings

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")


# 1. PATHS


DATA_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Data")
OUTPUT_DIR = Path("/content/drive/MyDrive/SAM_Thesis/Fig/P1_Corrected")

EASTWARD_CURRENT_FILES = [
    DATA_DIR / "EastwardSeaWaterVelocity_uo_2008_14 .nc",
    DATA_DIR / "EastwardSeaWaterVelocity_uo_2015_25 .nc",
]

NORTHWARD_CURRENT_FILES = [
    DATA_DIR / "NorthwardSeaWaterVelocity_uo_2008_14 .nc",
    DATA_DIR / "NorthwardSeaWaterVelocity_uo_2015_25 .nc",
]

OUT_PNG = (
    OUTPUT_DIR
    / "Seasonal_Ocean_Current_Speed_and_Vectors_2008_2025_1080dpi.png"
)
OUT_PDF = (
    OUTPUT_DIR
    / "Seasonal_Ocean_Current_Speed_and_Vectors_2008_2025.pdf"
)
OUT_STATS_CSV = (
    OUTPUT_DIR
    / "Seasonal_Ocean_Current_Speed_spatial_summary_2008_2025.csv"
)
OUT_SOURCE_CSV = (
    OUTPUT_DIR
    / "Seasonal_Ocean_Current_input_files_and_variables.csv"
)



# 2. ANALYSIS AND PLOT SETTINGS


START_YEAR = 2008
END_YEAR = 2025
SOUTHERN_LIMIT = -60.0
TARGET_PLOT_RESOLUTION_DEG = 0.25

SEASONS = {
    "Spring": {"months": [9, 10, 11], "abbreviation": "SON"},
    "Summer": {"months": [12, 1, 2], "abbreviation": "DJF"},
    "Autumn": {"months": [3, 4, 5], "abbreviation": "MAM"},
    "Winter": {"months": [6, 7, 8], "abbreviation": "JJA"},
}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)"]

# Current values outside this generous physical range are treated as invalid.
# This is applied after conversion to m s^-1.
MAX_ABSOLUTE_CURRENT_MPS = 5.0

# Shared shading scale. Leave as None for an automatic common 98th percentile.
# Set a number such as 0.30 to enforce a fixed maximum across related figures.
COLOR_MAX_MPS = None
COLOR_PERCENTILE = 98.0
# Publication-style sequential ocean palette: near-white/cyan for weak flow,
# progressing through blue to deep navy for strong flow. This avoids the harsh
# purple-to-yellow appearance of viridis while remaining monotonic in lightness.
COLOR_CMAP = LinearSegmentedColormap.from_list(
    "southern_ocean_speed",
    [
        "#F7FCFD",
        "#D9F0F2",
        "#A6DCE5",
        "#67B8CE",
        "#2B8CBE",
        "#1764A0",
        "#08306B",
    ],
    N=256,
)

# Streamline density and appearance. The streamline layer displays current
# direction only; current magnitude is represented by the unchanged shading.
# Increase STREAM_DENSITY slightly for more lines, or decrease it if the map
# looks crowded. A scalar density gives consistent spacing in both directions.
STREAM_DENSITY = 1.55
STREAM_COLOR = (0.05, 0.05, 0.05, 0.82)
STREAM_LINEWIDTH = 0.58
STREAM_ARROWSIZE = 0.72
STREAM_ARROWSTYLE = "-|>"
STREAM_MINLENGTH = 0.10
STREAM_MAXLENGTH = 4.0

# Directions become unstable where the seasonal vector mean is nearly zero.
# Those cells are omitted from the streamline calculation only; the speed
# shading retains every valid cell.
MIN_STREAM_SPEED_MPS = 0.015

FIGURE_SIZE = (12.4, 9.2)
SAVE_DPI = 1080



# 3. FILE AND DATASET HELPERS


def resolve_current_file(path):
    """Accept supplied current filenames with common spacing/name variants."""
    path = Path(path)
    path_string = str(path)
    candidates = [
        path,
        Path(path_string.replace(" .nc", ".nc")),
        Path(
            path_string.replace(
                "NorthwardSeaWaterVelocity_uo_",
                "NorthwardSeaWaterVelocity_vo_",
            )
        ),
        Path(
            path_string.replace(
                "NorthwardSeaWaterVelocity_uo_",
                "NorthwardSeaWaterVelocity_vo_",
            ).replace(" .nc", ".nc")
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def open_dataset_robust(path):
    """Open lazily with Dask when available, otherwise without chunks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Current input file not found: {path}")

    try:
        return xr.open_dataset(
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
        return xr.open_dataset(
            path,
            decode_times=True,
            decode_timedelta=False,
            chunks=None,
        )


def standardize_dataset(dataset):
    """Standardize coordinate names, longitude convention and ordering."""
    aliases = {
        "valid_time": "time",
        "time_counter": "time",
        "month": "time",
        "latitude": "lat",
        "nav_lat": "lat",
        "longitude": "lon",
        "nav_lon": "lon",
    }
    available = set(dataset.coords) | set(dataset.dims)
    rename = {
        old: new
        for old, new in aliases.items()
        if old in available and new not in available
    }
    if rename:
        dataset = dataset.rename(rename)

    missing = {"time", "lat", "lon"}.difference(dataset.coords)
    if missing:
        raise ValueError(
            f"Required coordinate(s) missing: {sorted(missing)}. "
            f"Coordinates found: {list(dataset.coords)}"
        )

    if dataset["lat"].ndim != 1 or dataset["lon"].ndim != 1:
        raise ValueError(
            "This script requires one-dimensional latitude and longitude "
            "coordinates."
        )

    dataset = dataset.drop_vars(["number", "expver"], errors="ignore")

    if float(dataset["lon"].max()) > 180.0:
        dataset = dataset.assign_coords(
            lon=((dataset["lon"] + 180.0) % 360.0) - 180.0
        )

    dataset = dataset.sortby("lon").sortby("lat").sortby("time")
    dataset = dataset.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    dataset = dataset.where(dataset["lat"] <= SOUTHERN_LIMIT, drop=True)
    return dataset


def find_variable(dataset, candidates, descriptive_name):
    """Find a data variable case-insensitively."""
    lower_lookup = {
        str(variable).lower(): variable for variable in dataset.data_vars
    }
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
        match = lower_lookup.get(str(candidate).lower())
        if match is not None:
            return match

    raise ValueError(
        f"Could not identify {descriptive_name}. Tried {candidates}. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def select_surface_layer(data_array, component_name):
    """Select the shallowest current level when a depth dimension exists."""
    depth_names = {
        "depth",
        "depthu",
        "depthv",
        "lev",
        "level",
        "z",
    }
    depth_dimensions = [
        dimension
        for dimension in data_array.dims
        if str(dimension).lower() in depth_names
    ]

    if not depth_dimensions:
        print(f"{component_name}: no depth dimension; using supplied field.")
        return data_array, np.nan, "not present"

    depth_dimension = depth_dimensions[0]
    depth_values = np.asarray(data_array[depth_dimension].values, dtype=float)
    if depth_values.size == 0 or not np.isfinite(depth_values).any():
        raise ValueError(
            f"{component_name}: unusable depth coordinate {depth_dimension}."
        )

    surface_index = int(np.nanargmin(np.abs(depth_values)))
    surface_depth = float(depth_values[surface_index])
    print(
        f"{component_name}: selected shallowest "
        f"{depth_dimension}={surface_depth:g}."
    )
    return (
        data_array.isel({depth_dimension: surface_index}, drop=True),
        surface_depth,
        depth_dimension,
    )


def month_start_index(values):
    """Convert timestamps to unique calendar-month start timestamps."""
    return (
        pd.DatetimeIndex(pd.to_datetime(values))
        .to_period("M")
        .to_timestamp()
    )


def convert_component_to_mps(data_array, component_name):
    """Convert common velocity units to metres per second."""
    original_units = str(data_array.attrs.get("units", "")).strip()
    compact_units = (
        original_units.lower()
        .replace(" ", "")
        .replace("second", "s")
        .replace("sec", "s")
        .replace("−", "-")
    )

    factor = 1.0
    if "cm" in compact_units:
        factor = 0.01
    elif "mm" in compact_units:
        factor = 0.001
    elif "knot" in compact_units or compact_units in {"kt", "kts"}:
        factor = 0.514444
    elif original_units == "":
        print(
            f"WARNING: {component_name} has no units attribute; assuming m s^-1."
        )
    elif "m" not in compact_units or "s" not in compact_units:
        raise ValueError(
            f"Unsupported {component_name} units: {original_units!r}. "
            "Add the correct conversion in convert_component_to_mps()."
        )

    converted = (data_array * factor).astype("float32")
    converted.attrs = dict(data_array.attrs)
    converted.attrs["original_units"] = original_units or "not provided"
    converted.attrs["units"] = "m s-1"
    converted.attrs["conversion_factor_to_m_s-1"] = factor
    converted.name = component_name
    return converted


def align_to_reference_grid(data_array, reference, label):
    """Interpolate only when latitude/longitude coordinates differ."""
    same_lat = np.array_equal(
        data_array["lat"].values, reference["lat"].values
    )
    same_lon = np.array_equal(
        data_array["lon"].values, reference["lon"].values
    )
    if same_lat and same_lon:
        return data_array

    print(f"{label}: interpolating to the reference current grid.")
    return data_array.interp(
        lat=reference["lat"],
        lon=reference["lon"],
        method="linear",
    )


def load_current_component(file_paths, candidates, component_name):
    """Load and concatenate both temporal files for one current component."""
    pieces = []
    opened_datasets = []
    source_records = []

    for file_path in file_paths:
        print(f"\nOpening {component_name}: {file_path}")
        dataset = standardize_dataset(open_dataset_robust(file_path))
        variable_name = find_variable(dataset, candidates, component_name)
        component = dataset[variable_name].astype("float32")
        component, surface_depth, depth_dimension = select_surface_layer(
            component, component_name
        )

        # Remove harmless singleton dimensions, but stop on ambiguous data.
        for dimension in list(component.dims):
            if dimension not in {"time", "lat", "lon"}:
                if component.sizes[dimension] == 1:
                    component = component.isel({dimension: 0}, drop=True)
                else:
                    raise ValueError(
                        f"{component_name}: unsupported remaining dimension "
                        f"{dimension!r} with size {component.sizes[dimension]}."
                    )

        original_units = str(component.attrs.get("units", "not provided"))
        component = convert_component_to_mps(component, component_name)

        monthly_time = month_start_index(component["time"].values)
        component = component.assign_coords(time=monthly_time)
        if monthly_time.duplicated().any():
            component = component.groupby("time").mean(skipna=True)

        if pieces:
            component = align_to_reference_grid(
                component,
                pieces[0],
                f"{component_name} {Path(file_path).name}",
            )

        pieces.append(component)
        opened_datasets.append(dataset)
        source_records.append(
            {
                "Component": component_name,
                "Input_file": str(file_path),
                "NetCDF_variable": variable_name,
                "Original_units": original_units,
                "Converted_units": "m s-1",
                "Depth_dimension": depth_dimension,
                "Selected_surface_depth": surface_depth,
            }
        )

    combined = xr.concat(
        pieces,
        dim="time",
        join="outer",
        combine_attrs="override",
    ).sortby("time")

    combined_time = month_start_index(combined["time"].values)
    combined = combined.assign_coords(time=combined_time)
    if combined_time.duplicated().any():
        combined = combined.groupby("time").mean(skipna=True)

    combined = combined.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    combined.name = component_name
    combined.attrs["units"] = "m s-1"

    expected_months = 12 * (END_YEAR - START_YEAR + 1)
    actual_months = combined.sizes.get("time", 0)
    print(
        f"{component_name}: combined months={actual_months}/{expected_months}; "
        f"period={str(combined.time.min().values)[:10]} to "
        f"{str(combined.time.max().values)[:10]}"
    )
    if actual_months != expected_months:
        print(
            f"WARNING: expected {expected_months} unique months for "
            f"{component_name}, found {actual_months}."
        )

    return combined, opened_datasets, source_records


def coarsen_to_resolution(data_array, target_resolution):
    """Reduce plotting cost while retaining the original coordinate system."""
    longitude = np.asarray(data_array["lon"].values, dtype=float)
    latitude = np.asarray(data_array["lat"].values, dtype=float)
    lon_resolution = float(np.nanmedian(np.abs(np.diff(longitude))))
    lat_resolution = float(np.nanmedian(np.abs(np.diff(latitude))))

    lon_factor = max(1, int(round(target_resolution / lon_resolution)))
    lat_factor = max(1, int(round(target_resolution / lat_resolution)))
    print(
        f"{data_array.name}: grid={data_array.sizes['lat']}x"
        f"{data_array.sizes['lon']}, resolution≈({lat_resolution:.4f}, "
        f"{lon_resolution:.4f})°, coarsen=({lat_factor}, {lon_factor})"
    )

    if lat_factor > 1 or lon_factor > 1:
        data_array = data_array.coarsen(
            lat=lat_factor,
            lon=lon_factor,
            boundary="trim",
        ).mean(skipna=True)
    return data_array



# 4. SEASONAL CALCULATION HELPERS

def seasonal_year_coordinate(time_coordinate, months):
    """Assign December to the following DJF season year."""
    season_year = time_coordinate.dt.year
    if 12 in months and 1 in months:
        season_year = xr.where(
            time_coordinate.dt.month == 12,
            season_year + 1,
            season_year,
        )
    return season_year.rename("season_year")


def complete_season_years(time_values, months):
    """Return season years containing all requested calendar months."""
    dates = pd.DatetimeIndex(pd.to_datetime(time_values))
    selected = dates[dates.month.isin(months)]
    season_year = selected.year.to_numpy(copy=True)
    if 12 in months and 1 in months:
        season_year[selected.month == 12] += 1

    table = pd.DataFrame(
        {
            "season_year": season_year,
            "calendar_month": selected.month,
        }
    )
    counts = table.groupby("season_year")["calendar_month"].nunique()
    complete = counts.loc[counts == len(set(months))].index.to_numpy(dtype=int)
    return complete


def seasonal_climatology(data_array, months, label):
    """Mean of complete yearly seasonal means over the analysis period."""
    selected = data_array.where(
        data_array["time"].dt.month.isin(months), drop=True
    )
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"No months available for {label}.")

    valid_years = complete_season_years(selected["time"].values, months)
    valid_years = valid_years[
        (valid_years >= START_YEAR) & (valid_years <= END_YEAR)
    ]
    if valid_years.size == 0:
        raise ValueError(f"No complete season years available for {label}.")

    season_year = seasonal_year_coordinate(selected["time"], months)
    selected = selected.assign_coords(season_year=season_year)
    yearly_means = selected.groupby("season_year").mean("time", skipna=True)
    yearly_means = yearly_means.sel(season_year=valid_years)
    climatology = yearly_means.mean("season_year", skipna=True)
    climatology.attrs = dict(data_array.attrs)
    climatology.attrs["season_years_used"] = ",".join(map(str, valid_years))
    climatology.attrs["number_of_seasons"] = int(valid_years.size)

    print(
        f"{label}: {valid_years.size} complete seasons "
        f"({valid_years.min()}–{valid_years.max()})"
    )
    return climatology, valid_years


def area_weighted_spatial_mean(data_array):
    weights = np.cos(np.deg2rad(data_array["lat"]))
    return data_array.weighted(weights).mean(
        dim=("lat", "lon"), skipna=True
    )


def prepare_streamline_components(u_field, v_field, minimum_speed):
    """Mask invalid or near-zero seasonal vectors before stream tracing."""
    vector_speed = np.hypot(u_field, v_field)
    supported = (
        np.isfinite(u_field)
        & np.isfinite(v_field)
        & np.isfinite(vector_speed)
        & (vector_speed >= minimum_speed)
    )

    valid_cells = int(supported.sum().compute().item())
    if valid_cells == 0:
        raise RuntimeError(
            "No valid current vectors remain for streamline tracing. "
            "Reduce MIN_STREAM_SPEED_MPS if the seasonal currents are weak."
        )

    return u_field.where(supported), v_field.where(supported), valid_cells



# 5. MAP HELPERS

def make_polar_axis(figure, subplot_spec):
    axis = figure.add_subplot(
        subplot_spec, projection=ccrs.SouthPolarStereo()
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    vertices = np.vstack([np.sin(theta), np.cos(theta)]).T
    boundary = mpath.Path(vertices * 0.5 + np.array([0.5, 0.5]))
    axis.set_boundary(boundary, transform=axis.transAxes)
    axis.set_extent(
        [-180.0, 180.0, -90.0, SOUTHERN_LIMIT],
        crs=ccrs.PlateCarree(),
    )

    longitude_grid = [
        -150, -120, -90, -60, -30, 0,
        30, 60, 90, 120, 150, 180,
    ]
    latitude_grid = [-60, -70, -80]
    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        linestyle=":",
        color="0.20",
        alpha=0.50,
        zorder=8,
    )
    gridlines.xlocator = mticker.FixedLocator(longitude_grid)
    gridlines.ylocator = mticker.FixedLocator(latitude_grid)

    # Edge labels; 0° is omitted to preserve title clearance.
    for longitude in [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180]:
        if longitude < 0:
            text = f"{abs(longitude)}°W"
        elif longitude == 180:
            text = "180°"
        else:
            text = f"{longitude}°E"
        axis.text(
            longitude,
            -57.8,
            text,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )

    # Latitude labels are placed consistently away from the densest vectors.
    for latitude in [-70, -80]:
        axis.text(
            15,
            latitude,
            f"{abs(latitude)}°S",
            transform=ccrs.PlateCarree(),
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="0.20",
            zorder=20,
        )
    return axis


def add_land_and_coastline(axis):
    """Draw land after streamlines so any accidental land flow is hidden."""
    axis.add_feature(
        cfeature.LAND,
        facecolor="0.82",
        edgecolor="black",
        linewidth=0.55,
        zorder=7,
    )
    axis.coastlines(
        resolution="110m",
        color="black",
        linewidth=0.75,
        zorder=9,
    )



# 6. MAIN ANALYSIS AND FIGURE

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eastward_files = [
        resolve_current_file(path) for path in EASTWARD_CURRENT_FILES
    ]
    northward_files = [
        resolve_current_file(path) for path in NORTHWARD_CURRENT_FILES
    ]
    all_required_files = eastward_files + northward_files
    missing = [str(path) for path in all_required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ocean-current input file(s):\n" + "\n".join(missing)
        )

    print("All four current files found:")
    for path in all_required_files:
        print("  ", path)

    opened_datasets = []
    try:
        eastward, eastward_datasets, eastward_records = load_current_component(
            eastward_files,
            [
                "uo",
                "eastward_sea_water_velocity",
                "eastward_velocity",
                "u",
            ],
            "Eastward Current",
        )
        northward, northward_datasets, northward_records = load_current_component(
            northward_files,
            [
                "vo",
                "northward_sea_water_velocity",
                "northward_velocity",
                "v",
            ],
            "Northward Current",
        )
        opened_datasets.extend(eastward_datasets + northward_datasets)

        # Interpolate a staggered/different vo grid to the uo grid once.
        northward = align_to_reference_grid(
            northward, eastward, "Northward Current"
        )

        # Retain only calendar months present in both vector components.
        eastward, northward = xr.align(
            eastward,
            northward,
            join="inner",
        )
        if eastward.sizes.get("time", 0) == 0:
            raise RuntimeError("uo and vo have no overlapping calendar months.")

        # Physical cleaning in m s^-1.
        valid_vector = (
            np.isfinite(eastward)
            & np.isfinite(northward)
            & (np.abs(eastward) <= MAX_ABSOLUTE_CURRENT_MPS)
            & (np.abs(northward) <= MAX_ABSOLUTE_CURRENT_MPS)
        )
        eastward = eastward.where(valid_vector)
        northward = northward.where(valid_vector)

        # Calculate magnitude before spatial or seasonal averaging.
        speed = np.sqrt(eastward**2 + northward**2).astype("float32")
        speed.name = "Ocean Current Speed"
        speed.attrs["units"] = "m s-1"
        speed.attrs["definition"] = "sqrt(uo^2 + vo^2) at each grid cell"

        # Calculate speed first, then coarsen all fields identically for plotting.
        eastward = coarsen_to_resolution(
            eastward, TARGET_PLOT_RESOLUTION_DEG
        )
        northward = coarsen_to_resolution(
            northward, TARGET_PLOT_RESOLUTION_DEG
        )
        speed = coarsen_to_resolution(speed, TARGET_PLOT_RESOLUTION_DEG)

        seasonal_fields = {}
        summary_rows = []

        for season_name in SEASON_ORDER:
            months = SEASONS[season_name]["months"]
            abbreviation = SEASONS[season_name]["abbreviation"]
            print(f"\nCalculating {season_name} ({abbreviation})...")

            speed_mean, speed_years = seasonal_climatology(
                speed, months, f"{season_name} speed"
            )
            u_mean, u_years = seasonal_climatology(
                eastward, months, f"{season_name} uo"
            )
            v_mean, v_years = seasonal_climatology(
                northward, months, f"{season_name} vo"
            )

            if not (
                np.array_equal(speed_years, u_years)
                and np.array_equal(speed_years, v_years)
            ):
                raise RuntimeError(
                    f"Season-year mismatch among speed, uo and vo for "
                    f"{season_name}."
                )

            speed_mean = speed_mean.compute()
            u_mean = u_mean.compute()
            v_mean = v_mean.compute()

            seasonal_fields[season_name] = {
                "speed": speed_mean,
                "u": u_mean,
                "v": v_mean,
                "years": speed_years,
            }

            finite_speed = speed_mean.values[np.isfinite(speed_mean.values)]
            spatial_mean = float(
                area_weighted_spatial_mean(speed_mean).values
            )
            summary_rows.append(
                {
                    "Season": season_name,
                    "Months": abbreviation,
                    "First_complete_season_year": int(speed_years.min()),
                    "Last_complete_season_year": int(speed_years.max()),
                    "Number_of_complete_seasons": int(speed_years.size),
                    "Area_weighted_mean_speed_m_s": spatial_mean,
                    "Spatial_median_speed_m_s": float(
                        np.nanmedian(finite_speed)
                    ),
                    "Spatial_95th_percentile_speed_m_s": float(
                        np.nanpercentile(finite_speed, 95.0)
                    ),
                    "Spatial_maximum_speed_m_s": float(
                        np.nanmax(finite_speed)
                    ),
                }
            )

        # One common speed scale across all four panels.
        all_finite_speed = np.concatenate(
            [
                seasonal_fields[name]["speed"].values[
                    np.isfinite(seasonal_fields[name]["speed"].values)
                ]
                for name in SEASON_ORDER
            ]
        )
        if all_finite_speed.size == 0:
            raise RuntimeError("All seasonal current-speed fields are empty.")

        if COLOR_MAX_MPS is None:
            color_max = float(
                np.nanpercentile(all_finite_speed, COLOR_PERCENTILE)
            )
        else:
            color_max = float(COLOR_MAX_MPS)
        if not np.isfinite(color_max) or color_max <= 0.0:
            raise RuntimeError(f"Invalid shared color maximum: {color_max}")

        print(f"\nShared color scale: 0 to {color_max:.4f} m s^-1")
        color_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
            }
        )

        figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
        grid = GridSpec(
            nrows=2,
            ncols=3,
            figure=figure,
            width_ratios=[1.0, 1.0, 0.055],
            left=0.040,
            right=0.935,
            bottom=0.075,
            top=0.945,
            wspace=0.085,
            hspace=0.155,
        )
        panel_specs = [
            grid[0, 0],
            grid[0, 1],
            grid[1, 0],
            grid[1, 1],
        ]

        last_shading = None

        for panel_index, season_name in enumerate(SEASON_ORDER):
            abbreviation = SEASONS[season_name]["abbreviation"]
            fields = seasonal_fields[season_name]
            speed_field = fields["speed"]
            u_field = fields["u"]
            v_field = fields["v"]

            axis = make_polar_axis(figure, panel_specs[panel_index])

            last_shading = axis.pcolormesh(
                speed_field["lon"].values,
                speed_field["lat"].values,
                speed_field.values,
                transform=ccrs.PlateCarree(),
                cmap=COLOR_CMAP,
                norm=color_norm,
                shading="auto",
                rasterized=True,
                zorder=1,
            )

            u_stream, v_stream, valid_stream_cells = (
                prepare_streamline_components(
                    u_field,
                    v_field,
                    MIN_STREAM_SPEED_MPS,
                )
            )
            print(
                f"{season_name}: tracing streamlines from "
                f"{valid_stream_cells:,} supported grid cells "
                f"(minimum vector-mean speed="
                f"{MIN_STREAM_SPEED_MPS:.3f} m s^-1)."
            )

            # Cartopy transforms the eastward/northward components from the
            # geographic source grid into the South Polar Stereographic map
            # before tracing. Arrowheads along the continuous lines indicate
            # direction; their length or density must not be read as speed.
            axis.streamplot(
                u_stream["lon"].values,
                u_stream["lat"].values,
                u_stream.values,
                v_stream.values,
                transform=ccrs.PlateCarree(),
                density=STREAM_DENSITY,
                color=STREAM_COLOR,
                linewidth=STREAM_LINEWIDTH,
                arrowsize=STREAM_ARROWSIZE,
                arrowstyle=STREAM_ARROWSTYLE,
                minlength=STREAM_MINLENGTH,
                maxlength=STREAM_MAXLENGTH,
                integration_direction="both",
                broken_streamlines=True,
                zorder=6,
            )

            # Land is drawn after the streamlines. This hides any source-grid
            # values over Antarctica while leaving ocean flow unobstructed.
            add_land_and_coastline(axis)

            axis.set_title(
                f"{season_name} ({abbreviation})",
                fontsize=14,
                fontweight="bold",
                pad=11,
            )
            axis.text(
                -0.055,
                1.025,
                PANEL_LETTERS[panel_index],
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                zorder=30,
            )

        # A vertically centred colorbar with its own reserved column.
        colorbar_grid = grid[:, 2].subgridspec(
            3, 1, height_ratios=[1.25, 4.0, 1.25]
        )
        colorbar_axis = figure.add_subplot(colorbar_grid[1, 0])
        colorbar = figure.colorbar(
            last_shading,
            cax=colorbar_axis,
            extend="max",
        )
        colorbar.set_label(
            "Seasonal mean ocean-current speed (m s$^{-1}$)",
            fontsize=11,
            fontweight="bold",
            labelpad=9,
        )
        colorbar.ax.tick_params(labelsize=9, width=0.8)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")

        figure.text(
            0.475,
            0.024,
            (
                "Shading: mean current-speed magnitude   |   "
                "Streamlines: seasonal mean current direction   |   "
                f"Period: {START_YEAR}–{END_YEAR}"
            ),
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
        )

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

        pd.DataFrame(summary_rows).to_csv(OUT_STATS_CSV, index=False)
        pd.DataFrame(eastward_records + northward_records).to_csv(
            OUT_SOURCE_CSV, index=False
        )

        print("\nSaved outputs:")
        print("PNG:", OUT_PNG)
        print("PDF:", OUT_PDF)
        print("Seasonal statistics:", OUT_STATS_CSV)
        print("Current-source record:", OUT_SOURCE_CSV)

    finally:
        for dataset in opened_datasets:
            dataset.close()
        gc.collect()


if __name__ == "__main__":
    main()
