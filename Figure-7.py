# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip -q install cartopy scipy xarray dask netCDF4 h5netcdf
!pip -q install xarray dask scipy scikit-learn statsmodels h5netcdf netCDF4


# FIGURE 6 — FINAL SIX-PREDICTOR SEA-WISE REGRESSION

# This version DOES NOT recompute OHC from Argo profiles.
# It reads the already measured monthly OHC file and rebuilds the
# final six-predictor regression figure + complete diagnostics.
#
# FINAL MODEL (13 separate sea-wise models):
#   SIC anomaly ~ SST + Wind Speed + MLD + 2 m Air Temperature
#                 + OHC(0–100 m) + Ocean Current Speed
#
# Main Figure 6:
#   (a) Sea-wise standardized beta coefficients
#   (b) Coefficient-based relative importance
#   (c) Observed vs blocked-CV predicted SIC anomalies
#   (d) Sea-wise blocked cross-validated R²
#
# FINAL LAYOUT CORRECTION:
#   * The beta colorbar is positioned close to panel (a).
#   * Its label is placed on the left of the colorbar.
#   * A dedicated spacer prevents overlap with panel (c)'s y-axis label.
#
# Supplementary outputs:
#   - Baseline 5-predictor vs final 6-predictor blocked-CV comparison
#   - VIF by sea
#   - predictor correlation matrices
#   - bootstrap 95% CIs for standardized beta
#   - final-model beta/performance tables
#
# IMPORTANT:
#   * These are 13 separate sea-wise regressions, NOT one pooled model.
#   * Panel (c) combines out-of-fold predictions from those 13 models.
#   * Relative importance is normalized mean |beta|, NOT variance explained.
#
# Colab install (run once if needed):
# !pip -q install xarray dask scipy scikit-learn statsmodels h5netcdf netCDF4


import os
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")


# 1. PATHS

DATA_DIR = "/content/drive/MyDrive/SAM_Thesis/Data"
out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

# All new Figure 6 graphics, CSVs, caption, and diagnostics are saved here.
OUTPUT_DIR = os.path.join(out_dir, "P1_Corrected", "Fig6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Existing measured OHC inputs remain in their original folder.
OHC_INPUT_DIR = os.path.join(out_dir, "Fig5_OHC_supplementary")

sic_file = os.path.join(DATA_DIR, "OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc")
sst_file = os.path.join(DATA_DIR, "OSTIA_SST_monthly_2008_2025_SO.nc")
wind_file = os.path.join(DATA_DIR, "Wind_U_V_era.nc")
two_m_temp_file = os.path.join(DATA_DIR, "2mT_MSLP_era.nc")
mld_file = os.path.join(DATA_DIR, "MLD_monthly_2008_2025_SO.nc")

# Two time segments for each surface-ocean current component.
EASTWARD_CURRENT_FILES = [
    os.path.join(DATA_DIR, "EastwardSeaWaterVelocity_uo_2008_14 .nc"),
    os.path.join(DATA_DIR, "EastwardSeaWaterVelocity_uo_2015_25 .nc"),
]

NORTHWARD_CURRENT_FILES = [
    os.path.join(DATA_DIR, "NorthwardSeaWaterVelocity_uo_2008_14 .nc"),
    os.path.join(DATA_DIR, "NorthwardSeaWaterVelocity_uo_2015_25 .nc"),
]


def resolve_current_file(file_path):
    """Accept common current-file naming variants without guessing silently."""

    candidates = [
        file_path,
        file_path.replace(" .nc", ".nc"),
        file_path.replace(
            "NorthwardSeaWaterVelocity_uo_",
            "NorthwardSeaWaterVelocity_vo_",
        ),
        file_path.replace(
            "NorthwardSeaWaterVelocity_uo_",
            "NorthwardSeaWaterVelocity_vo_",
        ).replace(" .nc", ".nc"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return file_path


EASTWARD_CURRENT_FILES = [
    resolve_current_file(path)
    for path in EASTWARD_CURRENT_FILES
]

NORTHWARD_CURRENT_FILES = [
    resolve_current_file(path)
    for path in NORTHWARD_CURRENT_FILES
]


# SAVED / MEASURED OHC FILE
# This should be the wide monthly table created previously, with:
#   index/time + WED, KHV, RLS, LAZ, COS, COO, DAV, MAW, DUR,
#   SOM, ROS, AMU, BEL
# and values in GJ m^-2 for 0–100 m OHC above freezing.
OHC_CSV = os.path.join(
    OHC_INPUT_DIR,
    "Argo_monthly_OHC_0_100m_GJ_m2.csv"
)

# Optional profile-count file generated together with the measured OHC.
# If present, months with fewer than MIN_ARGO_PROFILES_PER_MONTH are masked.
OHC_COUNT_CSV = os.path.join(
    OHC_INPUT_DIR,
    "Argo_monthly_OHC_profile_counts.csv"
)

FIG_OUT = os.path.join(
    OUTPUT_DIR,
    "Fig6_Six_Predictor_Regression_CV_FINAL_1080dpi.png"
)

FIG_PDF = os.path.join(
    OUTPUT_DIR,
    "Fig6_Six_Predictor_Regression_CV_FINAL.pdf"
)

required_files = [
    sic_file,
    sst_file,
    wind_file,
    two_m_temp_file,
    mld_file,
    OHC_CSV,
] + EASTWARD_CURRENT_FILES + NORTHWARD_CURRENT_FILES

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing input file: {f}")


# 2. SETTINGS
START_YEAR = 2008
END_YEAR = 2025
TARGET_RES_DEG = 0.25

# OHC coverage threshold. Because OHC is already measured, this only masks
# low-profile-count months if the count file exists; it does NOT recompute OHC.
MIN_ARGO_PROFILES_PER_MONTH = 1

# Monthly anomaly climatology requirement
MIN_CLIM_OBS_PER_CALENDAR_MONTH = 3

# Existing five-predictor baseline and new six-predictor final model.
BASELINE_PREDICTORS = [
    "SST",
    "Wind Speed",
    "MLD",
    "Air Temperature",
    "OHC_0_100m",
]

FINAL_PREDICTORS = [
    "SST",
    "Wind Speed",
    "MLD",
    "Air Temperature",
    "OHC_0_100m",
    "Ocean Current Speed",
]

MIN_MODEL_SAMPLES = 36
N_CV_BLOCKS = 6
CV_GAP_MONTHS = 3

# Bootstrap confidence intervals
N_BOOTSTRAP = 500
RANDOM_SEED = 42


# 3. 13 ANTARCTIC MARGINAL SEAS
sea_info = [
    ("WED", "Weddell Sea",             -60,  -20),
    ("KHV", "King Haakon VII Sea",     -20,    0),
    ("RLS", "Riiser-Larsen Sea",         0,   10),
    ("LAZ", "Lazarev Sea",              10,   30),
    ("COS", "Cosmonauts Sea",           30,   50),
    ("COO", "Cooperation Sea",          50,   70),
    ("DAV", "Davis Sea",                70,   90),
    ("MAW", "Mawson Sea",               90,  130),
    ("DUR", "D'Urville Sea",            130,  150),
    ("SOM", "Somov Sea",                150,  170),
    ("ROS", "Ross Sea",                 170, -130),
    ("AMU", "Amundsen Sea",            -130, -100),
    ("BEL", "Bellingshausen Sea",      -100,  -60),
]

SEA_CODES = [x[0] for x in sea_info]
SEA_NAMES = {x[0]: x[1] for x in sea_info}


# 4. DATA HELPERS
def standardize_ds(ds):
    rename_dict = {}

    if "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "time" not in ds.coords:
        raise ValueError("No time coordinate found.")
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("No lat/lon coordinate found.")

    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(
            lon=(((ds["lon"] + 180) % 360) - 180)
        )

    ds = ds.sortby("lon").sortby("lat").sortby("time")
    ds = ds.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    ds = ds.where(ds["lat"] <= -60, drop=True)

    return ds


def find_var(ds, candidates):
    lower_lookup = {
        str(variable).lower(): variable
        for variable in ds.data_vars
    }

    for v in candidates:
        if v in ds.data_vars:
            return v

        if str(v).lower() in lower_lookup:
            return lower_lookup[str(v).lower()]

    raise ValueError(
        f"Variable not found. Tried {candidates}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def select_surface_current_layer(da, component_name):
    """Select the shallowest available ocean-current depth level."""

    depth_dimensions = [
        dimension
        for dimension in da.dims
        if str(dimension).lower() in {
            "depth",
            "depthu",
            "depthv",
            "lev",
            "level",
            "z",
        }
    ]

    if not depth_dimensions:
        print(f"{component_name}: no depth dimension; using supplied surface field.")
        return da

    depth_dimension = depth_dimensions[0]
    depth_values = np.asarray(da[depth_dimension].values, dtype=float)

    if depth_values.size == 0 or not np.isfinite(depth_values).any():
        raise ValueError(
            f"{component_name}: unusable depth coordinate {depth_dimension}."
        )

    surface_index = int(np.nanargmin(np.abs(depth_values)))
    selected_depth = float(depth_values[surface_index])

    print(
        f"{component_name}: selected shallowest {depth_dimension}="
        f"{selected_depth:g}."
    )

    return da.isel({depth_dimension: surface_index}, drop=True)


def load_current_component(file_paths, variable_candidates, component_name):
    """
    Load both 2008–2014 and 2015–2025 files for one current component.

    The files are concatenated by time, duplicate calendar months are averaged,
    and the returned list of open datasets is closed at the end of the script.
    """

    pieces = []
    opened_datasets = []
    variable_names = []

    for file_path in file_paths:
        print(f"Opening {component_name}: {file_path}")

        dataset = standardize_ds(
            xr.open_dataset(
                file_path,
                decode_times=True,
                decode_timedelta=False,
                chunks="auto",
            )
        )

        variable_name = find_var(dataset, variable_candidates)
        variable_names.append(variable_name)

        component = dataset[variable_name].astype("float32")
        component = select_surface_current_layer(component, component_name)

        # Remove any remaining singleton non-spatial dimensions.
        for dimension in list(component.dims):
            if dimension not in {"time", "lat", "lon"}:
                if component.sizes[dimension] == 1:
                    component = component.isel({dimension: 0}, drop=True)
                else:
                    raise ValueError(
                        f"{component_name}: unsupported remaining dimension "
                        f"{dimension} with size {component.sizes[dimension]}."
                    )

        monthly_time = month_start_index(component["time"].values)
        component = component.assign_coords(time=monthly_time)

        if monthly_time.duplicated().any():
            component = component.groupby("time").mean(skipna=True)

        # The two time segments must share one spatial grid before temporal
        # concatenation. Interpolate only if their coordinates genuinely
        # differ; identical grids are left untouched.
        if pieces:
            same_lat = np.array_equal(
                component["lat"].values,
                pieces[0]["lat"].values,
            )
            same_lon = np.array_equal(
                component["lon"].values,
                pieces[0]["lon"].values,
            )

            if not (same_lat and same_lon):
                print(
                    f"{component_name}: aligning {os.path.basename(file_path)} "
                    "to the first segment's grid."
                )
                component = component.interp(
                    lat=pieces[0]["lat"],
                    lon=pieces[0]["lon"],
                    method="linear",
                )

        pieces.append(component)
        opened_datasets.append(dataset)

    combined = xr.concat(
        pieces,
        dim="time",
        join="outer",
        combine_attrs="override",
    )

    combined = combined.sortby("time")

    combined_months = month_start_index(combined["time"].values)
    combined = combined.assign_coords(time=combined_months)

    if combined_months.duplicated().any():
        combined = combined.groupby("time").mean(skipna=True)

    combined = combined.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )

    combined.name = component_name

    print(
        f"{component_name}: variables={variable_names}, "
        f"months={combined.sizes.get('time', 0)}, "
        f"period={str(combined.time.min().values)[:10]} to "
        f"{str(combined.time.max().values)[:10]}"
    )

    return combined, opened_datasets, variable_names


def coarsen_to_target_resolution(da, target_res=0.25):
    lon = da["lon"].values
    lat = da["lat"].values

    lon_res = float(np.nanmedian(np.abs(np.diff(lon))))
    lat_res = float(np.nanmedian(np.abs(np.diff(lat))))

    lon_factor = max(1, int(round(target_res / lon_res)))
    lat_factor = max(1, int(round(target_res / lat_res)))

    print(
        f"{da.name}: grid={da.sizes['lat']}x{da.sizes['lon']} | "
        f"resolution≈({lat_res:.4f}, {lon_res:.4f})° | "
        f"coarsen factors=({lat_factor}, {lon_factor})"
    )

    if lat_factor > 1 or lon_factor > 1:
        da = da.coarsen(
            lat=lat_factor,
            lon=lon_factor,
            boundary="trim"
        ).mean(skipna=True)

    return da


def convert_sic_to_percent(da):
    sample = float(
        da.isel(time=slice(0, min(12, da.sizes["time"])))
        .max(skipna=True)
        .compute()
    )

    if sample <= 1.5:
        da = da * 100.0
        print("SIC converted from fraction to percent.")
    else:
        print("SIC already appears to be percent.")

    da = da.clip(min=0, max=100)
    da.name = "SIC"
    return da


def convert_kelvin_to_celsius_if_needed(da, name):
    sample = float(da.isel(time=0).mean(skipna=True).compute())

    if sample > 100:
        da = da - 273.15
        print(f"{name}: converted Kelvin to °C.")
    else:
        print(f"{name}: no Kelvin conversion needed.")

    da.name = name
    return da


def apply_common_ocean_mask(da, ocean_mask):
    """
    Apply one static SST-derived ocean mask to every gridded predictor.
    This removes Antarctic land cells from ERA5 wind and T2m sector means.
    """
    mask_float = ocean_mask.astype("float32")

    same_lat = np.array_equal(mask_float["lat"].values, da["lat"].values)
    same_lon = np.array_equal(mask_float["lon"].values, da["lon"].values)

    if not (same_lat and same_lon):
        mask_float = mask_float.interp(
            lat=da["lat"],
            lon=da["lon"],
            method="nearest"
        )

    return da.where(mask_float > 0.5)


def select_lon_sector(da, lon_min, lon_max):
    lon = da["lon"]

    if lon_min < lon_max:
        return da.where((lon >= lon_min) & (lon < lon_max), drop=True)

    # Ross Sea sector crosses the international date line
    return da.where((lon >= lon_min) | (lon < lon_max), drop=True)


def area_mean_da(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    weights = weights.where(np.isfinite(weights), 0)

    return da.weighted(weights).mean(
        dim=("lat", "lon"),
        skipna=True
    )


def month_start_index(time_values):
    return (
        pd.DatetimeIndex(pd.to_datetime(time_values))
        .to_period("M")
        .to_timestamp()
    )


def sea_wise_monthly_means(da, var_name):
    print(f"\nCalculating common-ocean-mask sea-wise monthly means: {var_name}")

    sea_series = {}

    for code, fullname, lon_min, lon_max in sea_info:
        sub = select_lon_sector(da, lon_min, lon_max)

        if sub.sizes.get("lon", 0) == 0 or sub.sizes.get("lat", 0) == 0:
            print(f"  WARNING {code}: empty sector")
            continue

        ts = area_mean_da(sub).astype("float32").compute()
        idx = month_start_index(ts["time"].values)

        s = pd.Series(
            ts.values,
            index=idx,
            name=code
        )

        s = s.groupby(level=0).mean().sort_index()
        sea_series[code] = s

        print(f"  {code}: valid months={int(s.notna().sum())}")
        gc.collect()

    full_index = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-01",
        freq="MS"
    )

    df = pd.DataFrame(sea_series).reindex(full_index)
    df = df.reindex(columns=SEA_CODES)
    df.index.name = "time"

    return df.replace([np.inf, -np.inf], np.nan)


def monthly_anomaly_df(df, min_clim_obs=3):
    """
    Remove each sea's calendar-month climatology.
    Missing months remain missing; no temporal interpolation is performed.
    """
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        s = df[col].astype(float)
        result = pd.Series(np.nan, index=s.index, dtype=float)

        for month in range(1, 13):
            mask = s.index.month == month
            vals = s.loc[mask]
            valid = vals.dropna()

            if len(valid) >= min_clim_obs:
                result.loc[mask] = vals - valid.mean()

        out[col] = result

    return out


# 5. LOAD SAVED / MEASURED OHC
def load_saved_ohc(ohc_csv, count_csv=None):
    print("\n============================================================")
    print("Loading previously measured Argo OHC 0–100 m...")
    print("============================================================")
    print("OHC file:", ohc_csv)

    raw = pd.read_csv(ohc_csv)

    # Robust time-column detection
    possible_time_cols = [
        c for c in raw.columns
        if str(c).lower() in ["time", "month", "date", "datetime", "unnamed: 0"]
    ]

    if len(possible_time_cols) == 0:
        # Assume first column is time if it is not one of the sea codes
        first_col = raw.columns[0]
        if first_col not in SEA_CODES:
            time_col = first_col
        else:
            raise ValueError(
                "Could not identify time column in saved OHC CSV. "
                f"Columns: {list(raw.columns)}"
            )
    else:
        time_col = possible_time_cols[0]

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).copy()
    raw[time_col] = raw[time_col].dt.to_period("M").dt.to_timestamp()

    # Wide-format file expected from previous OHC workflow
    missing_seas = [s for s in SEA_CODES if s not in raw.columns]

    if missing_seas:
        # Also support a long file with columns: time/month, sea, OHC...
        lower_map = {str(c).lower(): c for c in raw.columns}
        sea_col = lower_map.get("sea")

        value_candidates = [
            c for c in raw.columns
            if "ohc" in str(c).lower() and c != time_col
        ]

        if sea_col is not None and len(value_candidates) >= 1:
            value_col = value_candidates[0]
            ohc_df = raw.pivot_table(
                index=time_col,
                columns=sea_col,
                values=value_col,
                aggfunc="mean"
            )
        else:
            raise ValueError(
                "Saved OHC file is neither the expected wide 13-sea table "
                "nor a recognizable long OHC table.\n"
                f"Missing sea columns: {missing_seas}\n"
                f"Columns found: {list(raw.columns)}"
            )
    else:
        ohc_df = raw.set_index(time_col)[SEA_CODES].copy()

    # Ensure numeric values and monthly index
    for sea in SEA_CODES:
        ohc_df[sea] = pd.to_numeric(ohc_df[sea], errors="coerce")

    ohc_df = ohc_df.groupby(level=0).mean().sort_index()

    full_index = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-01",
        freq="MS"
    )

    ohc_df = ohc_df.reindex(index=full_index, columns=SEA_CODES)
    ohc_df.index.name = "time"

    # Optional profile-count filtering, if the saved count table is available.
    count_df = None

    if count_csv is not None and os.path.exists(count_csv):
        print("Profile-count file found; applying monthly count threshold:", count_csv)

        cnt_raw = pd.read_csv(count_csv)

        possible_time_cols = [
            c for c in cnt_raw.columns
            if str(c).lower() in ["time", "month", "date", "datetime", "unnamed: 0"]
        ]

        if len(possible_time_cols) == 0:
            cnt_time_col = cnt_raw.columns[0]
        else:
            cnt_time_col = possible_time_cols[0]

        cnt_raw[cnt_time_col] = pd.to_datetime(cnt_raw[cnt_time_col], errors="coerce")
        cnt_raw = cnt_raw.dropna(subset=[cnt_time_col])
        cnt_raw[cnt_time_col] = cnt_raw[cnt_time_col].dt.to_period("M").dt.to_timestamp()

        available_seas = [s for s in SEA_CODES if s in cnt_raw.columns]

        if len(available_seas) == len(SEA_CODES):
            count_df = cnt_raw.set_index(cnt_time_col)[SEA_CODES].copy()

            for sea in SEA_CODES:
                count_df[sea] = pd.to_numeric(count_df[sea], errors="coerce")

            count_df = count_df.groupby(level=0).mean().sort_index()
            count_df = count_df.reindex(index=full_index, columns=SEA_CODES)

            ohc_df = ohc_df.where(count_df >= MIN_ARGO_PROFILES_PER_MONTH)
        else:
            print(
                "WARNING: profile-count file does not contain all 13 sea columns; "
                "count filtering skipped."
            )
    else:
        print("No profile-count file used; keeping all finite measured OHC months.")

    print("\nMeasured OHC coverage after loading/filtering:")

    coverage_rows = []

    for sea in SEA_CODES:
        n = int(ohc_df[sea].notna().sum())
        pct = 100.0 * n / len(full_index)

        coverage_rows.append({
            "Sea": sea,
            "Name": SEA_NAMES[sea],
            "Valid_OHC_months": n,
            "Total_months": len(full_index),
            "Coverage_percent": pct,
        })

        print(f"  {sea}: {n}/{len(full_index)} months ({pct:.1f}%)")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(
        os.path.join(OUTPUT_DIR, "Loaded_OHC_coverage_summary.csv"),
        index=False
    )

    return ohc_df, count_df, coverage_df


# 6. REGRESSION HELPERS
def fit_standardized_full(df, predictors):
    valid = df[["SIC"] + predictors].dropna().copy()

    if len(valid) < MIN_MODEL_SAMPLES:
        return None

    X = valid[predictors].values.astype(float)
    y = valid["SIC"].values.astype(float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xz = x_scaler.fit_transform(X)
    yz = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    model = LinearRegression().fit(Xz, yz)

    return {
        "beta": model.coef_.copy(),
        "n": len(valid),
        "index": valid.index,
    }


def blocked_cv_predictions(df, predictors):
    """
    Contiguous-year blocked cross-validation.

    Each fold holds out one contiguous block of years. A temporal gap is
    excluded from training around the held-out interval. Scaling is fitted
    using the training data only inside each fold.
    """
    valid = df[["SIC"] + predictors].dropna().sort_index().copy()

    if len(valid) < MIN_MODEL_SAMPLES:
        return None

    years = np.array(sorted(valid.index.year.unique()))

    if len(years) < 4:
        return None

    n_blocks = min(N_CV_BLOCKS, len(years))
    year_blocks = [b for b in np.array_split(years, n_blocks) if len(b) > 0]

    pred = pd.Series(np.nan, index=valid.index, dtype=float)
    fold_id = pd.Series(np.nan, index=valid.index, dtype=float)

    for fold, test_years in enumerate(year_blocks, start=1):
        test_mask = valid.index.year.isin(test_years)
        test = valid.loc[test_mask]

        if len(test) == 0:
            continue

        test_start = test.index.min()
        test_end = test.index.max()

        gap_start = test_start - pd.DateOffset(months=CV_GAP_MONTHS)
        gap_end = test_end + pd.DateOffset(months=CV_GAP_MONTHS)

        train = valid.loc[~test_mask].copy()
        train = train.loc[
            (train.index < gap_start) |
            (train.index > gap_end)
        ]

        min_train = max(24, 5 * len(predictors))

        if len(train) < min_train:
            continue

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = x_scaler.fit_transform(train[predictors].values)
        y_train_z = y_scaler.fit_transform(train[["SIC"]].values).ravel()

        model = LinearRegression().fit(X_train, y_train_z)

        X_test = x_scaler.transform(test[predictors].values)
        y_test_z_pred = model.predict(X_test)
        y_test_pred = y_scaler.inverse_transform(
            y_test_z_pred.reshape(-1, 1)
        ).ravel()

        pred.loc[test.index] = y_test_pred
        fold_id.loc[test.index] = fold

    good = pred.notna()

    if good.sum() < max(20, len(predictors) * 4):
        return None

    obs = valid.loc[good, "SIC"]
    prd = pred.loc[good]

    return {
        "observed": obs,
        "predicted": prd,
        "fold": fold_id.loc[good],
        "r2": r2_score(obs, prd),
        "rmse": np.sqrt(mean_squared_error(obs, prd)),
        "mae": mean_absolute_error(obs, prd),
        "n_oof": int(good.sum()),
        "n_total": len(valid),
    }


def calculate_vif(df, predictors):
    valid = df[predictors].dropna().copy()

    if len(valid) < max(20, 4 * len(predictors)):
        return pd.DataFrame(columns=["Predictor", "VIF"])

    X = sm.add_constant(
        valid[predictors].astype(float),
        has_constant="add"
    )

    rows = []

    for i, predictor in enumerate(predictors, start=1):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan

        rows.append({
            "Predictor": predictor,
            "VIF": vif
        })

    return pd.DataFrame(rows)


def block_bootstrap_beta(df, predictors, n_boot=500, seed=42):
    """
    Resample complete calendar years with replacement to preserve
    within-year monthly dependence while estimating beta uncertainty.
    """
    valid = df[["SIC"] + predictors].dropna().sort_index().copy()

    if len(valid) < MIN_MODEL_SAMPLES:
        return None

    years = np.array(sorted(valid.index.year.unique()))

    if len(years) < 4:
        return None

    rng = np.random.default_rng(seed)
    betas = []

    for _ in range(n_boot):
        sampled_years = rng.choice(
            years,
            size=len(years),
            replace=True
        )

        parts = [
            valid.loc[valid.index.year == y]
            for y in sampled_years
        ]

        boot = pd.concat(parts, axis=0, ignore_index=True)

        if len(boot) < MIN_MODEL_SAMPLES:
            continue

        X = boot[predictors].values.astype(float)
        y = boot["SIC"].values.astype(float)

        try:
            Xz = StandardScaler().fit_transform(X)
            yz = StandardScaler().fit_transform(
                y.reshape(-1, 1)
            ).ravel()

            beta = LinearRegression().fit(Xz, yz).coef_

            if np.all(np.isfinite(beta)):
                betas.append(beta)

        except Exception:
            pass

    if len(betas) < max(50, n_boot * 0.25):
        return None

    arr = np.asarray(betas)

    return {
        "mean": np.nanmean(arr, axis=0),
        "low": np.nanpercentile(arr, 2.5, axis=0),
        "high": np.nanpercentile(arr, 97.5, axis=0),
        "n_boot_valid": len(arr),
    }


# 7. LOAD MEASURED OHC

ohc_df, ohc_count_df, ohc_coverage = load_saved_ohc(
    OHC_CSV,
    OHC_COUNT_CSV
)


# 8. OPEN / PREPARE GRIDDED VARIABLES

print("\nOpening gridded datasets...")

ds_sic = standardize_ds(
    xr.open_dataset(sic_file, decode_times=True, chunks={"time": 12})
)
ds_sst = standardize_ds(
    xr.open_dataset(sst_file, decode_times=True, chunks={"time": 12})
)
ds_wind = standardize_ds(
    xr.open_dataset(wind_file, decode_times=True, chunks={"time": 12})
)
ds_t2m = standardize_ds(
    xr.open_dataset(two_m_temp_file, decode_times=True, chunks={"time": 12})
)
ds_mld = standardize_ds(
    xr.open_dataset(mld_file, decode_times=True, chunks={"time": 12})
)

eastward_current, eastward_current_datasets, eastward_variable_names = (
    load_current_component(
        EASTWARD_CURRENT_FILES,
        [
            "uo",
            "eastward_sea_water_velocity",
            "eastward_velocity",
            "u",
        ],
        "Eastward Current",
    )
)

northward_current, northward_current_datasets, northward_variable_names = (
    load_current_component(
        NORTHWARD_CURRENT_FILES,
        [
            "vo",
            "northward_sea_water_velocity",
            "northward_velocity",
            "v",
        ],
        "Northward Current",
    )
)

# Put uo and vo on the same grid and retain only their common months before
# calculating scalar speed. This calculation is performed at every grid cell:
#     current speed = sqrt(uo**2 + vo**2)
# The grid-cell speeds are subsequently area-averaged by marginal sea. This is
# not sqrt(mean(uo)**2 + mean(vo)**2), which would underestimate speed where
# current direction varies spatially.
same_current_lat = np.array_equal(
    eastward_current["lat"].values,
    northward_current["lat"].values,
)
same_current_lon = np.array_equal(
    eastward_current["lon"].values,
    northward_current["lon"].values,
)

if not (same_current_lat and same_current_lon):
    print(
        "Northward Current: interpolating to the eastward-current grid "
        "before speed calculation."
    )
    northward_current = northward_current.interp(
        lat=eastward_current["lat"],
        lon=eastward_current["lon"],
        method="linear",
    )

eastward_current, northward_current = xr.align(
    eastward_current,
    northward_current,
    join="inner",
)

ocean_current_speed = np.sqrt(
    eastward_current ** 2 + northward_current ** 2
).astype("float32")
ocean_current_speed.name = "Ocean Current Speed"
ocean_current_speed.attrs["long_name"] = "surface ocean current speed"
ocean_current_speed.attrs["units"] = eastward_current.attrs.get(
    "units",
    northward_current.attrs.get("units", "native current units"),
)

print(
    "Ocean Current Speed calculated from all four files: "
    "sqrt(uo² + vo²)."
)
print(
    "Ocean Current Speed common months:",
    ocean_current_speed.sizes.get("time", 0),
)

sic_var = find_var(
    ds_sic,
    ["SIF", "sic", "SIC", "sea_ice_fraction", "siconc", "ice_conc"]
)
sst_var = find_var(
    ds_sst,
    ["SST", "sst", "analysed_sst", "thetao"]
)
mld_var = find_var(
    ds_mld,
    ["MLD", "mld", "mlotst", "mlotst_mean"]
)
u10_var = find_var(
    ds_wind,
    ["u10", "U10", "u", "u10m", "eastward_wind"]
)
v10_var = find_var(
    ds_wind,
    ["v10", "V10", "v", "v10m", "northward_wind"]
)
t2m_var = find_var(
    ds_t2m,
    ["t2m", "T2M", "2m_temperature", "air_temperature", "t"]
)

print("\nVariables used:")
print("SIC:", sic_var)
print("SST:", sst_var)
print("MLD:", mld_var)
print("U10:", u10_var)
print("V10:", v10_var)
print("T2m:", t2m_var)
print("Eastward-current variables:", eastward_variable_names)
print("Northward-current variables:", northward_variable_names)

sic = convert_sic_to_percent(
    ds_sic[sic_var].astype("float32")
)

sst = convert_kelvin_to_celsius_if_needed(
    ds_sst[sst_var].astype("float32"),
    "SST"
)

mld = ds_mld[mld_var].astype("float32")
mld.name = "MLD"

u10 = ds_wind[u10_var].astype("float32")
v10 = ds_wind[v10_var].astype("float32")
wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
wind_speed.name = "Wind Speed"

t2m = convert_kelvin_to_celsius_if_needed(
    ds_t2m[t2m_var].astype("float32"),
    "Air Temperature"
)

# Coarsen before sector averaging
sic = coarsen_to_target_resolution(sic, TARGET_RES_DEG)
sst = coarsen_to_target_resolution(sst, TARGET_RES_DEG)
mld = coarsen_to_target_resolution(mld, TARGET_RES_DEG)
wind_speed = coarsen_to_target_resolution(wind_speed, TARGET_RES_DEG)
t2m = coarsen_to_target_resolution(t2m, TARGET_RES_DEG)
ocean_current_speed = coarsen_to_target_resolution(
    ocean_current_speed,
    TARGET_RES_DEG,
)
eastward_current = coarsen_to_target_resolution(
    eastward_current,
    TARGET_RES_DEG,
)
northward_current = coarsen_to_target_resolution(
    northward_current,
    TARGET_RES_DEG,
)

# Common ocean mask from valid OSTIA SST cells.
# This removes ERA5 Antarctic land cells from wind/T2m means.
ocean_mask = np.isfinite(sst).any("time")

sic = apply_common_ocean_mask(sic, ocean_mask)
sst = apply_common_ocean_mask(sst, ocean_mask)
mld = apply_common_ocean_mask(mld, ocean_mask)
wind_speed = apply_common_ocean_mask(wind_speed, ocean_mask)
t2m = apply_common_ocean_mask(t2m, ocean_mask)
ocean_current_speed = apply_common_ocean_mask(
    ocean_current_speed,
    ocean_mask,
)
eastward_current = apply_common_ocean_mask(eastward_current, ocean_mask)
northward_current = apply_common_ocean_mask(northward_current, ocean_mask)

print("\nCommon SST-derived ocean mask applied to all gridded variables.")


# 9. SEA-WISE MONTHLY MEANS + MONTHLY ANOMALIES

sic_df = sea_wise_monthly_means(sic, "SIC")
sst_df = sea_wise_monthly_means(sst, "SST")
wind_df = sea_wise_monthly_means(wind_speed, "Wind Speed")
mld_df = sea_wise_monthly_means(mld, "MLD")
t2m_df = sea_wise_monthly_means(t2m, "Air Temperature")
ocean_current_speed_df = sea_wise_monthly_means(
    ocean_current_speed,
    "Ocean Current Speed",
)
eastward_current_df = sea_wise_monthly_means(
    eastward_current,
    "Eastward Current",
)
northward_current_df = sea_wise_monthly_means(
    northward_current,
    "Northward Current",
)

master_index = sic_df.index
ohc_df = ohc_df.reindex(master_index)
ocean_current_speed_df = ocean_current_speed_df.reindex(master_index)
eastward_current_df = eastward_current_df.reindex(master_index)
northward_current_df = northward_current_df.reindex(master_index)

sic_anom = monthly_anomaly_df(
    sic_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
sst_anom = monthly_anomaly_df(
    sst_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
wind_anom = monthly_anomaly_df(
    wind_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
mld_anom = monthly_anomaly_df(
    mld_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
t2m_anom = monthly_anomaly_df(
    t2m_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
ohc_anom = monthly_anomaly_df(
    ohc_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH
)
ocean_current_speed_anom = monthly_anomaly_df(
    ocean_current_speed_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH,
)
eastward_current_anom = monthly_anomaly_df(
    eastward_current_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH,
)
northward_current_anom = monthly_anomaly_df(
    northward_current_df,
    MIN_CLIM_OBS_PER_CALENDAR_MONTH,
)

ohc_anom.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Loaded_Argo_OHC_0_100m_monthly_anomalies_GJ_m2.csv"
    )
)

ocean_current_speed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Ocean_Current_Speed_seawise_monthly_2008_2025.csv",
    )
)

ocean_current_speed_anom.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Ocean_Current_Speed_seawise_monthly_anomalies_2008_2025.csv",
    )
)

eastward_current_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Eastward_Current_seawise_monthly_2008_2025.csv",
    )
)

northward_current_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Northward_Current_seawise_monthly_2008_2025.csv",
    )
)

eastward_current_anom.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Eastward_Current_seawise_monthly_anomalies_2008_2025.csv",
    )
)

northward_current_anom.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Northward_Current_seawise_monthly_anomalies_2008_2025.csv",
    )
)

# One table containing every raw sea-wise monthly series used by the analysis.
# This is deliberately long-format so it can be used directly in a
# supplementary table, R, Python, or spreadsheet pivot table.
all_monthly_frames = []

for sea in SEA_CODES:
    monthly_frame = pd.DataFrame(
        {
            "time": master_index,
            "Sea": sea,
            "Sea_name": SEA_NAMES[sea],
            "SIC_percent": sic_df[sea].reindex(master_index).values,
            "SST_degC": sst_df[sea].reindex(master_index).values,
            "Wind_speed_m_s": wind_df[sea].reindex(master_index).values,
            "MLD_m": mld_df[sea].reindex(master_index).values,
            "Air_temperature_degC": t2m_df[sea].reindex(master_index).values,
            "OHC_0_100m_GJ_m2": ohc_df[sea].reindex(master_index).values,
            "Ocean_current_speed_native_units": (
                ocean_current_speed_df[sea].reindex(master_index).values
            ),
            "Eastward_current_native_units": (
                eastward_current_df[sea].reindex(master_index).values
            ),
            "Northward_current_native_units": (
                northward_current_df[sea].reindex(master_index).values
            ),
        }
    )
    all_monthly_frames.append(monthly_frame)

all_variables_monthly_df = pd.concat(
    all_monthly_frames,
    ignore_index=True,
)

all_variables_monthly_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "All_variables_seawise_monthly_means_2008_2025_long.csv",
    ),
    index=False,
)

current_coverage_rows = []

for sea in SEA_CODES:
    current_coverage_rows.append(
        {
            "Sea": sea,
            "Sea_name": SEA_NAMES[sea],
            "Expected_months": len(master_index),
            "Eastward_current_valid_months": int(
                eastward_current_df[sea].notna().sum()
            ),
            "Northward_current_valid_months": int(
                northward_current_df[sea].notna().sum()
            ),
            "Ocean_current_speed_valid_months": int(
                ocean_current_speed_df[sea].notna().sum()
            ),
            "Both_current_components_valid_months": int(
                (
                    eastward_current_df[sea].notna()
                    & northward_current_df[sea].notna()
                ).sum()
            ),
        }
    )

pd.DataFrame(current_coverage_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Ocean_current_monthly_coverage_by_sea.csv",
    ),
    index=False,
)

current_source_rows = []

for component_name, file_paths, variable_names in [
    (
        "Eastward Current",
        EASTWARD_CURRENT_FILES,
        eastward_variable_names,
    ),
    (
        "Northward Current",
        NORTHWARD_CURRENT_FILES,
        northward_variable_names,
    ),
]:
    for file_path, variable_name in zip(file_paths, variable_names):
        if component_name == "Eastward Current":
            component_units = eastward_current.attrs.get("units", "not provided")
        else:
            component_units = northward_current.attrs.get("units", "not provided")

        current_source_rows.append(
            {
                "Component": component_name,
                "Input_file": file_path,
                "NetCDF_variable": variable_name,
                "Units_attribute": component_units,
                "Depth_selection": "shallowest available level",
            }
        )

pd.DataFrame(current_source_rows).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Ocean_current_input_files_and_variables.csv",
    ),
    index=False,
)

model_specification_df = pd.DataFrame(
    [
        {
            "Model": "Baseline_5_predictor",
            "Response": "Monthly SIC anomaly (%)",
            "Predictor_count": len(BASELINE_PREDICTORS),
            "Predictors": " | ".join(BASELINE_PREDICTORS),
            "Sample_rule": "Complete rows for all final-six variables",
        },
        {
            "Model": "Final_6_predictor",
            "Response": "Monthly SIC anomaly (%)",
            "Predictor_count": len(FINAL_PREDICTORS),
            "Predictors": " | ".join(FINAL_PREDICTORS),
            "Sample_rule": "Complete rows for all final-six variables",
        },
    ]
)

model_specification_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Fig6_model_specification_and_predictors.csv",
    ),
    index=False,
)


# 10. FIT FINAL MODEL + SUPPLEMENTARY DIAGNOSTICS

beta_final = pd.DataFrame(
    index=SEA_CODES,
    columns=FINAL_PREDICTORS,
    dtype=float
)

cv_r2_final = pd.Series(index=SEA_CODES, dtype=float)
cv_rmse_final = pd.Series(index=SEA_CODES, dtype=float)
cv_mae_final = pd.Series(index=SEA_CODES, dtype=float)
cv_n_final = pd.Series(index=SEA_CODES, dtype=float)

all_oof_obs = []
all_oof_pred = []
final_oof_frames = []

baseline_oof_obs = []
baseline_oof_pred = []
baseline_oof_frames = []

comparison_rows = []
vif_rows = []
corr_rows = []
ci_rows = []
model_input_frames = []

for sea in SEA_CODES:
    print("\n============================================================")
    print(f"Regression: {sea} — {SEA_NAMES[sea]}")
    print("============================================================")

    df_model = pd.DataFrame({
        "SIC": sic_anom[sea],
        "SST": sst_anom[sea],
        "Wind Speed": wind_anom[sea],
        "MLD": mld_anom[sea],
        "Air Temperature": t2m_anom[sea],
        "OHC_0_100m": ohc_anom[sea],
        "Ocean Current Speed": ocean_current_speed_anom[sea],
    }).sort_index()

    export_frame = df_model.copy()
    export_frame.insert(0, "Sea", sea)
    export_frame.insert(1, "Sea_name", SEA_NAMES[sea])
    export_frame.index.name = "time"
    model_input_frames.append(export_frame.reset_index())

    # Use identical complete rows for baseline-five versus final-six
    # comparison. This prevents changes in data availability from being
    # mistaken for changes in model skill.
    common = df_model[["SIC"] + FINAL_PREDICTORS].dropna().copy()

    print(f"Complete OHC-and-current-matched months: {len(common)}")

    if len(common) < MIN_MODEL_SAMPLES:
        print(
            f"WARNING: {sea} has only {len(common)} complete months; "
            f"minimum required={MIN_MODEL_SAMPLES}. Final model skipped."
        )

        comparison_rows.append({
            "Sea": sea,
            "N_common": len(common),
            "Baseline5_CV_R2": np.nan,
            "Baseline5_CV_RMSE": np.nan,
            "Baseline5_CV_MAE": np.nan,
            "Final6_CV_R2": np.nan,
            "Final6_CV_RMSE": np.nan,
            "Final6_CV_MAE": np.nan,
            "Delta_R2_Final6_minus_Baseline5": np.nan,
            "Delta_RMSE_Final6_minus_Baseline5": np.nan,
        })

        continue

    # Full standardized coefficients for panel (a)
    full_final = fit_standardized_full(
        common,
        FINAL_PREDICTORS
    )

    if full_final is not None:
        beta_final.loc[sea, FINAL_PREDICTORS] = full_final["beta"]

    # Blocked temporal cross-validation
    cv_baseline = blocked_cv_predictions(
        common,
        BASELINE_PREDICTORS
    )

    cv_final = blocked_cv_predictions(
        common,
        FINAL_PREDICTORS
    )

    if cv_final is not None:
        cv_r2_final.loc[sea] = cv_final["r2"]
        cv_rmse_final.loc[sea] = cv_final["rmse"]
        cv_mae_final.loc[sea] = cv_final["mae"]
        cv_n_final.loc[sea] = cv_final["n_oof"]

        all_oof_obs.append(cv_final["observed"].values)
        all_oof_pred.append(cv_final["predicted"].values)
        final_oof_frames.append(
            pd.DataFrame(
                {
                    "time": cv_final["observed"].index,
                    "Sea": sea,
                    "Sea_name": SEA_NAMES[sea],
                    "Model": "Final_6_predictor",
                    "Observed_SIC_anomaly": cv_final["observed"].values,
                    "Predicted_SIC_anomaly": cv_final["predicted"].values,
                    "CV_fold": cv_final["fold"].values,
                }
            )
        )

        print(
            f"Final model blocked-CV: "
            f"R²={cv_final['r2']:.3f}, "
            f"RMSE={cv_final['rmse']:.3f}, "
            f"MAE={cv_final['mae']:.3f}, "
            f"N={cv_final['n_oof']}"
        )
    else:
        print("Final-model CV could not be calculated.")

    if cv_baseline is not None:
        baseline_oof_obs.append(cv_baseline["observed"].values)
        baseline_oof_pred.append(cv_baseline["predicted"].values)
        baseline_oof_frames.append(
            pd.DataFrame(
                {
                    "time": cv_baseline["observed"].index,
                    "Sea": sea,
                    "Sea_name": SEA_NAMES[sea],
                    "Model": "Baseline_5_predictor",
                    "Observed_SIC_anomaly": cv_baseline["observed"].values,
                    "Predicted_SIC_anomaly": cv_baseline["predicted"].values,
                    "CV_fold": cv_baseline["fold"].values,
                }
            )
        )

        print(
            f"Baseline five-predictor model blocked-CV: "
            f"R²={cv_baseline['r2']:.3f}, "
            f"RMSE={cv_baseline['rmse']:.3f}, "
            f"MAE={cv_baseline['mae']:.3f}, "
            f"N={cv_baseline['n_oof']}"
        )

    comparison_rows.append({
        "Sea": sea,
        "N_common": len(common),
        "Baseline5_CV_R2": cv_baseline["r2"] if cv_baseline else np.nan,
        "Baseline5_CV_RMSE": cv_baseline["rmse"] if cv_baseline else np.nan,
        "Baseline5_CV_MAE": cv_baseline["mae"] if cv_baseline else np.nan,
        "Final6_CV_R2": cv_final["r2"] if cv_final else np.nan,
        "Final6_CV_RMSE": cv_final["rmse"] if cv_final else np.nan,
        "Final6_CV_MAE": cv_final["mae"] if cv_final else np.nan,
        "Delta_R2_Final6_minus_Baseline5": (
            cv_final["r2"] - cv_baseline["r2"]
            if (cv_baseline is not None and cv_final is not None)
            else np.nan
        ),
        "Delta_RMSE_Final6_minus_Baseline5": (
            cv_final["rmse"] - cv_baseline["rmse"]
            if (cv_baseline is not None and cv_final is not None)
            else np.nan
        ),
    })

    # VIF for baseline and final models using identical rows.
    for model_name, predictors in [
        ("Baseline_5_predictor", BASELINE_PREDICTORS),
        ("Final_6_predictor", FINAL_PREDICTORS),
    ]:
        vif = calculate_vif(common, predictors)

        for _, row in vif.iterrows():
            vif_rows.append({
                "Sea": sea,
                "Model": model_name,
                "Predictor": row["Predictor"],
                "VIF": row["VIF"],
                "N": len(common),
            })

    # Predictor correlation matrix for final model
    corr = common[FINAL_PREDICTORS].corr()

    for v1 in FINAL_PREDICTORS:
        for v2 in FINAL_PREDICTORS:
            corr_rows.append({
                "Sea": sea,
                "Variable_1": v1,
                "Variable_2": v2,
                "r": corr.loc[v1, v2],
                "N": len(common),
            })

    # Year-block bootstrap beta confidence intervals
    for model_name, predictors in [
        ("Baseline_5_predictor", BASELINE_PREDICTORS),
        ("Final_6_predictor", FINAL_PREDICTORS),
    ]:
        ci = block_bootstrap_beta(
            common,
            predictors,
            n_boot=N_BOOTSTRAP,
            seed=RANDOM_SEED + SEA_CODES.index(sea)
        )

        if ci is not None:
            for k, predictor in enumerate(predictors):
                ci_rows.append({
                    "Sea": sea,
                    "Model": model_name,
                    "Predictor": predictor,
                    "Bootstrap_mean_beta": ci["mean"][k],
                    "CI_2.5": ci["low"][k],
                    "CI_97.5": ci["high"][k],
                    "CI_excludes_zero": bool(
                        (ci["low"][k] > 0) or
                        (ci["high"][k] < 0)
                    ),
                    "N_boot_valid": ci["n_boot_valid"],
                    "N_common": len(common),
                })


# 11. SAVE SUPPLEMENTARY TABLES

comparison_df = pd.DataFrame(comparison_rows)
vif_df = pd.DataFrame(vif_rows)
corr_df = pd.DataFrame(corr_rows)
ci_df = pd.DataFrame(ci_rows)
model_input_long_df = pd.concat(model_input_frames, ignore_index=True)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Baseline5_vs_Final6_blocked_CV_identical_months.csv"
    ),
    index=False
)

vif_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "VIF_by_sea_Baseline5_and_Final6.csv"
    ),
    index=False
)

corr_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final6_predictor_correlation_matrix_long_by_sea.csv"
    ),
    index=False
)

ci_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Bootstrap_beta_95CI_by_sea_Baseline5_and_Final6.csv"
    ),
    index=False
)

beta_final.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final6_standardized_beta_by_sea.csv"
    )
)

performance_final = pd.DataFrame({
    "CV_R2": cv_r2_final,
    "CV_RMSE": cv_rmse_final,
    "CV_MAE": cv_mae_final,
    "N_OOF": cv_n_final,
})

performance_final.index.name = "Sea"

performance_final.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final6_blocked_CV_performance_by_sea.csv"
    )
)

model_input_long_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final6_model_input_monthly_anomalies_all_seas_long.csv",
    ),
    index=False,
)


# 12. COMBINED OUT-OF-FOLD PERFORMANCE

if len(all_oof_obs) == 0:
    raise RuntimeError(
        "No sea produced enough complete OHC-and-current-matched observations "
        "for blocked CV. Check OHC/current coverage and MIN_MODEL_SAMPLES."
    )

all_oof_obs = np.concatenate(all_oof_obs)
all_oof_pred = np.concatenate(all_oof_pred)

if len(baseline_oof_obs) == 0:
    raise RuntimeError(
        "No sea produced baseline-model out-of-fold predictions on the "
        "common six-predictor sample."
    )

baseline_oof_obs = np.concatenate(baseline_oof_obs)
baseline_oof_pred = np.concatenate(baseline_oof_pred)

combined_cv_r2 = r2_score(
    all_oof_obs,
    all_oof_pred
)

combined_cv_rmse = np.sqrt(
    mean_squared_error(
        all_oof_obs,
        all_oof_pred
    )
)

combined_cv_mae = mean_absolute_error(
    all_oof_obs,
    all_oof_pred
)

baseline_combined_cv_r2 = r2_score(
    baseline_oof_obs,
    baseline_oof_pred,
)
baseline_combined_cv_rmse = np.sqrt(
    mean_squared_error(
        baseline_oof_obs,
        baseline_oof_pred,
    )
)
baseline_combined_cv_mae = mean_absolute_error(
    baseline_oof_obs,
    baseline_oof_pred,
)

print("\n============================================================")
print("COMBINED OUT-OF-FOLD FINAL-MODEL PERFORMANCE")
print("(combined predictions from 13 separate sea-wise models)")
print("============================================================")
print(f"R²   = {combined_cv_r2:.3f}")
print(f"RMSE = {combined_cv_rmse:.3f} SIC percentage points")
print(f"MAE  = {combined_cv_mae:.3f} SIC percentage points")
print(f"N    = {len(all_oof_obs)}")

print("\nCOMBINED OUT-OF-FOLD BASELINE FIVE-PREDICTOR PERFORMANCE")
print(f"R²   = {baseline_combined_cv_r2:.3f}")
print(f"RMSE = {baseline_combined_cv_rmse:.3f} SIC percentage points")
print(f"MAE  = {baseline_combined_cv_mae:.3f} SIC percentage points")
print(f"N    = {len(baseline_oof_obs)}")

combined_performance_df = pd.DataFrame(
    [
        {
            "Model": "Baseline_5_predictor",
            "Combined_OOF_R2": baseline_combined_cv_r2,
            "Combined_OOF_RMSE": baseline_combined_cv_rmse,
            "Combined_OOF_MAE": baseline_combined_cv_mae,
            "Combined_OOF_N": len(baseline_oof_obs),
            "Predictors": " | ".join(BASELINE_PREDICTORS),
            "CV_blocks": N_CV_BLOCKS,
            "CV_gap_months": CV_GAP_MONTHS,
        },
        {
            "Model": "Final_6_predictor",
            "Combined_OOF_R2": combined_cv_r2,
            "Combined_OOF_RMSE": combined_cv_rmse,
            "Combined_OOF_MAE": combined_cv_mae,
            "Combined_OOF_N": len(all_oof_obs),
            "Predictors": " | ".join(FINAL_PREDICTORS),
            "CV_blocks": N_CV_BLOCKS,
            "CV_gap_months": CV_GAP_MONTHS,
        }
    ]
)

combined_performance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Baseline5_vs_Final6_combined_out_of_fold_performance.csv",
    ),
    index=False,
)

oof_predictions_df = pd.concat(
    baseline_oof_frames + final_oof_frames,
    ignore_index=True,
)
oof_predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Baseline5_and_Final6_all_out_of_fold_predictions.csv",
    ),
    index=False,
)


# 13. COEFFICIENT-BASED RELATIVE IMPORTANCE

beta_values = beta_final.loc[
    SEA_CODES,
    FINAL_PREDICTORS
].astype(float).values

mean_abs_beta = np.nanmean(
    np.abs(beta_values),
    axis=0
)

importance_pct = (
    100.0 * mean_abs_beta /
    np.nansum(mean_abs_beta)
)

importance_df = pd.DataFrame({
    "Predictor": FINAL_PREDICTORS,
    "Mean_abs_standardized_beta": mean_abs_beta,
    "Normalized_relative_importance_percent": importance_pct,
})

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final6_coefficient_based_relative_importance.csv"
    ),
    index=False
)


# 14. PLOT FINAL FIGURE 6

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig = plt.figure(figsize=(17.2, 11.4), facecolor="white")

gs = fig.add_gridspec(
    nrows=3,
    ncols=4,
    # Column 0: heatmap; column 1: nearby colourbar;
    # column 2: controlled spacer; column 3: panels b-d.
    width_ratios=[1.78, 0.055, 0.16, 1.28],
    height_ratios=[0.90, 1.10, 0.95],
    left=0.055,
    right=0.975,
    bottom=0.120,
    top=0.950,
    # A small GridSpec gap keeps the colourbar close to panel (a).
    # The dedicated spacer column separates it from panel (c)'s y-label.
    wspace=0.08,
    hspace=0.58,
)


# (a) Standardized beta heatmap

ax_a = fig.add_subplot(gs[:, 0])

finite_beta = beta_values[np.isfinite(beta_values)]
max_abs = np.nanmax(np.abs(finite_beta)) if finite_beta.size else 1.0
beta_lim = max(1.0, float(np.ceil(max_abs * 10) / 10.0))

beta_norm = TwoSlopeNorm(
    vmin=-beta_lim,
    vcenter=0,
    vmax=beta_lim
)

im = ax_a.imshow(
    beta_values,
    cmap="RdBu_r",
    norm=beta_norm,
    aspect="auto"
)

ax_a.set_title(
    "(a) Sea-wise standardized regression coefficients (β)",
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=10
)

ax_a.set_xticks(np.arange(len(FINAL_PREDICTORS)))
ax_a.set_xticklabels(
    [
        "SST",
        "Wind\nspeed",
        "MLD",
        "2 m air\ntemp.",
        "OHC\n0–100 m",
        "Ocean current\nspeed",
    ],
    rotation=0,
    fontsize=8.3,
    fontweight="bold"
)

ax_a.set_yticks(np.arange(len(SEA_CODES)))
ax_a.set_yticklabels(SEA_CODES, fontsize=10)

ax_a.set_ylabel(
    "Antarctic marginal sea",
    fontsize=11,
    fontweight="bold"
)

for i in range(len(SEA_CODES)):
    for j in range(len(FINAL_PREDICTORS)):
        val = beta_values[i, j]

        if np.isfinite(val):
            text_color = (
                "white"
                if abs(val) > 0.60 * beta_lim
                else "black"
            )

            ax_a.text(
                j,
                i,
                f"{val:+.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold"
            )
        else:
            ax_a.text(
                j,
                i,
                "NA",
                ha="center",
                va="center",
                fontsize=7,
                color="0.4"
            )

ax_a.set_xticks(
    np.arange(-0.5, len(FINAL_PREDICTORS), 1),
    minor=True
)

ax_a.set_yticks(
    np.arange(-0.5, len(SEA_CODES), 1),
    minor=True
)

ax_a.grid(
    which="minor",
    color="0.70",
    linewidth=0.5
)

ax_a.tick_params(
    which="minor",
    bottom=False,
    left=False
)

cax_beta = fig.add_subplot(gs[:, 1])

cb = fig.colorbar(
    im,
    cax=cax_beta,
    extend="both"
)

cb.set_label(
    "Standardized β",
    fontsize=10,
    fontweight="bold",
    labelpad=4,
)

cb.ax.tick_params(labelsize=9)

# Keep the colourbar label away from panel (c). Ticks remain on the right,
# while the vertical label occupies the compact heatmap-colourbar gap.
cb.ax.yaxis.set_label_position("left")
cb.ax.yaxis.set_ticks_position("right")


# (b) Coefficient-based importance

ax_b = fig.add_subplot(gs[0, 3])
bar_x = np.arange(len(FINAL_PREDICTORS))

bars = ax_b.bar(
    bar_x,
    importance_pct,
    width=0.65,
    edgecolor="black",
    linewidth=0.5
)

ax_b.set_title(
    "(b) Coefficient-based relative importance",
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=10
)

ax_b.set_xticks(bar_x)
ax_b.set_xticklabels(
    ["SST", "Wind", "MLD", "T2m", "OHC", "Current"],
    fontsize=8.2,
    fontweight="bold"
)

ax_b.set_ylabel(
    "Normalized mean |β| (%)",
    fontsize=10,
    fontweight="bold"
)

if np.isfinite(importance_pct).any():
    imp_max = float(np.nanmax(importance_pct))
    ax_b.set_ylim(0, imp_max * 1.30)
else:
    imp_max = 100.0
    ax_b.set_ylim(0, 100)

ax_b.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45
)

for b, val in zip(bars, importance_pct):
    if np.isfinite(val):
        ax_b.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + imp_max * 0.025,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


# (c) Observed vs blocked-CV predicted SIC anomalies

ax_c = fig.add_subplot(gs[1, 3])

ax_c.scatter(
    all_oof_obs,
    all_oof_pred,
    s=20,
    alpha=0.55,
    edgecolor="none"
)

min_val = float(
    np.nanmin([
        np.nanmin(all_oof_obs),
        np.nanmin(all_oof_pred)
    ])
)

max_val = float(
    np.nanmax([
        np.nanmax(all_oof_obs),
        np.nanmax(all_oof_pred)
    ])
)

pad = 0.05 * (
    max_val - min_val
    if max_val > min_val
    else 1.0
)

plot_min = min_val - pad
plot_max = max_val + pad

ax_c.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    color="black",
    linewidth=1.0,
    label="1:1 line"
)

# Descriptive fit through out-of-fold points only
coef = np.polyfit(
    all_oof_obs,
    all_oof_pred,
    1
)

fit_x = np.linspace(
    plot_min,
    plot_max,
    100
)

fit_y = coef[0] * fit_x + coef[1]

ax_c.plot(
    fit_x,
    fit_y,
    color="red",
    linewidth=1.2,
    label="OOF fit"
)

ax_c.set_title(
    "(c) Observed vs blocked-CV predicted SIC anomalies",
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=10
)

ax_c.set_xlabel(
    "Observed SIC anomaly (%)",
    fontsize=10,
    fontweight="bold"
)

ax_c.set_ylabel(
    "Cross-validated predicted SIC anomaly (%)",
    fontsize=10,
    fontweight="bold"
)

ax_c.set_xlim(plot_min, plot_max)
ax_c.set_ylim(plot_min, plot_max)

ax_c.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.45
)

ax_c.legend(
    loc="upper left",
    fontsize=8,
    frameon=False
)

ax_c.text(
    0.97,
    0.05,
    f"Combined out-of-fold\n"
    f"R² = {combined_cv_r2:.2f}\n"
    f"RMSE = {combined_cv_rmse:.2f}\n"
    f"MAE = {combined_cv_mae:.2f}\n"
    f"N = {len(all_oof_obs)}",
    transform=ax_c.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.4",
        linewidth=0.8
    )
)


# (d) Sea-wise blocked-CV R²

ax_d = fig.add_subplot(gs[2, 3])

r2_vals = cv_r2_final.loc[SEA_CODES].astype(float).values
x_sea = np.arange(len(SEA_CODES))

bars_d = ax_d.bar(
    x_sea,
    np.nan_to_num(r2_vals, nan=0.0),
    width=0.65,
    edgecolor="black",
    linewidth=0.4
)

for bar, val in zip(bars_d, r2_vals):
    if not np.isfinite(val):
        bar.set_hatch("//")
        bar.set_alpha(0.25)

ax_d.axhline(
    0,
    color="black",
    linewidth=0.8
)

ax_d.set_title(
    "(d) Sea-wise blocked cross-validated R²",
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=10
)

ax_d.set_xticks(x_sea)
ax_d.set_xticklabels(
    SEA_CODES,
    rotation=45,
    ha="right",
    fontsize=8
)

ax_d.set_ylabel(
    "Cross-validated R²",
    fontsize=10,
    fontweight="bold"
)

finite_r2 = r2_vals[np.isfinite(r2_vals)]

if finite_r2.size:
    ymin = min(
        0.0,
        float(np.floor(np.nanmin(finite_r2) * 10) / 10 - 0.1)
    )

    ymax = max(
        1.0,
        float(np.ceil(np.nanmax(finite_r2) * 10) / 10 + 0.1)
    )
else:
    ymin = 0.0
    ymax = 1.0

ax_d.set_ylim(ymin, ymax)

ax_d.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45
)

for x, val in zip(x_sea, r2_vals):
    if np.isfinite(val):
        offset = 0.03 * (ymax - ymin)
        va = "bottom" if val >= 0 else "top"
        y_text = val + offset if val >= 0 else val - offset

        ax_d.text(
            x,
            y_text,
            f"{val:.2f}",
            ha="center",
            va=va,
            fontsize=7
        )
    else:
        ax_d.text(
            x,
            0.02,
            "NA",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.4"
        )


# 15. SAVE FIGURE

fig.savefig(
    FIG_OUT,
    dpi=1080,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.08,
)

fig.savefig(
    FIG_PDF,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.08,
)

plt.show()

print("\nFigure saved:")
print(FIG_OUT)
print(FIG_PDF)


# 16. SAVE CAPTION TEMPLATE

caption = f"""Figure 6. Sea-wise standardized multiple regression of monthly Antarctic sea-ice concentration (SIC) anomalies during {START_YEAR}–{END_YEAR}. The final six-predictor model includes sea-surface temperature (SST), 10-m wind speed, mixed-layer depth (MLD), 2-m air temperature, previously calculated Argo-derived 0–100 m ocean heat content (OHC) above the local seawater freezing point, and surface ocean-current speed calculated at each grid cell as the vector magnitude sqrt(uo² + vo²). (a) Standardized regression coefficients (beta) from 13 separate sea-wise models. (b) Coefficient-based relative importance, calculated as the normalized mean absolute standardized coefficient across marginal seas; this metric is not a partition of explained variance. (c) Combined observed and out-of-fold predicted SIC anomalies from blocked temporal cross-validation of the separate sea-wise models. (d) Sea-wise blocked cross-validated R-squared. All gridded predictors were averaged over a common SST-derived ocean mask. The existing five-predictor model and the final six-predictor model were evaluated using identical months with complete OHC and ocean-current-speed data."""

with open(
    os.path.join(OUTPUT_DIR, "Figure6_Six_Predictor_caption_template.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(caption)

print("\nSuggested Figure 6 caption:\n")
print(caption)

print("\nSupplementary outputs saved in:")
print(OUTPUT_DIR)


# 17. CLOSE DATASETS

ds_sic.close()
ds_sst.close()
ds_wind.close()
ds_t2m.close()
ds_mld.close()

for dataset in eastward_current_datasets:
    dataset.close()

for dataset in northward_current_datasets:
    dataset.close()

plt.close(fig)

gc.collect()

print("\nDONE.")
