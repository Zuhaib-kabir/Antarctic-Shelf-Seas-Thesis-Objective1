# 1) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

!pip -q install dask netCDF4 h5netcdf

# Fig. 2 — Sea-wise monthly SIC time series, 2008–2025
# 3 columns × 5 rows
# Last row: BEL in 1st column, legend in columns 2–3
#
# METHODS
# 1. The 12-month mean is trailing/right-aligned:
#       rolling(window=12, min_periods=12).mean()
#
# 2. The linear trend is calculated from complete annual mean
#    SIC values (12 valid monthly observations per year).
#
# 3. Trend slopes are reported in SIC percentage points per
#    year and per decade.


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import linregress, t



# 2. File paths

CSV_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "SIC_seawise_monthly_2008_2025_LOW_RAM.csv"
)

OUT_PATH = (
    "/content/drive/MyDrive/SAM_Thesis/Fig/"
    "Fig6_Monthly_SIC_seawise_2008_2025.png"
)

TREND_CSV_PATH = (
    "/content/drive/MyDrive/SAM_Thesis/Fig/"
    "Supplementary_Table_SIC_Linear_Trends_2008_2025.csv"
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TREND_CSV_PATH), exist_ok=True)


# 3. Analysis settings
START_YEAR = 2008
END_YEAR = 2025
SAVE_DPI = 1080 


# 4. Sea order
sea_order = [
    "WED", "KHV", "RLS",
    "LAZ", "COS", "COO",
    "DAV", "MAW", "DUR",
    "SOM", "ROS", "AMU",
    "BEL",
]


# 5. Read and validate CSV
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Input CSV not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE, parse_dates=["time"])

required_columns = ["time"] + sea_order

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required CSV columns: {missing_columns}"
    )

df = df[required_columns].copy()
df = df.sort_values("time").reset_index(drop=True)

if df["time"].isna().any():
    raise ValueError("Invalid or missing timestamps found in 'time'.")

# Check duplicate calendar months, even if timestamps differ by day.
month_periods = df["time"].dt.to_period("M")

if month_periods.duplicated().any():
    duplicate_months = (
        month_periods[month_periods.duplicated(keep=False)]
        .astype(str)
        .unique()
        .tolist()
    )

    raise ValueError(
        "Duplicate monthly timestamps found for: "
        f"{duplicate_months}"
    )

# Restrict explicitly to the requested analysis period.
df = df.loc[
    (df["time"].dt.year >= START_YEAR)
    & (df["time"].dt.year <= END_YEAR)
].copy()

df = df.reset_index(drop=True)

# Confirm that all expected calendar months are present.
actual_months = pd.PeriodIndex(
    df["time"].dt.to_period("M").unique(),
    freq="M",
)

expected_months = pd.period_range(
    start=f"{START_YEAR}-01",
    end=f"{END_YEAR}-12",
    freq="M",
)

missing_months = expected_months.difference(actual_months)
extra_months = actual_months.difference(expected_months)

if len(missing_months) > 0:
    raise ValueError(
        "Missing calendar months in the input CSV: "
        f"{missing_months.astype(str).tolist()}"
    )

if len(extra_months) > 0:
    raise ValueError(
        "Unexpected calendar months in the input CSV: "
        f"{extra_months.astype(str).tolist()}"
    )

# Convert SIC columns to numeric.
for sea in sea_order:
    df[sea] = pd.to_numeric(df[sea], errors="coerce")

    if df[sea].notna().sum() == 0:
        raise ValueError(
            f"Column '{sea}' contains no valid numeric SIC values."
        )

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Start:", df["time"].min())
print("End:", df["time"].max())
print("Expected number of months:", len(expected_months))



# 6. Trailing 12-month rolling mean — consistent with Eq. 5
roll_df = df.copy()

for sea in sea_order:
    # Right-aligned/trailing rolling mean:
    # current month plus the previous 11 months.
    #
    # The first 11 months remain NaN because a complete
    # 12-month window is required.
    roll_df[sea] = (
        df[sea]
        .rolling(
            window=12,
            min_periods=12,
            center=False,
        )
        .mean()
    )



# 7. Calculate annual means and linear trends
df["year"] = df["time"].dt.year

annual_mean_by_sea = {}
trend_models = {}
trend_results = []

for sea in sea_order:

    annual_data = (
        df.groupby("year", as_index=False)
        .agg(
            annual_mean_sic=(sea, "mean"),
            valid_months=(sea, "count"),
        )
    )

    # Only complete annual means are used in the regression.
    annual_data = annual_data.loc[
        annual_data["valid_months"] == 12
    ].copy()

    annual_data = annual_data.dropna(
        subset=["annual_mean_sic"]
    )

    if len(annual_data) < 3:
        raise ValueError(
            f"{sea}: fewer than three complete annual means "
            "are available for trend calculation."
        )

    x_year = annual_data["year"].to_numpy(dtype=float)
    y_annual = annual_data["annual_mean_sic"].to_numpy(dtype=float)

    regression = linregress(x_year, y_annual)

    n_years = len(x_year)
    degrees_of_freedom = n_years - 2

    # Two-sided 95% confidence interval for the slope.
    t_critical = t.ppf(0.975, degrees_of_freedom)

    slope_ci_lower = (
        regression.slope
        - t_critical * regression.stderr
    )

    slope_ci_upper = (
        regression.slope
        + t_critical * regression.stderr
    )

    first_year = int(x_year.min())
    last_year = int(x_year.max())

    fitted_sic_first_year = (
        regression.intercept
        + regression.slope * first_year
    )

    fitted_sic_last_year = (
        regression.intercept
        + regression.slope * last_year
    )

    total_fitted_change = (
        fitted_sic_last_year
        - fitted_sic_first_year
    )

    if regression.slope > 0:
        trend_direction = "Increasing"
    elif regression.slope < 0:
        trend_direction = "Decreasing"
    else:
        trend_direction = "No change"

    significant = regression.pvalue < 0.05

    annual_mean_by_sea[sea] = annual_data

    trend_models[sea] = {
        "slope": regression.slope,
        "intercept": regression.intercept,
    }

    trend_results.append(
        {
            "Sea": sea,
            "Start_year": first_year,
            "End_year": last_year,
            "N_complete_years": n_years,
            "Trend_SIC_percentage_points_per_year":
                regression.slope,
            "Trend_SIC_percentage_points_per_decade":
                regression.slope * 10.0,
            "Slope_standard_error_per_year":
                regression.stderr,
            "Slope_95CI_lower_per_year":
                slope_ci_lower,
            "Slope_95CI_upper_per_year":
                slope_ci_upper,
            "R_squared":
                regression.rvalue ** 2,
            "P_value":
                regression.pvalue,
            "Trend_direction":
                trend_direction,
            "Significant_at_p_less_than_0.05":
                "Yes" if significant else "No",
            "Fitted_SIC_at_start_year_percent":
                fitted_sic_first_year,
            "Fitted_SIC_at_end_year_percent":
                fitted_sic_last_year,
            "Fitted_change_start_to_end_percentage_points":
                total_fitted_change,
            "Regression_equation":
                (
                    f"SIC = {regression.intercept:.6f} "
                    f"+ ({regression.slope:.6f} × Year)"
                ),
        }
    )



# 8. Save trend results for the supplementary table

trend_results_df = pd.DataFrame(trend_results)

trend_results_df.to_csv(
    TREND_CSV_PATH,
    index=False,
    float_format="%.6f",
)

print("\nSea-wise annual-mean SIC linear trends:")
print(
    trend_results_df[
        [
            "Sea",
            "Trend_SIC_percentage_points_per_year",
            "Trend_SIC_percentage_points_per_decade",
            "R_squared",
            "P_value",
            "Significant_at_p_less_than_0.05",
        ]
    ].to_string(index=False)
)

print("\nTrend table saved:", TREND_CSV_PATH)



# 9. Plot settings
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
    }
)

monthly_color = "#6FA3FF"   # light blue
rolling_color = "#1426C8"   # dark blue
trend_color = "#D7191C"     # red

tick_years = pd.to_datetime(
    [
        "2008-01-01",
        "2013-01-01",
        "2019-01-01",
        "2025-01-01",
    ]
)

tick_labels = ["2008", "2013", "2019", "2025"]

# Decimal-year coordinates used only to evaluate the red line
# at every monthly timestamp.
decimal_year = (
    df["time"].dt.year
    + (df["time"].dt.month - 1) / 12.0
)



# 10. Figure layout

fig = plt.figure(
    figsize=(11.5, 14.0),
    dpi=300,
)

gs = GridSpec(
    nrows=5,
    ncols=3,
    figure=fig,
    hspace=0.50,
    wspace=0.28,
)

positions = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2),
    (4, 0),
]


# 11. Draw the 13 sea-wise time series
for index, sea in enumerate(sea_order):

    row, column = positions[index]
    axis = fig.add_subplot(gs[row, column])

    # Monthly SIC.
    axis.plot(
        df["time"],
        df[sea],
        color=monthly_color,
        linewidth=0.9,
        alpha=0.75,
        label="Monthly SIC (%)",
        zorder=1,
    )

    # Trailing 12-month mean.
    axis.plot(
        roll_df["time"],
        roll_df[sea],
        color=rolling_color,
        linewidth=2.2,
        label="Trailing 12-month mean",
        zorder=2,
    )

    # Linear trend calculated from complete annual means.
    trend_sic = (
        trend_models[sea]["intercept"]
        + trend_models[sea]["slope"] * decimal_year
    )

    axis.plot(
        df["time"],
        trend_sic,
        color=trend_color,
        linewidth=2.0,
        linestyle="-",
        label="Linear trend of annual mean SIC",
        zorder=3,
    )

    axis.set_title(
        sea,
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    axis.set_ylim(0, 100)
    axis.set_xlim(
        df["time"].min(),
        df["time"].max(),
    )

    axis.set_xticks(tick_years)
    axis.set_xticklabels(
        tick_labels,
        fontsize=9,
    )

    axis.set_yticks([0, 25, 50, 75, 100])

    axis.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
    )

    if column == 0:
        axis.set_ylabel(
            "SIC (%)",
            fontsize=11,
            fontweight="bold",
        )
    else:
        axis.set_ylabel("")

    if row == 4:
        axis.set_xlabel(
            "Year",
            fontsize=10,
            fontweight="bold",
        )
    else:
        axis.set_xlabel("")

    axis.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


# 12. Legend panel — bottom row, columns 2–3

legend_axis = fig.add_subplot(gs[4, 1:3])
legend_axis.set_xlim(0, 1)
legend_axis.set_ylim(0, 1)
legend_axis.axis("off")

legend_box = FancyBboxPatch(
    (0.06, 0.10),
    0.88,
    0.78,
    boxstyle="round,pad=0.02,rounding_size=0.035",
    linewidth=0.9,
    edgecolor="0.20",
    facecolor="white",
    transform=legend_axis.transAxes,
)

legend_axis.add_patch(legend_box)

# Monthly SIC legend.
legend_axis.plot(
    [0.14, 0.36],
    [0.68, 0.68],
    transform=legend_axis.transAxes,
    color=monthly_color,
    linewidth=1.2,
    alpha=0.75,
)

legend_axis.text(
    0.41,
    0.68,
    "Monthly SIC (%)",
    transform=legend_axis.transAxes,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=10.5,
)

# Trailing mean legend.
legend_axis.plot(
    [0.14, 0.36],
    [0.49, 0.49],
    transform=legend_axis.transAxes,
    color=rolling_color,
    linewidth=3.0,
)

legend_axis.text(
    0.41,
    0.49,
    "Trailing 12-month mean",
    transform=legend_axis.transAxes,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=10.5,
)

# Linear trend legend.
legend_axis.plot(
    [0.14, 0.36],
    [0.30, 0.30],
    transform=legend_axis.transAxes,
    color=trend_color,
    linewidth=2.4,
)

legend_axis.text(
    0.41,
    0.30,
    "Linear trend of annual mean SIC",
    transform=legend_axis.transAxes,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=10.5,
)



# 13. Save figure

fig.savefig(
    OUT_PATH,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()
plt.close(fig)

print("\nFigure saved:", OUT_PATH)
print("Supplementary trend CSV saved:", TREND_CSV_PATH)
