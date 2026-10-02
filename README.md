# Antarctic Shelf Seas Thesis — Objective 1

This repository contains the Python workflows used for **Objective 1 of the Antarctic shelf-seas thesis**, focusing on **Southern Ocean sea-ice concentration (SIC), its seasonal and interannual variability, trends, and relationships with atmospheric and oceanic drivers**.

The scripts are organized around the main and appendix figures used in the Objective 1 analysis. They are written primarily for **Google Colab** and use data stored in Google Drive under the thesis directory structure.

---

## Objective 1 analysis scope

The repository examines Antarctic SIC during **2008–2025** using a combination of:

- sea-wise monthly SIC time series;
- seasonal SIC climatology and anomalies;
- spatial, sea-wise, and seasonal trend analysis;
- atmospheric and oceanic driver correlations;
- Argo-derived upper-ocean heat content;
- multivariate sea-wise regression;
- blocked cross-validation;
- lagged SIC–driver relationships;
- seasonal atmospheric-energy fields;
- mixed-layer, snowfall, SST, and sea-ice-thickness context;
- seasonal surface-ocean currents and streamlines.

Several analyses use the same **13 Antarctic marginal-sea sectors**:

1. Weddell Sea (WED)
2. King Haakon VII Sea (KHV)
3. Riiser-Larsen Sea (RLS)
4. Lazarev Sea (LAZ)
5. Cosmonauts Sea (COS)
6. Cooperation Sea (COO)
7. Davis Sea (DAV)
8. Mawson Sea (MAW)
9. D'Urville Sea (DUR)
10. Somov Sea (SOM)
11. Ross Sea (ROS)
12. Amundsen Sea (AMU)
13. Bellingshausen Sea (BEL)

The Ross Sea sector is handled as a date-line-crossing longitude sector where required.

---

## Repository contents

| Script | Main purpose |
|---|---|
| [`Figure-3.py`](./Figure-3.py) | Sea-wise monthly SIC time series and linear trend statistics |
| [`Figure-4.py`](./Figure-4.py) | Spatial, sea-wise, and seasonal SIC trend analysis using Sen slopes and significance testing |
| [`Figure-5-and-6.py`](./Figure-5-and-6.py) | Seasonal SIC-driver spatial correlations for six atmospheric/oceanic variables with FDR control |
| [`Figure-7.py`](./Figure-7.py) | Final six-predictor sea-wise SIC regression with blocked cross-validation and diagnostics |
| [`Figure-8.py`](./Figure-8.py) | 2025 seasonal SIC anomalies with wind vectors and sea-level-pressure contours |
| [`Figure-9.py`](./Figure-9.py) | Sector-wise lag correlations between SIC and six environmental drivers |
| [`Figure-A1.py`](./Figure-A1.py) | Seasonal SIC climatology and 2025 SIC anomaly maps |
| [`Figure-A2.py`](./Figure-A2.py) | Seasonal Antarctic surface-energy climatology |
| [`Figure-A3.py`](./Figure-A3.py) | Seasonal oceanic and cryospheric controls on Antarctic SIC |
| [`Figure-A4.py`](./Figure-A4.py) | Seasonal surface-current speed and streamline maps |

---

## `Figure-3.py` — Sea-wise monthly SIC variability and trends

This workflow analyzes the monthly SIC time series for all 13 Antarctic marginal seas over **2008–2025**.

### Main methods

- reads the sea-wise monthly SIC table;
- validates the complete monthly time axis;
- calculates a **trailing/right-aligned 12-month mean**;
- calculates annual mean SIC using only years with all 12 monthly observations;
- fits a linear trend to complete annual means;
- reports trend slopes in SIC percentage points per year and per decade;
- calculates a two-sided 95% confidence interval for the slope.

### Main input

```text
SIC_seawise_monthly_2008_2025_LOW_RAM.csv
```

### Statistical output

```text
Supplementary_Table_SIC_Linear_Trends_2008_2025.csv
```

---

## `Figure-4.py` — Antarctic SIC trends

This script performs corrected trend analysis for SIC at spatial, sea-wise, and seasonal scales.

### Panels

- spatial Sen's slope;
- sea-wise annual Sen's slope with 95% confidence intervals;
- seasonal Sen's slope.

### Statistical methods

- **Theil–Sen / Sen slope** rather than ordinary least squares for the principal trend estimates;
- **trend-free prewhitening** to account for serial autocorrelation;
- Mann–Kendall / Kendall-based significance testing;
- **Benjamini–Hochberg false-discovery-rate correction** for spatial significance;
- complete 12-month annual means;
- complete 3-month seasonal means;
- removal of incomplete DJF seasons at the temporal boundaries.

The principal SIC source is:

```text
OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
```

---

## `Figure-5-and-6.py` — Seasonal SIC-driver spatial correlations

This combined workflow creates the corrected spatial correlation analyses between SIC and six environmental drivers.

### Driver variables

The first three rows analyze:

- sea-surface temperature (SST);
- 10-m wind speed;
- mixed-layer depth (MLD).

The second group analyzes:

- 2-m air temperature;
- Argo ocean heat content (OHC, 0–100 m);
- surface-ocean-current speed.

### Seasonal structure

The calculations are organized for Southern Hemisphere seasons:

```text
Spring = SON
Summer = DJF
Autumn = MAM
Winter = JJA
```

### Statistical workflow

For newly calculated driver panels, the script applies:

1. monthly climatological anomaly calculation;
2. grid-cell Pearson correlation with SIC anomalies;
3. lag-1-autocorrelation-adjusted effective sample size;
4. two-sided adjusted p-values;
5. panel-wise Benjamini–Hochberg FDR correction at `q < 0.05`.

### Argo OHC treatment

Monthly Argo OHC anomalies are mapped using a **local objective analysis / optimal interpolation** method with a Gaussian horizontal covariance. Mapping-error and local-observation-support thresholds are used to avoid unsupported extrapolation.

The workflow also produces a combined FDR/statistics table for the six analyzed variables.

---

## `Figure-7.py` — Six-predictor sea-wise regression

This script builds the final **six-predictor SIC regression analysis** using 13 separate sea-wise models.

The model is:

```text
SIC anomaly ~ SST
            + Wind Speed
            + MLD
            + 2 m Air Temperature
            + OHC (0–100 m)
            + Ocean Current Speed
```

### Important model design

- each Antarctic sea is modeled separately;
- the workflow does **not** use one pooled regression for all seas;
- measured monthly Argo OHC is read from the existing OHC product rather than recomputed from raw profiles;
- predictors are standardized for coefficient comparison;
- model performance is assessed using **blocked cross-validation**.

### Main diagnostics

- standardized beta coefficients;
- coefficient-based relative importance;
- observed versus out-of-fold predicted SIC anomalies;
- sea-wise blocked-CV R²;
- baseline five-predictor versus final six-predictor comparison;
- variance inflation factors (VIF);
- predictor correlation matrices;
- bootstrap 95% confidence intervals for standardized beta coefficients;
- final model performance tables.

The code explicitly notes that its relative-importance metric is based on normalized mean absolute standardized coefficients and is **not** the same as variance explained.

---

## `Figure-8.py` — Dynamic atmospheric forcing of SIC

This workflow visualizes seasonal atmospheric forcing during **2025**.

Each panel combines:

- **background:** 2025 SIC anomaly;
- **arrows:** 2025 seasonal mean 10-m wind vectors;
- **contours:** 2025 seasonal mean sea-level pressure.

The SIC anomaly is calculated relative to the **2008–2025 climatological period**.

Principal inputs include:

```text
OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
Wind_U_V_era.nc
2mT_MSLP_era.nc
```

---

## `Figure-9.py` — Sector-wise lag correlations

This script evaluates lagged relationships between SIC and six environmental drivers for the 13 Antarctic marginal seas.

### Driver panels

- SIC vs SST;
- SIC vs wind speed;
- SIC vs MLD;
- SIC vs 2-m air temperature;
- SIC vs Argo OHC (0–100 m);
- SIC vs surface-ocean-current speed.

### Lag definition

The workflow calculates:

```text
r(lag) = corr[SIC(t), driver(t + lag)]
```

for lags from **−6 to +6 months**.

### Corrections and controls

- one SST-derived static ocean mask is applied consistently across the gridded variables;
- each sector's calendar-month climatology is removed before lag correlation;
- p-values are adjusted for serial correlation using an effective sample size based on lag-1 autocorrelation;
- Benjamini–Hochberg FDR is applied across sector × lag tests within each driver panel;
- significance markers represent **FDR-adjusted q < 0.05**;
- ocean-current speed is calculated grid-cell by grid-cell as:

```text
sqrt(uo² + vo²)
```

### OHC source priority

The code uses the following priority:

1. existing sea-wise Argo OHC CSV;
2. existing gridded Argo OHC cache;
3. raw Argo profiles as a fallback.

The workflow saves detailed lag statistics, zero-lag summaries, peak-lag summaries, FDR-significant results, and current-speed support tables.

---

# Appendix / supporting figure workflows

## `Figure-A1.py` — Seasonal SIC and SIC anomalies

This script uses the monthly OSTIA SIC/SIF product to calculate:

- seasonal SIC for 2025;
- the 2008–2025 seasonal climatology;
- 2025 seasonal SIC anomalies.

Southern Hemisphere seasons are used, with December assigned to the following DJF season-year.

Principal input:

```text
OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
```

---

## `Figure-A2.py` — Seasonal Antarctic surface-energy climatology

This workflow produces seasonal climatologies for:

- 2-m air temperature;
- turbulent surface heat flux (latent + sensible);
- downward surface radiation (shortwave + longwave).

### Methodological details retained in the code

- only complete three-month seasons are retained;
- December is assigned to the following DJF season-year;
- monthly means are weighted by the number of days in each month when forming seasonal means;
- ERA5 monthly-averaged accumulated energy fields are converted from J m⁻² to W m⁻² by division by 86,400 s;
- turbulent heat-flux sign conventions are handled explicitly.

Principal inputs include:

```text
2mT_MSLP_era.nc
heatflux_latent_sensible_era.nc
SW_LW_era.nc
```

The script also saves summary and source-variable CSV tables.

---

## `Figure-A3.py` — Seasonal oceanic and cryospheric controls

This workflow provides seasonal physical context for Antarctic SIC using:

- SST;
- snowfall;
- mixed-layer depth;
- sea-ice thickness.

It uses Southern Hemisphere seasonal climatologies for **2008–2025** and combines shaded fields with contour information to display coupled oceanic and cryospheric conditions.

Principal inputs include:

```text
MLD_monthly_2008_2025_SO.nc
P_Snow_era.nc
OSTIA_SST_monthly_2008_2025_SO.nc
GLORYS_SIT_sithick_monthly_2008_2025_SO.nc
```

---

## `Figure-A4.py` — Seasonal ocean-current speed and streamlines

This script calculates seasonal mean Southern Ocean surface-current speed for **2008–2025**.

### Method

1. concatenate the 2008–2014 and 2015–2025 eastward-current files;
2. concatenate the corresponding northward-current files;
3. select the shallowest available depth level;
4. align northward velocity to the eastward-velocity grid when needed;
5. calculate monthly grid-cell speed as:

```text
sqrt(uo² + vo²)
```

6. form complete-season means by year;
7. average seasonal means over the study period;
8. plot speed as shading and the seasonal mean current vector as streamlines.

The script explicitly distinguishes:

```text
mean[sqrt(uo² + vo²)]
```

from:

```text
sqrt(mean(uo)² + mean(vo)²)
```

The former is used for current-speed shading, while streamlines represent the direction of the seasonal mean vector.

---

## Principal datasets referenced by the repository

The scripts use a combination of observational, reanalysis, and derived products, including:

- OSTIA sea-ice concentration / sea-ice fraction;
- OSTIA sea-surface temperature;
- ERA5 10-m winds;
- ERA5 2-m air temperature and mean sea-level pressure;
- ERA5 latent and sensible heat fluxes;
- ERA5 downward shortwave and longwave radiation;
- ERA5 snowfall;
- mixed-layer depth;
- GLORYS sea-ice thickness;
- eastward and northward ocean-current velocity;
- Argo hydrographic profiles and derived 0–100 m OHC.

The repository contains the analysis code; the large source datasets themselves are expected to exist in the configured Google Drive paths.

---

## Google Drive directory structure

The scripts primarily reference:

```text
/content/drive/MyDrive/SAM_Thesis/Data/
/content/drive/MyDrive/SAM_Thesis/Fig/
/content/drive/MyDrive/SAM_Thesis/Fig/P1_Corrected/
```

Before running the workflows, confirm that the required files exist at the configured paths.

---

## Python environment

The scripts are designed primarily for **Google Colab**.

Packages used across the repository include:

```text
numpy
pandas
xarray
dask
netCDF4
h5netcdf
scipy
matplotlib
cartopy
scikit-learn
statsmodels
gsw
```

Several scripts contain Colab/Jupyter commands such as `!pip` or `%pip`. For that reason, they should be run in a compatible notebook/Colab environment unless those commands are converted to standard Python package-installation code.

---

## Typical workflow

1. Open the required script in Google Colab.
2. Mount Google Drive.
3. Confirm that the required source datasets exist under the configured paths.
4. Install any missing packages.
5. Run the script from top to bottom.
6. Review the generated figure, CSV, PDF, NetCDF, or diagnostic outputs in the configured thesis output directories.

Several scripts also create supplementary statistical tables and diagnostic files in addition to their primary figure.

---

## Reproducibility notes

The workflows retain important analysis controls directly in the source code, including:

- the main **2008–2025** analysis period;
- Southern Hemisphere season definitions;
- common 13-sea sector boundaries;
- explicit treatment of DJF season-years;
- complete-year and complete-season requirements where specified;
- longitude normalization;
- effective-sample-size corrections for autocorrelation where specified;
- false-discovery-rate correction where specified;
- blocked cross-validation in the multivariate regression workflow;
- explicit OHC observation-support and mapping controls.

Users reproducing the analyses should preserve these settings unless intentionally conducting a sensitivity analysis.

---

## Repository structure

```text
.
├── Figure-3.py
├── Figure-4.py
├── Figure-5-and-6.py
├── Figure-7.py
├── Figure-8.py
├── Figure-9.py
├── Figure-A1.py
├── Figure-A2.py
├── Figure-A3.py
├── Figure-A4.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## License

This repository is distributed under the license provided in [`LICENSE`](./LICENSE).

---

## Repository scope

This repository provides the formatted and reproducible Python analysis workflows for **Objective 1**, documenting the seasonal, interannual, trend, driver-correlation, regression, lag-correlation, and physical-context analyses of Antarctic sea-ice concentration across the Southern Ocean.
