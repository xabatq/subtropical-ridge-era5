# Input data

All input files go into one folder (default `data/`, or the folder given by `--data-dir` / `HADLEY_DATA_DIR`). The file names below are the ones the scripts expect. Analysis period: **January 1966 – December 2025** (720 months), monthly means, global 0.25° grid, NetCDF.

## ERA5 (Copernicus Climate Data Store, https://cds.climate.copernicus.eu)

Downloads from the CDS may produce differently named files; rename them to the names below. Choose *Product type: Monthly averaged reanalysis*, *Time: 00:00*, all months, years 1966–2025 (for the V file 1965–2025 also works), *Data format: NetCDF*. Keep the default global area unless stated.

| File name | CDS dataset | Variable(s) | Levels / area |
|---|---|---|---|
| `W500.nc` | ERA5 monthly averaged data on **pressure levels** | Vertical velocity (ω, `w`) | 500 hPa |
| `w300-w700.nc` | ERA5 monthly averaged data on pressure levels | Vertical velocity | 300 and 700 hPa |
| `V(Levels).nc` | ERA5 monthly averaged data on pressure levels | V-component of wind (`v`) | 16 levels: 100, 200, 300, 400, 500, 600, 700, 800, 825, 850, 875, 900, 925, 950, 975, 1000 hPa; area N 65°, S −65°, W −180°, E 180° (about 16 GB) |
| `data_stream-moda_stepType-avgua.nc` | ERA5 monthly averaged data on **single levels** | Total cloud cover (`tcc`) | surface |
| `data_stream-moda_stepType-avgad.nc` | ERA5 monthly averaged data on single levels | Surface solar radiation downwards (`ssrd`) | surface |
| `windspeed.nc` | ERA5 monthly averaged data on single levels | 10 m wind **speed** (`si10`) | surface |
| `P-SLP.nc` | ERA5 monthly averaged data on single levels | Mean sea level pressure (`msl`) and Surface pressure (`sp`) | surface |
| static file (any name, e.g. `static.nc`) | ERA5 hourly data on single levels (one time step) | Land-sea mask (`lsm`) and Geopotential (`z`) | surface |

Important: the wind speed must be taken from `si10`. Wind speed is **never** derived from monthly-mean u and v.

The ERA5 DOIs are 10.24381/cds.6860a573 (pressure levels) and 10.24381/cds.f17050d7 (single levels). If a request is too large for the CDS, split it by decade and concatenate the files (the scripts only need one file per name, with a continuous monthly time axis).

Run `python tools/inspect_era5_files.py` after downloading: it prints the dimensions, time axis (missing months, `expver`), levels, units and grid of every file, and compares the grid with the static fields.

## Satellite cloud data (optional; used for the independent checks)

* **CERES EBAF Ed4.2.1** (NASA LaRC ASDC, https://ceres-tool.larc.nasa.gov/ord-tool/): TOA monthly means, 1° × 1°, variable `cldarea_total_daynight`, from 2000-03 onward. Expected name: `CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202608.nc` (change `CERES_FILE` in `scripts/04_ceres_cloud_check.py` if your subset has a different end date).
* **CLARA-A3 (CM SAF)** monthly cloud fraction `CFCmm…nc`, 0.25°, 1979–2020 (https://wui.cmsaf.eu): put all monthly files in `data/CLARA-A3/`.

## Climate indices (downloaded automatically)

`03a_download_enso.py` downloads ONI/SOI/MEI from NOAA; `10_pdo_sam_check.py` downloads the PDO (NOAA NCEI, ERSSTv5) and the AAO (NOAA CPC). If the machine has no internet access, download the raw files manually and place them in `data/enso_raw/` with the names printed by the script.

## Folder after a full run

```
data/
  W500.nc  w300-w700.nc  V(Levels).nc  P-SLP.nc  windspeed.nc
  data_stream-moda_stepType-avgua.nc  data_stream-moda_stepType-avgad.nc  static.nc
  CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202608.nc   CLARA-A3/CFCmm*.nc
  enso_raw/  enso_monthly_1966_2025.csv  climate_modes_monthly.csv
  outputs/      all tables, caches and figures of the analysis
  articleoutpt/ the figures of the paper
```
