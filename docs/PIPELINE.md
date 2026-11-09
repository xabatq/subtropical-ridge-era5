# Pipeline: scripts, inputs and outputs

Run order (this is the order used by `scripts/run_all.py`): 00 → 03a → 01 → 02 → 03 → 04 → 05 → 07 → 08 → 09 → 10 → 12 → 13 → 06 → 14.
`<data>` is the input folder; outputs go to `<data>/outputs`.

| Step | Script | Main inputs | Main outputs | Used for |
|---|---|---|---|---|
| 00 | `00_make_static_fields.py` | ERA5 static file (`lsm`, `z`) | `static_fields.nc` | ocean and mountain masks |
| 03a | `03a_download_enso.py` | internet (NOAA) | `enso_monthly_1966_2025.csv` | ENSO removal |
| 01 | `01_extract_zonal_means.py` | `W500.nc`, cloud, radiation, `windspeed.nc`, static | `zonal_means_masked.nc`, `climatology.nc` | zonal and basin means (ocean, land, sectors) |
| 02 | `02_hadley_latitude_analysis.py` | `zonal_means_masked.nc` | `peak_latitudes_by_year_masked.csv`, `calm_belt_min_speed_by_year.csv`, `trends_masked.csv` | ridge latitudes of ω500, cloud and wind and their trends |
| 03 | `03_enso_adjusted_trends.py` | outputs of 02, ENSO | `enso_adjusted_trends.csv`, `enso_breakpoint_1979.csv` | ENSO-adjusted trends, 1979 step |
| 04 | `04_ceres_cloud_check.py` | CERES, `zonal_means_masked.nc` | `ceres_vs_era5_cloud_ridge.csv`, `ceres_era5_ridge_by_year.csv` | independent cloud check (2000–) |
| 05 | `05_subperiod_breakpoints.py` | outputs of 02, ENSO | `era5_subperiod_trends.csv`, `era5_breakpoint_scan.csv` | sub-periods and break-point scan |
| 07 | `07_clara_cloud_check.py` | CLARA-A3 files | `clara_ridge_by_year.csv`, `clara_vs_era5_cloud_ridge.csv` | independent cloud check (1979–2020) |
| 08 | `08_basin_trends_and_full_stats.py` | outputs of 02, ENSO | `main_trends_full_table.csv`, `basin_trends.csv` | OLS with CI, Theil–Sen, Mann–Kendall, FDR (42-test family), basins (48-test family) |
| 09 | `09_smoothing_sensitivity.py` | `zonal_means_masked.nc` | `smoothing_sensitivity.csv`, `smoothing_sensitivity_by_year.csv` | 5/9/13-point smoothing |
| 10 | `10_pdo_sam_check.py` | outputs of 02, internet | `climate_modes_monthly.csv`, `pdo_sam_adjusted_trends.csv` | PDO and AAO sensitivity (exploratory, no multiple-test correction) |
| 12 | `12_levels_and_slp_ridge.py` | `w300-w700.nc`, `P-SLP.nc` | `levels_slp_trends_ocean.csv`, `levels_slp_basin_trends.csv`, `levels_slp_ridge_by_year.csv`, `levels_slp_correlations.csv`, `levels_slp_mean_latitude.csv`, `slp_high_trends.csv`, `slp_high_centres_by_year.csv`, `fig_levels_slp_ridge.png`, `fig_slp_highs.png` | omega at 700/300 hPa, SLP ridge, individual highs |
| 13 | `13_mass_streamfunction_hadley_edge.py` | `V(Levels).nc`, `P-SLP.nc`, outputs of 12 and 02 | `psi_monthly_1966_2025.npz`, `hadley_edge_by_year.csv`, `hadley_edge_trends.csv`, `hadley_vs_ridge_*.csv`, `fig_psi_climatology.png`, `fig_hadley_edge.png` | mass streamfunction and Hadley-cell edge |
| 06 | `06_article_figures.py` | outputs of 01–05, 07, 08 | `articleoutpt/fig1…fig7` | paper figures 1–7 |
| 14 | `14_global_hadley_map_figure.py` | outputs of 12 and 13 | `fig_global_hadley_ridge_map.png` (+ copies in `articleoutpt`) | global map figure |

## Conventions

* Latitude trends are in degrees of |latitude| per decade; **positive = poleward** in both hemispheres.
* ω is positive for subsidence. ψ is in 10⁹ kg s⁻¹; ψ > 0 is the Northern Hemisphere Hadley-cell sense.
* `at_window_edge` / `edge_fraction` mark series whose extremum lies at the edge of the 15–45° search window; series with `edge_fraction > 0.3` are flagged and should not be interpreted.
* `fdr_family_size` (and, where given, `fdr_family`) states which tests the Benjamini–Hochberg *q* refers to. Test families: 42 (main ocean zonal series), 48 / 64 (basins without / with SLP), 48 (vertical-structure table), 45 (individual highs), 18 (Hadley-edge series).
* Caches: delete `outputs/*.npz` (and `clara_ocean_zonal_1deg.npz`) to force re-reading of the input files.

## Known limitations

* The mass streamfunction uses monthly-mean winds (no eddy term), monthly-mean surface pressure and only 65°S–65°N.
* Wind-speed and cloud trends in a reanalysis can contain observing-system changes (notably around 1979); the scripts report the 1979 step explicitly.
* PDO/AAO regressions are exploratory (regressors are correlated and the tests are not corrected for multiplicity).
