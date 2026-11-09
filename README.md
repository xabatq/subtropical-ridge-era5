# Subtropical ridge latitude in ERA5 (1966–2025)

Analysis code for the paper **"Is the Subtropical Ridge Moving Poleward? Indicator and Period Dependence of Ocean Ridge-Latitude Trends in ERA5, 1966–2025, with CERES and CLARA-A3 Checks"** (Ghamari and Seyedzahedi, submitted to *Climate Dynamics*).

The scripts locate the ocean subtropical ridge in the ERA5 reanalysis with several related but distinct diagnostics, estimate their latitude trends with autocorrelation-adjusted and false-discovery-rate-controlled statistics, and test whether the results are robust to the indicator, the period, the estimator, ENSO/PDO/AAO, the smoothing window, the ocean basin and independent satellite data.

> **Input data are not included.** ERA5, CERES EBAF and CLARA-A3 are free but large (tens of GB); see [`docs/DATA.md`](docs/DATA.md) for the exact download specifications. The small result tables of the paper are in [`results/`](results/).

## What is computed

| Diagnostic | Definition (ocean only unless stated) |
|---|---|
| Subsidence ridge | latitude of maximum ω at 500 hPa (main) and 700 / 300 hPa (sensitivity), positive downward |
| Cloud-minimum ridge | latitude of minimum total cloud cover (ERA5; CERES EBAF and CLARA-A3 as independent checks) |
| Calm-wind ridge | latitude of minimum 10 m wind **speed** (from `si10`, never from monthly-mean u and v) |
| Sea-level-pressure ridge | latitude of maximum SLP; also the position and central pressure of the five individual subtropical highs |
| Hadley-cell edge | zero of the 500 hPa meridional mass streamfunction (all longitudes), computed from ERA5 v on 16 levels |

Each ridge latitude is estimated for 15–45° windows in each hemisphere, after a 9-point running mean, with two estimators: the parabolic-interpolated extremum (*argmax*) and the half-range *centroid*. Trends are in degrees of |latitude| per decade (positive = poleward) for 1966–2025 and 1979–2025, with ENSO (ONI) removed by regression, 95 % confidence intervals using a lag-1 effective sample size, Theil–Sen and Mann–Kendall checks, and Benjamini–Hochberg *q*-values within explicitly defined test families (the family size is stored in every output table).

## Main findings (see the paper for details and caveats)

* The diagnostics place the ridge at different latitudes (the SLP ridge lies about 5–8° poleward of the subsidence maximum) and their trends disagree.
* Cloud-based ridge latitudes move poleward significantly in 1979–2025 (0.15–0.55° per decade), SLP does so in the Southern Hemisphere (about +0.15° per decade, *q* = 0.016), but subsidence at 700, 500 and 300 hPa shows no significant shift after FDR control.
* The annual Hadley-cell edge does not shift significantly in either hemisphere; the Southern Hemisphere summer edge does.
* The cloud signal is confined to 1979–1999 and absent in 2000–2025 in ERA5, CERES EBAF and CLARA-A3, and a step in the calm-belt wind speed near 1979 suggests observing-system effects.

## Repository layout

```
scripts/        the analysis chain (00 … 14) and run_all.py
tools/          helper scripts: inspect_era5_files.py (prints the structure of the NetCDF files),
                check_references.py (checks the DOIs of a .bib file against Crossref)
docs/           DATA.md (data download), PIPELINE.md (inputs/outputs of every script), GITHUB_GUIDE_FA.md (publishing guide, Persian)
data/           put the downloaded input files here (not tracked by git)
results/        the derived tables of the paper (CSV)
```

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/subtropical-ridge-era5.git
cd subtropical-ridge-era5
conda env create -f environment.yml && conda activate subtropical-ridge   # or: pip install -r requirements.txt
# 1. download the input files into data/  (docs/DATA.md)
# 2. run everything (several hours, mostly reading the NetCDF files):
python scripts/run_all.py --data-dir data
# or a single step, e.g. the new indicators and the Hadley edge:
python scripts/run_all.py --data-dir data --only 12 13 14
```

The data folder can also be set with the environment variable `HADLEY_DATA_DIR`. Results are written to `<data>/outputs`, the paper figures to `<data>/articleoutpt`. Slow extraction steps (01, 07, 12, 13) cache their result (`.nc`/`.npz` files in `outputs`), so a second run takes minutes; delete the cache to force a re-read.

Requirements: Python ≥ 3.10; 16 GB RAM recommended; about 100 GB free disk space for the input data. `basemap` is only needed for Figure 1.

## Reproducibility notes

* Months are matched by (year, month), not by file position; the scripts check that all 720 months of 1966–2025 are present and that `expver` is `0001`.
* The longitude axis of some ERA5 downloads is −180…179.75 and of others 0…359.75; the scripts reorder it explicitly and check the grid against the static fields.
* Python/xarray versions can change floating-point details in the last digits; trends should agree to the printed precision.
* The scripts print an error and stop (instead of guessing) if an input file does not have the expected layout.

## Citation

Please cite the paper and the software archive:

> Ghamari, K. and Seyedzahedi, A. (2026). *Subtropical ridge latitude in ERA5 (1966–2025): analysis code.* Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

(`CITATION.cff` is provided; GitHub shows a "Cite this repository" button.)

## Data sources and acknowledgements

* **ERA5**: Hersbach et al. (2023), Copernicus Climate Change Service (C3S) Climate Data Store, doi:10.24381/cds.6860a573 (pressure levels) and doi:10.24381/cds.f17050d7 (single levels).
* **CERES EBAF Ed4.2.1**: NASA Langley Atmospheric Science Data Center.
* **CLARA-A3**: EUMETSAT CM SAF, doi:10.5676/EUM_SAF_CM/CLARA_AVHRR/V003.
* **ENSO (ONI), PDO, AAO indices**: NOAA CPC / NCEI / PSL (downloaded by the scripts).

The code was written with the help of an AI assistant (Claude, Anthropic); the authors designed, ran and verified the analyses and take full responsibility for the code and results.

## License

Code: MIT (see `LICENSE`). The result tables in `results/` are released under CC BY 4.0. Input data remain subject to the licenses of their providers.

## Contact

Khabat Ghamari (corresponding author) — ghamari.kh@ut.ac.ir
Kurdistan Provincial Meteorological Office, Iran Meteorological Organization, Sanandaj, Iran
# subtropical-ridge-era5
Code for ocean subtropical-ridge latitude trends in ERA5 (1966–2025) with CERES and CLARA-A3 checks
