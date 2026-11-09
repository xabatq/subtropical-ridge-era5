# -*- coding: utf-8 -*-
"""
Step 1 (version 3): extract zonal/sector means and climatologies from ERA5

Analysis period: 1966-2025 (60 years = two 30-year periods)
    period 1: 1966-1995
    period 2: 1996-2025

Prerequisite: 00_plot_static_fields.py has been run (outputs/static_fields.nc exists).

Inputs (in DATA_DIR):
    W500.nc                               -> w   (Pa/s, positive = subsidence)
    data_stream-moda_stepType-avgua.nc    -> tcc  (cloud cover only; u10 and v10 are not read)
    data_stream-moda_stepType-avgad.nc    -> ssrd (J/m2 per day -> W/m2)
    windspeed.nc                          -> monthly mean 10 m wind SPEED (si10)
    outputs/static_fields.nc              -> lsm and height_m

Masks (each one is a 'sector' in the output):
    all        : all grid points
    nomtn1000  : points with height < 1000 m   (sensitivity test)
    nomtn2500  : points with height < 2500 m   (main mountain mask)
    ocean      : ocean only (lsm < 0.1)
    atlantic, pacific_w, pacific_c, pacific_e, indian : ocean basin sectors

Outputs (in DATA_DIR/outputs):
    zonal_means_masked.nc
    climatology.nc

Run:  python 01_extract_zonal_means.py     (takes tens of minutes)
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

Y0, Y1 = 1966, 2025          # 60 years
P1 = (1966, 1995)            # first period (30 years)
P2 = (1996, 2025)            # second period (30 years)

F_W = os.path.join(DATA_DIR, "W500.nc")
F_UA = os.path.join(DATA_DIR, "data_stream-moda_stepType-avgua.nc")
F_AD = os.path.join(DATA_DIR, "data_stream-moda_stepType-avgad.nc")
F_SI = os.path.join(DATA_DIR, "windspeed.nc")
F_STATIC = os.path.join(OUT_DIR, "static_fields.nc")
SI10_VAR = None            # None = auto-detect the variable name inside windspeed.nc

H_LIST = [1000.0, 2500.0]  # height thresholds (m) for the mountain mask
OCEAN_LSM = 0.1            # lsm below this = ocean
MIN_COVER = 0.25           # minimum share of valid points in each latitude row (otherwise NaN)

# Ocean basin sectors (longitude 0 to 360)
SECTOR_LON = {
    "atlantic":  (300.0, 350.0),
    "pacific_w": (150.0, 200.0),
    "pacific_c": (200.0, 250.0),
    "pacific_e": (250.0, 290.0),
    "indian":    (55.0, 95.0),
}

VARS = ["w500", "wspd", "tcc", "ssrd"]

# ---------------------------------------------------------------
# Open the files
# ---------------------------------------------------------------
for f in (F_W, F_UA, F_AD, F_SI, F_STATIC):
    if not os.path.exists(f):
        raise SystemExit(f"File not found: {f}")

dsW = xr.open_dataset(F_W)
dsUA = xr.open_dataset(F_UA)
dsAD = xr.open_dataset(F_AD)
dsSI = xr.open_dataset(F_SI)

assert float(dsW["pressure_level"].values[0]) == 500.0, "Pressure level of the W500 file is not 500 hPa!"

lat = dsW["latitude"].values            # descending (90 to -90)
lon = dsW["longitude"].values           # 0 to 359.75
nlat, nlon = lat.size, lon.size

months = pd.period_range(f"{Y0}-01", f"{Y1}-12", freq="M")
nM = len(months)                         # 720 months


def tname_of(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def start_index(ds, name):
    """Index of month Y0-01 in the file; compared by month (hour 00 or 06 does not matter)."""
    p = pd.DatetimeIndex(ds[tname_of(ds)].values).to_period("M")
    pos = pd.Series(np.arange(len(p)), index=p)
    idx = pos.reindex(months)
    if idx.isna().any():
        raise RuntimeError(f"{name}: some months of {Y0}-{Y1} are missing from the file "
                           f"(file range: {p.min()} to {p.max()}).")
    idx = idx.values.astype(int)
    if not np.all(np.diff(idx) == 1):
        raise RuntimeError(f"{name}: months are not contiguous.")
    if "expver" in ds:
        ev = np.asarray(ds["expver"].values)[idx]
        if np.any(ev != "0001"):
            print(f"Warning: {name} contains expver values other than 0001 in the selected range ({np.unique(ev)}).")
    return int(idx[0])


I0 = {"w": start_index(dsW, "W500"),
      "ua": start_index(dsUA, "avgua"),
      "ad": start_index(dsAD, "avgad"),
      "si": start_index(dsSI, "windspeed.nc")}

# --- wind speed file ---
print("\n=== windspeed.nc ===")
print(dsSI)


def detect_si_var(ds):
    if SI10_VAR is not None:
        return SI10_VAR
    for c in ("si10", "wind_speed", "10si", "ws10", "windspeed"):
        if c in ds.data_vars:
            return c
    for v in ds.data_vars:
        ln = str(ds[v].attrs.get("long_name", "")).lower()
        if "wind speed" in ln and ds[v].ndim >= 3:
            return v
    cand = [v for v in ds.data_vars if ds[v].ndim >= 3]
    if len(cand) == 1:
        return cand[0]
    raise RuntimeError(f"Could not identify the wind speed variable: {list(ds.data_vars)}. "
                       f"Set SI10_VAR manually.")


SI_VAR = detect_si_var(dsSI)
SI_TN = tname_of(dsSI)
print(f"\nWind speed variable: {SI_VAR}  units={dsSI[SI_VAR].attrs.get('units', '-')}  "
      f"long_name={dsSI[SI_VAR].attrs.get('long_name', '-')}")
si_lat = dsSI["latitude"].values
si_lon = dsSI["longitude"].values
SI_FLIP = bool(si_lat[0] < si_lat[-1])           # flip if latitude is ascending
si_lat_cmp = si_lat[::-1] if SI_FLIP else si_lat
assert si_lat_cmp.shape == lat.shape and np.allclose(si_lat_cmp, lat), \
    "The latitude grid of windspeed.nc differs from the other data."
assert si_lon.shape == lon.shape and np.allclose(si_lon, lon), \
    "The longitude grid of windspeed.nc differs from the other data."

# ---------------------------------------------------------------
# Masks
# ---------------------------------------------------------------
st = xr.open_dataset(F_STATIC)
assert st["lsm"].shape == (nlat, nlon), "Dimensions of static_fields.nc do not match the data."
assert np.allclose(st["latitude"].values[::-1], lat), "Latitude of static_fields differs from the data."
assert np.allclose(st["longitude"].values, lon), "Longitude of static_fields differs from the data."
lsm = st["lsm"].values[::-1]            # descending like the data
hgt = st["height_m"].values[::-1]
ocean = lsm < OCEAN_LSM


def lon_band(a, b):
    return ((lon >= a) & (lon < b))[None, :] & np.ones((nlat, 1), bool)


MASKS, NCOLS = {}, {}
MASKS["all"] = np.ones((nlat, nlon), bool)
NCOLS["all"] = nlon
for h in H_LIST:
    nm = f"nomtn{int(h)}"
    MASKS[nm] = hgt < h
    NCOLS[nm] = nlon
MASKS["ocean"] = ocean
NCOLS["ocean"] = nlon
for name, (a, b) in SECTOR_LON.items():
    MASKS[name] = ocean & lon_band(a, b)
    NCOLS[name] = int(((lon >= a) & (lon < b)).sum())

sec_names = list(MASKS)
MASKF = {k: v.astype("float32") for k, v in MASKS.items()}
DEN = {k: MASKF[k].sum(axis=1) for k in sec_names}              # number of valid points in each latitude row
OK_ROW = {k: DEN[k] >= MIN_COVER * NCOLS[k] for k in sec_names}


def read_year(var, y):
    """Monthly fields of one year (12 x lat x lon) as float32."""
    k = (y - Y0) * 12
    if var == "w500":
        sl = slice(I0["w"] + k, I0["w"] + k + 12)
        return dsW["w"].isel(pressure_level=0, valid_time=sl).values.astype("float32")
    if var == "wspd":
        da = dsSI[SI_VAR]
        for d in list(da.dims):
            if d not in (SI_TN, "latitude", "longitude"):
                da = da.isel({d: 0})
        sl = slice(I0["si"] + k, I0["si"] + k + 12)
        arr = da.isel({SI_TN: sl}).values.astype("float32")
        return arr[:, ::-1, :] if SI_FLIP else arr
    if var == "tcc":
        sl = slice(I0["ua"] + k, I0["ua"] + k + 12)
        return dsUA["tcc"].isel(valid_time=sl).values.astype("float32")
    if var == "ssrd":
        sl = slice(I0["ad"] + k, I0["ad"] + k + 12)
        return (dsAD["ssrd"].isel(valid_time=sl).values / 86400.0).astype("float32")
    raise ValueError(var)


Z = {v: np.full((len(sec_names), nM, nlat), np.nan, dtype="float32") for v in VARS}
SUM = {v: {t: np.zeros((nlat, nlon), dtype="float64") for t in ("all", "p1", "p2")} for v in VARS}
CNT = {"all": 0, "p1": 0, "p2": 0}

t0 = time.time()
for var in VARS:
    warned = False
    for y in range(Y0, Y1 + 1):
        arr = read_year(var, y)                          # (12, lat, lon)
        if not warned and not np.isfinite(arr).all():
            print(f"Warning: variable {var} (year {y}) contains NaN; the affected latitude rows become NaN.")
            warned = True
        k = (y - Y0) * 12
        for si, name in enumerate(sec_names):
            num = np.einsum("tyx,yx->ty", arr, MASKF[name], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                zm_ = num / DEN[name][None, :]
            zm_[:, ~OK_ROW[name]] = np.nan
            Z[var][si, k:k + 12, :] = zm_
        s = arr.sum(axis=0, dtype="float64")
        SUM[var]["all"] += s
        if P1[0] <= y <= P1[1]:
            SUM[var]["p1"] += s
        if P2[0] <= y <= P2[1]:
            SUM[var]["p2"] += s
        if var == VARS[0]:
            CNT["all"] += 12
            if P1[0] <= y <= P1[1]:
                CNT["p1"] += 12
            if P2[0] <= y <= P2[1]:
                CNT["p2"] += 12
        if y % 5 == 0:
            print(f"{var}: year {y} done  (elapsed {time.time() - t0:.0f} s)", flush=True)

assert CNT["all"] == 720 and CNT["p1"] == 360 and CNT["p2"] == 360, CNT

# ---------------------------------------------------------------
# Sanity check: mean wind speed over land and ocean
# ---------------------------------------------------------------
sp = SUM["wspd"]["all"] / CNT["all"]
wa = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
subtrop = ((np.abs(lat) >= 15) & (np.abs(lat) <= 45))[:, None] & np.ones((nlat, nlon), bool)
print("\n=== Area-mean 10 m wind speed (windspeed.nc), 1966-2025 ===")
for region, rm in (("global", np.ones((nlat, nlon), bool)), ("subtropics 15-45 deg", subtrop)):
    for sname, sm in (("land", lsm >= 0.5), ("ocean", ocean)):
        mk = rm & sm
        print(f"  {region:22s} {sname:5s}: {(wa * sp * mk).sum() / (wa * mk).sum():5.2f} m/s")

# ---------------------------------------------------------------
# Saving (ascending latitude)
# ---------------------------------------------------------------
lat_asc = lat[::-1]

zm = xr.Dataset(
    {v: (("sector", "time", "latitude"), Z[v][:, :, ::-1]) for v in VARS},
    coords={"sector": sec_names, "time": months.to_timestamp(), "latitude": lat_asc},
)
zm["w500"].attrs.update(units="Pa s-1", note="positive = subsidence")
zm["wspd"].attrs.update(units="m s-1", note="monthly mean of 10-m wind speed (windspeed.nc)")
zm["tcc"].attrs.update(units="0-1")
zm["ssrd"].attrs.update(units="W m-2", note="J m-2 / 86400")
zm.attrs["H_LIST_m"] = str(H_LIST)
zm.attrs["ocean_lsm_threshold"] = OCEAN_LSM
zm.attrs["min_cover"] = MIN_COVER
zm.attrs["sector_lon_0_360"] = str(SECTOR_LON)
zm.attrs["period_all"] = f"{Y0}-{Y1}"
zm.to_netcdf(os.path.join(OUT_DIR, "zonal_means_masked.nc"))

clim = xr.Dataset(coords={"latitude": lat_asc, "longitude": lon})
for v in VARS:
    for tag in ("all", "p1", "p2"):
        clim[f"{v}_{tag}"] = (("latitude", "longitude"),
                              (SUM[v][tag] / CNT[tag])[::-1].astype("float32"))
clim.attrs.update(period_all=f"{Y0}-{Y1}", period_p1=str(P1), period_p2=str(P2))
clim.to_netcdf(os.path.join(OUT_DIR, "climatology.nc"))

print("\nDone. Outputs in:", OUT_DIR)
print(" - zonal_means_masked.nc")
print(" - climatology.nc")
