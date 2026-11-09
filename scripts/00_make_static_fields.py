# -*- coding: utf-8 -*-
"""
00_make_static_fields.py
Builds outputs/static_fields.nc (land-sea mask and surface height on the ERA5 0.25-degree grid), which all later
scripts use for the ocean and mountain masks.

Input : one ERA5 NetCDF file in DATA_DIR that contains the time-invariant fields
            lsm   land-sea mask (0-1)
            z     surface geopotential (m2/s2)   [or 'orog' in m]
        (Copernicus CDS: "ERA5 hourly data on single levels", variables "Land-sea mask" and "Geopotential",
         one time step, global, 0.25 x 0.25 degrees, NetCDF.)  The file is found automatically: the first *.nc file
        in DATA_DIR that contains 'lsm' and 'z' (or 'orog'). It can also be given as the first argument.
Output: DATA_DIR/outputs/static_fields.nc with variables lsm and height_m (= z / 9.80665), latitude in ASCENDING order
        (-90..90) and longitude 0..359.75 (the layout the other scripts expect).

If outputs/static_fields.nc already exists it is not overwritten; the script only checks that it is consistent.
Run:  python 00_make_static_fields.py [static_file.nc]
"""
import glob
import os
import sys
import numpy as np
import xarray as xr

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
OUT = os.path.join(OUT_DIR, "static_fields.nc")
G = 9.80665
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(OUT):
    st = xr.open_dataset(OUT)
    lat = np.asarray(st["latitude"].values, float)
    lon = np.asarray(st["longitude"].values, float)
    ok = (lat[0] < lat[-1]) and lon[0] == 0.0 and lon.size == 1440 and lat.size == 721
    print(f"{OUT} already exists ({lat.size} x {lon.size}); layout {'OK' if ok else 'UNEXPECTED'}. Nothing to do.")
    sys.exit(0 if ok else 1)

cands = [sys.argv[1]] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))
src = None
for f in cands:
    try:
        ds = xr.open_dataset(f)
    except Exception:
        continue
    if "lsm" in ds.data_vars and ("z" in ds.data_vars or "orog" in ds.data_vars):
        src = f
        break
if src is None:
    raise SystemExit("No NetCDF file with 'lsm' and 'z' (or 'orog') found in " + DATA_DIR +
                     ". Download the static fields from the CDS (see docs/DATA.md) and run again.")
print("Using", src)


def field2d(da):
    """First time step / member of a (possibly time-dependent) field -> 2-D array (lat, lon)."""
    for d in list(da.dims):
        if d not in ("latitude", "longitude"):
            da = da.isel({d: 0})
    return np.asarray(da.values, dtype="float64")


lat = np.asarray(ds["latitude"].values, float)
lon = np.asarray(ds["longitude"].values, float)
lsm = field2d(ds["lsm"])
if "z" in ds.data_vars:
    height = field2d(ds["z"]) / G
else:
    height = field2d(ds["orog"])
# latitude ascending
if lat[0] > lat[-1]:
    lat, lsm, height = lat[::-1], lsm[::-1], height[::-1]
# longitude 0 .. 359.75
order = np.argsort(np.mod(lon, 360.0), kind="stable")
lon, lsm, height = np.mod(lon, 360.0)[order], lsm[:, order], height[:, order]
if lat.size != 721 or lon.size != 1440:
    print(f"Warning: grid is {lat.size} x {lon.size}, not the 721 x 1440 grid of the other files.")
out = xr.Dataset({"lsm": (("latitude", "longitude"), lsm.astype("float32")),
                  "height_m": (("latitude", "longitude"), height.astype("float32"))},
                 coords={"latitude": lat, "longitude": lon})
out["lsm"].attrs.update(long_name="land-sea mask", units="1")
out["height_m"].attrs.update(long_name="surface height (geopotential / g)", units="m")
out.to_netcdf(OUT)
print("Written:", OUT, f"(land fraction {float((lsm >= 0.5).mean()):.3f}, max height {float(height.max()):.0f} m)")
