# -*- coding: utf-8 -*-
"""
inspect_era5_files.py
Prints the structure of the new ERA5 files so that the analysis scripts can be written for the real layout:

    w300-w700.nc   omega at 300 and 700 hPa
    P-SLP.nc       surface pressure and mean sea-level pressure
    V(Levels).nc   meridional wind on 11 pressure levels

For every file: dimensions, coordinates (name, size, first/last values, order, spacing), variables
(dims, dtype, units, long_name, packing attributes, fill value), time axis (range, number of months,
gaps, duplicates, expver), a quick statistics check of one time step (min / mean / max, NaN fraction),
and whether the horizontal grid is identical to static_fields.nc.

Output: printed on screen and saved in DATA_DIR/outputs/inspect_report.txt  .
Run:  python inspect_era5_files.py
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
FILES = ["w300-w700.nc", "P-SLP.nc", "V(Levels).nc"]
os.makedirs(OUT_DIR, exist_ok=True)

lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


def describe_coord(name, v):
    a = np.asarray(v.values)
    if a.ndim != 1 or a.size == 0:
        return f"    {name}: shape {a.shape}"
    if np.issubdtype(a.dtype, np.number):
        d = np.diff(a.astype(float)) if a.size > 1 else np.array([np.nan])
        order = "ascending" if np.all(d > 0) else ("descending" if np.all(d < 0) else "not monotonic")
        step = f"step {np.nanmedian(np.abs(d)):.4g}" if a.size > 1 else ""
        return f"    {name}: n={a.size}, {a[0]} .. {a[-1]}, {order}, {step}, units={v.attrs.get('units', '-')}"
    return f"    {name}: n={a.size}, {a[0]} .. {a[-1]}, dtype={a.dtype}"


static = None
sp = os.path.join(OUT_DIR, "static_fields.nc")
if os.path.exists(sp):
    static = xr.open_dataset(sp)

for fn in FILES:
    path = os.path.join(DATA_DIR, fn)
    out("=" * 100)
    out(f"FILE: {fn}")
    if not os.path.exists(path):
        cands = glob.glob(os.path.join(DATA_DIR, fn.split(".")[0][:4] + "*.nc"))
        out(f"  NOT FOUND at {path}. Similar files in the folder: {[os.path.basename(c) for c in cands]}")
        continue
    out(f"  size: {os.path.getsize(path) / 1e6:.1f} MB")
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        out(f"  could not open: {e}")
        continue
    out(f"  global attributes: { {k: str(v)[:80] for k, v in list(ds.attrs.items())[:8]} }")
    out(f"  dimensions: {dict(ds.sizes)}")
    out("  coordinates:")
    for c in ds.coords:
        out(describe_coord(c, ds[c]))
    out("  variables:")
    for v in ds.data_vars:
        da = ds[v]
        enc = da.encoding
        out(f"    {v}: dims={da.dims}, shape={da.shape}, dtype={da.dtype}")
        out(f"       units={da.attrs.get('units', '-')}, long_name={da.attrs.get('long_name', '-')}, "
            f"standard_name={da.attrs.get('standard_name', '-')}")
        out(f"       packing: scale_factor={enc.get('scale_factor', '-')}, add_offset={enc.get('add_offset', '-')}, "
            f"_FillValue={enc.get('_FillValue', '-')}, on-disk dtype={enc.get('dtype', '-')}")

    # time axis
    tname = next((c for c in ["valid_time", "time"] if c in ds.coords), None)
    if tname is not None:
        t = pd.DatetimeIndex(ds[tname].values)
        out(f"  time axis '{tname}': {t.min():%Y-%m-%d %H:%M} .. {t.max():%Y-%m-%d %H:%M}, n={len(t)}")
        mm = pd.PeriodIndex(t, freq="M")
        full = pd.period_range(mm.min(), mm.max(), freq="M")
        miss = full.difference(mm)
        out(f"    unique months: {mm.nunique()}, missing months: {len(miss)} {list(miss.astype(str))[:6]}, "
            f"duplicates: {int(mm.duplicated().sum())}, monotonic: {bool(t.is_monotonic_increasing)}")
        out(f"    hours of day present: {sorted(set(t.hour))}, day of month present: {sorted(set(t.day))[:5]}")
    if "expver" in ds.coords or "expver" in ds.variables:
        ev = np.unique(np.asarray(ds["expver"].values))
        out(f"  expver values: {ev}  (more than one value means ERA5/ERA5T are mixed)")
    if "number" in ds.coords:
        out(f"  ensemble 'number': {np.asarray(ds['number'].values)}")

    # quick numbers
    out("  quick statistics (first time step, all levels if present):")
    for v in ds.data_vars:
        da = ds[v]
        sel = da
        if tname in da.dims:
            sel = da.isel({tname: 0})
        if "expver" in sel.dims:
            sel = sel.isel(expver=0)
        vals = np.asarray(sel.values, dtype=float)
        out(f"    {v}: min={np.nanmin(vals):.5g}  mean={np.nanmean(vals):.5g}  max={np.nanmax(vals):.5g}  "
            f"NaN fraction={np.isnan(vals).mean():.4f}")
        lev = next((c for c in ["pressure_level", "level", "plev"] if c in sel.dims), None)
        if lev is not None:
            for k in range(sel.sizes[lev]):
                x = np.asarray(sel.isel({lev: k}).values, dtype=float)
                out(f"       {lev}={float(sel[lev].values[k]):g}: min={np.nanmin(x):.4g} mean={np.nanmean(x):.4g} "
                    f"max={np.nanmax(x):.4g} NaN={np.isnan(x).mean():.4f}")

    # grid comparison with static fields
    if static is not None:
        la = next((c for c in ["latitude", "lat"] if c in ds.coords), None)
        lo = next((c for c in ["longitude", "lon"] if c in ds.coords), None)
        if la and lo:
            a1, a2 = np.sort(np.asarray(ds[la].values, float)), np.sort(np.asarray(static["latitude"].values, float))
            b1, b2 = np.asarray(ds[lo].values, float), np.asarray(static["longitude"].values, float)
            same_lat = a1.shape == a2.shape and np.allclose(a1, a2)
            same_lon = b1.shape == b2.shape and np.allclose(b1, b2)
            out(f"  grid identical to static_fields.nc: latitude {same_lat}, longitude {same_lon}")
    ds.close()

out("=" * 100)
out("Done. Please send DATA_DIR/outputs/inspect_report.txt")
with open(os.path.join(OUT_DIR, "inspect_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
