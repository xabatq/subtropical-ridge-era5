# -*- coding: utf-8 -*-
"""
07_clara_cloud_check.py
Independent test of the cloud trend: CLARA-A3 (AVHRR, 1979-2020 in your download) versus ERA5 and CERES.

Inputs:
  CLARA-A3/CFCmmYYYYMM01000000319AVPOS01GL.nc   (one file per month)
  outputs/zonal_means_masked.nc , outputs/static_fields.nc          (from 01 and 00)
  outputs/ceres_era5_ridge_by_year.csv                              (from 04; optional, for the three-way comparison)

Steps:
  1) read the files, ocean-only zonal mean, bin to a 1-degree grid (cached in an npz file; the second run is fast)
  2) cloud-minimum latitude (argmin and centroid) with the same methods as 02 and 04
  3) compare with ERA5 for 1979-2020 and the sub-periods 1979-1999 and 2000-2020
Outputs: clara_ridge_by_year.csv , clara_vs_era5_cloud_ridge.csv , fig_clara_three_way.png
"""
import os
import re
import warnings
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")   # empty-slice / constant-input warnings are expected for some seasons

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
CLARA_DIR = os.path.join(DATA_DIR, "CLARA-A3")
OUT_DIR = os.path.join(DATA_DIR, "outputs")
CACHE = os.path.join(OUT_DIR, "clara_ocean_zonal_1deg.npz")
FILE_GLOB = "CFCmm*.nc"
VAR_NAME = None                       # if auto-detection is wrong, put the variable name here
OCEAN_LSM = 0.1
MIN_COVER = 0.25
NSMOOTH = 3                           # 1-degree points (same as 04)
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"full": None, "1979-1999": (1979, 1999), "2000-end": (2000, 9999)}
LAT1 = np.arange(-89.5, 90, 1.0)

# ---------------------------------------------------------------
# 1) Read CLARA
# ---------------------------------------------------------------
files = sorted(glob.glob(os.path.join(CLARA_DIR, FILE_GLOB)))
if not files:
    raise SystemExit(f"No file matching {FILE_GLOB} in {CLARA_DIR}.")
pat = re.compile(r"CFCmm(\d{4})(\d{2})")
times = []
for f in files:
    m = pat.search(os.path.basename(f))
    if not m:
        raise SystemExit(f"Cannot parse the file name: {f}")
    times.append(pd.Timestamp(int(m.group(1)), int(m.group(2)), 1))
times = pd.DatetimeIndex(times)
assert times.is_unique and times.is_monotonic_increasing, "Months are duplicated or out of order"
full = pd.date_range(times[0], times[-1], freq="MS")
if len(full) != len(times):
    print(f"[warning] {len(full) - len(times)} months are missing from the file sequence; they become NaN.")
print(f"{len(files)} files: {times[0]:%Y-%m} to {times[-1]:%Y-%m}")

# Ocean mask: ERA5 lsm (nearest neighbour)
st = xr.open_dataset(os.path.join(OUT_DIR, "static_fields.nc"))
s_lat = np.asarray(st["latitude"].values, float)
s_lon = np.asarray(st["longitude"].values, float)
s_lsm = np.asarray(st["lsm"].values, float)
if s_lat[0] > s_lat[-1]:
    s_lat, s_lsm = s_lat[::-1], s_lsm[::-1]


def find_var(ds):
    if VAR_NAME:
        return VAR_NAME
    names = list(ds.data_vars)
    for n in names:
        if n.lower() == "cfc":
            return n
    for n in names:
        if "cfc" in n.lower():
            return n
    for n in names:
        ln = str(ds[n].attrs.get("long_name", "")).lower()
        if "cloud" in ln and "fraction" in ln:
            return n
    raise SystemExit(f"Cloud variable not found; variables in the file: {names}. Set VAR_NAME.")


def coord(ds, kind):
    for n in list(ds.dims) + list(ds.variables):
        if n.lower().startswith(kind):
            return n
    raise SystemExit(f"Coordinate {kind} not found")


def build_cache():
    prof = np.full((len(files), len(LAT1)), np.nan)
    ocean = None
    info_done = False
    for k, f in enumerate(files):
        ds = xr.open_dataset(f)
        vn = find_var(ds)
        da = ds[vn]
        latn, lonn = coord(ds, "lat"), coord(ds, "lon")
        lat = np.asarray(ds[latn].values, float)
        lon = np.asarray(ds[lonn].values, float)
        da2 = da.squeeze()
        arr = np.asarray(da2.values, float)
        if arr.ndim != 2:
            raise SystemExit(f"Array {vn} is not 2-D after squeeze: {arr.shape}, dims={da.dims}")
        if str(da2.dims[0]).lower().startswith("lon"):   # (lon, lat) -> (lat, lon)
            arr = arr.T
        if lat[0] > lat[-1]:
            lat, arr = lat[::-1], arr[::-1]
        if not info_done:
            u = da.attrs.get("units", "?")
            print(f"Variable: {vn} | dims={da.dims} | grid {arr.shape} | units={u} | "
                  f"lat {lat[0]:.3f}..{lat[-1]:.3f} | lon {lon[0]:.3f}..{lon[-1]:.3f}")
            li = np.abs(s_lat[None, :] - lat[:, None]).argmin(axis=1)
            lj = np.abs(((s_lon[None, :] - lon[:, None] + 180) % 360) - 180).argmin(axis=1)
            ocean = s_lsm[np.ix_(li, lj)] < OCEAN_LSM
            bins = np.clip(np.floor(lat + 90).astype(int), 0, 179)
            info_done = True
        mx = np.nanmax(arr) if np.isfinite(arr).any() else np.nan
        if mx <= 1.0001:
            arr = arr * 100.0                                 # fraction -> percent
        arr = np.where((arr >= 0) & (arr <= 100), arr, np.nan)
        m = ocean & np.isfinite(arr)
        cnt = m.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            row = np.where(m, arr, 0.0).sum(axis=1) / cnt
        row[cnt < MIN_COVER * arr.shape[1]] = np.nan
        for b in range(180):
            v = row[bins == b]
            v = v[np.isfinite(v)]
            if len(v):
                prof[k, b] = v.mean()
        ds.close() if hasattr(ds, "close") else None
        if (k + 1) % 60 == 0:
            print(f"  {k + 1}/{len(files)} read")
    return prof


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    if len(z["times"]) == len(times) and str(z["times"][0]) == str(times[0]):
        print("Cache found; skipping file reading:", CACHE)
        c_prof = z["prof"]
    else:
        c_prof = build_cache()
        np.savez(CACHE, prof=c_prof, times=times.values.astype("datetime64[ns]"))
else:
    c_prof = build_cache()
    np.savez(CACHE, prof=c_prof, times=times.values.astype("datetime64[ns]"))

c_lat = LAT1
print(f"Mean ocean cloud 15-45 (CLARA): {np.nanmean(c_prof[:, (c_lat >= 15) & (c_lat <= 45)]):.1f}%")

# ---------------------------------------------------------------
# ERA5 tcc on the same 1-degree grid
# ---------------------------------------------------------------
WTS5 = np.array([0.5, 1, 1, 1, 0.5])
zm = xr.open_dataset(os.path.join(OUT_DIR, "zonal_means_masked.nc"))
e_lat = np.asarray(zm["latitude"].values, float)
e_time = pd.DatetimeIndex(zm["time"].values)
e_arr = np.asarray(zm["tcc"].sel(sector="ocean").values, float)
if e_lat[0] > e_lat[-1]:
    e_lat, e_arr = e_lat[::-1], e_arr[:, ::-1]


def era_to_1deg(a):
    if abs(abs(e_lat[1] - e_lat[0]) - 0.25) < 1e-6 and len(e_lat) == 721:
        num = np.zeros((a.shape[0], len(c_lat)))
        den = np.zeros_like(num)
        for k, w in enumerate(WTS5):
            ri = (np.round((c_lat + 90) / 0.25).astype(int) + (k - 2)).clip(0, 720)
            v = a[:, ri]
            ok = np.isfinite(v)
            num += w * np.where(ok, v, 0.0)
            den += w * ok
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[den < 0.5 * WTS5.sum()] = np.nan
        return out
    out = np.full((a.shape[0], len(c_lat)), np.nan)
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i])
        if ok.sum() > 2:
            out[i] = np.interp(c_lat, e_lat[ok], a[i][ok], left=np.nan, right=np.nan)
    return out


e_prof_all = 100.0 * era_to_1deg(e_arr)
# Align the months
pos_c = {(t.year, t.month): i for i, t in enumerate(times)}
pos_e = {(t.year, t.month): i for i, t in enumerate(e_time)}
keys = sorted(set(pos_c) & set(pos_e))
tt = pd.DatetimeIndex([pd.Timestamp(y, m, 1) for y, m in keys])
Cp = c_prof[[pos_c[k] for k in keys]]
Ep = e_prof_all[[pos_e[k] for k in keys]]
print(f"Common CLARA-ERA5 months: {len(tt)} ({tt[0]:%Y-%m} to {tt[-1]:%Y-%m})")


# ---------------------------------------------------------------
# Functions (same logic as 02/04)
# ---------------------------------------------------------------
def nan_smooth(f, size, axis=0, mode="nearest"):
    m = np.isfinite(f).astype(float)
    n_ = uniform_filter1d(np.where(m > 0, f, 0.0), size, axis=axis, mode=mode)
    d_ = uniform_filter1d(m, size, axis=axis, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = n_ / d_
    out[d_ < 0.5] = np.nan
    return out


def prep(profile, window):
    y = nan_smooth(np.asarray(profile, float), NSMOOTH)
    s = (c_lat >= window[0]) & (c_lat <= window[1])
    x, y = c_lat[s], -y[s]
    if np.isfinite(y).mean() < 0.8:
        return None
    return x, y


def argmax_lat(x, y):
    i = int(np.nanargmax(y))
    edge = i in (0, len(y) - 1)
    if edge or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(x[i]), edge
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(x[i] + off * (x[1] - x[0])), False


def centroid_lat(x, y):
    lo, hi = np.nanmin(y), np.nanmax(y)
    thr = lo + 0.5 * (hi - lo)
    w = np.where(np.isfinite(y), np.clip(y - thr, 0, None), 0.0)
    return float((w * x).sum() / w.sum()) if w.sum() > 0 else np.nan


def season_series(arr, tt, months):
    m, y = tt.month.values, tt.year.values
    if months == [12, 1, 2]:
        y = y + (m == 12)
    sel = np.isin(m, months)
    years, out = [], []
    for yr in np.unique(y[sel]):
        idx = sel & (y == yr)
        if idx.sum() == len(months):
            years.append(int(yr))
            out.append(np.nanmean(arr[idx], axis=0))
    return np.array(years), np.array(out)


def trend(years, v):
    ok = np.isfinite(v)
    yrs, v = years[ok].astype(float), v[ok]
    n = len(v)
    if n < 10:
        return dict(n=n, ols=np.nan, p=np.nan, sen=np.nan)
    o = stats.linregress(yrs, v)
    res = v - (o.intercept + o.slope * yrs)
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    ne = max(n * (1 - r1) / (1 + r1), 3.0)
    se = o.stderr * np.sqrt((n - 2) / (ne - 2))
    p = 2 * stats.t.sf(abs(o.slope / se), ne - 2)
    return dict(n=n, ols=o.slope * 10, p=p, sen=stats.theilslopes(v, yrs)[0] * 10)


rows = []
for src, prof in [("CLARA", Cp), ("ERA5", Ep)]:
    for sname, months in SEASONS.items():
        years, sp = season_series(prof, tt, months)
        for hemi, win in WINDOWS.items():
            for yr, p in zip(years, sp):
                pr = prep(p, win)
                if pr is None:
                    continue
                x, y = pr
                la, edge = argmax_lat(x, y)
                rows.append((src, hemi, sname, "argmin", int(yr), abs(la), edge))
                rows.append((src, hemi, sname, "centroid", int(yr), abs(centroid_lat(x, y)), edge))
yy = pd.DataFrame(rows, columns=["source", "hemi", "season", "method", "year", "abs_lat", "at_window_edge"])
yy.to_csv(os.path.join(OUT_DIR, "clara_ridge_by_year.csv"), index=False)

# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
res = []
for (hemi, sname, meth), g in yy.groupby(["hemi", "season", "method"]):
    a = g[g.source == "CLARA"].set_index("year")["abs_lat"]
    b = g[g.source == "ERA5"].set_index("year")["abs_lat"]
    yr_all = a.index.intersection(b.index)
    for pname, rng_ in PERIODS.items():
        yr = yr_all if rng_ is None else yr_all[(yr_all >= rng_[0]) & (yr_all <= rng_[1])]
        if len(yr) < 10:
            continue
        aa, bb = a.loc[yr], b.loc[yr]
        r_p = stats.pearsonr(aa, bb)
        ad = aa - np.polyval(np.polyfit(yr, aa, 1), yr)
        bd = bb - np.polyval(np.polyfit(yr, bb, 1), yr)
        ta, tb = trend(yr.values, aa.values), trend(yr.values, bb.values)
        res.append(dict(hemi=hemi, season=sname, method=meth, period=pname, n=len(yr),
                        years=f"{yr.min()}-{yr.max()}", mean_clara=aa.mean(), mean_era5=bb.mean(),
                        bias_clara_minus_era5=(aa - bb).mean(), pearson_r=r_p[0],
                        r_detrended=stats.pearsonr(ad, bd)[0],
                        trend_clara=ta["ols"], p_clara=ta["p"], sen_clara=ta["sen"],
                        trend_era5=tb["ols"], p_era5=tb["p"], sen_era5=tb["sen"]))
rs = pd.DataFrame(res)
rs.to_csv(os.path.join(OUT_DIR, "clara_vs_era5_cloud_ridge.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
cols = ["hemi", "season", "period", "n", "pearson_r", "r_detrended", "bias_clara_minus_era5",
        "trend_clara", "p_clara", "trend_era5", "p_era5"]
print("\n=== CLARA versus ERA5 - cloud-minimum latitude (centroid), ocean ===")
print(rs[rs.method == "centroid"][cols].round(3).to_string(index=False))
print("\n=== argmin (for the appendix) ===")
print(rs[rs.method == "argmin"][cols].round(3).to_string(index=False))

# ---------------------------------------------------------------
# Figure: series (CLARA, ERA5, and CERES if available)
# ---------------------------------------------------------------
cer = None
pc = os.path.join(OUT_DIR, "ceres_era5_ridge_by_year.csv")
if os.path.exists(pc):
    cer = pd.read_csv(pc)
    cer = cer[cer.source == "CERES"]
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, h in enumerate(["NH", "SH"]):
    for j, se in enumerate(["Annual", "DJF", "JJA"]):
        a = ax[i, j]
        for src, c, lw in [("ERA5", "tab:orange", 1.1), ("CLARA", "tab:green", 1.1)]:
            g = yy[(yy.hemi == h) & (yy.season == se) & (yy.method == "centroid") & (yy.source == src)].sort_values("year")
            a.plot(g.year, g.abs_lat, "-o", ms=2.5, lw=lw, color=c, label=f"{src} (CLARA-A3)" if src == "CLARA" else src)
        if cer is not None:
            g = cer[(cer.hemi == h) & (cer.season == se) & (cer.method == "centroid")].sort_values("year")
            a.plot(g.year, g.abs_lat, "-o", ms=2.5, lw=1.1, color="tab:blue", label="CERES EBAF")
        a.axvspan(1979, 1999.5, color="0.9", zorder=0)
        a.set_title(f"{'Northern' if h == 'NH' else 'Southern'} Hemisphere - {se}", fontsize=10)
        a.set_xlabel("Year")
        a.set_ylabel("|Latitude| of cloud-minimum centroid (deg)", fontsize=8)
        a.grid(alpha=0.3)
        if i == 0 and j == 0:
            a.legend(fontsize=8)
fig.suptitle("Ocean cloud-minimum latitude: ERA5, CLARA-A3 and CERES EBAF (grey: 1979-1999)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_clara_three_way.png"), dpi=150)
plt.close(fig)
print("\nDone. Outputs in:", OUT_DIR)
print(" - clara_ridge_by_year.csv, clara_vs_era5_cloud_ridge.csv, fig_clara_three_way.png, cache:", CACHE)
