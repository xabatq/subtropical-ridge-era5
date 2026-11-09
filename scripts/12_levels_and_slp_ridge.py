# -*- coding: utf-8 -*-
"""
12_levels_and_slp_ridge.py
Additional ridge indicators from the new ERA5 files:

    w300-w700.nc   omega at 700 and 300 hPa   (the 500 hPa result already exists in the main analysis)
    P-SLP.nc       mean sea-level pressure (msl) and surface pressure (sp)

Part A (slow, run once, results are cached):
    * ocean zonal means (lsm < 0.1, MIN_COVER 0.25 per latitude row, same rule as script 01) of w700, w300 and
      msl for the whole ocean and for the five ocean basin sectors of script 01;
    * msl on a 1-degree ocean grid (for the position of the individual subtropical highs).
    Caches: outputs/levels_slp_zonal_means.npz, outputs/slp_ocean_1deg.npz  (delete them to force a re-read).

Part B (fast):
    * ridge latitude of omega-700, omega-300 and SLP (argmax with parabolic interpolation + half-range centroid,
      15-45 deg windows, 9-point smoothing - identical to the main analysis) per year and season;
    * ENSO(ONI)-adjusted OLS trends with 95% CI (lag-1 effective sample size), Theil-Sen, Mann-Kendall,
      Benjamini-Hochberg q; omega-500 (from peak_latitudes_by_year_masked.csv) is added for comparison, so that the
      FDR family for the vertical-structure table is 4 levels x 2 hemispheres x 3 seasons x 2 estimators = 48
      tests per period. The q-values of the main 42-test family (script 08) are NOT changed by this script;
    * the same analysis for the basin sectors (annual);
    * correlation of the new ridge series with the 500 hPa series;
    * individual SLP highs (North Pacific, North Atlantic, South Pacific, South Atlantic, South Indian):
      latitude, longitude and central pressure of the maximum of the smoothed ocean SLP field in a fixed box.

Inputs : DATA_DIR/w300-w700.nc, DATA_DIR/P-SLP.nc, DATA_DIR/outputs/static_fields.nc,
         DATA_DIR/outputs/peak_latitudes_by_year_masked.csv, DATA_DIR/enso_monthly_1966_2025.csv
Outputs (DATA_DIR/outputs):
    levels_slp_ridge_by_year.csv, levels_slp_trends_ocean.csv, levels_slp_basin_trends.csv,
    levels_slp_mean_latitude.csv, levels_slp_correlations.csv,
    slp_high_centres_by_year.csv, slp_high_trends.csv,
    fig_levels_slp_ridge.png, fig_slp_highs.png

Run:  python 12_levels_and_slp_ridge.py      (Part A: tens of minutes, Part B: about a minute)
"""
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import uniform_filter1d, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
F_W = os.path.join(DATA_DIR, "w300-w700.nc")
F_P = os.path.join(DATA_DIR, "P-SLP.nc")
F_STATIC = os.path.join(OUT_DIR, "static_fields.nc")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
PEAK_CSV = os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv")
CACHE_ZM = os.path.join(OUT_DIR, "levels_slp_zonal_means.npz")
CACHE_SLP = os.path.join(OUT_DIR, "slp_ocean_1deg.npz")

Y0, Y1 = 1966, 2025
OCEAN_LSM = 0.1
MIN_COVER = 0.25
NSMOOTH = 9
SECTOR_LON = {"atlantic": (300.0, 350.0), "pacific_w": (150.0, 200.0), "pacific_c": (200.0, 250.0),
              "pacific_e": (250.0, 290.0), "indian": (55.0, 95.0)}
SECTORS = ["ocean"] + list(SECTOR_LON)
BASINS = list(SECTOR_LON)
LEVELS = {"w700": 700.0, "w300": 300.0}          # variable name -> pressure level (hPa)
NEW_VARS = ["w700", "w300", "msl"]
ALL_VARS = ["w700", "w500", "w300", "msl"]
KIND = {"w700": "max", "w500": "max", "w300": "max", "msl": "max"}
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
MIN_N = 20
MAX_EDGE = 0.3

# boxes for the individual subtropical highs: (lat_min, lat_max, lon_start, lon_end) with lon in 0..360 (may wrap)
HIGH_BOXES = {
    "N Pacific high":  (20.0, 45.0, 160.0, 250.0),
    "N Atlantic high": (20.0, 45.0, 290.0, 350.0),
    "S Pacific high":  (-45.0, -15.0, 220.0, 290.0),
    "S Atlantic high": (-45.0, -15.0, 320.0, 10.0),
    "S Indian high":   (-45.0, -15.0, 55.0, 110.0),
}
HIGH_SMOOTH_DEG = 5.0

months = pd.period_range(f"{Y0}-01", f"{Y1}-12", freq="M")
nM = len(months)


# ===============================================================
# Part A: extraction (cached)
# ===============================================================
def tname_of(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def start_index(ds, name):
    """Index of month Y0-01 in the file; months are matched by (year, month), the hour does not matter."""
    p = pd.DatetimeIndex(ds[tname_of(ds)].values).to_period("M")
    pos = pd.Series(np.arange(len(p)), index=p)
    idx = pos.reindex(months)
    if idx.isna().any():
        raise RuntimeError(f"{name}: some months of {Y0}-{Y1} are missing (file range {p.min()} to {p.max()}).")
    idx = idx.values.astype(int)
    if not np.all(np.diff(idx) == 1):
        raise RuntimeError(f"{name}: months are not contiguous.")
    if "expver" in ds:
        ev = np.asarray(ds["expver"].values)[idx]
        if np.any(ev != "0001"):
            print(f"Warning: {name} contains expver values other than 0001 ({np.unique(ev)}).")
    return int(idx[0])


def extract():
    for f in (F_W, F_P, F_STATIC):
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")
    dsW = xr.open_dataset(F_W)
    dsP = xr.open_dataset(F_P)
    st = xr.open_dataset(F_STATIC)

    lat = np.asarray(dsP["latitude"].values, float)       # descending 90 .. -90
    lon = np.asarray(dsP["longitude"].values, float)      # 0 .. 359.75
    nlat, nlon = lat.size, lon.size
    assert np.allclose(np.asarray(dsW["latitude"].values, float), lat), "w file: latitude differs from P-SLP.nc"
    # w file: longitude -180 .. 179.75 -> reorder to 0 .. 359.75
    lonW = np.asarray(dsW["longitude"].values, float)
    order_w = np.argsort(np.mod(lonW, 360.0), kind="stable")
    assert np.allclose(np.mod(lonW, 360.0)[order_w], lon), "w file: longitudes do not match after reordering"
    print("w file longitude reordering needed:", not np.array_equal(order_w, np.arange(nlon)))

    # static fields (ascending latitude in the file -> flip to descending)
    assert st["lsm"].shape == (nlat, nlon), "static_fields.nc has a different grid"
    assert np.allclose(np.asarray(st["latitude"].values, float)[::-1], lat), "static latitude differs"
    assert np.allclose(np.asarray(st["longitude"].values, float), lon), "static longitude differs"
    ocean = st["lsm"].values[::-1] < OCEAN_LSM

    lev = np.asarray(dsW["pressure_level"].values, float)
    kidx = {}
    for v, p in LEVELS.items():
        hit = np.where(np.isclose(lev, p))[0]
        if len(hit) != 1:
            raise SystemExit(f"Level {p} hPa not found in w300-w700.nc (levels: {lev})")
        kidx[v] = int(hit[0])
    print("level index:", kidx)

    I0W, I0P = start_index(dsW, "w300-w700.nc"), start_index(dsP, "P-SLP.nc")

    def lon_band(a, b):
        return ((lon >= a) & (lon < b))[None, :] & np.ones((nlat, 1), bool)

    masks, ncols = {"ocean": ocean}, {"ocean": nlon}
    for nm, (a, b) in SECTOR_LON.items():
        masks[nm] = ocean & lon_band(a, b)
        ncols[nm] = int(((lon >= a) & (lon < b)).sum())
    maskf = {k: v.astype("float32") for k, v in masks.items()}
    den = {k: maskf[k].sum(axis=1) for k in SECTORS}
    ok_row = {k: den[k] >= MIN_COVER * ncols[k] for k in SECTORS}

    Z = {v: np.full((len(SECTORS), nM, nlat), np.nan, dtype="float32") for v in NEW_VARS}

    # 1-degree block means of msl over ocean points
    dlat = abs(lat[1] - lat[0])
    B = max(1, int(round(1.0 / dlat)))
    nb_lat = (nlat - 1) // B
    nb_lon = nlon // B
    rows = slice(1, 1 + nb_lat * B)                       # drop the pole row so that the blocks tile exactly
    lat_b = lat[rows].reshape(nb_lat, B).mean(axis=1)
    lon_b = lon[:nb_lon * B].reshape(nb_lon, B).mean(axis=1)
    ocean_rows = ocean[rows, :nb_lon * B].astype("float32")
    cnt_b = ocean_rows.reshape(nb_lat, B, nb_lon, B).sum(axis=(1, 3))
    slp_b = np.full((nM, nb_lat, nb_lon), np.nan, dtype="float32")

    def zonal(arr, name):
        num = np.einsum("tyx,yx->ty", arr, maskf[name], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = num / den[name][None, :]
        z[:, ~ok_row[name]] = np.nan
        return z

    t0 = time.time()
    for y in range(Y0, Y1 + 1):
        k = (y - Y0) * 12
        for v in LEVELS:
            sl = slice(I0W + k, I0W + k + 12)
            arr = dsW["w"].isel(pressure_level=kidx[v], valid_time=sl).values.astype("float32")[:, :, order_w]
            if not np.isfinite(arr).all():
                print(f"Warning: {v} year {y} contains NaN; affected rows become NaN.")
            for si, name in enumerate(SECTORS):
                Z[v][si, k:k + 12, :] = zonal(arr, name)
        sl = slice(I0P + k, I0P + k + 12)
        arr = dsP["msl"].isel(valid_time=sl).values.astype("float32")
        if not np.isfinite(arr).all():
            print(f"Warning: msl year {y} contains NaN; affected rows become NaN.")
        for si, name in enumerate(SECTORS):
            Z["msl"][si, k:k + 12, :] = zonal(arr, name)
        a = arr[:, rows, :nb_lon * B] * ocean_rows[None]
        sm = a.reshape(12, nb_lat, B, nb_lon, B).sum(axis=(2, 4), dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            blk = sm / cnt_b[None]
        blk[:, cnt_b < 0.5 * B * B] = np.nan
        slp_b[k:k + 12] = blk
        if y % 5 == 0:
            print(f"year {y} done (elapsed {time.time() - t0:.0f} s)", flush=True)

    # sanity check: ocean-mean values
    for v in NEW_VARS:
        print(f"ocean-mean of {v} over all latitudes/months: {np.nanmean(Z[v][0]):.5g}")
    np.savez_compressed(CACHE_ZM, lat=lat, sectors=np.array(SECTORS), **{v: Z[v] for v in NEW_VARS})
    np.savez_compressed(CACHE_SLP, lat=lat_b, lon=lon_b, slp=slp_b)
    print("cache written:", CACHE_ZM, CACHE_SLP)


if not (os.path.exists(CACHE_ZM) and os.path.exists(CACHE_SLP)):
    extract()
else:
    print("Using cached extraction (delete the .npz files in outputs to re-read the NetCDF files).")

# ===============================================================
# Part B: ridge latitudes and statistics
# ===============================================================
zc = np.load(CACHE_ZM, allow_pickle=False)
lats = np.asarray(zc["lat"], float)
sector_names = [str(s) for s in zc["sectors"]]
ZM = {v: np.asarray(zc[v], float) for v in NEW_VARS}
tt = pd.DatetimeIndex(months.to_timestamp())
enso = pd.read_csv(ENSO_CSV)


def enso_season(mlist, col="ONI"):
    d = enso[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if mlist == [12, 1, 2] else 0)
    d = d[d.Month.isin(mlist)]
    g = d.groupby("Y")[col]
    return g.mean()[g.count() == len(mlist)]


ENSO_S = {s: enso_season(m) for s, m in SEASONS.items()}


def nan_smooth(f, size, mode="nearest"):
    m = np.isfinite(f).astype(float)
    num = uniform_filter1d(np.where(m > 0, f, 0.0), size, mode=mode)
    dn = uniform_filter1d(m, size, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / dn
    out[dn < 0.5] = np.nan
    return out


def prep(profile, window, kind):
    y = nan_smooth(np.asarray(profile, float), NSMOOTH)
    sel = (lats >= window[0]) & (lats <= window[1])
    x, y = lats[sel], y[sel]
    if kind == "min":
        y = -y
    if np.isfinite(y).mean() < 0.8:
        return None
    return x, y


def argmax_lat(x, y):
    i = int(np.nanargmax(y))
    edge = i in (0, len(y) - 1)
    if edge or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(x[i]), edge
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    return float(x[i] + off * (x[1] - x[0])), False


def centroid_lat(x, y):
    lo, hi = np.nanmin(y), np.nanmax(y)
    thr = lo + 0.5 * (hi - lo)
    w = np.where(np.isfinite(y), np.clip(y - thr, 0, None), 0.0)
    return float((w * x).sum() / w.sum()) if w.sum() > 0 else np.nan


def season_series(arr, times, mlist):
    """Seasonal means per year; DJF is assigned to the year of its January. Works on any trailing dimensions."""
    m, yy = times.month.values, times.year.values
    if mlist == [12, 1, 2]:
        yy = yy + (m == 12)
    sel = np.isin(m, mlist)
    years, out = [], []
    for yr in np.unique(yy[sel]):
        idx = sel & (yy == yr)
        if idx.sum() == len(mlist):
            years.append(int(yr))
            out.append(np.nanmean(arr[idx], axis=0))
    return np.array(years), np.array(out)


def ols_adj(y, X, j):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = n - k
    se = np.sqrt((res @ res / dof * np.linalg.inv(X.T @ X))[j, j])
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), k + 1.5)
    se_adj = se * np.sqrt(dof / (n_eff - k))
    p = 2 * stats.t.sf(abs(beta[j] / se_adj), n_eff - k)
    tc = stats.t.ppf(0.975, n_eff - k)
    return beta[j], p, beta[j] - tc * se_adj, beta[j] + tc * se_adj, n_eff, beta


def design(years, e=None):
    t = (years - 1995.5) / 10.0
    cols = [np.ones_like(t), t]
    if e is not None:
        cols.append(e)
    return np.column_stack(cols)


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qv = np.empty_like(pv)
    qv[o] = np.minimum(r, 1.0)
    q[ok] = qv
    return q


def analyse(s, season, ya, yb):
    """Raw and ENSO-adjusted trend statistics of one yearly series (units per decade)."""
    s = s[(s.index >= ya) & (s.index <= yb)].dropna()
    e_all = ENSO_S[season]
    common = s.index.intersection(e_all.index)
    if len(common) < MIN_N:
        return None
    yrs = common.values.astype(float)
    y = s.loc[common].values
    e = e_all.loc[common].values
    e = (e - e.mean()) / e.std(ddof=1)
    out = dict(n=len(y))
    b, p, lo, hi, ne, _ = ols_adj(y, design(yrs), 1)
    out.update(ols_raw=b, ols_raw_lo=lo, ols_raw_hi=hi, p_raw=p, n_eff_raw=ne)
    b, p, lo, hi, ne, beta = ols_adj(y, design(yrs, e), 1)
    y_adj = y - beta[2] * e
    sl = stats.theilslopes(y_adj, yrs, 0.95)
    out.update(ols_adj=b, ols_adj_lo=lo, ols_adj_hi=hi, p_adj=p, n_eff_adj=ne,
               sen_adj=sl[0] * 10, sen_adj_lo=sl[2] * 10, sen_adj_hi=sl[3] * 10,
               mk_p_adj=stats.kendalltau(yrs, y_adj)[1])
    return out


# ---- yearly ridge latitudes of the new variables ----
rows = []
for var in NEW_VARS:
    for si, sec in enumerate(sector_names):
        for sname, mlist in SEASONS.items():
            if sec != "ocean" and sname != "Annual":
                continue
            years, prof = season_series(ZM[var][si], tt, mlist)
            for hemi, win in WINDOWS.items():
                for yr, pr_ in zip(years, prof):
                    pr = prep(pr_, win, KIND[var])
                    if pr is None:
                        continue
                    x, y = pr
                    la_a, edge = argmax_lat(x, y)
                    la_c = centroid_lat(x, y)
                    rows.append((sec, hemi, sname, var, "argmax", int(yr), la_a, abs(la_a), edge))
                    if np.isfinite(la_c):
                        rows.append((sec, hemi, sname, var, "centroid", int(yr), la_c, abs(la_c), edge))
cols = ["sector", "hemi", "season", "var", "method", "year", "lat", "abs_lat", "at_window_edge"]
NEW = pd.DataFrame(rows, columns=cols)

# omega-500 from the main analysis (same estimators, same smoothing) for comparison
PK = pd.read_csv(PEAK_CSV)
W500 = PK[(PK["var"] == "w500") & (PK["sector"].isin(SECTORS))][cols].copy()
if W500.empty:
    print("Warning: no w500 rows in peak_latitudes_by_year_masked.csv; comparison with 500 hPa is skipped.")
ALL = pd.concat([NEW, W500], ignore_index=True)
ALL.to_csv(os.path.join(OUT_DIR, "levels_slp_ridge_by_year.csv"), index=False)


def trend_table(df, sectors, seasons=None):
    out = []
    for (sec, hemi, seas, var, meth), g in df[df.sector.isin(sectors)].groupby(["sector", "hemi", "season", "var", "method"]):
        if seasons is not None and seas not in seasons:
            continue
        s = g.set_index("year")["abs_lat"]
        for pname, (ya, yb) in PERIODS.items():
            r = analyse(s, seas, ya, yb)
            if r is None:
                continue
            r.update(sector=sec, hemi=hemi, season=seas, var=var, method=meth, period=pname,
                     edge_fraction=float(g.at_window_edge.mean()))
            r["edge_flag"] = bool(r["edge_fraction"] > MAX_EDGE)
            out.append(r)
    T = pd.DataFrame(out)
    T["q_adj"] = np.nan
    T["fdr_family_size"] = 0
    for pname, idx in T.groupby("period").groups.items():
        T.loc[idx, "q_adj"] = bh_fdr(T.loc[idx, "p_adj"].values)
        T.loc[idx, "fdr_family_size"] = int(np.isfinite(T.loc[idx, "p_adj"]).sum())
    return T


TO = trend_table(ALL, ["ocean"])
TO = TO[["sector", "hemi", "season", "var", "method", "period", "n", "edge_fraction", "edge_flag",
         "ols_raw", "ols_raw_lo", "ols_raw_hi", "p_raw",
         "ols_adj", "ols_adj_lo", "ols_adj_hi", "p_adj", "n_eff_adj", "sen_adj", "sen_adj_lo", "sen_adj_hi",
         "mk_p_adj", "q_adj", "fdr_family_size"]].sort_values(["period", "hemi", "season", "method", "var"])
TO.to_csv(os.path.join(OUT_DIR, "levels_slp_trends_ocean.csv"), index=False)

TB = trend_table(ALL, BASINS, ["Annual"])
TB.to_csv(os.path.join(OUT_DIR, "levels_slp_basin_trends.csv"), index=False)

# mean latitude (climatology of the ridge latitude) per variable
ML = (ALL[ALL.sector == "ocean"].groupby(["season", "hemi", "var", "method"])["abs_lat"]
      .agg(mean_abs_lat="mean", sd="std", n="count").reset_index())
ML.to_csv(os.path.join(OUT_DIR, "levels_slp_mean_latitude.csv"), index=False)

# correlation of the new ridge series with the 500 hPa series (annual, ocean)
cr = []
A = ALL[(ALL.sector == "ocean") & (ALL.season == "Annual")]
for (hemi, meth), g in A.groupby(["hemi", "method"]):
    P = g.pivot_table(index="year", columns="var", values="abs_lat")
    if "w500" not in P:
        continue
    for pname, (ya, yb) in PERIODS.items():
        Q = P[(P.index >= ya) & (P.index <= yb)]
        for v in [c for c in P.columns if c != "w500"]:
            d = Q[["w500", v]].dropna()
            if len(d) < MIN_N:
                continue
            tr = np.arange(len(d), dtype=float)
            dd = d.apply(lambda c: c - np.polyval(np.polyfit(tr, c.values, 1), tr))
            cr.append(dict(hemi=hemi, method=meth, period=pname, var=v, n=len(d),
                           r_with_w500=d.corr().iloc[0, 1], r_detrended=dd.corr().iloc[0, 1]))
pd.DataFrame(cr).to_csv(os.path.join(OUT_DIR, "levels_slp_correlations.csv"), index=False)

# ===============================================================
# Individual subtropical highs from the 1-degree ocean SLP field
# ===============================================================
sc = np.load(CACHE_SLP, allow_pickle=False)
latb, lonb, SLP = np.asarray(sc["lat"], float), np.asarray(sc["lon"], float), np.asarray(sc["slp"], float) / 100.0  # hPa
res = abs(latb[1] - latb[0])
fsize = max(1, int(round(HIGH_SMOOTH_DEG / res)))


def box_mask(box):
    la0, la1, lo0, lo1 = box
    in_lat = (latb >= la0) & (latb <= la1)
    span = (lo1 - lo0) % 360.0
    in_lon = ((lonb - lo0) % 360.0) <= span
    return in_lat[:, None] & in_lon[None, :]


def nan_smooth2(f, size):
    m = np.isfinite(f).astype(float)
    num = uniform_filter(np.where(m > 0, f, 0.0), size, mode=("nearest", "wrap"))
    dn = uniform_filter(m, size, mode=("nearest", "wrap"))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / dn
    out[dn < 0.5] = np.nan
    return out


def parab(a, b, c):
    d = a - 2 * b + c
    return 0.5 * (a - c) / d if d != 0 and np.isfinite(d) else 0.0


def high_centre(field, box):
    bm = box_mask(box)
    sm = nan_smooth2(field, fsize)
    f = np.where(bm, sm, np.nan)
    if np.isfinite(f).mean() == 0 or np.isfinite(f[bm]).mean() < 0.3:
        return None
    iy, ix = np.unravel_index(np.nanargmax(f), f.shape)
    # edge: the maximum touches the box boundary (a neighbour is outside the box or not finite)
    nlo = f.shape[1]
    edge = iy == 0 or iy == f.shape[0] - 1
    if not edge:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jx = (ix + dx) % nlo
                if not (bm[iy + dy, jx] and np.isfinite(f[iy + dy, jx])):
                    edge = True
    la, lo = latb[iy], lonb[ix]
    if not edge:
        la += parab(f[iy - 1, ix], f[iy, ix], f[iy + 1, ix]) * (latb[1] - latb[0])
        lo += parab(f[iy, (ix - 1) % f.shape[1]], f[iy, ix], f[iy, (ix + 1) % f.shape[1]]) * (lonb[1] - lonb[0])
    return float(la), float(lo % 360.0), float(f[iy, ix]), edge


rows = []
for sname, mlist in SEASONS.items():
    years, fld = season_series(SLP, tt, mlist)
    for bname, box in HIGH_BOXES.items():
        for yr, f in zip(years, fld):
            r = high_centre(f, box)
            if r is None:
                continue
            la, lo, pc, edge = r
            lon_ref = (box[2] + ((box[3] - box[2]) % 360.0) / 2.0) % 360.0
            rows.append((bname, sname, int(yr), la, lo, pc, abs(la), ((lo - lon_ref + 180.0) % 360.0) - 180.0, edge))
HC = pd.DataFrame(rows, columns=["high", "season", "year", "lat", "lon", "central_hPa", "abs_lat",
                                 "lon_rel_box_centre", "at_box_edge"])
HC.to_csv(os.path.join(OUT_DIR, "slp_high_centres_by_year.csv"), index=False)

rows = []
for (bname, seas), g in HC.groupby(["high", "season"]):
    for metric, col in [("abs_lat_poleward", "abs_lat"), ("lon_eastward", "lon_rel_box_centre"),
                        ("central_hPa", "central_hPa")]:
        s = g.set_index("year")[col]
        for pname, (ya, yb) in PERIODS.items():
            r = analyse(s, seas, ya, yb)
            if r is None:
                continue
            r.update(high=bname, season=seas, metric=metric, period=pname, edge_fraction=float(g.at_box_edge.mean()))
            r["edge_flag"] = bool(r["edge_fraction"] > MAX_EDGE)
            rows.append(r)
HT = pd.DataFrame(rows)
HT["q_adj"] = np.nan
HT["fdr_family_size"] = 0
for pname, idx in HT.groupby("period").groups.items():
    HT.loc[idx, "q_adj"] = bh_fdr(HT.loc[idx, "p_adj"].values)
    HT.loc[idx, "fdr_family_size"] = int(np.isfinite(HT.loc[idx, "p_adj"]).sum())
HT = HT[["high", "season", "metric", "period", "n", "edge_fraction", "edge_flag", "ols_raw", "p_raw",
         "ols_adj", "ols_adj_lo", "ols_adj_hi", "p_adj", "sen_adj", "mk_p_adj", "q_adj", "fdr_family_size"]]
HT.to_csv(os.path.join(OUT_DIR, "slp_high_trends.csv"), index=False)

# ===============================================================
# Figures
# ===============================================================
COL = {"w700": "#1b9e77", "w500": "#7570b3", "w300": "#d95f02", "msl": "#222222"}
LAB = {"w700": r"$\omega$ 700 hPa", "w500": r"$\omega$ 500 hPa", "w300": r"$\omega$ 300 hPa", "msl": "SLP"}

fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))
for j, hemi in enumerate(["NH", "SH"]):
    ax = axs[0, j]
    for v in ALL_VARS:
        g = ALL[(ALL.sector == "ocean") & (ALL.season == "Annual") & (ALL.hemi == hemi) &
                (ALL["var"] == v) & (ALL.method == "centroid")].set_index("year")["abs_lat"].sort_index()
        if g.empty:
            continue
        a = g - g.mean()
        ax.plot(a.index, a.rolling(5, center=True, min_periods=3).mean(), color=COL[v], lw=1.8, label=LAB[v])
    ax.axvline(1979, color="grey", ls=":", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{hemi}: annual ridge latitude (half-range centroid), 5-yr mean", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly of |latitude| (deg; + = poleward)")
    if j == 0:
        ax.legend(fontsize=8, frameon=False)
    ax = axs[1, j]
    d = TO[(TO.hemi == hemi) & (TO.season == "Annual")]
    ylabels, y = [], 0
    for meth in ["centroid", "argmax"]:
        for v in ALL_VARS:
            for pn, mk, off in [("1979-2025", "o", 0.12), ("1966-2025", "s", -0.12)]:
                r = d[(d["var"] == v) & (d.method == meth) & (d.period == pn)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.ols_adj, y + off, xerr=[[r.ols_adj - r.ols_adj_lo], [r.ols_adj_hi - r.ols_adj]],
                            fmt=mk, color=COL[v], mfc=COL[v] if r.p_adj < 0.05 else "white", ms=5, lw=1.2, capsize=2)
            ylabels.append(f"{meth} - {LAB[v]}")
            y += 1
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ENSO-adjusted trend (deg/decade, + = poleward), 95% CI")
    ax.set_title(f"{hemi}: circles 1979-2025, squares 1966-2025 (filled: p<0.05)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_levels_slp_ridge.png"), dpi=300)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.8))
titles = {"abs_lat_poleward": "Latitude of high (deg/decade, + = poleward)",
          "lon_eastward": "Longitude of high (deg/decade, + = eastward)",
          "central_hPa": "Central pressure (hPa/decade)"}
names = list(HIGH_BOXES)
for ax, metric in zip(axs, titles):
    for i, b in enumerate(names):
        for k, (pn, mk) in enumerate([("1979-2025", "o"), ("1966-2025", "s")]):
            r = HT[(HT.high == b) & (HT.season == "Annual") & (HT.metric == metric) & (HT.period == pn)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.ols_adj, i + (0.12 if k == 0 else -0.12),
                        xerr=[[r.ols_adj - r.ols_adj_lo], [r.ols_adj_hi - r.ols_adj]], fmt=mk, color="k",
                        mfc="k" if r.p_adj < 0.05 else "white", ms=5, lw=1.2, capsize=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if ax is axs[0] else [""] * len(names), fontsize=9)
    ax.invert_yaxis()
    ax.set_title(titles[metric], fontsize=9)
    ax.set_xlabel("ENSO-adjusted trend, annual (95% CI)")
fig.suptitle("Individual subtropical highs (ocean SLP maxima): circles 1979-2025, squares 1966-2025; filled: p<0.05",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_slp_highs.png"), dpi=300)
plt.close(fig)

# ===============================================================
# Summary on screen
# ===============================================================
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 500)
print("\n=== Mean ridge latitude (annual, ocean, |deg|) ===")
print(ML[ML.season == "Annual"].pivot_table(index=["hemi", "method"], columns="var", values="mean_abs_lat").round(2))
for pn in PERIODS:
    print(f"\n=== ENSO-adjusted trends, annual, ocean, {pn} (deg/decade, + = poleward; q within 48-test family) ===")
    d = TO[(TO.period == pn) & (TO.season == "Annual")]
    for r in d.itertuples():
        flag = "  [edge]" if r.edge_flag else ""
        print(f"  {r.hemi} {r.var:5s} {r.method:8s} {r.ols_adj:+.2f} [{r.ols_adj_lo:+.2f},{r.ols_adj_hi:+.2f}] "
              f"p={r.p_adj:.3f} q={r.q_adj:.3f} Sen={r.sen_adj:+.2f}{flag}")
print("\n=== Correlation with the 500 hPa ridge series (annual) ===")
print(pd.DataFrame(cr).round(2).to_string(index=False))
print("\n=== Individual SLP highs, annual, 1979-2025 (ENSO-adjusted) ===")
d = HT[(HT.season == "Annual") & (HT.period == "1979-2025")]
for r in d.itertuples():
    flag = "  [edge]" if r.edge_flag else ""
    print(f"  {r.high:16s} {r.metric:17s} {r.ols_adj:+.2f} [{r.ols_adj_lo:+.2f},{r.ols_adj_hi:+.2f}] "
          f"p={r.p_adj:.3f} q={r.q_adj:.3f}{flag}")
print("\nDone. Outputs in:", OUT_DIR)
