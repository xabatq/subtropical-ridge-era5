# -*- coding: utf-8 -*-
"""
09_smoothing_sensitivity.py
Sensitivity of the ocean ridge-latitude trends to the smoothing window along latitude.

The main analysis (02) smooths each zonal-mean profile with a 9-point running mean (2.25 deg at 0.25 deg
resolution). Here the whole chain (smoothing -> argmax / half-range centroid -> annual series ->
ENSO(ONI)-adjusted OLS trend with autocorrelation-adjusted p) is repeated with 5, 9 and 13 points
(1.25, 2.25 and 3.25 deg).

Input  (DATA_DIR/outputs): zonal_means_masked.nc   (from 01; sector 'ocean', variables w500, tcc, wspd)
       (DATA_DIR):         enso_monthly_1966_2025.csv
Output (DATA_DIR/outputs): smoothing_sensitivity.csv, smoothing_sensitivity_by_year.csv

Run:  python 09_smoothing_sensitivity.py     (about a minute)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import uniform_filter1d

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
SECTOR = "ocean"
NSMOOTH_LIST = [5, 9, 13]
PEAK_VARS = {"w500": "max", "tcc": "min", "wspd": "min"}
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
MIN_N = 20

zm = xr.open_dataset(os.path.join(OUT_DIR, "zonal_means_masked.nc"))
lats = np.asarray(zm["latitude"].values, float)
t = pd.DatetimeIndex(zm["time"].values)
enso = pd.read_csv(ENSO_CSV)


def enso_season(months, col="ONI"):
    d = enso[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")[col]
    return g.mean()[g.count() == len(months)]


ENSO_S = {s: enso_season(m) for s, m in SEASONS.items()}


def nan_smooth(f, size, axis=0, mode="nearest"):
    m = np.isfinite(f).astype(float)
    num = uniform_filter1d(np.where(m > 0, f, 0.0), size, axis=axis, mode=mode)
    den = uniform_filter1d(m, size, axis=axis, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


def prep(profile, window, kind, nsmooth):
    y = nan_smooth(np.asarray(profile, float), nsmooth)
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
            out.append(arr[idx].mean(axis=0))
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
    return beta[j], p, beta[j] - tc * se_adj, beta[j] + tc * se_adj


rows_y = []
for nsm in NSMOOTH_LIST:
    for var, kind in PEAK_VARS.items():
        arr = np.asarray(zm[var].sel(sector=SECTOR).values, float)
        for sname, months in SEASONS.items():
            years, prof = season_series(arr, t, months)
            for hemi, win in WINDOWS.items():
                for yr, p in zip(years, prof):
                    pr = prep(p, win, kind, nsm)
                    if pr is None:
                        continue
                    x, y = pr
                    la_a, edge = argmax_lat(x, y)
                    la_c = centroid_lat(x, y)
                    rows_y.append((nsm, hemi, sname, var, "argmax", int(yr), abs(la_a), edge))
                    if np.isfinite(la_c):
                        rows_y.append((nsm, hemi, sname, var, "centroid", int(yr), abs(la_c), edge))
    print(f"smoothing {nsm} points done", flush=True)
Y = pd.DataFrame(rows_y, columns=["nsmooth", "hemi", "season", "var", "method", "year", "abs_lat", "at_window_edge"])
Y.to_csv(os.path.join(OUT_DIR, "smoothing_sensitivity_by_year.csv"), index=False)

rows = []
for (nsm, hemi, seas, var, meth), g in Y.groupby(["nsmooth", "hemi", "season", "var", "method"]):
    s = g.set_index("year")["abs_lat"]
    for pname, (ya, yb) in PERIODS.items():
        ss = s[(s.index >= ya) & (s.index <= yb)].dropna()
        e_all = ENSO_S[seas]
        common = ss.index.intersection(e_all.index)
        if len(common) < MIN_N:
            continue
        yrs = common.values.astype(float)
        y = ss.loc[common].values
        e = e_all.loc[common].values
        e = (e - e.mean()) / e.std(ddof=1)
        tt = (yrs - 1995.5) / 10.0
        b, p, lo, hi = ols_adj(y, np.column_stack([np.ones_like(tt), tt, e]), 1)
        rows.append(dict(nsmooth=nsm, width_deg=0.25 * nsm, hemi=hemi, season=seas, var=var, method=meth,
                         period=pname, n=len(y), trend_adj=b, ci_lo=lo, ci_hi=hi, p_adj=p,
                         edge_fraction=g.at_window_edge.mean()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT_DIR, "smoothing_sensitivity.csv"), index=False)

pd.set_option("display.width", 220)
print("\n=== ENSO-adjusted trend (deg/decade) and p, annual, 1979-2025, for 5 / 9 / 13-point smoothing ===")
q = R[(R.season == "Annual") & (R.period == "1979-2025")]
for (hemi, var, meth), g in q.groupby(["hemi", "var", "method"]):
    cells = [f"{int(r.nsmooth):2d}pt {r.trend_adj:+.2f} (p={r.p_adj:.3f})" for r in g.sort_values("nsmooth").itertuples()]
    print(f"  {hemi} {var:5s} {meth:8s} " + "   ".join(cells))
agree = q.pivot_table(index=["hemi", "var", "method"], columns="nsmooth", values="trend_adj")
agree["same_sign_all"] = (np.sign(agree).nunique(axis=1) == 1)
print("\nSame sign for all three windows:", int(agree.same_sign_all.sum()), "of", len(agree))
print("\nDone. Outputs in:", OUT_DIR)
