# -*- coding: utf-8 -*-
"""
04_ceres_cloud_check.py
Compare the ERA5 'cloud-minimum latitude' with CERES EBAF Ed4.2.1 (day+night cloud cover) over the ocean.

Inputs:
  CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202608.nc   (1x1 degree, 318 months from 2000-03)
  outputs/zonal_means_masked.nc                   (from 01; sector ocean, variable tcc)
  outputs/static_fields.nc                        (land/sea mask)

Notes:
  * Both series are built on the same 1-degree grid with the same ocean mask and the same smoothing (3 points).
  * Complete years only (Annual: 2001-2025 because CERES starts in March 2000).
  * Absolute cloud values differ between ERA5 and CERES; only the 'minimum latitude' and its change are compared.
  * The 25-year period is short: interpret trends with caution and together with the correlation of the series.
Outputs: ceres_vs_era5_cloud_ridge.csv , ceres_era5_ridge_by_year.csv , fig12 , fig13
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
CERES_FILE = os.path.join(DATA_DIR, "CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202608.nc")
CERES_START = "2000-03"
Y_END = 2025
NSMOOTH = 3                      # in 1-degree points
OCEAN_LSM = 0.1
MIN_COVER = 0.25                 # minimum fraction of ocean columns in each latitude row
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
WTS5 = np.array([0.5, 1, 1, 1, 0.5])


# ---------------------------------------------------------------
# CERES
# ---------------------------------------------------------------
ds = xr.open_dataset(CERES_FILE, decode_times=False)
cands = []
for name in ds.data_vars:
    da = ds[name]
    if len(da.dims) == 3:
        cands.append(name)
print("3-D variables in the CERES file:", cands)
pref = [n for n in cands if "cldarea_total_daynight" in n.lower()]
if not pref:
    pref = [n for n in cands if "cldarea" in n.lower() and "total" in n.lower()] or \
           [n for n in cands if "cloud" in n.lower() or "cld" in n.lower()]
if not pref:
    raise SystemExit("Cloud variable not found; set the variable name by hand in VAR_NAME.")
VAR_NAME = pref[0]
print("Selected cloud variable:", VAR_NAME, " (if wrong, set VAR_NAME by hand)")
da = ds[VAR_NAME]
dims = list(da.dims)
tdim = [d for d in dims if "time" in d.lower()][0]
latdim = [d for d in dims if d.lower().startswith("lat")][0]
londim = [d for d in dims if d.lower().startswith("lon")][0]
cl = np.asarray(da.values, dtype=float)
cl = np.transpose(cl, [dims.index(tdim), dims.index(latdim), dims.index(londim)])
c_lat = np.asarray(ds[latdim].values, dtype=float)
c_lon = np.asarray(ds[londim].values, dtype=float)
if c_lat[0] > c_lat[-1]:
    c_lat = c_lat[::-1]
    cl = cl[:, ::-1, :]
cl = np.where((cl >= 0) & (cl <= 100), cl, np.nan)
nt = cl.shape[0]
if nt < 300:
    raise SystemExit(f"Unexpected number of time steps: {nt}")
c_time = pd.period_range(CERES_START, periods=nt, freq="M").to_timestamp()
print(f"CERES: {nt} months, {c_time[0]:%Y-%m} to {c_time[-1]:%Y-%m}, grid {len(c_lat)}x{len(c_lon)}")
keep = c_time <= pd.Timestamp(f"{Y_END}-12-01")
cl, c_time = cl[keep], c_time[keep]

# ---------------------------------------------------------------
# Ocean mask on the 1-degree grid (from ERA5 lsm)
# ---------------------------------------------------------------
st = xr.open_dataset(os.path.join(OUT_DIR, "static_fields.nc"))
s_lat = np.asarray(st["latitude"].values, dtype=float)
s_lon = np.asarray(st["longitude"].values, dtype=float)
lsm = np.asarray(st["lsm"].values, dtype=float)
if s_lat[0] > s_lat[-1]:
    s_lat, lsm = s_lat[::-1], lsm[::-1]
dlat = abs(s_lat[1] - s_lat[0])


def lsm_to_1deg():
    if abs(dlat - 0.25) < 1e-6 and len(s_lat) == 721 and len(s_lon) == 1440:
        out = np.zeros((len(c_lat), len(c_lon)))
        wsum = WTS5.sum()
        for ia, wa in enumerate(WTS5):
            ri = (np.round((c_lat + 90) / 0.25).astype(int) + (ia - 2)).clip(0, 720)
            for ib, wb in enumerate(WTS5):
                ci = (np.round(c_lon / 0.25).astype(int) + (ib - 2)) % 1440
                out += wa * wb * lsm[np.ix_(ri, ci)]
        return out / (wsum * wsum)
    ri = np.abs(s_lat[None, :] - c_lat[:, None]).argmin(axis=1)
    ci = np.abs(((s_lon[None, :] - c_lon[:, None] + 180) % 360) - 180).argmin(axis=1)
    return lsm[np.ix_(ri, ci)]


lsm1 = lsm_to_1deg()
ocean1 = lsm1 < OCEAN_LSM
print(f"Ocean cells at 1 degree: {ocean1.sum()} of {ocean1.size}")

# CERES zonal mean over the ocean
M = ocean1[None, :, :] & np.isfinite(cl)
num = np.where(M, cl, 0.0).sum(axis=2)
cnt = M.sum(axis=2)
with np.errstate(invalid="ignore", divide="ignore"):
    c_prof = num / cnt
c_prof[cnt < MIN_COVER * len(c_lon)] = np.nan          # (time, lat)

# ---------------------------------------------------------------
# ERA5 tcc (ocean zonal mean) -> 1-degree grid -> x100 (%)
# ---------------------------------------------------------------
zm = xr.open_dataset(os.path.join(OUT_DIR, "zonal_means_masked.nc"))
e_lat = np.asarray(zm["latitude"].values, dtype=float)
e_time = pd.DatetimeIndex(zm["time"].values)
e_arr = np.asarray(zm["tcc"].sel(sector="ocean").values, dtype=float)      # (time, lat)
if e_lat[0] > e_lat[-1]:
    e_lat, e_arr = e_lat[::-1], e_arr[:, ::-1]


def era_to_1deg(a):
    if abs(abs(e_lat[1] - e_lat[0]) - 0.25) < 1e-6 and len(e_lat) == 721:
        num = np.zeros((a.shape[0], len(c_lat)))
        den = np.zeros_like(num)
        for k, w in enumerate(WTS5):
            ri = (np.round((c_lat + 90) / 0.25).astype(int) + (k - 2)).clip(0, 720)
            v = a[:, ri]
            m = np.isfinite(v)
            num += w * np.where(m, v, 0.0)
            den += w * m
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[den < 0.5 * WTS5.sum()] = np.nan
        return out
    out = np.full((a.shape[0], len(c_lat)), np.nan)          # coarse grid (test): interpolation
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i])
        if ok.sum() > 2:
            out[i] = np.interp(c_lat, e_lat[ok], a[i][ok], left=np.nan, right=np.nan)
    return out


e_prof = 100.0 * era_to_1deg(e_arr)
sel = (e_time >= pd.Timestamp(f"{CERES_START}-01")) & (e_time <= pd.Timestamp(f"{Y_END}-12-01"))
e_prof, e_time = e_prof[sel], e_time[sel]
# Align the months
common = c_time.intersection(e_time)
c_prof = c_prof[c_time.isin(common)]
e_prof = e_prof[e_time.isin(common)]
tt = common
print(f"Common months: {len(tt)} ({tt[0]:%Y-%m} to {tt[-1]:%Y-%m})")
print(f"Mean ocean cloud 15-45: CERES {np.nanmean(c_prof[:, (c_lat>=15)&(c_lat<=45)]):.1f}%  "
      f"ERA5 {np.nanmean(e_prof[:, (c_lat>=15)&(c_lat<=45)]):.1f}%")


# ---------------------------------------------------------------
# Functions (same logic as 02)
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
    x, y = c_lat[s], -y[s]                       # minimum -> maximum
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


def trend(years, y):
    ok = np.isfinite(y)
    yrs, v = years[ok].astype(float), y[ok]
    n = len(v)
    if n < 10:
        return dict(n=n, ols=np.nan, p=np.nan, sen=np.nan, sen_lo=np.nan, sen_hi=np.nan)
    ols = stats.linregress(yrs, v)
    res = v - (ols.intercept + ols.slope * yrs)
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), 3.0)
    se = ols.stderr * np.sqrt((n - 2) / (n_eff - 2))
    p = 2 * stats.t.sf(abs(ols.slope / se), n_eff - 2)
    sen = stats.theilslopes(v, yrs)
    return dict(n=n, ols=ols.slope * 10, p=p, sen=sen[0] * 10, sen_lo=sen[2] * 10, sen_hi=sen[3] * 10)


# ---------------------------------------------------------------
# Annual series of the cloud-minimum latitude
# ---------------------------------------------------------------
yrs_by = {}
rows_y = []
for src, prof in [("CERES", c_prof), ("ERA5", e_prof)]:
    for sname, months in SEASONS.items():
        years, sp = season_series(prof, tt, months)
        for hemi, win in WINDOWS.items():
            for yr, p in zip(years, sp):
                pr = prep(p, win)
                if pr is None:
                    continue
                x, y = pr
                la, edge = argmax_lat(x, y)
                lc = centroid_lat(x, y)
                rows_y.append((src, hemi, sname, "argmin", int(yr), abs(la), edge))
                rows_y.append((src, hemi, sname, "centroid", int(yr), abs(lc), edge))
yy = pd.DataFrame(rows_y, columns=["source", "hemi", "season", "method", "year", "abs_lat", "at_window_edge"])
yy.to_csv(os.path.join(OUT_DIR, "ceres_era5_ridge_by_year.csv"), index=False)

# ---------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------
res = []
for (hemi, sname, meth), g in yy.groupby(["hemi", "season", "method"]):
    a = g[g.source == "CERES"].set_index("year")["abs_lat"]
    b = g[g.source == "ERA5"].set_index("year")["abs_lat"]
    yr = a.index.intersection(b.index)
    if len(yr) < 10:
        continue
    a, b = a.loc[yr], b.loc[yr]
    r_p = stats.pearsonr(a, b)
    r_s = stats.spearmanr(a, b)
    ad = a - np.polyval(np.polyfit(yr, a, 1), yr)
    bd = b - np.polyval(np.polyfit(yr, b, 1), yr)
    r_dt = stats.pearsonr(ad, bd)
    ta, tb, td = trend(yr.values, a.values), trend(yr.values, b.values), trend(yr.values, (a - b).values)
    res.append(dict(hemi=hemi, season=sname, method=meth, n=len(yr), years=f"{yr.min()}-{yr.max()}",
                    mean_ceres=a.mean(), mean_era5=b.mean(), bias_ceres_minus_era5=(a - b).mean(),
                    rmse=float(np.sqrt(((a - b) ** 2).mean())),
                    pearson_r=r_p[0], pearson_p=r_p[1], spearman_r=r_s[0], r_detrended=r_dt[0],
                    trend_ceres=ta["ols"], p_ceres=ta["p"], sen_ceres=ta["sen"],
                    trend_era5=tb["ols"], p_era5=tb["p"], sen_era5=tb["sen"],
                    trend_diff=td["ols"], p_diff=td["p"]))
rs = pd.DataFrame(res)
rs.to_csv(os.path.join(OUT_DIR, "ceres_vs_era5_cloud_ridge.csv"), index=False)
pd.set_option("display.width", 220)
print("\n=== CERES versus ERA5: cloud-minimum latitude over the ocean (unit: degrees and degrees/decade) ===")
print(rs[["hemi", "season", "method", "n", "mean_ceres", "mean_era5", "pearson_r", "r_detrended",
          "trend_ceres", "p_ceres", "trend_era5", "p_era5", "trend_diff", "p_diff"]].round(3).to_string(index=False))

# ---------------------------------------------------------------
# Figure 12: time series
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, hemi in enumerate(["NH", "SH"]):
    for j, sname in enumerate(["Annual", "DJF", "JJA"]):
        ax = axes[i, j]
        for src, col in [("CERES", "tab:blue"), ("ERA5", "tab:orange")]:
            g = yy[(yy.hemi == hemi) & (yy.season == sname) & (yy.method == "argmin") & (yy.source == src)]
            g = g.sort_values("year")
            ax.plot(g.year, g.abs_lat, "-o", ms=3, color=col, lw=1.1, label=src)
            if len(g) >= 10:
                sl = stats.linregress(g.year, g.abs_lat)
                ax.plot(g.year, sl.intercept + sl.slope * g.year, "--", color=col, lw=1)
        r = rs[(rs.hemi == hemi) & (rs.season == sname) & (rs.method == "argmin")]
        ttl = f"{'Northern' if hemi == 'NH' else 'Southern'} Hemisphere - {sname}"
        if len(r):
            r = r.iloc[0]
            ttl += (f"\nr = {r.pearson_r:.2f} (detrended {r.r_detrended:.2f})  |  trend CERES {r.trend_ceres:+.2f}, "
                    f"ERA5 {r.trend_era5:+.2f} deg/dec")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("|Latitude| of min cloud cover (deg)")
        ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle("Latitude of minimum ocean cloud cover: CERES EBAF vs ERA5 (2000-2025)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig12_ceres_vs_era5_cloud_ridge.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 13: zonal-mean profile
# ---------------------------------------------------------------
m_ok = (tt.year >= 2001)
fig, ax = plt.subplots(figsize=(7, 6))
for src, prof, col in [("CERES", c_prof, "tab:blue"), ("ERA5", e_prof, "tab:orange")]:
    ax.plot(np.nanmean(prof[m_ok], axis=0), c_lat, color=col, label=src)
for hemi, win in WINDOWS.items():
    for src, col in [("CERES", "tab:blue"), ("ERA5", "tab:orange")]:
        g = yy[(yy.hemi == hemi) & (yy.season == "Annual") & (yy.method == "argmin") & (yy.source == src)]
        if len(g):
            ax.axhline(np.sign(win[0] + win[1]) * g.abs_lat.mean(), color=col, ls="--", lw=0.9)
ax.set_xlabel("Total cloud cover over ocean (%)")
ax.set_ylabel("Latitude (deg)")
ax.set_ylim(-60, 60)
ax.set_title("Time-mean ocean zonal profile, 2001-2025\n(dashed: mean latitude of minimum)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig13_ceres_era5_cloud_profile.png"), dpi=150)
plt.close(fig)

print("\nDone. Outputs in:", OUT_DIR)
print(" - ceres_vs_era5_cloud_ridge.csv, ceres_era5_ridge_by_year.csv, fig12, fig13")
