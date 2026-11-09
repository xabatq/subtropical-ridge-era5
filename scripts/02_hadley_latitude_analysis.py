# -*- coding: utf-8 -*-
"""
Step 2 (final version 2): subsidence latitude, wind calm belt and their trends

Indicators used to locate the ridge (for each hemisphere and each year):
    w500 : maximum subsidence (omega)           tcc  : minimum total cloud cover
    wspd : minimum mean 10 m wind SPEED (windspeed.nc; from 1966)
(ssrd was dropped from the ridge detection; it is kept in the maps. No quantity is built from monthly-mean u and v.)

Wind analyses: latitude of minimum speed (calm belt) + 'calm-belt intensity' (the minimum speed itself) and their trends.
Methods: argmax with parabolic interpolation + half-maximum centroid
Trend: |latitude|; positive = poleward. Two periods: 1966-2025 and 1979-2025 (satellite era)
Significance: autocorrelation correction + FDR. All text inside the figures is in English.

Run:  python 02_hadley_latitude_analysis.py     (after step 1)
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

# ---------------------------------------------------------------
# Analysis settings (adjustable)
# ---------------------------------------------------------------
MAIN_SECTOR = "ocean"
PEAK_VARS = {"w500": "max", "tcc": "min", "wspd": "min"}   # to add radiation: "ssrd": "max"
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
NSMOOTH = 9
MAP_H_MAX = 2500.0          # only used to grey out mountains in the maps (m)
COLORS = {"w500": "tab:red", "tcc": "tab:gray", "wspd": "tab:green", "ssrd": "tab:orange"}

HEMI_NAMES = {"NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}
SEASON_NAMES = {"Annual": "Annual", "DJF": "DJF (winter in NH)", "JJA": "JJA (summer in NH)"}
SECTOR_NAMES = {"all": "All points", "nomtn1000": "Terrain < 1000 m", "nomtn2500": "Terrain < 2500 m",
                "ocean": "Ocean only", "atlantic": "Atlantic", "pacific_w": "W Pacific",
                "pacific_c": "C Pacific", "pacific_e": "E Pacific", "indian": "Indian"}

VAR_LABELS = {"w500": "Max 500-hPa subsidence (omega)",
              "tcc": "Min total cloud cover",
              "wspd": "Min 10-m wind speed",
              "ssrd": "Max surface solar radiation"}
VAR_NAMES = {"w500": "500-hPa omega",
             "wspd": "10-m wind speed (monthly mean of speed)",
             "tcc": "Total cloud cover",
             "ssrd": "Surface solar radiation (SSRD)"}

zm = xr.open_dataset(os.path.join(OUT_DIR, "zonal_means_masked.nc"))
lats = zm["latitude"].values
t = pd.DatetimeIndex(zm["time"].values)


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def nan_smooth(f, size, axis, mode="nearest"):
    m = np.isfinite(f).astype(float)
    num = uniform_filter1d(np.where(m > 0, f, 0.0), size, axis=axis, mode=mode)
    den = uniform_filter1d(m, size, axis=axis, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


def prep(profile, window, kind):
    y = nan_smooth(np.asarray(profile, dtype=float), NSMOOTH, 0)
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
    if w.sum() == 0:
        return np.nan
    return float((w * x).sum() / w.sum())


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


def trend(years, y):
    ok = np.isfinite(y)
    yrs, v = years[ok].astype(float), y[ok]
    n = len(v)
    if n < 10:
        return None
    ols = stats.linregress(yrs, v)
    resid = v - (ols.intercept + ols.slope * yrs)
    r1 = max(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), 3.0)
    se_adj = ols.stderr * np.sqrt((n - 2) / (n_eff - 2))
    p_adj = 2 * stats.t.sf(abs(ols.slope / se_adj), n_eff - 2)
    sen = stats.theilslopes(v, yrs)
    _, p_mk = stats.kendalltau(yrs, v)
    return dict(n=n, mean_value=v.mean(), std=v.std(ddof=1),
                ols_per_decade=ols.slope * 10, ols_p=ols.pvalue, lag1_autocorr=r1,
                n_eff=n_eff, ols_p_adj=p_adj,
                sen_per_decade=sen[0] * 10, sen_lo=sen[2] * 10, sen_hi=sen[3] * 10,
                mk_p=p_mk)


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty_like(pv)
    qv[order] = np.minimum(ranked, 1.0)
    q[ok] = qv
    return q


# ---------------------------------------------------------------
# Part 1: peak latitude of each indicator in each year + wind calm-belt intensity
# ---------------------------------------------------------------
rows, calm_rows = [], []
for sector in zm["sector"].values:
    for var, kind in PEAK_VARS.items():
        arr = zm[var].sel(sector=sector).values
        for sname, months in SEASONS.items():
            years, prof = season_series(arr, t, months)
            for hemi, win in WINDOWS.items():
                for yr, p in zip(years, prof):
                    pr = prep(p, win, kind)
                    if pr is None:
                        continue
                    x, y = pr
                    la_a, edge = argmax_lat(x, y)
                    la_c = centroid_lat(x, y)
                    rows.append((str(sector), hemi, sname, var, "argmax", int(yr), la_a, abs(la_a), edge))
                    if np.isfinite(la_c):
                        rows.append((str(sector), hemi, sname, var, "centroid", int(yr), la_c, abs(la_c), edge))
                    if var == "wspd":
                        calm_rows.append((str(sector), hemi, sname, int(yr), float(-np.nanmax(y))))

df = pd.DataFrame(rows, columns=["sector", "hemi", "season", "var", "method", "year",
                                 "lat", "abs_lat", "at_window_edge"])
df.to_csv(os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv"), index=False)
calm = pd.DataFrame(calm_rows, columns=["sector", "hemi", "season", "year", "min_wspd"])
calm.to_csv(os.path.join(OUT_DIR, "calm_belt_min_speed_by_year.csv"), index=False)

# ---------------------------------------------------------------
# Trends
# ---------------------------------------------------------------
trs = []
for key, g in df.groupby(["sector", "hemi", "season", "var", "method"]):
    for pname, (ya, yb) in PERIODS.items():
        gg = g[(g.year >= ya) & (g.year <= yb)]
        r = trend(gg["year"].values, gg["abs_lat"].values)
        if r is None:
            continue
        r.update(sector=key[0], hemi=key[1], season=key[2], var=key[3], method=key[4],
                 period=pname, edge_fraction=gg["at_window_edge"].mean())
        trs.append(r)
tr = pd.DataFrame(trs)
tr["q_ols_adj"] = bh_fdr(tr["ols_p_adj"].values)
tr["q_mk"] = bh_fdr(tr["mk_p"].values)
tr = tr.rename(columns={"mean_value": "mean_abs_lat", "ols_per_decade": "ols_deg_per_decade",
                        "sen_per_decade": "sen_deg_per_decade"})
cols = ["sector", "hemi", "season", "var", "method", "period", "n", "mean_abs_lat", "std",
        "ols_deg_per_decade", "ols_p", "lag1_autocorr", "n_eff", "ols_p_adj", "q_ols_adj",
        "sen_deg_per_decade", "sen_lo", "sen_hi", "mk_p", "q_mk", "edge_fraction"]
tr = tr[cols]
tr.to_csv(os.path.join(OUT_DIR, "trends_masked.csv"), index=False)

ctr = []
for key, g in calm.groupby(["sector", "hemi", "season"]):
    for pname, (ya, yb) in PERIODS.items():
        gg = g[(g.year >= ya) & (g.year <= yb)]
        r = trend(gg["year"].values, gg["min_wspd"].values)
        if r is None:
            continue
        r.update(sector=key[0], hemi=key[1], season=key[2], period=pname)
        ctr.append(r)
ct = pd.DataFrame(ctr)
ct["q_ols_adj"] = bh_fdr(ct["ols_p_adj"].values)
ct["q_mk"] = bh_fdr(ct["mk_p"].values)
ct = ct.rename(columns={"mean_value": "mean_min_wspd_mps", "ols_per_decade": "ols_mps_per_decade",
                        "sen_per_decade": "sen_mps_per_decade"})
ct.to_csv(os.path.join(OUT_DIR, "calm_belt_speed_trends_masked.csv"), index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def show_lat(var, method="argmax"):
    s = tr[(tr.sector == MAIN_SECTOR) & (tr["var"] == var) & (tr.method == method)]
    print(f"\n=== Latitude trend ({var}, {method}), sector {MAIN_SECTOR} - positive = poleward ===")
    print(s[["hemi", "season", "period", "n", "mean_abs_lat", "ols_deg_per_decade", "ols_p_adj", "q_ols_adj",
             "sen_deg_per_decade", "sen_lo", "sen_hi", "mk_p", "q_mk", "edge_fraction"]]
          .round(3).to_string(index=False))


show_lat("w500")
show_lat("wspd")
print(f"\n=== Calm-belt intensity trend (minimum speed, m/s per decade), sector {MAIN_SECTOR} ===")
s = ct[ct.sector == MAIN_SECTOR]
print(s[["hemi", "season", "period", "n", "mean_min_wspd_mps", "ols_mps_per_decade", "ols_p_adj", "q_ols_adj",
         "sen_mps_per_decade", "sen_lo", "sen_hi", "mk_p", "q_mk"]].round(4).to_string(index=False))

bad = tr[(tr.edge_fraction > 0.2) & (tr.period == "1966-2025") & (tr.method == "argmax")]
if len(bad):
    print("\nWarning: in these cases the peak sits on the edge of the search window in more than 20% of years:")
    print(bad[["sector", "hemi", "season", "var", "edge_fraction"]].round(2).to_string(index=False))


def get_tr(sector, hemi, season, var, method, period="1966-2025"):
    r = tr[(tr.sector == sector) & (tr.hemi == hemi) & (tr.season == season) &
           (tr["var"] == var) & (tr.method == method) & (tr.period == period)]
    return r.iloc[0] if len(r) else None


def series(sector, hemi, season, var, method):
    return df[(df.sector == sector) & (df.hemi == hemi) & (df.season == season) &
              (df["var"] == var) & (df.method == method)].sort_values("year")


# ---------------------------------------------------------------
# Figure 1: time series of peak latitude (main sector)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for i, hemi in enumerate(WINDOWS):
    for j, sname in enumerate(SEASONS):
        ax = axes[i, j]
        for var in PEAK_VARS:
            g = series(MAIN_SECTOR, hemi, sname, var, "argmax")
            ax.plot(g.year, g.abs_lat, color=COLORS[var], lw=2.2 if var == "w500" else 0.9,
                    alpha=1 if var == "w500" else 0.7, label=VAR_LABELS[var])
        gc = series(MAIN_SECTOR, hemi, sname, "w500", "centroid")
        ax.plot(gc.year, gc.abs_lat, color="darkred", lw=1.2, ls=":", label="Omega, half-max centroid")
        r = get_tr(MAIN_SECTOR, hemi, sname, "w500", "argmax")
        g = series(MAIN_SECTOR, hemi, sname, "w500", "argmax")
        if r is not None and len(g):
            s = stats.linregress(g.year, g.abs_lat)
            ax.plot(g.year, s.intercept + s.slope * g.year, "k--", lw=1)
            ax.set_title(f"{HEMI_NAMES[hemi]} - {SEASON_NAMES[sname]}\n"
                         f"omega trend: {r.ols_deg_per_decade:+.2f} deg/decade "
                         f"(autocorr-adj. p = {r.ols_p_adj:.3f}, q = {r.q_ols_adj:.2f})", fontsize=9)
        ax.set_ylabel("|Latitude| (deg)")
        if i == 1:
            ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Latitude of subsidence maximum / Hadley descending branch "
             f"(ERA5, {SECTOR_NAMES[MAIN_SECTOR].lower()} zonal mean)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_peak_latitude_timeseries.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 2: effect of masks and ocean sectors (annual)
# ---------------------------------------------------------------
groups = {"Masks (zonal means)": ["all", "nomtn1000", "nomtn2500", "ocean"],
          "Ocean sectors": ["atlantic", "pacific_w", "pacific_c", "pacific_e", "indian"]}
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for i, (gname, secs) in enumerate(groups.items()):
    for j, hemi in enumerate(WINDOWS):
        ax = axes[i, j]
        for sector in secs:
            g = series(sector, hemi, "Annual", "w500", "argmax")
            r = get_tr(sector, hemi, "Annual", "w500", "argmax")
            if not len(g) or r is None:
                continue
            ax.plot(g.year, g.abs_lat,
                    label=f"{SECTOR_NAMES[sector]} (Sen {r.sen_deg_per_decade:+.2f} deg/dec)")
        ax.set_title(f"{HEMI_NAMES[hemi]} - {gname} (annual)")
        ax.set_ylabel("|Latitude| of max omega (deg)")
        if i == 1:
            ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_masks_and_sectors_w500.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 3: Hovmoller of omega (main sector)
# ---------------------------------------------------------------
w = zm["w500"].sel(sector=MAIN_SECTOR).values
years, prof = season_series(w, t, SEASONS["Annual"])
fig, ax = plt.subplots(figsize=(11, 5))
sel = (lats >= -70) & (lats <= 70)
pc = ax.pcolormesh(years, lats[sel], np.ma.masked_invalid(prof[:, sel].T), cmap="RdBu_r",
                   vmin=-0.03, vmax=0.03, shading="auto")
fig.colorbar(pc, label="500-hPa omega (Pa/s); positive = subsidence")
for hemi in WINDOWS:
    ga = series(MAIN_SECTOR, hemi, "Annual", "w500", "argmax")
    gc = series(MAIN_SECTOR, hemi, "Annual", "w500", "centroid")
    ax.plot(ga.year, ga.lat, "k-", lw=1.5, label="Max (parabolic)" if hemi == "NH" else None)
    ax.plot(gc.year, gc.lat, color="gold", lw=1.2, ls="--",
            label="Half-max centroid" if hemi == "NH" else None)
ax.legend(loc="upper left", fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("Latitude (deg)")
ax.set_title(f"{SECTOR_NAMES[MAIN_SECTOR]} zonal-mean 500-hPa omega and latitude of max subsidence (annual)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_hovmoller_omega.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 4: forest plot of Theil-Sen trends (omega peak)
# ---------------------------------------------------------------
sectors_all = list(zm["sector"].values)
combos = [("argmax", "1966-2025", "tab:red", "o"), ("argmax", "1979-2025", "tab:red", "s"),
          ("centroid", "1966-2025", "tab:blue", "o"), ("centroid", "1979-2025", "tab:blue", "s")]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
for i, hemi in enumerate(WINDOWS):
    for j, sname in enumerate(SEASONS):
        ax = axes[i, j]
        for k, (meth, per, col, mk) in enumerate(combos):
            ys, xs, lo, hi = [], [], [], []
            for si, sector in enumerate(sectors_all):
                r = get_tr(sector, hemi, sname, "w500", meth, per)
                if r is None:
                    continue
                ys.append(si + (k - 1.5) * 0.15)
                xs.append(r.sen_deg_per_decade)
                lo.append(r.sen_deg_per_decade - r.sen_lo)
                hi.append(r.sen_hi - r.sen_deg_per_decade)
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt=mk, color=col, ms=4, lw=1,
                        mfc=col if per == "1966-2025" else "none", label=f"{meth}, {per}")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(sectors_all)))
        ax.set_yticklabels([SECTOR_NAMES.get(s, s) for s in sectors_all], fontsize=8)
        ax.set_title(f"{HEMI_NAMES[hemi]} - {SEASON_NAMES[sname]}", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.set_xlabel("Theil-Sen trend of |lat| of max omega (deg/decade), 95% CI")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_forest_trends.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 8: time-mean zonal profile of 10-m wind speed
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
for sector, ls in (("ocean", "-"), ("all", "--")):
    sp = zm["wspd"].sel(sector=sector).mean("time", skipna=True).values
    ax.plot(sp, lats, ls, color="tab:green", label=f"10-m wind speed - {SECTOR_NAMES[sector]}")
for hemi in WINDOWS:
    g = series("ocean", hemi, "Annual", "w500", "argmax")
    ax.axhline(g.lat.mean(), color="tab:red", ls="--", lw=1,
               label="Mean latitude of max omega (ocean)" if hemi == "NH" else None)
for lat0 in (-45, -15, 15, 45):
    ax.axhline(lat0, color="0.6", lw=0.6, ls=":")
ax.set_xlabel("Wind speed (m/s)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Time-mean zonal profile of 10-m wind speed")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig8_wind_profile.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 9: strength of the calm belt (minimum zonal-mean wind speed)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for i, hemi in enumerate(WINDOWS):
    for j, sname in enumerate(SEASONS):
        ax = axes[i, j]
        g = calm[(calm.sector == MAIN_SECTOR) & (calm.hemi == hemi) &
                 (calm.season == sname)].sort_values("year")
        r = ct[(ct.sector == MAIN_SECTOR) & (ct.hemi == hemi) & (ct.season == sname) &
               (ct.period == "1966-2025")]
        ax.plot(g.year, g.min_wspd, color="tab:green")
        if len(g) > 2:
            s = stats.linregress(g.year, g.min_wspd)
            ax.plot(g.year, s.intercept + s.slope * g.year, "k--", lw=1)
        ttl = f"{HEMI_NAMES[hemi]} - {SEASON_NAMES[sname]}"
        if len(r):
            r = r.iloc[0]
            ttl += (f"\ntrend: {r.ols_mps_per_decade:+.3f} m/s per decade "
                    f"(adj. p = {r.ols_p_adj:.3f}, q = {r.q_ols_adj:.2f})")
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("Min zonal-mean wind speed (m/s)")
        if i == 1:
            ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
fig.suptitle(f"Strength of the calm belt (horse latitudes), {SECTOR_NAMES[MAIN_SECTOR].lower()}")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig9_calm_belt_min_speed.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Part 2: maps with Basemap
# ---------------------------------------------------------------
try:
    from mpl_toolkits.basemap import Basemap
except Exception as e:
    print("\nBasemap could not be imported, maps are not produced:", e)
    raise SystemExit

cl = xr.open_dataset(os.path.join(OUT_DIR, "climatology.nc"))
clat, clon = cl["latitude"].values, cl["longitude"].values
st = xr.open_dataset(os.path.join(OUT_DIR, "static_fields.nc"))
st = st.reindex(latitude=clat, longitude=clon, method="nearest", tolerance=0.05)
lsm2, hgt2 = st["lsm"].values, st["height_m"].values
ocean2 = lsm2 < 0.1
mountain = hgt2 > MAP_H_MAX


def ridge_per_lon(field, window, kind="max"):
    """Peak latitude at each longitude; ocean only; points at the window edge or coast edge are dropped."""
    f = np.where(ocean2, field, np.nan)
    f = nan_smooth(f, 9, 0, "nearest")
    f = nan_smooth(f, 9, 1, "wrap")
    sel = (clat >= window[0]) & (clat <= window[1])
    sub = f[sel]
    if kind == "min":
        sub = -sub
    allnan = ~np.isfinite(sub).any(axis=0)
    i = np.where(np.isfinite(sub), sub, -np.inf).argmax(axis=0)
    j = np.arange(sub.shape[1])
    n = sub.shape[0]
    up = sub[np.clip(i + 1, 0, n - 1), j]
    dn = sub[np.clip(i - 1, 0, n - 1), j]
    r = clat[sel][i].astype(float)
    bad = allnan | (i == 0) | (i == n - 1) | ~np.isfinite(up) | ~np.isfinite(dn)
    r[bad] = np.nan
    return r


def draw(ax, fig, field, title, cmap, vmin, vmax, ridges=None, label="", ridge_color="k"):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.82")
    m = Basemap(projection="cyl", llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=0, urcrnrlon=360, resolution="c", ax=ax)
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(range(-60, 61, 30), labels=[1, 0, 0, 0], fontsize=7, linewidth=0.2)
    m.drawmeridians(range(0, 361, 60), labels=[0, 0, 0, 1], fontsize=7, linewidth=0.2)
    X, Y = np.meshgrid(clon[::2], clat[::2])
    pc = m.pcolormesh(X, Y, np.ma.masked_invalid(field[::2, ::2]), latlon=True, cmap=cm,
                      vmin=vmin, vmax=vmax, shading="auto")
    if ridges is not None:
        for r in ridges:
            m.scatter(clon, r, s=2, c=ridge_color, latlon=True, zorder=5)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
    cb.set_label(label, fontsize=8)


specs = {
    "w500": dict(cmap="RdBu_r", lim=(-0.08, 0.08), label="Pa/s (positive = subsidence)"),
    "wspd": dict(cmap="viridis", lim=(0, 12), label="m/s"),
    "tcc": dict(cmap="Blues", lim=(0, 1), label="Cloud cover fraction"),
    "ssrd": dict(cmap="YlOrRd", lim=(80, 300), label="W/m$^2$"),
}


def masked(field, v):
    return np.where(mountain, np.nan, field) if v == "w500" else field


# Figure 5: climatology
w_all = masked(cl["w500_all"].values, "w500")
ridges_w = [ridge_per_lon(w_all, WINDOWS[h]) for h in WINDOWS]
ridges_ws = [ridge_per_lon(cl["wspd_all"].values, WINDOWS[h], "min") for h in WINDOWS]
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, (v, sp) in zip(axes.ravel(), specs.items()):
    title = f"Mean {VAR_NAMES[v]}"
    rid, col = None, "k"
    if v == "w500":
        title += f" (1966-2025)\nblack dots: ocean-only latitude of max omega; grey: terrain > {MAP_H_MAX:.0f} m"
        rid = ridges_w
    elif v == "wspd":
        title += " (1966-2025)\nwhite dots: ocean-only latitude of min wind speed (calm belt)"
        rid, col = ridges_ws, "w"
    else:
        title += " (1966-2025)"
    draw(ax, fig, masked(cl[f"{v}_all"].values, v), title, sp["cmap"], sp["lim"][0], sp["lim"][1],
         ridges=rid, label=sp["label"], ridge_color=col)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_climatology_maps.png"), dpi=150)
plt.close(fig)

# Figure 6: difference (1996-2025) minus (1966-1995)
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, (v, sp) in zip(axes.ravel(), specs.items()):
    d = masked(cl[f"{v}_p2"].values - cl[f"{v}_p1"].values, v)
    lim = np.nanpercentile(np.abs(d), 98)
    p1 = "1966-1995"
    draw(ax, fig, d, f"Change in {VAR_NAMES[v]}\n(1996-2025) minus ({p1})",
         "RdBu_r", -lim, lim, label=sp["label"])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_difference_maps.png"), dpi=150)
plt.close(fig)

# Figure 7: ocean-only ridges by longitude (omega max and calm-belt minimum)
fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex=True)
for j, (v, kind, ttl) in enumerate((("w500", "max", "Latitude of max omega"),
                                    ("wspd", "min", "Latitude of min wind speed (calm belt)"))):
    for i, hemi in enumerate(WINDOWS):
        ax = axes[i, j]
        r1 = ridge_per_lon(masked(cl[f"{v}_p1"].values, v), WINDOWS[hemi], kind)
        r2 = ridge_per_lon(masked(cl[f"{v}_p2"].values, v), WINDOWS[hemi], kind)
        ax.plot(clon, r1, ".", ms=2, label="1966-1995")
        ax.plot(clon, r2, ".", ms=2, label="1996-2025")
        ax.set_title(f"{HEMI_NAMES[hemi]} - {ttl}", fontsize=9)
        ax.set_ylabel("Latitude (deg)")
        ax.grid(alpha=0.3)
        ax.legend(markerscale=4, fontsize=8)
        if i == 1:
            ax.set_xlabel("Longitude (deg E, 0-360)")
fig.suptitle("Ocean-only ridge latitudes at each longitude")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_ridge_by_longitude.png"), dpi=150)
plt.close(fig)

print("\nDone. Tables and figures in:", OUT_DIR)
print(" - trends_masked.csv, calm_belt_speed_trends_masked.csv, peak_latitudes_by_year_masked.csv")
print(" - fig1 ... fig9")
