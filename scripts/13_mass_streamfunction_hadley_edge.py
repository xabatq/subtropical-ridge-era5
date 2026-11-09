# -*- coding: utf-8 -*-
"""
13_mass_streamfunction_hadley_edge.py
Meridional mass streamfunction from ERA5 monthly-mean meridional wind on 16 pressure levels, the latitude of the
Hadley-cell edge, and its comparison with the ridge indicators of the paper.

    psi(phi, p) = (2 pi a cos(phi) / g) * integral from the top of the atmosphere down to p of [v] dp'

Implementation (see the notes in the output):
  * [v] is the zonal mean over ALL longitudes (land and ocean), not the ocean-only mean used for the ridge,
    because the mass flux of a latitude circle needs the whole circle.
  * Each pressure level represents a layer bounded by the mid-points between neighbouring levels (top boundary
    50 hPa). The weight of a layer in a column is its thickness limited by the surface pressure sp (from P-SLP.nc),
    so that below-ground data are not used. Monthly-mean sp is an approximation of the instantaneous surface pressure.
  * Mass-balance correction: at each latitude and month, the net mass flux through the latitude circle is removed
    by subtracting a uniform (mass-weighted) wind from all layers, so that psi = 0 at the surface.
  * Unit of psi: 10^9 kg/s.  Sign: psi > 0 for the Northern Hemisphere Hadley cell (poleward flow aloft),
    psi < 0 for the Southern Hemisphere Hadley cell.
  * The monthly-mean v contains only the mean meridional circulation; eddy contributions are not included.
  * The data cover 65S-65N only; the layer above 50 hPa is not included.

Hadley-cell edge: the latitude, poleward of the extremum of psi(500 hPa) in 5-35 deg of each hemisphere, where
psi(500 hPa) changes sign (linear interpolation; psi smoothed with a 9-point running mean along latitude).
Sensitivity: 300 and 700 hPa. The edge, the latitude of the extremum and its strength are computed for annual, DJF and
JJA means. Trends: ENSO(ONI)-adjusted OLS with 95% CI (lag-1 effective sample size), Theil-Sen, Mann-Kendall and
Benjamini-Hochberg q (families are stated in the output tables). Trends in deg/decade of |latitude|, + = poleward.

The edge is then compared with the ridge latitudes (omega at 700/500/300 hPa, sea-level pressure, cloud, wind):
mean position, annual correlation (raw and detrended) and trends.

Inputs : DATA_DIR/V(Levels).nc  (16 levels, 65N-65S, 1965 or 1966 to 2025), DATA_DIR/P-SLP.nc (sp),
         DATA_DIR/enso_monthly_1966_2025.csv, DATA_DIR/outputs/levels_slp_ridge_by_year.csv (script 12),
         DATA_DIR/outputs/peak_latitudes_by_year_masked.csv (script 02)
Outputs (DATA_DIR/outputs): psi_monthly_1966_2025.npz (cache), hadley_edge_by_year.csv, hadley_edge_trends.csv,
         hadley_vs_ridge_mean_latitude.csv, hadley_vs_ridge_correlations.csv, hadley_vs_ridge_trends.csv,
         fig_psi_climatology.png, fig_hadley_edge.png

Run:  python 13_mass_streamfunction_hadley_edge.py     (extraction: tens of minutes; analysis: about a minute)
"""
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
F_V = os.path.join(DATA_DIR, "V(Levels).nc")
F_P = os.path.join(DATA_DIR, "P-SLP.nc")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
LEVELS_CSV = os.path.join(OUT_DIR, "levels_slp_ridge_by_year.csv")
PEAK_CSV = os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv")
CACHE = os.path.join(OUT_DIR, "psi_monthly_1966_2025.npz")

Y0, Y1 = 1966, 2025
EXPECTED_LEVELS = [100, 200, 300, 400, 500, 600, 700, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
A_EARTH = 6.371e6
G = 9.80665
NSMOOTH = 9
EXT_RANGE = (5.0, 35.0)       # search range (|latitude|) for the extremum of psi
EDGE_LEVELS = [500, 300, 700]  # first = main definition
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
MIN_N = 20
MAX_EDGE = 0.3

months = pd.period_range(f"{Y0}-01", f"{Y1}-12", freq="M")
nM = len(months)


# ===============================================================
# Part A: streamfunction (cached)
# ===============================================================
def tname_of(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def start_index(ds, name):
    p = pd.DatetimeIndex(ds[tname_of(ds)].values).to_period("M")
    pos = pd.Series(np.arange(len(p)), index=p)
    idx = pos.reindex(months)
    if idx.isna().any():
        miss = [str(m) for m in months[idx.isna().values]][:6]
        raise RuntimeError(f"{name}: months of {Y0}-{Y1} are missing, for example {miss} "
                           f"(file range {p.min()} to {p.max()}).")
    idx = idx.values.astype(int)
    if not np.all(np.diff(idx) == 1):
        raise RuntimeError(f"{name}: months are not contiguous.")
    if "expver" in ds:
        ev = np.asarray(ds["expver"].values)[idx]
        if np.any(ev != "0001"):
            print(f"Warning: {name} contains expver values other than 0001 ({np.unique(ev)}).")
    return int(idx[0])


def extract():
    for f in (F_V, F_P):
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")
    dsV = xr.open_dataset(F_V)
    dsP = xr.open_dataset(F_P)
    if "v" not in dsV.data_vars:
        raise SystemExit(f"Variable 'v' not found in {F_V}: {list(dsV.data_vars)}")

    lev_file = np.asarray(dsV["pressure_level"].values, float)
    order_lev = np.argsort(lev_file)                       # ascending pressure (top first)
    lev = lev_file[order_lev]
    if sorted(lev.tolist()) != sorted(float(x) for x in EXPECTED_LEVELS):
        raise SystemExit(f"Pressure levels in the file {lev.tolist()} differ from the expected {EXPECTED_LEVELS}")
    nlev = lev.size

    latV = np.asarray(dsV["latitude"].values, float)
    lonV = np.asarray(dsV["longitude"].values, float)
    latP = np.asarray(dsP["latitude"].values, float)
    lonP = np.asarray(dsP["longitude"].values, float)
    if latV.max() < 60 or latV.min() > -60:
        raise SystemExit(f"Latitude range of V is {latV.min()} .. {latV.max()}; at least about 60S-60N is needed.")
    ridx = np.array([int(np.argmin(np.abs(latP - x))) for x in latV])
    if not np.allclose(latP[ridx], latV):
        raise SystemExit("Latitudes of V(Levels).nc are not on the grid of P-SLP.nc.")
    order_lon = np.argsort(np.mod(lonV, 360.0), kind="stable")
    if lonV.size != lonP.size or not np.allclose(np.mod(lonV, 360.0)[order_lon], lonP):
        raise SystemExit("Longitudes of V(Levels).nc do not match P-SLP.nc after reordering.")
    print(f"levels (hPa): {lev.tolist()}")
    print(f"latitude range {latV.max()} .. {latV.min()} ({latV.size} rows), longitude reordering needed: "
          f"{not np.array_equal(order_lon, np.arange(lonV.size))}")

    I0V, I0P = start_index(dsV, "V(Levels).nc"), start_index(dsP, "P-SLP.nc")

    # layer boundaries (hPa): top 50, mid-points between levels, bottom = surface pressure
    b = np.empty(nlev + 1)
    b[0] = lev[0] - 0.5 * (lev[1] - lev[0])
    b[1:nlev] = 0.5 * (lev[:-1] + lev[1:])
    b[nlev] = 1.0e9                                          # replaced by sp inside the column
    # position of each level inside its layer (for interpolating psi to the level itself)
    b_low = np.where(np.arange(nlev) < nlev - 1, b[1:], lev[-1])
    frac = np.clip((lev - b[:-1]) / (b_low - b[:-1]), 0, 1)

    coslat = np.cos(np.deg2rad(latV))
    PSI = np.full((nM, nlev, latV.size), np.nan, dtype="float32")
    t0 = time.time()
    for y in range(Y0, Y1 + 1):
        k = (y - Y0) * 12
        slP = slice(I0P + k, I0P + k + 12)
        slV = slice(I0V + k, I0V + k + 12)
        sp = dsP["sp"].isel(valid_time=slP).values[:, ridx, :].astype("float32") / 100.0     # hPa, (12, lat, lon)
        VF = np.zeros((12, nlev, latV.size))                 # zonal mean of v*dp   (m/s * hPa)
        MM = np.zeros((12, nlev, latV.size))                 # zonal mean of dp     (hPa)
        for kk in range(nlev):
            v = dsV["v"].isel(pressure_level=int(order_lev[kk]), valid_time=slV).values[:, :, order_lon]
            v = v.astype("float32")
            if not np.isfinite(v).all():
                v = np.where(np.isfinite(v), v, 0.0)
                print(f"Warning: NaN in v (year {y}, level {lev[kk]:g}); replaced by 0 with zero weight.")
            hi = np.minimum(b[kk + 1], sp)
            dp = np.clip(hi - b[kk], 0.0, None).astype("float32")
            VF[:, kk, :] = (v * dp).mean(axis=2, dtype="float64")
            MM[:, kk, :] = dp.mean(axis=2, dtype="float64")
        tot = VF.sum(axis=1, keepdims=True)
        msum = MM.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            Vc = VF - tot * MM / msum                        # mass-balance correction
        F = (2 * np.pi * A_EARTH * coslat[None, None, :] / G) * Vc * 100.0 / 1e9   # 1e9 kg/s per layer
        cum = np.cumsum(F, axis=1)                           # psi at the lower boundary of each layer
        prev = np.concatenate([np.zeros_like(cum[:, :1]), cum[:, :-1]], axis=1)
        PSI[k:k + 12] = (prev + (cum - prev) * frac[None, :, None]).astype("float32")
        resid = np.abs(cum[:, -1, :]).max()
        if y % 5 == 0:
            print(f"year {y} done (elapsed {time.time() - t0:.0f} s); max |psi| at the bottom after correction: "
                  f"{resid:.2e}", flush=True)
    np.savez_compressed(CACHE, psi=PSI, lat=latV, lev=lev)
    print("cache written:", CACHE)
    print(f"psi extremes: {np.nanmin(PSI):.1f} .. {np.nanmax(PSI):.1f}  (10^9 kg/s)")


if not os.path.exists(CACHE):
    extract()
else:
    print("Using cached psi (delete the .npz file in outputs to recompute).")

# ===============================================================
# Part B: Hadley-cell edge and statistics
# ===============================================================
zc = np.load(CACHE, allow_pickle=False)
PSI = np.asarray(zc["psi"], float)
lats = np.asarray(zc["lat"], float)
levs = np.asarray(zc["lev"], float)
tt = pd.DatetimeIndex(months.to_timestamp())
enso = pd.read_csv(ENSO_CSV)


def enso_season(mlist, col="ONI"):
    d = enso[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if mlist == [12, 1, 2] else 0)
    d = d[d.Month.isin(mlist)]
    g = d.groupby("Y")[col]
    return g.mean()[g.count() == len(mlist)]


ENSO_S = {s: enso_season(m) for s, m in SEASONS.items()}


def season_series(arr, times, mlist):
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
    out.update(ols_raw=b, ols_raw_lo=lo, ols_raw_hi=hi, p_raw=p)
    b, p, lo, hi, ne, beta = ols_adj(y, design(yrs, e), 1)
    y_adj = y - beta[2] * e
    sl = stats.theilslopes(y_adj, yrs, 0.95)
    out.update(ols_adj=b, ols_adj_lo=lo, ols_adj_hi=hi, p_adj=p, n_eff_adj=ne,
               sen_adj=sl[0] * 10, sen_adj_lo=sl[2] * 10, sen_adj_hi=sl[3] * 10,
               mk_p_adj=stats.kendalltau(yrs, y_adj)[1])
    return out


def smooth(f, size):
    return uniform_filter1d(f, size, mode="nearest")


def hadley_edge(psi_lat_profile, hemi):
    """Edge, extremum latitude and strength of the Hadley cell of one hemisphere from psi at one level.
    Returns (edge, ext_lat, ext_value, flags) with |latitude| in degrees; NaN when no cell/crossing is found."""
    sgn = 1.0 if hemi == "NH" else -1.0
    sel = (lats >= 0) if hemi == "NH" else (lats <= 0)
    x = np.abs(lats[sel])
    y = sgn * smooth(psi_lat_profile[sel], NSMOOTH)
    o = np.argsort(x)
    x, y = x[o], y[o]
    win = (x >= EXT_RANGE[0]) & (x <= EXT_RANGE[1])
    if not win.any() or not np.isfinite(y[win]).any():
        return np.nan, np.nan, np.nan, True, False
    iw = np.where(win)[0]
    i0 = iw[int(np.nanargmax(y[iw]))]
    ext_edge = i0 in (iw[0], iw[-1])
    if not (y[i0] > 0):
        return np.nan, x[i0], y[i0], ext_edge, False
    j = i0 + 1
    while j < len(x) and y[j] > 0:
        j += 1
    if j >= len(x):
        return np.nan, x[i0], y[i0], ext_edge, False
    edge = x[j - 1] + (x[j] - x[j - 1]) * y[j - 1] / (y[j - 1] - y[j])
    return float(edge), float(x[i0]), float(y[i0]), ext_edge, True


rows = []
for lv in EDGE_LEVELS:
    kl = int(np.where(np.isclose(levs, lv))[0][0])
    for sname, mlist in SEASONS.items():
        years, prof = season_series(PSI[:, kl, :], tt, mlist)
        for hemi in ("NH", "SH"):
            for yr, p in zip(years, prof):
                e, xl, xv, ext_edge, found = hadley_edge(p, hemi)
                rows.append((hemi, sname, lv, int(yr), e, xl, abs(xv) if np.isfinite(xv) else np.nan, ext_edge, found))
E = pd.DataFrame(rows, columns=["hemi", "season", "level_hPa", "year", "edge_abs_lat", "extremum_abs_lat",
                                "extremum_strength", "extremum_at_search_edge", "edge_found"])
E.to_csv(os.path.join(OUT_DIR, "hadley_edge_by_year.csv"), index=False)
print("\nShare of years with a valid edge (psi at the given level):")
print(E.groupby(["level_hPa", "hemi", "season"]).edge_found.mean().unstack().round(2))

out = []
for (hemi, seas, lv), g in E.groupby(["hemi", "season", "level_hPa"]):
    for metric, col in [("edge_abs_lat", "edge_abs_lat"), ("extremum_abs_lat", "extremum_abs_lat"),
                        ("extremum_strength_1e9kg_s", "extremum_strength")]:
        s = g.set_index("year")[col]
        for pname, (ya, yb) in PERIODS.items():
            r = analyse(s, seas, ya, yb)
            if r is None:
                continue
            r.update(hemi=hemi, season=seas, level_hPa=lv, metric=metric, period=pname,
                     valid_fraction=float(g.edge_found.mean()),
                     search_edge_fraction=float(g.extremum_at_search_edge.mean()))
            out.append(r)
T = pd.DataFrame(out)
T["q_adj"] = np.nan
T["fdr_family"] = ""
T["fdr_family_size"] = 0
for (pname, metric_group), idx in T.assign(mg=np.where(T.metric == "edge_abs_lat", "edge", "other")).groupby(
        ["period", "mg"]).groups.items():
    T.loc[idx, "q_adj"] = bh_fdr(T.loc[idx, "p_adj"].values)
    T.loc[idx, "fdr_family"] = ("edge: 2 hemispheres x 3 seasons x 3 levels" if metric_group == "edge"
                                else "extremum latitude and strength: 2 x 3 x 3 levels x 2 metrics")
    T.loc[idx, "fdr_family_size"] = int(np.isfinite(T.loc[idx, "p_adj"]).sum())
T = T[["hemi", "season", "level_hPa", "metric", "period", "n", "valid_fraction", "search_edge_fraction",
       "ols_raw", "p_raw", "ols_adj", "ols_adj_lo", "ols_adj_hi", "p_adj", "n_eff_adj", "sen_adj", "sen_adj_lo",
       "sen_adj_hi", "mk_p_adj", "q_adj", "fdr_family", "fdr_family_size"]]
T.to_csv(os.path.join(OUT_DIR, "hadley_edge_trends.csv"), index=False)

# ---------------------------------------------------------------
# Comparison with the ridge indicators
# ---------------------------------------------------------------
RIDGE = []
if os.path.exists(LEVELS_CSV):
    L = pd.read_csv(LEVELS_CSV)
    L = L[(L.sector == "ocean") & (L.season == "Annual") & (L.method == "centroid")]
    for v, g in L.groupby(["var", "hemi"]):
        RIDGE.append((v[0], v[1], g.set_index("year")["abs_lat"]))
if os.path.exists(PEAK_CSV):
    P = pd.read_csv(PEAK_CSV)
    P = P[(P.sector == "ocean") & (P.season == "Annual") & (P.method == "centroid") & (P["var"].isin(["tcc", "wspd"]))]
    for v, g in P.groupby(["var", "hemi"]):
        RIDGE.append((v[0], v[1], g.set_index("year")["abs_lat"]))
if not RIDGE:
    print("Warning: no ridge series found (run scripts 02 and 12 first); the comparison is skipped.")
NAMES = {"w700": "omega 700", "w500": "omega 500", "w300": "omega 300", "msl": "SLP", "tcc": "cloud", "wspd": "wind"}

ed = E[(E.level_hPa == 500) & (E.season == "Annual")]
mean_rows, corr_rows, trend_rows = [], [], []
for hemi in ("NH", "SH"):
    se = ed[ed.hemi == hemi].set_index("year")["edge_abs_lat"].dropna()
    mean_rows.append(dict(hemi=hemi, indicator="Hadley edge (psi 500)", mean_abs_lat=se.mean(), sd=se.std(), n=len(se)))
    for v, h, s in [r for r in RIDGE if r[1] == hemi]:
        mean_rows.append(dict(hemi=hemi, indicator=NAMES[v] + " centroid", mean_abs_lat=s.mean(), sd=s.std(), n=len(s)))
        for pname, (ya, yb) in PERIODS.items():
            d = pd.concat([se, s], axis=1, keys=["edge", "ridge"]).dropna()
            d = d[(d.index >= ya) & (d.index <= yb)]
            if len(d) < MIN_N:
                continue
            tr = np.arange(len(d), dtype=float)
            dd = d.apply(lambda c: c - np.polyval(np.polyfit(tr, c.values, 1), tr))
            corr_rows.append(dict(hemi=hemi, indicator=NAMES[v], period=pname, n=len(d),
                                  r=d.corr().iloc[0, 1], r_detrended=dd.corr().iloc[0, 1]))
            r = analyse(s, "Annual", ya, yb)
            if r is not None:
                trend_rows.append(dict(hemi=hemi, indicator=NAMES[v] + " centroid", period=pname,
                                       ols_adj=r["ols_adj"], lo=r["ols_adj_lo"], hi=r["ols_adj_hi"], p_adj=r["p_adj"]))
    for pname, (ya, yb) in PERIODS.items():
        r = analyse(se, "Annual", ya, yb)
        if r is not None:
            trend_rows.append(dict(hemi=hemi, indicator="Hadley edge (psi 500)", period=pname,
                                   ols_adj=r["ols_adj"], lo=r["ols_adj_lo"], hi=r["ols_adj_hi"], p_adj=r["p_adj"]))
M = pd.DataFrame(mean_rows)
C = pd.DataFrame(corr_rows)
TR = pd.DataFrame(trend_rows)
M.to_csv(os.path.join(OUT_DIR, "hadley_vs_ridge_mean_latitude.csv"), index=False)
C.to_csv(os.path.join(OUT_DIR, "hadley_vs_ridge_correlations.csv"), index=False)
TR.to_csv(os.path.join(OUT_DIR, "hadley_vs_ridge_trends.csv"), index=False)

# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
levels_c = np.arange(-120, 121, 10)
for ax, (sname, mlist) in zip(axs, SEASONS.items()):
    sel = np.isin(tt.month.values, mlist)
    mean_psi = np.nanmean(PSI[sel], axis=0)
    cs = ax.contourf(lats, levs, mean_psi, levels=levels_c, cmap="RdBu_r", extend="both")
    ax.contour(lats, levs, mean_psi, levels=[0], colors="k", linewidths=1.2)
    kl = int(np.where(np.isclose(levs, 500))[0][0])
    for hemi in ("NH", "SH"):
        g = E[(E.level_hPa == 500) & (E.season == sname) & (E.hemi == hemi)].edge_abs_lat.mean()
        if np.isfinite(g):
            ax.plot([g if hemi == "NH" else -g] * 2, [100, 1000], color="green", ls="--", lw=1.2)
    ax.set_title(f"{sname}: mean mass streamfunction, 1966-2025", fontsize=10)
    ax.set_xlabel("Latitude")
    ax.set_xlim(-65, 65)
    ax.set_ylim(1000, 100)
axs[0].set_ylabel("Pressure (hPa)")
cb = fig.colorbar(cs, ax=axs, shrink=0.85, pad=0.01)
cb.set_label(r"$\psi$ ($10^{9}$ kg s$^{-1}$); red = NH Hadley cell sense")
fig.suptitle("Dashed green: mean Hadley-cell edge (zero of $\\psi_{500}$ poleward of its extremum)", fontsize=9)
fig.savefig(os.path.join(OUT_DIR, "fig_psi_climatology.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

COL = {"Hadley edge (psi 500)": "#000000", "omega 500 centroid": "#7570b3", "SLP centroid": "#d95f02",
       "cloud centroid": "#1b9e77", "wind centroid": "#e7298a", "omega 700 centroid": "#66a61e",
       "omega 300 centroid": "#a6761d"}
fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))
for j, hemi in enumerate(("NH", "SH")):
    ax = axs[0, j]
    se = ed[ed.hemi == hemi].set_index("year")["edge_abs_lat"].dropna()
    series = [("Hadley edge (psi 500)", se)] + [(NAMES[v] + " centroid", s) for v, h, s in RIDGE
                                                if h == hemi and v in ("w500", "msl", "tcc")]
    for nm, s in series:
        a = s - s.mean()
        ax.plot(a.index, a.rolling(5, center=True, min_periods=3).mean(), color=COL[nm], lw=1.8, label=nm)
    ax.axvline(1979, color="grey", ls=":", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{hemi}: annual latitude anomalies, 5-yr mean", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly of |latitude| (deg; + = poleward)")
    if j == 0:
        ax.legend(fontsize=8, frameon=False)
    ax = axs[1, j]
    d = TR[TR.hemi == hemi]
    names = list(dict.fromkeys(d.indicator))
    for i, nm in enumerate(names):
        for pn, mk, off in [("1979-2025", "o", 0.12), ("1966-2025", "s", -0.12)]:
            r = d[(d.indicator == nm) & (d.period == pn)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.ols_adj, i + off, xerr=[[r.ols_adj - r.lo], [r.hi - r.ols_adj]], fmt=mk,
                        color=COL.get(nm, "k"), mfc=COL.get(nm, "k") if r.p_adj < 0.05 else "white", ms=5, lw=1.2,
                        capsize=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ENSO-adjusted trend (deg/decade, + = poleward), 95% CI")
    ax.set_title(f"{hemi}: circles 1979-2025, squares 1966-2025 (filled: p<0.05)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_hadley_edge.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 400)
print("\n=== Mean position (annual, |deg|) ===")
print(M.round(2).to_string(index=False))
for pn in PERIODS:
    print(f"\n=== Hadley-cell edge trends, {pn} (ENSO-adjusted, deg/decade, + = poleward; q within the 18-test edge family) ===")
    d = T[(T.metric == "edge_abs_lat") & (T.period == pn)]
    for r in d.itertuples():
        print(f"  {r.hemi} {r.season:6s} psi{int(r.level_hPa):4d}  {r.ols_adj:+.2f} [{r.ols_adj_lo:+.2f},{r.ols_adj_hi:+.2f}] "
              f"p={r.p_adj:.3f} q={r.q_adj:.3f} Sen={r.sen_adj:+.2f} valid={r.valid_fraction:.2f}")
print("\n=== Annual edge (psi 500) versus ridge indicators: correlation ===")
print(C.round(2).to_string(index=False))
print("\n=== Extremum latitude / strength trends, annual, psi 500, 1979-2025 ===")
d = T[(T.season == "Annual") & (T.level_hPa == 500) & (T.period == "1979-2025") & (T.metric != "edge_abs_lat")]
for r in d.itertuples():
    print(f"  {r.hemi} {r.metric:26s} {r.ols_adj:+.3f} [{r.ols_adj_lo:+.3f},{r.ols_adj_hi:+.3f}] p={r.p_adj:.3f} q={r.q_adj:.3f}")
print("\nDone. Outputs in:", OUT_DIR)
