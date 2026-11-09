# -*- coding: utf-8 -*-
"""
03_enso_adjusted_trends.py
Remove the ENSO effect from the annual ridge-latitude / calm-belt-intensity series and test for a break near 1979.

Inputs (all from earlier runs):
  outputs/peak_latitudes_by_year_masked.csv   (from 02)
  outputs/calm_belt_min_speed_by_year.csv     (from 02)
  enso_monthly_1966_2025.csv                  (from 03a)

Models for each series y(t):
  (A) raw         : y = a + b*t
  (B) adjusted    : y = a + b*t + c*ENSO(t)         ENSO = ONI / SOI / MEI (same season/year)
  (C) 1979 step   : y = a + b*t + c*ENSO + d*[t>=1979]
t is in decades, so b is directly 'units per decade'.
p-values are adjusted with the lag-1 autocorrelation of the residuals (effective n), as in 02.
Outputs: enso_adjusted_trends.csv , enso_breakpoint_1979.csv , fig10 , fig11
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")

PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
SECTORS = ["ocean", "nomtn2500"]          # ocean ridge (main) and without height > 2500 m
VARS = ["w500", "tcc", "wspd"]
METHODS = ["argmax", "centroid"]
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
INDICES = {"ONI": ("ONI", 1966), "SOI": ("SOI", 1966), "MEI": ("MEI", 1979)}   # (column, first valid year)
BREAK_YEAR = 1979
MIN_N = 20
MAX_EDGE = 0.3          # series whose peak sits on the window edge in more than 30% of years are flagged

# ---------------------------------------------------------------
# Seasonal/annual ENSO
# ---------------------------------------------------------------
enso = pd.read_csv(ENSO_CSV)
enso["date"] = pd.to_datetime(dict(year=enso.Year, month=enso.Month, day=1))


def enso_season(col, months):
    d = enso[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")[col]
    ok = g.count() == len(months)
    return g.mean()[ok]


ENSO_S = {(ix, s): enso_season(col, m) for ix, (col, _) in INDICES.items() for s, m in SEASONS.items()}

# ---------------------------------------------------------------
# Regression with autocorrelation adjustment
# ---------------------------------------------------------------


def ols_adj(y, X, j):
    """Coefficient of column j with adjusted standard error; returns (beta, p_adj, R2, resid)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = n - k
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[j, j])
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), k + 1.5)
    se_adj = se * np.sqrt(dof / (n_eff - k))
    p = 2 * stats.t.sf(abs(beta[j] / se_adj), n_eff - k)
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return beta[j], p, r2, res, r1, n_eff


def fit_series(years, y, e=None, step=False):
    """Three cases: raw / with ENSO / with ENSO and step. t is in decades."""
    t = (years - 1995.5) / 10.0
    cols = [np.ones_like(t), t]
    if e is not None:
        cols.append(e)
    if step:
        cols.append((years >= BREAK_YEAR).astype(float))
    X = np.column_stack(cols)
    return X


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


# ---------------------------------------------------------------
# Load the series
# ---------------------------------------------------------------
pk = pd.read_csv(os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv"))
cb = pd.read_csv(os.path.join(OUT_DIR, "calm_belt_min_speed_by_year.csv"))

series = []     # (key dict, Series indexed by year, edge_fraction)
for (sec, hemi, seas, var, meth), g in pk.groupby(["sector", "hemi", "season", "var", "method"]):
    if sec in SECTORS and var in VARS and meth in METHODS:
        s = g.set_index("year")["abs_lat"]
        series.append((dict(sector=sec, hemi=hemi, season=seas, var=var, method=meth, unit="deg"),
                       s, g["at_window_edge"].mean()))
for (sec, hemi, seas), g in cb.groupby(["sector", "hemi", "season"]):
    if sec in SECTORS:
        series.append((dict(sector=sec, hemi=hemi, season=seas, var="calm_min_wspd", method="min", unit="m/s"),
                       g.set_index("year")["min_wspd"], 0.0))
print(f"{len(series)} series to analyse")

# ---------------------------------------------------------------
# Main analysis: raw versus adjusted
# ---------------------------------------------------------------
rows, brk = [], []
for key, s, edge in series:
    for pname, (ya, yb) in PERIODS.items():
        ss = s[(s.index >= ya) & (s.index <= yb)].dropna()
        for ix, (_, y_first) in INDICES.items():
            if ix == "MEI" and ya < y_first:
                continue
            e_all = ENSO_S[(ix, key["season"])]
            common = ss.index.intersection(e_all.index)
            if len(common) < MIN_N:
                continue
            yrs = common.values.astype(float)
            y = ss.loc[common].values
            e = e_all.loc[common].values
            e = (e - e.mean()) / e.std(ddof=1)            # ENSO coefficient = unit of y per 1 standard deviation
            # raw
            b0, p0, r20, _, _, _ = ols_adj(y, fit_series(yrs, y), 1)
            sen0 = stats.theilslopes(y, yrs)[0] * 10
            # with ENSO
            X1 = fit_series(yrs, y, e)
            b1, p1, r21, res1, r1, neff = ols_adj(y, X1, 1)
            c1, pc, _, _, _, _ = ols_adj(y, X1, 2)
            # Sen slope on the ENSO residual (non-parametric)
            beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            y_adj = y - beta[2] * e
            sen1 = stats.theilslopes(y_adj, yrs)[0] * 10
            tau_p = stats.kendalltau(yrs, y_adj)[1]
            rows.append(dict(**key, period=pname, index=ix, n=len(y), edge_fraction=edge,
                             trend_raw=b0, p_raw=p0, sen_raw=sen0,
                             trend_adj=b1, p_adj=p1, sen_adj=sen1, mk_p_adj=tau_p,
                             enso_coef_per_sd=c1, enso_p=pc,
                             R2_raw=r20, R2_adj=r21, n_eff=neff))
            # 1979 step only for the full period and ONI/SOI
            if pname == "1966-2025" and ix == "ONI":
                X2 = fit_series(yrs, y, e, step=True)
                d2, pd2, r22, res2, _, _ = ols_adj(y, X2, 3)
                b2, pb2, _, _, _, _ = ols_adj(y, X2, 1)
                # F-test of the step model against the no-step model (with a simple degrees-of-freedom adjustment)
                rss1, rss2 = res1 @ res1, res2 @ res2
                F = (rss1 - rss2) / (rss2 / (len(y) - X2.shape[1]))
                p_chow = stats.f.sf(F, 1, max(neff - X2.shape[1], 1.0))
                brk.append(dict(**key, n=len(y), step_1979=d2, step_p_adj=pd2, F_p_adj=p_chow,
                                trend_with_step=b2, trend_with_step_p=pb2, trend_adj_no_step=b1,
                                trend_adj_no_step_p=p1, R2_with_step=r22))

res = pd.DataFrame(rows)
res["q_adj"] = np.nan
for (pname, ix), idx in res.groupby(["period", "index"]).groups.items():
    res.loc[idx, "q_adj"] = bh_fdr(res.loc[idx, "p_adj"].values)
res.to_csv(os.path.join(OUT_DIR, "enso_adjusted_trends.csv"), index=False)
bk = pd.DataFrame(brk)
if len(bk):
    bk["step_q"] = bh_fdr(bk["step_p_adj"].values)
bk.to_csv(os.path.join(OUT_DIR, "enso_breakpoint_1979.csv"), index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)
show = ["sector", "hemi", "season", "var", "method", "period", "trend_raw", "p_raw", "trend_adj", "p_adj",
        "q_adj", "enso_coef_per_sd", "enso_p", "R2_adj"]
core = res[(res["index"] == "ONI") & (res.sector == "ocean") & (res.season == "Annual")]
print("\n=== Effect of removing ENSO (ONI) - ocean, annual (unit: degrees/decade or m/s/decade) ===")
print(core[show].round(3).to_string(index=False))
if len(bk):
    print("\n=== 1979 step test (ONI, 1966-2025) - significant cases (p<0.1) ===")
    sig = bk[bk.step_p_adj < 0.1][["sector", "hemi", "season", "var", "method", "step_1979", "step_p_adj",
                                   "trend_with_step", "trend_with_step_p", "trend_adj_no_step"]]
    print(sig.round(3).to_string(index=False) if len(sig) else "No case is significant")

# ---------------------------------------------------------------
# Figure 10: raw versus adjusted trend (ocean, ONI)
# ---------------------------------------------------------------
LAB = {"w500": "Max omega (500 hPa)", "tcc": "Min cloud cover", "wspd": "Min wind speed (lat)",
       "calm_min_wspd": "Calm-belt speed (m/s)"}
plot_items = [("tcc", "argmax"), ("w500", "argmax"), ("w500", "centroid"), ("wspd", "argmax")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
for r_i, hemi in enumerate(["NH", "SH"]):
    for c_i, seas in enumerate(["Annual", "DJF", "JJA"]):
        ax = axes[r_i, c_i]
        yy = 0
        yt, yl = [], []
        for var, meth in plot_items:
            for pname, col in [("1966-2025", "tab:blue"), ("1979-2025", "tab:orange")]:
                q = res[(res["index"] == "ONI") & (res.sector == "ocean") & (res.hemi == hemi) &
                        (res.season == seas) & (res["var"] == var) & (res.method == meth) &
                        (res.period == pname)]
                if q.empty:
                    continue
                q = q.iloc[0]
                ax.plot(q.trend_raw, yy + 0.12, "o", mfc="none", color=col)
                ax.plot(q.trend_adj, yy - 0.12, "s", color=col)
                if q.p_adj < 0.05:
                    ax.plot(q.trend_adj, yy - 0.12, "*", color="k", ms=9, zorder=5)
            yt.append(yy)
            yl.append(f"{LAB[var].split(' (')[0]}\n[{meth}]")
            yy += 1
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{'Northern' if hemi == 'NH' else 'Southern'} Hemisphere - {seas}", fontsize=10)
        ax.set_xlabel("Poleward shift (deg/decade)")
        ax.grid(alpha=0.3)
axes[0, 0].plot([], [], "o", mfc="none", color="gray", label="raw trend")
axes[0, 0].plot([], [], "s", color="gray", label="ENSO(ONI)-adjusted")
axes[0, 0].plot([], [], "*", color="k", label="adj. p < 0.05")
axes[0, 0].plot([], [], "s", color="tab:blue", label="1966-2025")
axes[0, 0].plot([], [], "s", color="tab:orange", label="1979-2025")
axes[0, 0].legend(fontsize=7, loc="best")
fig.suptitle("Ridge-latitude trends before and after removing ENSO (ocean only)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig10_enso_adjusted_trends.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 11: calm-belt intensity with the 1979 line (annual, ocean)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, hemi in zip(axes, ["NH", "SH"]):
    s = cb[(cb.sector == "ocean") & (cb.hemi == hemi) & (cb.season == "Annual")].set_index("year")["min_wspd"]
    e = ENSO_S[("ONI", "Annual")].reindex(s.index)
    ok = s.index[e.notna()]
    yrs = ok.values.astype(float)
    y = s.loc[ok].values
    ez = ((e.loc[ok] - e.loc[ok].mean()) / e.loc[ok].std(ddof=1)).values
    X = fit_series(yrs, y, ez)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_adj = y - beta[2] * ez
    ax.plot(yrs, y, color="tab:green", lw=1.2, label="calm-belt min speed")
    ax.plot(yrs, y_adj, color="k", lw=1.2, label="ENSO(ONI) removed")
    for ya, yb, col in [(1966, 2025, "tab:blue"), (1979, 2025, "tab:orange")]:
        m = (yrs >= ya) & (yrs <= yb)
        sl = stats.linregress(yrs[m], y_adj[m])
        ax.plot(yrs[m], sl.intercept + sl.slope * yrs[m], "--", color=col,
                label=f"{ya}-{yb}: {sl.slope * 10:+.3f} m/s/decade")
    ax.axvline(1979, color="red", lw=1, ls=":")
    ax.text(1979.5, ax.get_ylim()[0], " TOVS (late 1978)", color="red", va="bottom", fontsize=8)
    ax.set_title(f"{'Northern' if hemi == 'NH' else 'Southern'} Hemisphere - Annual")
    ax.set_xlabel("Year")
    ax.set_ylabel("Min zonal-mean wind speed, ocean (m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Calm-belt intensity: effect of ENSO and of the 1979 observing-system change", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig11_calm_belt_1979_check.png"), dpi=150)
plt.close(fig)

print("\nDone. Outputs in:", OUT_DIR)
print(" - enso_adjusted_trends.csv, enso_breakpoint_1979.csv")
print(" - fig10_enso_adjusted_trends.png, fig11_calm_belt_1979_check.png")
