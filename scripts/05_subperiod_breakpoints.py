# -*- coding: utf-8 -*-
"""
05_subperiod_breakpoints.py
Is the poleward trend of the ERA5 cloud/ridge uniform over the whole period, or does it come from one specific period?

1) Trends in sub-periods: 1966-1978 (before TOVS), 1979-1999, 2000-2025 (CERES era), 1979-2025
2) 'Broken-stick' regression (continuous trend with a slope change at year t0), scanning t0 over 1975..2012,
   with ENSO (ONI) removed as a covariate. Significance uses an AR(1) Monte Carlo simulation and the
   sup-F statistic, so the free scan over t0 is accounted for in the p-value.

Input: outputs/peak_latitudes_by_year_masked.csv , outputs/calm_belt_min_speed_by_year.csv ,
        enso_monthly_1966_2025.csv
Output: era5_subperiod_trends.csv , era5_breakpoint_scan.csv , fig14_ridge_breakpoints.png
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

SECTORS = ["ocean"]
VARS = ["tcc", "w500", "wspd"]
METHODS = ["centroid", "argmax"]
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
SUBPERIODS = {"1966-1978": (1966, 1978), "1979-1999": (1979, 1999),
              "2000-2025": (2000, 2025), "1979-2025": (1979, 2025)}
BREAK_RANGE = range(1975, 2013)
N_SIM = 400
SEED = 7
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------
# ENSO
# ---------------------------------------------------------------
enso = pd.read_csv(ENSO_CSV)


def enso_season(months):
    d = enso[["Year", "Month", "ONI"]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")["ONI"]
    return g.mean()[g.count() == len(months)]


ENSO_S = {s: enso_season(m) for s, m in SEASONS.items()}

# ---------------------------------------------------------------
# Series
# ---------------------------------------------------------------
pk = pd.read_csv(os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv"))
cb = pd.read_csv(os.path.join(OUT_DIR, "calm_belt_min_speed_by_year.csv"))
series = []
for (sec, hemi, seas, var, meth), g in pk.groupby(["sector", "hemi", "season", "var", "method"]):
    if sec in SECTORS and var in VARS and meth in METHODS:
        series.append((dict(sector=sec, hemi=hemi, season=seas, var=var, method=meth, unit="deg"),
                       g.set_index("year")["abs_lat"], g["at_window_edge"].mean()))
for (sec, hemi, seas), g in cb.groupby(["sector", "hemi", "season"]):
    if sec in SECTORS:
        series.append((dict(sector=sec, hemi=hemi, season=seas, var="calm_min_wspd", method="min", unit="m/s"),
                       g.set_index("year")["min_wspd"], 0.0))
print(f"{len(series)} series")


# ---------------------------------------------------------------
# Statistical tools
# ---------------------------------------------------------------
def trend_adj(years, y, e=None):
    """Linear trend (per decade) with autocorrelation adjustment; if e is given, ENSO is removed."""
    t = (years - years.mean()) / 10.0
    cols = [np.ones_like(t), t] + ([e] if e is not None else [])
    X = np.column_stack(cols)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (n - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    ne = max(n * (1 - r1) / (1 + r1), k + 1.5)
    p = 2 * stats.t.sf(abs(beta[1] / (se * np.sqrt((n - k) / (ne - k)))), ne - k)
    return beta[1], p, stats.theilslopes(y, years)[0] * 10


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


def design(years, t0=None, e=None):
    t = (years - 1995.5) / 10.0
    cols = [np.ones_like(t), t]
    if t0 is not None:
        cols.append(np.maximum(0.0, years - t0) / 10.0)
    if e is not None:
        cols.append(e)
    return np.column_stack(cols)


def resid_proj(X):
    return np.eye(len(X)) - X @ np.linalg.pinv(X)


def scan_break(years, y, e, breaks, sims=0):
    X0 = design(years, None, e)
    M0 = resid_proj(X0)
    rss0 = float(y @ M0 @ y)
    best = None
    Ms = {}
    for t0 in breaks:
        if (years < t0).sum() < 8 or (years > t0).sum() < 8:
            continue
        X1 = design(years, t0, e)
        M1 = resid_proj(X1)
        Ms[t0] = M1
        rss1 = float(y @ M1 @ y)
        if best is None or rss1 < best[1]:
            best = (t0, rss1)
    t0, rss1 = best
    k1 = design(years, t0, e).shape[1]
    n = len(y)
    F = (rss0 - rss1) / (rss1 / (n - k1))
    p_mc = np.nan
    if sims:
        beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        fit0, res0 = X0 @ beta0, y - X0 @ beta0
        phi = float(np.clip(np.corrcoef(res0[:-1], res0[1:])[0, 1], 0, 0.9))
        sig = res0.std(ddof=X0.shape[1]) * np.sqrt(1 - phi ** 2) if phi < 1 else res0.std()
        eps = rng.normal(0, sig, (n, sims))
        noise = np.zeros((n, sims))
        noise[0] = rng.normal(0, res0.std(), sims)
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + eps[i]
        Ysim = fit0[:, None] + noise
        r0 = ((M0 @ Ysim) ** 2).sum(axis=0)
        r1 = np.min([((M @ Ysim) ** 2).sum(axis=0) for M in Ms.values()], axis=0)
        Fs = (r0 - r1) / (r1 / (n - k1))
        p_mc = float((Fs >= F).mean())
    beta, *_ = np.linalg.lstsq(design(years, t0, e), y, rcond=None)
    return dict(t0=t0, F=F, p_mc=p_mc, slope_before=beta[1], slope_after=beta[1] + beta[2],
                rss_gain=1 - rss1 / rss0)


# ---------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------
rows_sub, rows_brk = [], []
for key, s, edge in series:
    e_all = ENSO_S[key["season"]]
    s = s.dropna()
    for pname, (ya, yb) in SUBPERIODS.items():
        ss = s[(s.index >= ya) & (s.index <= yb)]
        if len(ss) < 10:
            continue
        yrs = ss.index.values.astype(float)
        e = e_all.reindex(ss.index)
        use_e = e.notna().all()
        ez = ((e - e.mean()) / e.std(ddof=1)).values if use_e else None
        b, p, sen = trend_adj(yrs, ss.values, ez)
        rows_sub.append(dict(**key, period=pname, n=len(ss), edge_fraction=edge,
                             trend=b, p_adj=p, sen=sen))
    common = s.index.intersection(e_all.index)
    yrs = common.values.astype(float)
    y = s.loc[common].values
    e = e_all.loc[common].values
    ez = (e - e.mean()) / e.std(ddof=1)
    r = scan_break(yrs, y, ez, BREAK_RANGE, sims=N_SIM)
    rows_brk.append(dict(**key, n=len(y), edge_fraction=edge, **r))
    print("  ", key["hemi"], key["season"], key["var"], key["method"], "-> t0 =", r["t0"], f"p_mc={r['p_mc']:.3f}")

sub = pd.DataFrame(rows_sub)
sub["q_adj"] = bh_fdr(sub["p_adj"].values)
sub.to_csv(os.path.join(OUT_DIR, "era5_subperiod_trends.csv"), index=False)
brk = pd.DataFrame(rows_brk)
brk["q_mc"] = bh_fdr(brk["p_mc"].values)
brk.to_csv(os.path.join(OUT_DIR, "era5_breakpoint_scan.csv"), index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)
print("\n=== Sub-period trends (ENSO removed); cloud (tcc), centroid, ocean ===")
q = sub[(sub["var"] == "tcc") & (sub.method == "centroid")]
print(q.pivot_table(index=["hemi", "season"], columns="period", values="trend").round(3)
      [["1966-1978", "1979-1999", "2000-2025", "1979-2025"]].to_string())
print("\n(adjusted p)")
print(q.pivot_table(index=["hemi", "season"], columns="period", values="p_adj").round(3)
      [["1966-1978", "1979-1999", "2000-2025", "1979-2025"]].to_string())
print("\n=== Break-point scan (all series) - by Monte Carlo p ===")
cols = ["hemi", "season", "var", "method", "t0", "slope_before", "slope_after", "p_mc", "q_mc", "edge_fraction"]
print(brk.sort_values("p_mc")[cols].round(3).head(20).to_string(index=False))

# ---------------------------------------------------------------
# Figure 14: cloud (centroid) with broken stick
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, hemi in enumerate(["NH", "SH"]):
    for j, seas in enumerate(["Annual", "DJF", "JJA"]):
        ax = axes[i, j]
        row = next((r for r in zip(series) if False), None)
        sr = [(k, s) for k, s, _ in series if k["hemi"] == hemi and k["season"] == seas
              and k["var"] == "tcc" and k["method"] == "centroid"]
        if not sr:
            continue
        k, s = sr[0]
        s = s.dropna()
        e_all = ENSO_S[seas]
        common = s.index.intersection(e_all.index)
        yrs = common.values.astype(float)
        y = s.loc[common].values
        e = e_all.loc[common].values
        ez = (e - e.mean()) / e.std(ddof=1)
        b = brk[(brk.hemi == hemi) & (brk.season == seas) & (brk["var"] == "tcc") & (brk.method == "centroid")].iloc[0]
        X = design(yrs, b.t0, ez)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_adj = y - beta[3] * ez
        ax.plot(yrs, y_adj, "-o", ms=3, lw=1, color="tab:gray", label="ERA5 cloud-min centroid (ENSO removed)")
        fit = beta[0] + beta[1] * (yrs - 1995.5) / 10 + beta[2] * np.maximum(0, yrs - b.t0) / 10
        ax.plot(yrs, fit, "r-", lw=1.8, label="broken-stick fit")
        ax.axvline(b.t0, color="r", ls=":", lw=1)
        ax.axvline(1979, color="tab:blue", ls=":", lw=0.9)
        ax.axvline(2000, color="tab:green", ls=":", lw=0.9)
        ax.set_title(f"{'Northern' if hemi == 'NH' else 'Southern'} Hemisphere - {seas}\n"
                     f"break {int(b.t0)}: {b.slope_before:+.2f} -> {b.slope_after:+.2f} deg/dec (MC p = {b.p_mc:.2f})",
                     fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("|Latitude| (deg)")
        ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
fig.suptitle("ERA5 ocean cloud-minimum latitude: where does the poleward trend come from?\n"
             "(blue dotted: 1979, green dotted: 2000 = start of CERES)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig14_ridge_breakpoints.png"), dpi=150)
plt.close(fig)
print("\nDone. Outputs in:", OUT_DIR)
