# -*- coding: utf-8 -*-
"""
08_basin_trends_and_full_stats.py
Two additions requested in review:

 (A) Full statistics table for the main (ocean-only, zonal-mean) ridge series:
     OLS trend with 95% CI, Theil-Sen slope with 95% CI, Mann-Kendall p, autocorrelation-adjusted p,
     and Benjamini-Hochberg q. The FDR family is stated explicitly: ALL ocean-sector series
     (2 hemispheres x 3 seasons x [3 indicators x 2 estimators + calm-belt intensity] = 42 tests)
     within one period and one model type (raw or ENSO-adjusted).
 (B) The same ENSO-adjusted trend for each ocean basin separately
     (atlantic, pacific_w, pacific_c, pacific_e, indian), to test whether opposite-signed basin trends
     are hidden in the zonal mean.

Inputs  (DATA_DIR/outputs): peak_latitudes_by_year_masked.csv, calm_belt_min_speed_by_year.csv
        (DATA_DIR):         enso_monthly_1966_2025.csv
Outputs (DATA_DIR/outputs): main_trends_full_table.csv, basin_trends.csv, fig_basin_trends.png

Run:  python 08_basin_trends_and_full_stats.py     (takes a few seconds)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
BASINS = ["atlantic", "pacific_w", "pacific_c", "pacific_e", "indian"]
MIN_N = 20
MAX_EDGE = 0.3

enso = pd.read_csv(ENSO_CSV)


def enso_season(months, col="ONI"):
    d = enso[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")[col]
    return g.mean()[g.count() == len(months)]


ENSO_S = {s: enso_season(m) for s, m in SEASONS.items()}


def ols_adj(y, X, j):
    """Coefficient j, its autocorrelation-adjusted p-value and 95% CI (effective sample size, lag-1)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = n - k
    s2 = res @ res / dof
    se = np.sqrt((s2 * np.linalg.inv(X.T @ X))[j, j])
    r1 = max(np.corrcoef(res[:-1], res[1:])[0, 1], 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), k + 1.5)
    se_adj = se * np.sqrt(dof / (n_eff - k))
    p = 2 * stats.t.sf(abs(beta[j] / se_adj), n_eff - k)
    tcrit = stats.t.ppf(0.975, n_eff - k)
    return beta[j], p, beta[j] - tcrit * se_adj, beta[j] + tcrit * se_adj, n_eff, beta


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
    """Raw and ENSO(ONI)-adjusted statistics of one annual/seasonal series. Trends in units per decade."""
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
    sl = stats.theilslopes(y, yrs, 0.95)
    out.update(ols_raw=b, ols_raw_lo=lo, ols_raw_hi=hi, p_raw=p, n_eff_raw=ne,
               sen_raw=sl[0] * 10, sen_raw_lo=sl[2] * 10, sen_raw_hi=sl[3] * 10,
               mk_p_raw=stats.kendalltau(yrs, y)[1])
    b, p, lo, hi, ne, beta = ols_adj(y, design(yrs, e), 1)
    y_adj = y - beta[2] * e
    sl = stats.theilslopes(y_adj, yrs, 0.95)
    out.update(ols_adj=b, ols_adj_lo=lo, ols_adj_hi=hi, p_adj=p, n_eff_adj=ne,
               sen_adj=sl[0] * 10, sen_adj_lo=sl[2] * 10, sen_adj_hi=sl[3] * 10,
               mk_p_adj=stats.kendalltau(yrs, y_adj)[1])
    return out


pk = pd.read_csv(os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv"))
cb = pd.read_csv(os.path.join(OUT_DIR, "calm_belt_min_speed_by_year.csv"))


def collect(sectors, include_calm):
    items = []
    for (sec, hemi, seas, var, meth), g in pk.groupby(["sector", "hemi", "season", "var", "method"]):
        if sec in sectors:
            items.append((dict(sector=sec, hemi=hemi, season=seas, var=var, method=meth, unit="deg"),
                          g.set_index("year")["abs_lat"], g["at_window_edge"].mean()))
    if include_calm:
        for (sec, hemi, seas), g in cb.groupby(["sector", "hemi", "season"]):
            if sec in sectors:
                items.append((dict(sector=sec, hemi=hemi, season=seas, var="calm_min_wspd", method="min",
                                   unit="m/s"), g.set_index("year")["min_wspd"], 0.0))
    return items


# ---------------------------------------------------------------
# (A) full statistics table, main ocean sector
# ---------------------------------------------------------------
rows = []
for key, s, edge in collect(["ocean"], True):
    for pname, (ya, yb) in PERIODS.items():
        r = analyse(s, key["season"], ya, yb)
        if r:
            rows.append(dict(**key, period=pname, edge_fraction=edge, **r))
A = pd.DataFrame(rows)
for pname in PERIODS:
    idx = A.index[A.period == pname]
    n_fam = len(idx)
    for col in ["p_raw", "p_adj"]:
        A.loc[idx, "q_" + col] = bh_fdr(A.loc[idx, col].values)
    A.loc[idx, "fdr_family_size"] = n_fam
A.to_csv(os.path.join(OUT_DIR, "main_trends_full_table.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(f"(A) FDR family size per period/model: {int(A.fdr_family_size.iloc[0])} tests")
show = A[(A.season == "Annual") & (A.var != "calm_min_wspd") & (A.period == "1979-2025")]
print("\nAnnual, 1979-2025, raw (no ENSO removal): OLS [95% CI], Sen, MK p, p_adj(autocorr), q")
for _, r in show.iterrows():
    print(f"  {r.hemi} {r['var']:5s} {r.method:8s} OLS {r.ols_raw:+.2f} [{r.ols_raw_lo:+.2f},{r.ols_raw_hi:+.2f}]  "
          f"Sen {r.sen_raw:+.2f}  MK p={r.mk_p_raw:.3f}  p={r.p_raw:.3f}  q={r.q_p_raw:.3f}")

# ---------------------------------------------------------------
# (B) basins
# ---------------------------------------------------------------
rows = []
for key, s, edge in collect(BASINS, False):
    if key["season"] != "Annual":
        continue
    for pname, (ya, yb) in PERIODS.items():
        r = analyse(s, "Annual", ya, yb)
        if r:
            rows.append(dict(**key, period=pname, edge_fraction=edge, **r))
B = pd.DataFrame(rows)
B["q_p_adj"] = np.nan
for pname in PERIODS:
    idx = B.index[B.period == pname]
    B.loc[idx, "q_p_adj"] = bh_fdr(B.loc[idx, "p_adj"].values)
B["edge_flag"] = B.edge_fraction > MAX_EDGE
B.to_csv(os.path.join(OUT_DIR, "basin_trends.csv"), index=False)

print("\n(B) Basin trends, annual, ENSO-adjusted, 1979-2025 (deg/decade; * = edge fraction > 30%)")
for var in ["tcc", "w500", "wspd"]:
    for meth in ["centroid", "argmax"]:
        sub = B[(B["var"] == var) & (B.method == meth) & (B.period == "1979-2025")]
        if sub.empty:
            continue
        pv = sub.pivot_table(index="sector", columns="hemi", values="ols_adj").reindex(BASINS)
        pp = sub.pivot_table(index="sector", columns="hemi", values="p_adj").reindex(BASINS)
        ee = sub.pivot_table(index="sector", columns="hemi", values="edge_fraction").reindex(BASINS)
        print(f"\n  {var} / {meth}")
        for b in BASINS:
            cells = []
            for h in ["NH", "SH"]:
                if h in pv.columns and np.isfinite(pv.loc[b, h]):
                    star = "*" if ee.loc[b, h] > MAX_EDGE else " "
                    cells.append(f"{h} {pv.loc[b, h]:+.2f} (p={pp.loc[b, h]:.3f}){star}")
                else:
                    cells.append(f"{h}   n/a")
            print(f"    {b:10s} " + "   ".join(cells))

# sign agreement summary for the cloud and omega centroid
print("\nSign summary (1979-2025, centroid, annual): number of basins with positive / negative trend")
for var in ["tcc", "w500", "wspd"]:
    for h in ["NH", "SH"]:
        sub = B[(B["var"] == var) & (B.method == "centroid") & (B.period == "1979-2025") & (B.hemi == h)]
        print(f"  {var:5s} {h}: +{int((sub.ols_adj > 0).sum())} / -{int((sub.ols_adj < 0).sum())} "
              f"(significant at p<0.05: {int((sub.p_adj < 0.05).sum())})")

# ---------------------------------------------------------------
# Figure: basin forest plot (annual, centroid, ENSO adjusted, 1979-2025), zonal mean for reference
# ---------------------------------------------------------------
LAB = {"tcc": "Min cloud cover", "w500": "Max omega (500 hPa)", "wspd": "Min wind speed"}
NICE = {"atlantic": "Atlantic", "pacific_w": "W Pacific", "pacific_c": "C Pacific", "pacific_e": "E Pacific",
        "indian": "Indian", "ocean": "All oceans (zonal mean)"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, hemi in zip(axes, ["NH", "SH"]):
    yy, yt, yl = 0, [], []
    for var in ["tcc", "w500", "wspd"]:
        for sec in BASINS + ["ocean"]:
            src = B if sec != "ocean" else A[A.season == "Annual"]
            q = src[(src.sector == sec) & (src.hemi == hemi) & (src["var"] == var) & (src.method == "centroid")
                    & (src.period == "1979-2025")]
            if q.empty:
                yy += 1
                continue
            q = q.iloc[0]
            col = "k" if sec == "ocean" else ("tab:orange" if q.edge_fraction > MAX_EDGE else "tab:blue")
            ax.errorbar(q.ols_adj, yy, xerr=[[q.ols_adj - q.ols_adj_lo], [q.ols_adj_hi - q.ols_adj]],
                        fmt="s" if sec == "ocean" else "o", color=col, capsize=2, ms=5)
            yt.append(yy)
            yl.append(f"{LAB[var]} - {NICE[sec]}")
            yy += 1
        yy += 1
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-0.9, 0.9)           # a few poorly constrained estimates are off scale; they are listed in the CSV
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Poleward shift (deg/decade), centroid, ENSO removed")
    ax.set_title(f"{'Northern' if hemi == 'NH' else 'Southern'} Hemisphere, 1979-2025 (95% CI)")
    ax.grid(alpha=0.3)
fig.suptitle("Ridge-latitude trends by ocean basin (orange: peak at window edge in >30% of years)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_basin_trends.png"), dpi=150)
plt.close(fig)
print("\nDone. Outputs in:", OUT_DIR)
