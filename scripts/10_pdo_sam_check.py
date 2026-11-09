# -*- coding: utf-8 -*-
"""
10_pdo_sam_check.py
Do other modes of variability (PDO, Southern Annular Mode / Antarctic Oscillation) change the ridge trends?

 1) Downloads monthly PDO (NOAA NCEI ERSSTv5, 1854-present) and AAO (NOAA CPC, 1979-present; used as the SAM index)
    and stores them in DATA_DIR/enso_raw (raw) and DATA_DIR/climate_modes_monthly.csv (Year, Month, PDO, AAO).
    If a download fails, save the raw file by hand with the name given in the message and run again.
 2) Re-fits the annual/seasonal ocean ridge-latitude trends (series from 02) with
       M0: t + ONI                (as in 03)
       M1: t + ONI + PDO
       M2: t + ONI + PDO + AAO    (1979-2025 only, because AAO starts in 1979)
    and prints the trend, 95% CI and autocorrelation-adjusted p for each model.
 Indices are averaged over the same season as the ridge series and z-scored; the PDO is
 partly correlated with ENSO, so the correlation between regressors is reported.

Inputs  (DATA_DIR/outputs): peak_latitudes_by_year_masked.csv ;  (DATA_DIR): enso_monthly_1966_2025.csv
Outputs (DATA_DIR): climate_modes_monthly.csv ; (DATA_DIR/outputs): pdo_sam_adjusted_trends.csv

Run:  python 10_pdo_sam_check.py
"""
import os
import re
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
RAW_DIR = os.path.join(DATA_DIR, "enso_raw")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
MODES_CSV = os.path.join(DATA_DIR, "climate_modes_monthly.csv")
URLS = {
    "pdo.txt": ["https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
                "https://psl.noaa.gov/data/correlation/pdo.data"],
    "aao.txt": ["https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/"
                "monthly.aao.index.b79.current.ascii"],
}
SEASONS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}
PERIODS = {"1966-2025": (1966, 2025), "1979-2025": (1979, 2025)}
SECTOR = "ocean"
MIN_N = 20
os.makedirs(RAW_DIR, exist_ok=True)


def fetch(name):
    path = os.path.join(RAW_DIR, name)
    for url in URLS[name]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"[downloaded] {name} <- {url}")
            return path
        except Exception as e:
            print(f"[warning] could not download {url}: {e}")
    if os.path.exists(path):
        print(f"[warning] using the saved copy {path}")
        return path
    raise SystemExit(f"Could not obtain {name}. Download one of\n  " + "\n  ".join(URLS[name]) +
                     f"\nby hand and save it as {path}, then run again.")


def read_year_x12(path, miss_abs=9.0):
    """Rows 'year v1 ... v12' (PDO). Values with |v| >= miss_abs are treated as missing."""
    out = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        tok = line.replace(",", " ").split()
        if len(tok) >= 13 and re.fullmatch(r"\d{4}", tok[0]):
            try:
                v = [float(x) for x in tok[1:13]]
            except ValueError:
                continue
            out[int(tok[0])] = [np.nan if abs(x) >= miss_abs else x for x in v]
    if not out:
        raise SystemExit(f"No 'year + 12 values' rows found in {path}")
    return out


def read_year_month_value(path):
    """Rows 'year month value' (AAO)."""
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        tok = line.split()
        if len(tok) >= 3 and re.fullmatch(r"\d{4}", tok[0]) and re.fullmatch(r"\d{1,2}", tok[1]):
            try:
                rows.append((int(tok[0]), int(tok[1]), float(tok[2])))
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"No 'year month value' rows found in {path}")
    return pd.DataFrame(rows, columns=["Year", "Month", "AAO"])


pdo_t = read_year_x12(fetch("pdo.txt"))
pdo = pd.DataFrame([(y, m + 1, v) for y, vals in pdo_t.items() for m, v in enumerate(vals)],
                   columns=["Year", "Month", "PDO"])
aao = read_year_month_value(fetch("aao.txt"))
modes = pd.merge(pdo, aao, on=["Year", "Month"], how="outer").sort_values(["Year", "Month"])
modes = modes[(modes.Year >= 1966) & (modes.Year <= 2025)]
modes.to_csv(MODES_CSV, index=False)
print(f"Written: {MODES_CSV}  ({len(modes)} months; missing PDO {int(modes.PDO.isna().sum())}, "
      f"AAO {int(modes.AAO.isna().sum())} (AAO starts in 1979))")

enso = pd.read_csv(ENSO_CSV)[["Year", "Month", "ONI"]]
allm = pd.merge(enso, modes, on=["Year", "Month"], how="left")


def seasonal(col, months):
    d = allm[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")[col]
    ok = g.count() == len(months)
    return g.mean()[ok]


IDX = {s: {c: seasonal(c, m) for c in ["ONI", "PDO", "AAO"]} for s, m in SEASONS.items()}


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


def z(v):
    return (v - v.mean()) / v.std(ddof=1)


pk = pd.read_csv(os.path.join(OUT_DIR, "peak_latitudes_by_year_masked.csv"))
pk = pk[pk.sector == SECTOR]
MODELS = {"M0: t+ONI": ["ONI"], "M1: t+ONI+PDO": ["ONI", "PDO"], "M2: t+ONI+PDO+AAO": ["ONI", "PDO", "AAO"]}
rows = []
for (hemi, seas, var, meth), g in pk.groupby(["hemi", "season", "var", "method"]):
    s = g.set_index("year")["abs_lat"]
    for pname, (ya, yb) in PERIODS.items():
        ss = s[(s.index >= ya) & (s.index <= yb)].dropna()
        for mname, cols in MODELS.items():
            if "AAO" in cols and ya < 1979:
                continue
            common = ss.index
            for c in cols:
                common = common.intersection(IDX[seas][c].dropna().index)
            if len(common) < MIN_N:
                continue
            yrs = common.values.astype(float)
            y = ss.loc[common].values
            X = [np.ones_like(yrs), (yrs - 1995.5) / 10.0] + [z(IDX[seas][c].loc[common].values) for c in cols]
            X = np.column_stack(X)
            b, p, lo, hi = ols_adj(y, X, 1)
            corr = ""
            if len(cols) > 1:
                cm = np.corrcoef(X[:, 2:].T)
                corr = f"max|r| between regressors = {np.max(np.abs(cm - np.eye(len(cols)))):.2f}"
            rows.append(dict(hemi=hemi, season=seas, var=var, method=meth, period=pname, model=mname, n=len(y),
                             trend=b, ci_lo=lo, ci_hi=hi, p_adj=p, regressor_corr=corr))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT_DIR, "pdo_sam_adjusted_trends.csv"), index=False)

print("\n=== Annual ocean ridge trends, 1979-2025 (deg/decade, [95% CI], adjusted p) ===")
q = R[(R.season == "Annual") & (R.period == "1979-2025")]
for (hemi, var, meth), g in q.groupby(["hemi", "var", "method"]):
    cells = [f"{r.model.split(':')[0]} {r.trend:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] p={r.p_adj:.3f}" for r in g.itertuples()]
    print(f"  {hemi} {var:5s} {meth:8s} " + " | ".join(cells))
print("\nCorrelations between regressors (annual means, 1979-2025):")
a = pd.concat([IDX["Annual"][c] for c in ["ONI", "PDO", "AAO"]], axis=1, keys=["ONI", "PDO", "AAO"]).loc[1979:2025]
print(a.corr().round(2).to_string())
print("\nDone.")
