# -*- coding: utf-8 -*-
"""
06_article_figures.py
Draws all paper figures directly from the data / output tables (no old PNG is copied)
and saves them in articleoutpt. All figure text is in English. dpi=300.

  fig1_ridge_maps.png               omega and wind maps with ridge dots   <- climatology.nc + static_fields.nc  (Basemap)
  fig2_enso_adjusted_trends.png     raw / adjusted trends                 <- enso_adjusted_trends.csv
  fig3_subperiod_breakpoints.png    cloud + broken stick                  <- peak_latitudes_by_year_masked.csv, era5_breakpoint_scan.csv, ENSO
  fig4_ceres_vs_era5_centroid.png   CERES versus ERA5                     <- ceres_era5_ridge_by_year.csv, ceres_vs_era5_cloud_ridge.csv
  fig5_calm_belt_1979.png           calm-belt intensity and 1979 step     <- calm_belt_min_speed_by_year.csv, ENSO
  fig6_clara_three_way.png          ERA5/CLARA-A3/CERES                   <- clara_ridge_by_year.csv, clara_vs_era5_cloud_ridge.csv
  fig7_basin_trends.png             trends by ocean basin                 <- basin_trends.csv, main_trends_full_table.csv (from 08)

The inputs must already have been produced by 01..05, 07 and 08. A figure whose input is missing is skipped and reported at the end.
Run:  python 06_article_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ART_DIR = os.path.join(DATA_DIR, "articleoutpt")
ENSO_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
DPI = 300
MAP_H_MAX = 2500.0
WINDOWS = {"NH": (15.0, 45.0), "SH": (-45.0, -15.0)}
os.makedirs(ART_DIR, exist_ok=True)
missing, done = [], []
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
HN = {"NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}


def P(f):
    return os.path.join(OUT_DIR, f)


def have(*files):
    ok = True
    for f in files:
        if not os.path.exists(f if os.path.isabs(f) else P(f)):
            missing.append(f)
            ok = False
    return ok


def save(fig, name):
    fig.savefig(os.path.join(ART_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    done.append(name)
    print(name, "OK")


def enso_season(months, col="ONI"):
    e = pd.read_csv(ENSO_CSV)
    d = e[["Year", "Month", col]].copy()
    d["Y"] = d.Year + ((d.Month == 12) if months == [12, 1, 2] else 0)
    d = d[d.Month.isin(months)]
    g = d.groupby("Y")[col]
    return g.mean()[g.count() == len(months)]


SEAS = {"Annual": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8]}


def nan_smooth(f, size, axis=0, mode="nearest"):
    m = np.isfinite(f).astype(float)
    n_ = uniform_filter1d(np.where(m > 0, f, 0.0), size, axis=axis, mode=mode)
    d_ = uniform_filter1d(m, size, axis=axis, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = n_ / d_
    out[d_ < 0.5] = np.nan
    return out


# ================================================================
# Figure 1: maps (from the climatology data)
# ================================================================
def fig1():
    if not have("climatology.nc", "static_fields.nc"):
        return
    try:
        import xarray as xr
        from mpl_toolkits.basemap import Basemap
    except Exception as e:
        missing.append(f"fig1: xarray/Basemap not available ({e})")
        return
    cl = xr.open_dataset(P("climatology.nc"))
    clat, clon = cl["latitude"].values, cl["longitude"].values
    st = xr.open_dataset(P("static_fields.nc")).reindex(latitude=clat, longitude=clon,
                                                        method="nearest", tolerance=0.05)
    ocean2 = st["lsm"].values < 0.1
    mountain = st["height_m"].values > MAP_H_MAX

    def ridge_per_lon(field, window, kind="max"):
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
        r[allnan | (i == 0) | (i == n - 1) | ~np.isfinite(up) | ~np.isfinite(dn)] = np.nan
        return r

    def draw(ax, fig, field, title, cmap, vmin, vmax, ridges, label, color):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.82")
        m = Basemap(projection="cyl", llcrnrlat=-80, urcrnrlat=80, llcrnrlon=0, urcrnrlon=360,
                    resolution="c", ax=ax)
        m.drawcoastlines(linewidth=0.5)
        m.drawparallels(range(-60, 61, 30), labels=[1, 0, 0, 0], fontsize=7, linewidth=0.2)
        m.drawmeridians(range(0, 361, 60), labels=[0, 0, 0, 1], fontsize=7, linewidth=0.2)
        X, Y = np.meshgrid(clon[::2], clat[::2])
        pc = m.pcolormesh(X, Y, np.ma.masked_invalid(field[::2, ::2]), latlon=True, cmap=cm,
                          vmin=vmin, vmax=vmax, shading="auto")
        for r in ridges:
            m.scatter(clon, r, s=2, c=color, latlon=True, zorder=5)
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
        cb.set_label(label, fontsize=8)

    w = np.where(mountain, np.nan, cl["w500_all"].values)
    ws = cl["wspd_all"].values
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    draw(axes[0], fig, w, f"(a) Mean 500-hPa omega, 1966-2025\nblack dots: ocean-only latitude of maximum omega; "
         f"grey: terrain > {MAP_H_MAX:.0f} m", "RdBu_r", -0.08, 0.08,
         [ridge_per_lon(w, WINDOWS[h]) for h in WINDOWS], "Pa/s (positive = subsidence)", "k")
    draw(axes[1], fig, ws, "(b) Mean 10 m wind speed, 1966-2025\nwhite dots: ocean-only latitude of minimum wind speed",
         "viridis", 0, 12, [ridge_per_lon(ws, WINDOWS[h], "min") for h in WINDOWS], "m/s", "w")
    fig.tight_layout()
    save(fig, "fig1_ridge_maps.png")


# ================================================================
# Figure 2: raw and adjusted trends
# ================================================================
def fig2():
    if not have("enso_adjusted_trends.csv"):
        return
    res = pd.read_csv(P("enso_adjusted_trends.csv"))
    LAB = {"w500": "Max omega", "tcc": "Min cloud cover", "wspd": "Min wind speed (lat)"}
    items = [("tcc", "argmax"), ("w500", "argmax"), ("w500", "centroid"), ("wspd", "argmax")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r_i, hemi in enumerate(["NH", "SH"]):
        for c_i, seas in enumerate(["Annual", "DJF", "JJA"]):
            ax = axes[r_i, c_i]
            yy, yt, yl = 0, [], []
            for var, meth in items:
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
                yl.append(f"{LAB[var]}\n[{meth}]")
                yy += 1
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_yticks(yt)
            ax.set_yticklabels(yl, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{HN[hemi]} - {seas}")
            ax.set_xlabel("Poleward shift (deg/decade)")
            ax.grid(alpha=0.3)
    a = axes[0, 0]
    a.plot([], [], "o", mfc="none", color="gray", label="raw trend")
    a.plot([], [], "s", color="gray", label="ENSO (ONI) adjusted")
    a.plot([], [], "*", color="k", label="adj. p < 0.05")
    a.plot([], [], "s", color="tab:blue", label="1966-2025")
    a.plot([], [], "s", color="tab:orange", label="1979-2025")
    a.legend(fontsize=7)
    fig.suptitle("Ridge-latitude trends before and after removing ENSO (ocean only)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig2_enso_adjusted_trends.png")


# ================================================================
# Figure 3: ERA5 cloud with broken stick
# ================================================================
def fig3():
    if not have("peak_latitudes_by_year_masked.csv", "era5_breakpoint_scan.csv", ENSO_CSV):
        return
    pk = pd.read_csv(P("peak_latitudes_by_year_masked.csv"))
    brk = pd.read_csv(P("era5_breakpoint_scan.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, hemi in enumerate(["NH", "SH"]):
        for j, seas in enumerate(["Annual", "DJF", "JJA"]):
            ax = axes[i, j]
            g = pk[(pk.sector == "ocean") & (pk.hemi == hemi) & (pk.season == seas) &
                   (pk["var"] == "tcc") & (pk.method == "centroid")]
            b = brk[(brk.sector == "ocean") & (brk.hemi == hemi) & (brk.season == seas) &
                    (brk["var"] == "tcc") & (brk.method == "centroid")]
            if g.empty or b.empty:
                continue
            b = b.iloc[0]
            s = g.set_index("year")["abs_lat"].dropna()
            e_all = enso_season(SEAS[seas])
            common = s.index.intersection(e_all.index)
            yrs = common.values.astype(float)
            y = s.loc[common].values
            e = e_all.loc[common].values
            ez = (e - e.mean()) / e.std(ddof=1)
            t = (yrs - 1995.5) / 10.0
            X = np.column_stack([np.ones_like(t), t, np.maximum(0, yrs - b.t0) / 10.0, ez])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            ax.plot(yrs, y - beta[3] * ez, "-o", ms=3, lw=1, color="tab:gray",
                    label="ERA5 cloud-minimum centroid (ENSO removed)")
            ax.plot(yrs, X[:, :3] @ beta[:3], "r-", lw=1.8, label="broken-stick fit")
            ax.axvline(b.t0, color="r", ls=":", lw=1)
            ax.axvline(1979, color="tab:blue", ls=":", lw=0.9)
            ax.axvline(2000, color="tab:green", ls=":", lw=0.9)
            ax.set_title(f"{HN[hemi]} - {seas}\nbreak {int(b.t0)}: {b.slope_before:+.2f} -> "
                         f"{b.slope_after:+.2f} deg/dec (MC p = {b.p_mc:.2f})", fontsize=9)
            ax.set_xlabel("Year")
            ax.set_ylabel("|Latitude| (deg)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("ERA5 ocean cloud-minimum latitude: where does the poleward trend come from?\n"
                 "(blue dotted: 1979; green dotted: 2000, start of CERES)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_subperiod_breakpoints.png")


# ================================================================
# Figure 4: CERES versus ERA5
# ================================================================
def fig4():
    if not have("ceres_era5_ridge_by_year.csv", "ceres_vs_era5_cloud_ridge.csv"):
        return
    y = pd.read_csv(P("ceres_era5_ridge_by_year.csv"))
    s = pd.read_csv(P("ceres_vs_era5_cloud_ridge.csv"))
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for i, h in enumerate(["NH", "SH"]):
        for j, se in enumerate(["Annual", "DJF", "JJA"]):
            a = ax[i, j]
            for src, c in [("CERES", "tab:blue"), ("ERA5", "tab:orange")]:
                g = y[(y.hemi == h) & (y.season == se) & (y.method == "centroid") & (y.source == src)].sort_values("year")
                a.plot(g.year, g.abs_lat, "-o", ms=3, lw=1.1, color=c, label="CERES EBAF" if src == "CERES" else src)
                sl = stats.linregress(g.year, g.abs_lat)
                a.plot(g.year, sl.intercept + sl.slope * g.year, "--", color=c, lw=1)
            r = s[(s.hemi == h) & (s.season == se) & (s.method == "centroid")].iloc[0]
            a.set_title(f"{HN[h]} - {se}\nr = {r.pearson_r:.2f} (detrended {r.r_detrended:.2f}); "
                        f"trend CERES {r.trend_ceres:+.2f}, ERA5 {r.trend_era5:+.2f} deg/dec", fontsize=8.5)
            a.set_xlabel("Year")
            a.set_ylabel("|Latitude| of cloud-minimum centroid (deg)", fontsize=8)
            a.grid(alpha=0.3)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_ceres_vs_era5_centroid.png")


# ================================================================
# Figure 5: calm belt and the 1979 step
# ================================================================
def fig5():
    if not have("calm_belt_min_speed_by_year.csv", ENSO_CSV):
        return
    cb = pd.read_csv(P("calm_belt_min_speed_by_year.csv"))
    e_all = enso_season(SEAS["Annual"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, hemi in zip(axes, ["NH", "SH"]):
        s = cb[(cb.sector == "ocean") & (cb.hemi == hemi) & (cb.season == "Annual")].set_index("year")["min_wspd"]
        e = e_all.reindex(s.index)
        ok = s.index[e.notna()]
        yrs = ok.values.astype(float)
        y = s.loc[ok].values
        ez = ((e.loc[ok] - e.loc[ok].mean()) / e.loc[ok].std(ddof=1)).values
        X = np.column_stack([np.ones_like(yrs), (yrs - 1995.5) / 10.0, ez])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_adj = y - beta[2] * ez
        ax.plot(yrs, y, color="tab:green", lw=1.2, label="calm-belt minimum speed")
        ax.plot(yrs, y_adj, color="k", lw=1.2, label="ENSO (ONI) removed")
        for ya, yb, col in [(1966, 2025, "tab:blue"), (1979, 2025, "tab:orange")]:
            m = (yrs >= ya) & (yrs <= yb)
            sl = stats.linregress(yrs[m], y_adj[m])
            ax.plot(yrs[m], sl.intercept + sl.slope * yrs[m], "--", color=col,
                    label=f"{ya}-{yb}: {sl.slope * 10:+.3f} m/s/decade")
        ax.axvline(1979, color="red", lw=1, ls=":")
        ax.text(1979.5, ax.get_ylim()[0], " TOVS (late 1978)", color="red", va="bottom", fontsize=8)
        ax.set_title(f"{HN[hemi]} - Annual")
        ax.set_xlabel("Year")
        ax.set_ylabel("Minimum zonal-mean wind speed, ocean (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Calm-belt intensity: effect of ENSO and of the 1979 observing-system change", fontsize=12)
    fig.tight_layout()
    save(fig, "fig5_calm_belt_1979.png")


# ================================================================
# Figure 6: ERA5 / CLARA-A3 / CERES
# ================================================================
def fig6():
    if not have("clara_ridge_by_year.csv", "clara_vs_era5_cloud_ridge.csv"):
        return
    yy = pd.read_csv(P("clara_ridge_by_year.csv"))
    rs = pd.read_csv(P("clara_vs_era5_cloud_ridge.csv"))
    cer = None
    if os.path.exists(P("ceres_era5_ridge_by_year.csv")):
        cer = pd.read_csv(P("ceres_era5_ridge_by_year.csv"))
        cer = cer[cer.source == "CERES"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for i, h in enumerate(["NH", "SH"]):
        for j, se in enumerate(["Annual", "DJF", "JJA"]):
            a = ax[i, j]
            for src, c, lab in [("ERA5", "tab:orange", "ERA5"), ("CLARA", "tab:green", "CLARA-A3")]:
                g = yy[(yy.hemi == h) & (yy.season == se) & (yy.method == "centroid") & (yy.source == src)].sort_values("year")
                a.plot(g.year, g.abs_lat, "-o", ms=2.5, lw=1.1, color=c, label=lab)
            if cer is not None:
                g = cer[(cer.hemi == h) & (cer.season == se) & (cer.method == "centroid")].sort_values("year")
                a.plot(g.year, g.abs_lat, "-o", ms=2.5, lw=1.1, color="tab:blue", label="CERES EBAF")
            a.axvspan(1979, 1999.5, color="0.92", zorder=0)
            q = rs[(rs.hemi == h) & (rs.season == se) & (rs.method == "centroid")]
            t = []
            for per in ["1979-1999", "2000-end"]:
                r = q[q.period == per]
                if len(r):
                    r = r.iloc[0]
                    t.append(f"{per}: CLARA {r.trend_clara:+.2f}, ERA5 {r.trend_era5:+.2f}")
            a.set_title(f"{HN[h]} - {se}\n" + "; ".join(t) + " deg/dec", fontsize=8)
            a.set_xlabel("Year")
            a.set_ylabel("|Latitude| of cloud-minimum centroid (deg)", fontsize=8)
            a.grid(alpha=0.3)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    fig.suptitle("Ocean cloud-minimum latitude: ERA5, CLARA-A3 and CERES EBAF (grey: 1979-1999)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig6_clara_three_way.png")


# ================================================================
# Figure 7: basin-by-basin trends (centroid, ENSO adjusted, 1979-2025)
# ================================================================
def fig7():
    if not have("basin_trends.csv", "main_trends_full_table.csv"):
        return
    B = pd.read_csv(P("basin_trends.csv"))
    A = pd.read_csv(P("main_trends_full_table.csv"))
    A = A[A.season == "Annual"]
    BASINS = ["atlantic", "pacific_w", "pacific_c", "pacific_e", "indian"]
    LAB = {"tcc": "Min cloud cover", "w500": "Max omega (500 hPa)", "wspd": "Min wind speed"}
    NICE = {"atlantic": "Atlantic", "pacific_w": "W Pacific", "pacific_c": "C Pacific", "pacific_e": "E Pacific",
            "indian": "Indian", "ocean": "All oceans (zonal mean)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, hemi in zip(axes, ["NH", "SH"]):
        yy, yt, yl = 0, [], []
        for var in ["tcc", "w500", "wspd"]:
            for sec in BASINS + ["ocean"]:
                src = B if sec != "ocean" else A
                q = src[(src.sector == sec) & (src.hemi == hemi) & (src["var"] == var) &
                        (src.method == "centroid") & (src.period == "1979-2025")]
                if q.empty:
                    yy += 1
                    continue
                q = q.iloc[0]
                col = "k" if sec == "ocean" else ("tab:orange" if q.edge_fraction > 0.3 else "tab:blue")
                ax.errorbar(q.ols_adj, yy, xerr=[[q.ols_adj - q.ols_adj_lo], [q.ols_adj_hi - q.ols_adj]],
                            fmt="s" if sec == "ocean" else "o", color=col, capsize=2, ms=5)
                yt.append(yy)
                yl.append(f"{LAB[var]} - {NICE[sec]}")
                yy += 1
            yy += 1
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-0.9, 0.9)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Poleward shift (deg/decade), centroid, ENSO removed")
        ax.set_title(f"{HN[hemi]}, 1979-2025 (95% CI)")
        ax.grid(alpha=0.3)
    fig.suptitle("Ridge-latitude trends by ocean basin (orange: peak at window edge in >30% of years)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig7_basin_trends.png")


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
    try:
        f()
    except Exception as ex:
        missing.append(f"{f.__name__}: error - {type(ex).__name__}: {ex}")

print(f"\nFigures produced ({len(done)}): {', '.join(done)}\nFolder: {ART_DIR}")
if missing:
    print("\nIncomplete items:")
    for m in missing:
        print("  -", m)
