# -*- coding: utf-8 -*-
"""
14_global_hadley_map_figure.py
Global figure for the paper: mean ocean sea-level pressure (1966-2025, annual) with the mean latitudes of the
Hadley-cell edge (zero of the 500 hPa mass streamfunction) and of the ridge diagnostics (SLP, omega 500, cloud, wind),
the mean centres of the five subtropical highs, and the zonal-mean psi(500 hPa) profile on the same latitude axis.
Also copies the two figures of script 13 into the article folder.

Inputs (DATA_DIR/outputs): slp_ocean_1deg.npz (12), psi_monthly_1966_2025.npz (13), hadley_vs_ridge_mean_latitude.csv (13),
                           slp_high_centres_by_year.csv (12)
Outputs: DATA_DIR/outputs/fig_global_hadley_ridge_map.png and the same file in DATA_DIR/articleoutpt

Run:  python 14_global_hadley_map_figure.py
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
OUT_DIR = os.path.join(DATA_DIR, "outputs")
ART_DIR = os.path.join(DATA_DIR, "articleoutpt")
os.makedirs(ART_DIR, exist_ok=True)
LAT_LIM = 65.0

for f in ("slp_ocean_1deg.npz", "psi_monthly_1966_2025.npz", "hadley_vs_ridge_mean_latitude.csv",
          "slp_high_centres_by_year.csv"):
    if not os.path.exists(os.path.join(OUT_DIR, f)):
        raise SystemExit(f"Missing input: {os.path.join(OUT_DIR, f)} (run scripts 12 and 13 first)")

sc = np.load(os.path.join(OUT_DIR, "slp_ocean_1deg.npz"))
latb, lonb = np.asarray(sc["lat"], float), np.asarray(sc["lon"], float)
slp = np.nanmean(np.asarray(sc["slp"], float), axis=0) / 100.0           # hPa, land = NaN

pc = np.load(os.path.join(OUT_DIR, "psi_monthly_1966_2025.npz"))
psi, plat, plev = np.asarray(pc["psi"], float), np.asarray(pc["lat"], float), np.asarray(pc["lev"], float)
psi500 = np.nanmean(psi[:, int(np.where(np.isclose(plev, 500))[0][0]), :], axis=0)

M = pd.read_csv(os.path.join(OUT_DIR, "hadley_vs_ridge_mean_latitude.csv"))
HC = pd.read_csv(os.path.join(OUT_DIR, "slp_high_centres_by_year.csv"))
HC = HC[HC.season == "Annual"]

import matplotlib.patheffects as pe

STYLE = [("Hadley edge (psi 500)", "Hadley-cell edge ($\\psi_{500}=0$)", "k", "--", 2.6),
         ("SLP centroid", "SLP ridge", "#4b2c83", "-", 1.8),
         ("omega 500 centroid", "$\\omega_{500}$ ridge", "#0b6623", "-", 1.8),
         ("cloud centroid", "Cloud-minimum ridge", "#7f3b08", ":", 2.2),
         ("wind centroid", "Wind-minimum ridge", "#c51b7d", "-.", 1.8)]
HALO = [pe.withStroke(linewidth=4.0, foreground="white")]


def mean_lat(key, hemi):
    r = M[(M.hemi == hemi) & (M.indicator == key)]
    return None if r.empty else float(r.mean_abs_lat.iloc[0])


fig = plt.figure(figsize=(13.5, 6.8))
gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1], height_ratios=[20, 1.1], wspace=0.05, hspace=0.28)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
cax = fig.add_subplot(gs[1, 0])

# high pressure = cold (blue), low pressure = warm (red)
im = ax.pcolormesh(lonb, latb, slp, cmap="RdYlBu", shading="nearest", vmin=1008, vmax=1024)
ax.set_facecolor("#cfcfcf")
for key, lab, col, ls, lw in STYLE:
    for hemi, sgn in (("NH", 1), ("SH", -1)):
        v = mean_lat(key, hemi)
        if v is None:
            continue
        for a_, w_ in ((ax, lw), (ax2, lw * 0.8)):
            a_.axhline(sgn * v, color=col, ls=ls, lw=w_, path_effects=HALO,
                       label=lab if (hemi == "NH" and a_ is ax) else None)
mk = {"N Pacific high": "N Pacific", "N Atlantic high": "N Atlantic", "S Pacific high": "S Pacific",
      "S Atlantic high": "S Atlantic", "S Indian high": "S Indian"}
for h, g in HC.groupby("high"):
    lon_c = np.rad2deg(np.arctan2(np.sin(np.deg2rad(g.lon)).mean(), np.cos(np.deg2rad(g.lon)).mean())) % 360.0
    la = g.lat.mean()
    ax.plot(lon_c, la, marker="*", ms=16, mfc="gold", mec="k", mew=0.9, zorder=6)
    ha = "right" if lon_c > 330 else ("left" if lon_c < 30 else "center")
    ax.text(lon_c, la + (4.5 if la > 0 else -6.0), mk.get(h, h), ha=ha, fontsize=8, weight="bold", zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
ax.axhline(0, color="k", lw=0.5, alpha=0.5)
ax.set_xlim(0, 360)
ax.set_ylim(-LAT_LIM, LAT_LIM)
ax.set_xticks(np.arange(0, 361, 60))
ax.set_xticklabels(["0", "60E", "120E", "180", "120W", "60W", "0"])
yt = np.arange(-60, 61, 15)
ax.set_yticks(yt)
ax.set_yticklabels([f"{abs(v)}{'N' if v > 0 else ('S' if v < 0 else '')}" for v in yt])
ax.set_ylabel("Latitude")
ax.set_title("Mean ocean sea-level pressure, 1966-2025 (grey = land),\nsubtropical-high centres (stars) and mean latitudes of the ridge diagnostics", fontsize=10)
ax.legend(loc="lower left", fontsize=8, framealpha=0.92, ncol=2)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
cb.set_ticks(np.arange(1008, 1025, 2))
cb.set_label("Sea-level pressure (hPa); high pressure = blue")

ax2.plot(psi500, plat, color="k", lw=1.8, zorder=5)
ax2.axvline(0, color="grey", lw=0.8)
ax2.set_yticks(yt)
ax2.set_yticklabels([f"{abs(v)}{'N' if v > 0 else ('S' if v < 0 else '')}" for v in yt])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("Latitude (same axis as the map)")
ax2.grid(axis="y", color="#bbbbbb", lw=0.5)
ax2.set_xlabel(r"$\psi_{500}$ ($10^{9}$ kg s$^{-1}$)")
ax2.set_title("Zonal-mean mass\nstreamfunction at 500 hPa", fontsize=9)
fig.savefig(os.path.join(OUT_DIR, "fig_global_hadley_ridge_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

for f in ("fig_global_hadley_ridge_map.png", "fig_psi_climatology.png", "fig_hadley_edge.png"):
    src = os.path.join(OUT_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(ART_DIR, f))
        print("copied to articleoutpt:", f)
print("Done:", os.path.join(OUT_DIR, "fig_global_hadley_ridge_map.png"))
