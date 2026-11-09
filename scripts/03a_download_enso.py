# -*- coding: utf-8 -*-
"""
03a_download_enso.py
Download the ENSO indices from NOAA for 1966-2025 and build enso_monthly_1966_2025.csv

  ONI   : CPC  oni.ascii.txt   (3-month mean; DJF is assigned to January)
  SOI   : CPC  soi             ('STANDARDIZED' block, not the first 'ANOMALY' block)
  MEI.v2: PSL  meiv2.data      (available only from 1979; NaN before)

If your own file (enso_monthly_1996_2025.xlsx) is next to the script, the downloaded values for 1996-2025
are compared with it to make sure the right column/block was read.
If the internet is down, download the raw files by hand and put them in the enso_raw folder (names: oni.txt, soi.txt, meiv2.txt).
"""
import os
import re
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("HADLEY_DATA_DIR", "data")   # folder with the input files (see docs/DATA.md)
RAW_DIR = os.path.join(DATA_DIR, "enso_raw")
XLSX_USER = os.path.join(DATA_DIR, "enso_monthly_1996_2025.xlsx")
OUT_CSV = os.path.join(DATA_DIR, "enso_monthly_1966_2025.csv")
Y0, Y1 = 1966, 2025

URLS = {
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
    "meiv2": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
}
os.makedirs(RAW_DIR, exist_ok=True)


def fetch(name):
    path = os.path.join(RAW_DIR, name + ".txt")
    try:
        req = urllib.request.Request(URLS[name], headers={"User-Agent": "Mozilla/5.0 (research script)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode("utf-8", "replace")
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[downloaded] {name}")
        return txt
    except Exception as e:
        if os.path.exists(path):
            print(f"[warning] could not download {name} ({e}); using the saved copy.")
            with open(path, encoding="utf-8") as f:
                return f.read()
        raise SystemExit(f"Could not download {name}: {e}\nGet the file by hand from {URLS[name]} and save it as {path}.")


SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def parse_oni(txt):
    rows = []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) >= 4 and p[0] in SEAS:
            i = SEAS.index(p[0])          # DJF -> month 1 ... NDJ -> month 12
            rows.append((int(p[1]), i + 1, float(p[3])))
    return pd.DataFrame(rows, columns=["Year", "Month", "ONI"])


def parse_year_table(txt, start_after=None, missing=(-99.9, -999.0, -999.9, -99.99, -9999.0)):
    """Tables of 'year + 12 values'. If start_after is given, only the blocks after the first line containing that phrase."""
    lines = txt.splitlines()
    if start_after:
        idx = [i for i, l in enumerate(lines) if start_after.lower() in l.lower()]
        if idx:
            lines = lines[idx[0]:]
        else:
            print(f"[warning] phrase '{start_after}' not found; the last table block is used.")
    blocks, cur = [], []
    for ln in lines:
        p = ln.split()
        ok = (len(p) >= 13 and re.fullmatch(r"(19|20)\d\d", p[0]) and
              all(re.fullmatch(r"-?\d+(\.\d+)?", x) for x in p[1:13]))
        if ok:
            cur.append((int(p[0]), [float(x) for x in p[1:13]]))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if not blocks:
        raise ValueError("No year x month table found")
    blk = blocks[0] if start_after else blocks[-1]
    out = []
    for y, vals in blk:
        for m, v in enumerate(vals, 1):
            out.append((y, m, np.nan if any(abs(v - s) < 1e-6 for s in missing) else v))
    return pd.DataFrame(out, columns=["Year", "Month", "v"])


def build():
    oni = parse_oni(fetch("oni"))
    soi = parse_year_table(fetch("soi"), start_after="STANDARDIZED").rename(columns={"v": "SOI"})
    mei = parse_year_table(fetch("meiv2")).rename(columns={"v": "MEI"})
    df = oni.merge(soi, on=["Year", "Month"], how="outer").merge(mei, on=["Year", "Month"], how="outer")
    df = df[(df.Year >= Y0) & (df.Year <= Y1)].sort_values(["Year", "Month"]).reset_index(drop=True)
    full = pd.MultiIndex.from_product([range(Y0, Y1 + 1), range(1, 13)], names=["Year", "Month"])
    df = df.set_index(["Year", "Month"]).reindex(full).reset_index()
    df["Phase"] = np.where(df.ONI >= 0.5, "El Nino", np.where(df.ONI <= -0.5, "La Nina", "Neutral"))
    df.loc[df.ONI.isna(), "Phase"] = ""
    return df


def compare_with_user(df):
    if not os.path.exists(XLSX_USER):
        print("User xlsx file not found; comparison skipped.")
        return
    u = pd.read_excel(XLSX_USER)
    m = df.merge(u, on=["Year", "Month"], suffixes=("", "_user"))
    print(f"\nComparison with your file ({len(m)} overlapping months):")
    for c in ["ONI", "SOI", "MEI"]:
        d = (m[c] - m[c + "_user"]).abs()
        print(f"  {c}: max difference = {d.max():.3f}, mean = {d.mean():.4f}, months with difference > 0.05: {(d > 0.05).sum()}")
        if d.max() > 0.05:
            print("   -> the block/column or the index version probably differs; take your file as the reference.")
    print("  (comparison only; the values in your file are not replaced so that the series stay consistent.)")
    return u


if __name__ == "__main__":
    df = build()
    compare_with_user(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWritten: {OUT_CSV}  ({len(df)} months)")
    print("Months without a value:", {c: int(df[c].isna().sum()) for c in ["ONI", "SOI", "MEI"]})
