# -*- coding: utf-8 -*-
"""
run_all.py
Runs the whole analysis chain in the correct order and stops at the first failure.
Steps whose input files are missing are skipped with a warning if they are optional (CERES, CLARA-A3, internet
downloads) and stop the run if they are essential.

Usage (from the repository root):
    python scripts/run_all.py --data-dir /path/to/data                # everything
    python scripts/run_all.py --data-dir /path/to/data --only 12 13   # selected steps (use the step numbers)
    python scripts/run_all.py --data-dir /path/to/data --from 08      # from a step to the end
    python scripts/run_all.py --list                                  # show the steps and their inputs

The data folder is passed to every script through the environment variable HADLEY_DATA_DIR.
All results are written to <data-dir>/outputs and the paper figures to <data-dir>/articleoutpt.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (step, script, essential?, required files relative to the data folder, short description)
STEPS = [
    ("00", "00_make_static_fields.py", True, [], "land-sea mask and surface height (static_fields.nc)"),
    ("03a", "03a_download_enso.py", False, [], "download ENSO indices (needs internet)"),
    ("01", "01_extract_zonal_means.py", True,
     ["W500.nc", "data_stream-moda_stepType-avgua.nc", "data_stream-moda_stepType-avgad.nc", "windspeed.nc",
      "outputs/static_fields.nc"], "zonal and basin means of omega500, cloud, wind, radiation"),
    ("02", "02_hadley_latitude_analysis.py", True, ["outputs/zonal_means_masked.nc"],
     "ridge latitudes (argmax, centroid) and trends of omega500, cloud and wind"),
    ("03", "03_enso_adjusted_trends.py", True, ["outputs/peak_latitudes_by_year_masked.csv"],
     "ENSO-adjusted trends and the 1979 step"),
    ("04", "04_ceres_cloud_check.py", False, ["CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202608.nc"],
     "CERES EBAF comparison"),
    ("05", "05_subperiod_breakpoints.py", True, ["outputs/peak_latitudes_by_year_masked.csv"],
     "sub-period trends and break-point scan"),
    ("07", "07_clara_cloud_check.py", False, ["CLARA-A3"], "CLARA-A3 comparison"),
    ("08", "08_basin_trends_and_full_stats.py", True, ["outputs/peak_latitudes_by_year_masked.csv"],
     "full statistics table (CI, Sen, MK, FDR) and basin trends"),
    ("09", "09_smoothing_sensitivity.py", True, ["outputs/zonal_means_masked.nc"], "smoothing-window sensitivity"),
    ("10", "10_pdo_sam_check.py", False, ["outputs/peak_latitudes_by_year_masked.csv"],
     "PDO and AAO sensitivity (needs internet)"),
    ("12", "12_levels_and_slp_ridge.py", True, ["w300-w700.nc", "P-SLP.nc", "outputs/static_fields.nc"],
     "omega 700/300 hPa, sea-level-pressure ridge, individual subtropical highs"),
    ("13", "13_mass_streamfunction_hadley_edge.py", True, ["V(Levels).nc", "P-SLP.nc"],
     "mass streamfunction and Hadley-cell edge"),
    ("06", "06_article_figures.py", True, ["outputs/peak_latitudes_by_year_masked.csv"], "paper figures 1-7"),
    ("14", "14_global_hadley_map_figure.py", True, ["outputs/psi_monthly_1966_2025.npz"],
     "global map figure with the Hadley-cell edge"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("HADLEY_DATA_DIR", "data"))
    ap.add_argument("--only", nargs="*", help="run only these steps, e.g. --only 12 13")
    ap.add_argument("--from", dest="start", help="start at this step")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for st, sc, ess, req, d in STEPS:
            print(f"{st:>3}  {sc:<40} {'essential' if ess else 'optional ':<9}  {d}")
            for r in req:
                print(f"       needs {r}")
        return
    data = os.path.abspath(a.data_dir)
    if not os.path.isdir(data):
        sys.exit(f"Data folder not found: {data}")
    env = dict(os.environ, HADLEY_DATA_DIR=data, PYTHONIOENCODING="utf-8")
    names = [s[0] for s in STEPS]
    run = STEPS
    if a.only:
        bad = [x for x in a.only if x not in names]
        if bad:
            sys.exit(f"Unknown steps {bad}; valid: {names}")
        run = [s for s in STEPS if s[0] in a.only]
    elif a.start:
        if a.start not in names:
            sys.exit(f"Unknown step {a.start}; valid: {names}")
        run = STEPS[names.index(a.start):]
    t_all = time.time()
    for st, sc, ess, req, d in run:
        missing = [r for r in req if not os.path.exists(os.path.join(data, r))]
        # outputs of earlier steps are produced during this run, so only check them when the step is run alone
        missing = [m for m in missing if not (m.startswith("outputs/") and not a.only and not a.start)]
        if missing:
            msg = f"step {st} ({d}): missing {missing}"
            if ess:
                sys.exit("STOP: " + msg)
            print("SKIP (optional): " + msg)
            continue
        print("\n" + "=" * 90 + f"\nstep {st}: {sc}  -  {d}\n" + "=" * 90, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, sc)], env=env)
        if r.returncode != 0:
            if ess:
                sys.exit(f"STOP: step {st} failed (exit code {r.returncode}).")
            print(f"WARNING: optional step {st} failed (exit code {r.returncode}); continuing.")
        print(f"step {st} finished in {time.time() - t0:.0f} s", flush=True)
    print(f"\nAll requested steps finished in {(time.time() - t_all) / 60:.1f} min. Results: {os.path.join(data, 'outputs')}")


if __name__ == "__main__":
    main()
