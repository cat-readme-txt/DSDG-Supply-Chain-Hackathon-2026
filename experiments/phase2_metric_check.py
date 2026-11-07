"""Fit unresolved scoring conventions to all known public leaderboard scores."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "eda"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402
all_variants = runpy.run_path(str(ROOT / "methods" / "lean_split_67_74"
                                  / "eda" / "17_metric_variants.py"))["all_variants"]

SCORES = (44.95, 45.80, 46.21, 46.22, 48.55, 50.44, 54.10,
          60.66, 66.98, 67.26, 67.74, 84.27, 89.45, 91.36, 92.31, 92.62)
KEYS = ["avgcost", "init", "F_agg", "fill_agg", "calm", "div", "p_agg"]


def main() -> None:
    data = load_data(ROOT / "data" / "Phase 2 data")
    naive = read_submission(ROOT / "data" / "Phase 2 data" / "sample_submission.csv")
    results = None
    for score in SCORES:
        sub = read_submission(ROOT / "submissions" / f"submission{score:.2f}.csv")
        table = all_variants(sub, naive, data, range(209, 222))[KEYS+['score']]
        table = table.rename(columns={"score": f"actual_{score:.2f}"})
        results = table if results is None else results.merge(table, on=KEYS)
    candidates = sorted((ROOT / "submissions" / "candidates").glob("*w221.csv"))
    for path in candidates:
        sub = read_submission(path)
        table = all_variants(sub, naive, data, range(209, 222))[KEYS+['score']]
        results = results.merge(table.rename(columns={"score": path.name}), on=KEYS)
    error = np.stack([results[f"actual_{score:.2f}"].to_numpy(float)-score for score in SCORES], axis=1)
    results["rmse"] = np.sqrt((error**2).mean(axis=1))
    # Allow a shared affine calibration of each local scorer to the official
    # totals. Relative file-to-file differences can be more reliable than
    # absolute scores when a scorer convention is still omitted.
    predicted = np.stack([results[f"actual_{score:.2f}"].to_numpy(float) for score in SCORES], axis=1)
    actual = np.array(SCORES, dtype=float)
    xbar = predicted.mean(axis=1)
    ybar = actual.mean()
    slope = ((predicted-xbar[:, None])*(actual-ybar)).sum(axis=1) / ((predicted-xbar[:, None])**2).sum(axis=1)
    intercept = ybar - slope*xbar
    results["affine_rmse"] = np.sqrt(((intercept[:, None]+slope[:, None]*predicted-actual)**2).mean(axis=1))
    results["affine_slope"] = slope
    results["affine_intercept"] = intercept
    results = results.sort_values("rmse")
    path = ROOT / "reports" / "phase2_metric_check.json"
    path.write_text(results.to_json(orient="records", indent=2) + "\n")
    for _, row in results.head(15).iterrows():
        print({key: row[key] for key in KEYS}, "rmse", round(row.rmse, 3),
              "candidates", {candidate.name: round(row[candidate.name], 2)
                             for candidate in candidates})
    for _, row in results.nsmallest(5, "affine_rmse").iterrows():
        print("affine", {key: row[key] for key in KEYS}, "rmse", round(row.affine_rmse, 3),
              "slope", round(row.affine_slope, 3))
    print(path)


if __name__ == "__main__":
    main()
