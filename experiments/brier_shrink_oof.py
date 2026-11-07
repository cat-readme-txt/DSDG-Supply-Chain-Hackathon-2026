"""Fit Brier-optimal shrinkage on archived out-of-fold forecasts.

The archive contains forecasts generated at historical origins using only data
available then. Reconstruct the calm hedge used in the scored 67.74 file, fit
shrinkage toward the historical material rate on early origins, and reserve
later origins as an untouched chronological check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "methods" / "lean_split_67_74" / "reports" / "script_dense.csv"


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y-p)**2))


def alpha_fit(frame: pd.DataFrame) -> float:
    p = frame.base.to_numpy(float)
    rate = frame.naive.to_numpy(float)
    y = frame.y.to_numpy(float)
    direction = rate-p
    denominator = float((direction**2).sum())
    return float(np.clip((direction*(y-p)).sum()/denominator, 0, 1)) if denominator else 0.


def main() -> None:
    table = pd.read_csv(SOURCE)
    table["h"] = table.week-table.origin
    table["base"] = np.where((~table.story) & (table.h > 2), .5*table.script, table.script)
    table["bin"] = pd.cut(table.h, [0, 4, 8, 13], labels=["h1_4", "h5_8", "h9_13"])
    development = table[table.origin <= 130]
    holdout = table[table.origin >= 132]
    models = {"none": {}, "global": {"all": alpha_fit(development)},
              "by_horizon": {str(k): alpha_fit(g) for k, g in development.groupby("bin", observed=True)},
              "by_story": {str(k): alpha_fit(g) for k, g in development.groupby("story")}}
    results = []
    for split, data in (("development", development), ("later_holdout", holdout)):
        for name, weights in models.items():
            if name == "global":
                alpha = np.full(len(data), weights["all"])
            elif name == "by_horizon":
                alpha = data.bin.astype(str).map(weights).to_numpy(float)
            elif name == "by_story":
                alpha = data.story.astype(str).map(weights).to_numpy(float)
            else:
                alpha = np.zeros(len(data))
            p = data.base.to_numpy(float) * (1-alpha) + data.naive.to_numpy(float)*alpha
            for subset, mask in (("all", np.ones(len(data), bool)),
                                 ("story", data.story.to_numpy(bool)),
                                 ("background", ~data.story.to_numpy(bool))):
                if mask.sum():
                    results.append(dict(split=split, model=name, subset=subset,
                                        n=int(mask.sum()), brier=brier(data.y.to_numpy(float)[mask], p[mask]),
                                        baseline=brier(data.y.to_numpy(float)[mask],
                                                       data.base.to_numpy(float)[mask])))
    output = {"alphas": models, "results": results}
    path = ROOT / "reports" / "brier_shrink_oof_w208.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    for r in results:
        if r["subset"] == "all":
            print(r["split"], r["model"], "Brier", round(r["brier"], 5),
                  "base", round(r["baseline"], 5), "n", r["n"])
    print("alphas", models)
    print(path)


if __name__ == "__main__":
    main()
