"""Compare public diversification weights against graded-score-fitted metrics."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "eda"))
from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402

all_variants = runpy.run_path(str(ROOT / "methods" / "lean_split_67_74" / "eda"
                                  / "17_metric_variants.py"))["all_variants"]
KEYS = ["avgcost", "init", "F_agg", "fill_agg", "calm", "div", "p_agg"]
SCALES = (.25, .5, 1., 1.5, 2., 3.)


def main() -> None:
    data = load_data(ROOT / "data" / "Phase 2 data")
    naive = read_submission(ROOT / "data" / "Phase 2 data" / "sample_submission.csv")
    base = read_submission(ROOT / "submissions" / "submission89.45.csv")
    known = all_variants(base, naive, data, range(209, 222))[KEYS+['score']]
    fit = pd.read_json(ROOT / "reports" / "phase2_metric_check.json")[
        KEYS+['rmse', 'affine_rmse', 'affine_slope']]
    top = fit.nsmallest(20, "rmse")
    affine_top = fit.nsmallest(20, "affine_rmse")
    rows = []
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dsdg-qp-sweep-") as directory:
        for scale in SCALES:
            output = Path(directory) / f"scale_{scale}.csv"
            cmd = [sys.executable, "-m", "experiments.public_diversification_qp",
                   "--diversity-scale", str(scale), "--output", str(output)]
            subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            candidate = read_submission(output)
            table = all_variants(candidate, naive, data, range(209, 222))[KEYS+['score']]
            comparison = known.merge(table, on=KEYS, suffixes=("_base", "_candidate"))
            comparison = comparison.merge(top, on=KEYS)
            delta = comparison.score_candidate - comparison.score_base
            calibrated = known.merge(table, on=KEYS, suffixes=("_base", "_candidate"))
            calibrated = calibrated.merge(affine_top, on=KEYS)
            affine_delta = (calibrated.score_candidate-calibrated.score_base)*calibrated.affine_slope
            rows.append(dict(scale=scale, mean=float(delta.mean()), median=float(delta.median()),
                             minimum=float(delta.min()), maximum=float(delta.max()),
                             positive=int((delta > 0).sum()), variants=len(delta),
                             affine_mean=float(affine_delta.mean()),
                             affine_min=float(affine_delta.min())))
    report = ROOT / "reports" / "public_qp_scale_sweep.json"
    report.write_text(json.dumps(rows, indent=2)+"\n")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
