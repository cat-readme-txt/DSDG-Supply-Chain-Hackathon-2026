"""Fit a local public-score response from controlled graded policy pairs.

For files with the same forecasts, approximate official score differences by
-alpha * change in simulated cost + beta * change in diversification. Fit only
two orthogonal-ish controls and check other controls out of sample. This is a
response model for choosing a new public allocation, not the official scorer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402

CONTROLS = ((67.74, 67.26), (89.45, 91.36))
CHECKS = ((67.74, 66.98), (50.44, 54.10), (60.66, 67.74))
SCALES = (.25, .5, 1., 1.5, 2., 2.5, 3., 4.)


def main():
    data = load_data(ROOT/"data"/"Phase 2 data")
    reference = naive(data.until(208), range(209,222))
    cache = {}
    def metrics(path):
        path = Path(path)
        if path not in cache:
            cache[path] = score_window(read_submission(path), reference, data, range(209,222))
        return cache[path]
    def pair(a,b):
        x,y = metrics(a),metrics(b)
        return dict(cost_delta=y.cost-x.cost,div_delta=y.diversification-x.diversification,
                    fill_delta=y.fill_rate-x.fill_rate)
    def scored(score):
        return ROOT/"submissions"/f"submission{score:.2f}.csv"
    X=[];y=[]
    for a,b in CONTROLS:
        features=pair(scored(a),scored(b))
        X.append([-features["cost_delta"],features["div_delta"]])
        y.append(b-a)
    alpha,beta=np.linalg.solve(np.array(X),np.array(y))
    checks=[]
    for a,b in CHECKS:
        features=pair(scored(a),scored(b))
        predicted=-alpha*features["cost_delta"]+beta*features["div_delta"]
        checks.append(dict(pair=f"{a:.2f}->{b:.2f}",actual=b-a,predicted=float(predicted),
                           error=float(predicted-(b-a)),**features))
    base=scored(91.36)
    scales=[]
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dsdg-response-") as directory:
        for scale in SCALES:
            output=Path(directory)/f"scale_{scale}.csv"
            subprocess.run([sys.executable,"-m","experiments.public_diversification_qp",
                            "--diversity-scale",str(scale),"--output",str(output)],
                           cwd=ROOT,env=env,capture_output=True,text=True,check=True)
            features=pair(base,output)
            gain=-alpha*features["cost_delta"]+beta*features["div_delta"]
            scales.append(dict(scale=scale,predicted_gain=float(gain),predicted_score=float(91.36+gain),
                               **features))
    report=dict(alpha_cost_per_dollar=float(alpha),beta_diversification=float(beta),
                controls=[f"{a:.2f}->{b:.2f}" for a,b in CONTROLS],checks=checks,scales=scales)
    path=ROOT/"reports"/"score_surface_fit.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print("alpha",round(alpha,8),"beta",round(beta,3))
    print("checks",[(x["pair"],round(x["error"],3)) for x in checks])
    print("scales",[(x["scale"],round(x["predicted_score"],3)) for x in scales])
    print(path)


if __name__ == "__main__":
    main()
