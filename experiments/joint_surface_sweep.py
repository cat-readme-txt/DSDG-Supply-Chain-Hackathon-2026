"""Select a joint public order/share plan using controlled Kaggle responses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"methods"/"lean_split_67_74"/"src"))
from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402

PAIRS=((67.74,67.26),(89.45,91.36),(91.36,92.31),(92.31,92.62))
SCALES=(.8,1.,1.2,1.4,1.5,1.6,1.8,2.,2.2,2.5,3.)


def main():
    data=load_data(ROOT/"data"/"Phase 2 data")
    reference=naive(data.until(208),range(209,222))
    cache={}
    def metrics(path):
        p=Path(path)
        if p not in cache:
            cache[p]=score_window(read_submission(p),reference,data,range(209,222))
        return cache[p]
    def scored(score):
        return ROOT/"submissions"/f"submission{score:.2f}.csv"
    design=[];actual=[]
    for start,end in PAIRS:
        a,b=metrics(scored(start)),metrics(scored(end))
        design.append([-(b.cost-a.cost),b.diversification-a.diversification])
        actual.append(end-start)
    X=np.array(design);y=np.array(actual)
    models={"least_squares_4":np.linalg.lstsq(X,y,rcond=None)[0]}
    for idx in combinations(range(len(PAIRS)),3):
        models["leave_one_out_"+"_".join(map(str,idx))] = np.linalg.lstsq(X[list(idx)],y[list(idx)],rcond=None)[0]
    baseline=metrics(scored(92.62))
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    rows=[]
    with tempfile.TemporaryDirectory(prefix="dsdg-joint-sweep-") as directory:
        for scale in SCALES:
            path=Path(directory)/f"joint_{scale}.csv"
            subprocess.run([sys.executable,"-m","experiments.public_joint_order_search",
                            "--base",str(scored(92.62)),"--scale",str(scale),"--output",str(path)],
                           cwd=ROOT,env=env,capture_output=True,text=True,check=True)
            s=metrics(path)
            cost=s.cost-baseline.cost;div=s.diversification-baseline.diversification
            predictions={name:float(beta@[ -cost,div]) for name,beta in models.items()}
            rows.append(dict(scale=scale,cost_delta=float(cost),div_delta=float(div),
                             fill_delta=float(s.fill_rate-baseline.fill_rate),
                             local_delta=float(s.score-baseline.score),
                             predicted_gain=predictions,
                             robust_min=min(predictions.values())))
    report=dict(pairs=[f"{a:.2f}->{b:.2f}" for a,b in PAIRS],
                models={name:beta.tolist() for name,beta in models.items()},
                pair_residuals=(X@models["least_squares_4"]-y).tolist(),
                candidates=rows)
    path=ROOT/"reports"/"joint_surface_sweep.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    for row in rows:
        print(row["scale"],"cost",round(row["cost_delta"]),"div",round(row["div_delta"],4),
              "fitted",round(row["predicted_gain"]["least_squares_4"],3),
              "robust_min",round(row["robust_min"],3))
    print(path)


if __name__=="__main__":
    main()
