"""Test whether the joint public optimizer is trapped in a local solution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"methods"/"lean_split_67_74"/"src"))
from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402

STARTS=("scored92.31","scored92.62","fixed1.0","fixed2.0","fixed3.0")
TARGET_SCALES=(1.5,1.8,2.2)


def main():
    data=load_data(ROOT/"data"/"Phase 2 data")
    reference=naive(data.until(208),range(209,222))
    scored=ROOT/"submissions"/"submission92.62.csv"
    baseline=score_window(read_submission(scored),reference,data,range(209,222))
    response=json.loads((ROOT/"reports"/"joint_surface_sweep.json").read_text())
    models=response["models"]
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    rows=[]
    with tempfile.TemporaryDirectory(prefix="dsdg-multistart-") as directory:
        folder=Path(directory)
        sources={"scored92.31":ROOT/"submissions"/"submission92.31.csv",
                 "scored92.62":scored}
        for start in (1.,2.,3.):
            path=folder/f"fixed_{start}.csv"
            subprocess.run([sys.executable,"-m","experiments.public_diversification_qp",
                            "--diversity-scale",str(start),"--output",str(path)],
                           cwd=ROOT,env=env,capture_output=True,text=True,check=True)
            sources[f"fixed{start}"]=path
        for name in STARTS:
            for scale in TARGET_SCALES:
                path=folder/f"joint_{name}_{scale}.csv"
                subprocess.run([sys.executable,"-m","experiments.public_joint_order_search",
                                "--base",str(sources[name]),"--scale",str(scale),"--output",str(path)],
                               cwd=ROOT,env=env,capture_output=True,text=True,check=True)
                s=score_window(read_submission(path),reference,data,range(209,222))
                cost=s.cost-baseline.cost
                div=s.diversification-baseline.diversification
                gains={k:float(-v[0]*cost+v[1]*div) for k,v in models.items()}
                rows.append(dict(start=name,scale=scale,cost_delta=float(cost),
                                 div_delta=float(div),fill_delta=float(s.fill_rate-baseline.fill_rate),
                                 local_delta=float(s.score-baseline.score),
                                 gain_mean=float(sum(gains.values())/len(gains)),
                                 gain_min=min(gains.values()),gain_max=max(gains.values())))
    path=ROOT/"reports"/"public_multistart.json"
    path.write_text(json.dumps(rows,indent=2)+"\n")
    for row in sorted(rows,key=lambda x:-x["gain_min"]):
        print(row["start"],row["scale"],"cost",round(row["cost_delta"]),
              "div",round(row["div_delta"],4),"mean",round(row["gain_mean"],3),
              "min",round(row["gain_min"],3))
    print(path)


if __name__=="__main__":
    main()
