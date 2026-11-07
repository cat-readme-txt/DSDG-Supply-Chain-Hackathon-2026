"""Choose a public order plan per material by robust score-response optimization.

The library contains chronological-independent public hindsight variants only;
all public truth is released in Phase 2. Forecasts and private rows come from
the scored 92.62 file. A small binary program selects one plan per material to
maximize the minimum predicted gain across four controlled-score fits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"methods"/"lean_split_67_74"/"src"))
from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import policy_arrays, score_window  # noqa: E402
from dsdg_supply_chain.simulator import simulate, truth_for  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402
from experiments.public_diversification_qp import write_precise  # noqa: E402

FIXED_SCALES=(.5,1.,1.5,2.,3.)
JOINT_SCALES=(1.2,1.5,1.8,2.2,2.5)
JOINT_STARTS=("scored92.31","scored92.62","fixed1.0","fixed2.0","fixed3.0")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reports"/"generated_candidates"/"material_portfolio_w221.csv")
    parser.add_argument("--report",type=Path,default=ROOT/"reports"/"public_material_portfolio.json")
    args=parser.parse_args()
    data=load_data(ROOT/"data"/"Phase 2 data")
    reference=naive(data.until(208),range(209,222))
    baseline_path=ROOT/"submissions"/"submission92.62.csv"
    baseline=read_submission(baseline_path)
    models=json.loads((ROOT/"reports"/"joint_surface_sweep.json").read_text())["models"]
    model_names=list(models)
    coefficients=np.array([models[name] for name in model_names])
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    library={"scored92.62":baseline_path,"scored92.31":ROOT/"submissions"/"submission92.31.csv"}
    with tempfile.TemporaryDirectory(prefix="dsdg-portfolio-") as directory:
        folder=Path(directory)
        for scale in FIXED_SCALES:
            path=folder/f"fixed_{scale}.csv"
            subprocess.run([sys.executable,"-m","experiments.public_diversification_qp",
                            "--diversity-scale",str(scale),"--output",str(path)],
                           cwd=ROOT,env=env,capture_output=True,text=True,check=True)
            library[f"fixed{scale}"]=path
        for start in JOINT_STARTS:
            for scale in JOINT_SCALES:
                path=folder/f"joint_{start}_{scale}.csv"
                subprocess.run([sys.executable,"-m","experiments.public_joint_order_search",
                                "--base",str(library[start]),"--scale",str(scale),"--output",str(path)],
                               cwd=ROOT,env=env,capture_output=True,text=True,check=True)
                library[f"joint_{start}_{scale}"]=path
        frames={name:read_submission(path) for name,path in library.items()}
        options=[]
        for material in sorted(data.suppliers.material.unique()):
            tr=truth_for(data,material,range(209,222))
            q0,s0=policy_arrays(baseline,tr)
            z0=simulate(tr,q0,s0)
            d0=float(((1-(s0**2).sum(axis=0))*1.5).sum()/130)
            for name,frame in frames.items():
                q,s=policy_arrays(frame,tr)
                z=simulate(tr,q,s)
                if np.max(tr.demand-z.served)>1e-6:
                    continue
                div=float(((1-(s**2).sum(axis=0))*1.5).sum()/130)
                dc=float(z.total_cost-z0.total_cost)
                dd=div-d0
                gains=-coefficients[:,0]*dc+coefficients[:,1]*dd
                options.append(dict(material=material,variant=name,cost_delta=dc,div_delta=dd,
                                    gains=gains.tolist()))
        mats=sorted(data.suppliers.material.unique())
        N=len(options);K=len(model_names)
        objective=np.zeros(N+1);objective[-1]=-1
        A=np.zeros((len(mats)+K,N+1));lower=np.zeros(len(mats)+K);upper=np.zeros(len(mats)+K)
        for i,material in enumerate(mats):
            for j,opt in enumerate(options):
                if opt["material"]==material:A[i,j]=1
            lower[i]=upper[i]=1
        for k in range(K):
            for j,opt in enumerate(options):A[len(mats)+k,j]=-opt["gains"][k]
            A[len(mats)+k,-1]=1
            lower[len(mats)+k]=-np.inf
        bounds=Bounds(np.r_[np.zeros(N),-np.inf],np.r_[np.ones(N),np.inf])
        result=milp(objective,integrality=np.r_[np.ones(N),0],bounds=bounds,
                    constraints=LinearConstraint(A,lower,upper),options={"time_limit":60})
        if result.x is None:
            raise RuntimeError(result.message)
        selected=[options[j] for j in range(N) if result.x[j]>.5]
        composite=baseline.copy()
        for opt in selected:
            material=opt["material"]
            source=frames[opt["variant"]]
            mask=(composite.material==material)&(composite.week<=221)
            cols=["order_units",*SHARE_COLS]
            composite.loc[mask,cols]=source.loc[mask,cols].to_numpy()
        write_precise(composite,args.output)
    from pipeline import CompetitionData
    CompetitionData(ROOT/"data"/"Phase 2 data").validate_file(args.output)
    generated=read_submission(args.output)
    current=score_window(baseline,reference,data,range(209,222))
    new=score_window(generated,reference,data,range(209,222))
    dc=float(new.cost-current.cost);dd=float(new.diversification-current.diversification)
    gains=-coefficients[:,0]*dc+coefficients[:,1]*dd
    report=dict(output=str(args.output),model_names=model_names,selection=selected,
                cost_delta=dc,div_delta=dd,fill_delta=float(new.fill_rate-current.fill_rate),
                local_score_delta=float(new.score-current.score),
                predicted_gains=dict(zip(model_names,gains.tolist())),
                n_library=len(library),n_options=N)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print('selected',[(x['material'],x['variant']) for x in selected])
    print('cost',round(dc),'div',round(dd,4),'predicted',np.round(gains,3),'local',round(new.score-current.score,3))
    print(args.output,args.report)


if __name__=="__main__":
    main()
