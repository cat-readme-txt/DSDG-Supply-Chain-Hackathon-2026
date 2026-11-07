"""Compare possible diversification formulas using controlled public scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"methods"/"lean_split_67_74"/"src"))
from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import naive  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402

TRAIN=((67.74,67.26),(89.45,91.36),(91.36,92.31),(92.31,92.62))
CHECK=((67.74,66.98),(50.44,54.10))


def diversity(sub):
    rows=sub[sub.week.between(209,221)]
    shares=rows[SHARE_COLS].to_numpy(float)
    sq=(shares**2).sum(axis=1)
    ent=-np.where(shares>0,shares*np.log(np.maximum(shares,1e-20)),0).sum(axis=1)/np.log(3)
    metrics=dict(hhi=(1-sq)*1.5,entropy=ent,effective=(1/sq-1)/2,
                 max_share=1.5*(1-shares.max(axis=1)),
                 count=(np.count_nonzero(shares>1e-5,axis=1)-1)/2,
                 hhi_sqrt=np.sqrt(np.clip((1-sq)*1.5,0,1)),
                 hhi_square=((1-sq)*1.5)**2)
    w=rows.order_units.to_numpy(float)
    return {f"{name}_{agg}":float(v.mean() if agg=="row" else (v*w).sum()/w.sum())
            for name,v in metrics.items() for agg in ("row","weighted")}


def main():
    data=load_data(ROOT/"data"/"Phase 2 data")
    ref=naive(data.until(208),range(209,222))
    scores={};div={}
    for score in {x for pair in (*TRAIN,*CHECK) for x in pair}:
        sub=read_submission(ROOT/"submissions"/f"submission{score:.2f}.csv")
        scores[score]=score_window(sub,ref,data,range(209,222))
        div[score]=diversity(sub)
    keys=div[next(iter(div))]
    rows=[]
    for metric in keys:
        def design(pair):
            a,b=pair
            return [-(scores[b].cost-scores[a].cost),div[b][metric]-div[a][metric]]
        X=np.array([design(p) for p in TRAIN]);y=np.array([b-a for a,b in TRAIN])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        Z=np.array([design(p) for p in CHECK]);z=np.array([b-a for a,b in CHECK])
        rows.append(dict(metric=metric,cost_coef=float(coef[0]),div_coef=float(coef[1]),
                         training_rmse=float(np.sqrt(np.mean((X@coef-y)**2))),
                         check_errors=(Z@coef-z).tolist(),
                         check_rmse=float(np.sqrt(np.mean((Z@coef-z)**2)))))
    rows.sort(key=lambda r:r["training_rmse"]+r["check_rmse"])
    path=ROOT/"reports"/"diversification_metric_inference.json"
    path.write_text(json.dumps(rows,indent=2)+"\n")
    for r in rows:
        print(r["metric"],"train",round(r["training_rmse"],3),"check",round(r["check_rmse"],3),
              "coeff",round(r["cost_coef"],8),round(r["div_coef"],3))
    print(path)


if __name__=="__main__":
    main()
