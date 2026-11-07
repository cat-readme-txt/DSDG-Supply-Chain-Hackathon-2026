"""Chronologically test shared-lane corrections to material disruption risk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"methods"/"lean_split_67_74"/"src"))
from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import (BLEND_BASE,BLEND_PER_WEEK,
    material_probability,predict_disruption,supplier_forecast)  # noqa: E402
from experiments.event_calendar import calendar_for  # noqa: E402


def event_probability(event,week):
    value=0.
    for onset,weight in event["onset_weeks"].items():
        onset=int(onset)
        if onset<=week:
            value+=weight*event["duration_survival"].get(str(week-onset+1),0.)
    return float(event["conversion_probability"]*value)


def corrected_forecast(info,weeks,sup):
    origin=info.last_week
    base=material_probability(sup).set_index(["week","material"]).p.copy()
    changes=[]
    for event in calendar_for(info)["active"]:
        if event["type"] not in ("closure","strike"):
            continue
        materials={x["material"] for x in event["affected"]}
        for material in materials:
            countries={x["country"] for x in event["affected"] if x["material"]==material}
            if len(countries)<2:
                continue
            for week in weeks:
                q=event_probability(event,week)
                if q<=0:
                    continue
                s=sup[(sup.week==week)&(sup.material==material)]
                if s.empty:
                    continue
                p=s.p.to_numpy(float)
                target=s.country.isin(countries).to_numpy()
                residual=p.copy()
                residual[target]=np.clip((p[target]-q)/max(1-q,1e-8),0,1)
                joint=1-(1-q)*np.prod(1-residual)
                old=float(base.loc[(week,material)])
                base.loc[(week,material)]=joint
                changes.append(dict(week=week,material=material,source=old,corrected=float(joint),event=q))
    out=base.rename("p").reset_index()
    rate=out.material.map(info.outcomes.groupby("material").disrupted.mean())
    blend=np.clip(BLEND_BASE+BLEND_PER_WEEK*(out.week-origin-1),0,1)
    out["p"]=(1-blend)*out.p+blend*rate
    return out,changes


def main():
    data=load_data(ROOT/"data"/"Phase 2 data")
    rows=[]
    for origin in (*range(103,111),221):
        info=data.until(origin)
        weeks=range(origin+1,origin+14)
        feat=build_features(info)
        sup=supplier_forecast(info,weeks,feats=feat)
        base=predict_disruption(info,weeks,sup=sup).set_index(["week","material"]).p
        new,changes=corrected_forecast(info,weeks,sup)
        new=new.set_index(["week","material"]).p
        if not changes:
            continue
        if origin<221:
            truth=data.outcomes[data.outcomes.week.isin(weeks)].set_index(["week","material"]).disrupted
            keys=[(c["week"],c["material"]) for c in changes]
            old_error=float(np.mean([(base.loc[k]-truth.loc[k])**2 for k in keys]))
            new_error=float(np.mean([(new.loc[k]-truth.loc[k])**2 for k in keys]))
        else:
            old_error=new_error=None
        shifts=[new.loc[(c["week"],c["material"])]-base.loc[(c["week"],c["material"])]
                for c in changes]
        rows.append(dict(origin=origin,affected_rows=len(changes),old_brier=old_error,
                         new_brier=new_error,mean_probability_change=float(np.mean(shifts)),
                         changes=changes))
        print(origin,len(changes),old_error,new_error,rows[-1]["mean_probability_change"],flush=True)
    path=ROOT/"reports"/"correlated_brier.json"
    path.write_text(json.dumps(rows,indent=2)+"\n")
    print(path)


if __name__=="__main__":
    main()
