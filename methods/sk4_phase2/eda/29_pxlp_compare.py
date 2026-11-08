"""lean_split_pxlp vs lean_split on the 66 step-4 windows: alone, public/private halves, the pair with
insured_hedged under 3 metric readings, and windows with a live price-moving (source) shock at the
information week (a warned story not yet started, or a shock ongoing).
usage: uv run python eda/29_pxlp_compare.py [new_strategy]
"""
import json
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.backtest import rescore
from dsdg_supply_chain.data import WINDOW, load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.models.cost_path import SOURCE_TYPES
from dsdg_supply_chain.models.duration import shock_class
from dsdg_supply_chain.models.scripts import ScriptModel, _targets

new = sys.argv[1] if len(sys.argv) > 1 else "lean_split_pxlp"
R = {k: pd.read_csv(f"reports/backtest_{k}_step4.csv") for k in ["lean_split", new, "insured_hedged"]}
d = load_data()
feats = build_features(d)
cls = feats.assign(cls=shock_class(feats))


def live_source(w: int) -> set[str]:
    """Materials with a source shock ongoing at week w or warned by a live (not overdue) story."""
    now = cls[(cls.week == w) & (cls.hit == 1) & cls.cls.isin(SOURCE_TYPES)]
    mats = set(now.material)
    sm = ScriptModel.fit(d.until(w))
    for r in sm.open_stories.itertuples():
        if r.type in SOURCE_TYPES and sm.onset_pmf(r.type, r.stage, w - int(getattr(r, f"w{r.stage}")), 26).sum() > 0:
            mats |= set(_targets(d.suppliers, r.type, r.entity).material)
    return mats


a, b, ins = R["lean_split"], R[new], R["insured_hedged"]
info = np.where(a.window == "public", a.origin, a.origin + WINDOW)
live = [live_source(int(w)) for w in info]
flag = np.array([len(m) > 0 for m in live])

print(f"{'reading':24s} {'lean':>6s} {'new':>6s} {'d':>6s} {'pub d':>6s} {'priv d':>6s} {'pair':>6s} {'pair new':>8s} {'live d':>7s}")
for rd in [("pooled", "pooled", "zero"), ("pooled", "pooled", "one"), ("mean", "mean", "one")]:
    sa, sb, si = rescore(a, *rd), rescore(b, *rd), rescore(ins, *rd)
    dd = sb - sa
    print(f"{' '.join(rd):24s} {sa.mean():6.2f} {sb.mean():6.2f} {dd.mean():+6.2f} {dd[a.window == 'public'].mean():+6.2f} "
          f"{dd[a.window == 'private'].mean():+6.2f} {np.maximum(sa, si).mean():6.2f} {np.maximum(sb, si).mean():8.2f} "
          f"{dd[flag].mean():+7.2f}")
dd = b.score - a.score
print(f"\nour scorer: wins {(dd > 0.01).sum()}, losses {(dd < -0.01).sum()}, ties {(dd.abs() <= 0.01).sum()}; "
      f"F {a.F.mean():.3f} -> {b.F.mean():.3f}")


def mat_F(df, i, mats):
    m = pd.DataFrame(json.loads(df.per_material.iloc[i])).set_index("material").loc[sorted(mats)]
    return (m.naive_cost.sum() - m.cost.sum()) / (m.naive_cost.sum() - m.hindsight_cost.sum())


rows = []
for i in np.flatnonzero(flag):
    rows.append(dict(origin=a.origin[i], window=a.window[i], info=info[i], mats=",".join(sorted(live[i])),
                     lean=a.score[i], new=b.score[i], d=dd[i], F_lean=a.F[i], F_new=b.F[i],
                     Fm_lean=mat_F(a, i, live[i]), Fm_new=mat_F(b, i, live[i])))
t = pd.DataFrame(rows)
print(f"\n{len(t)} windows with a live source shock at the information week "
      f"(Fm = pooled F over the affected materials):")
print(t.round(3).to_string(index=False))
print(t[["lean", "new", "d", "F_lean", "F_new", "Fm_lean", "Fm_new"]].mean().round(3).to_string())
big = pd.DataFrame(dict(origin=a.origin, window=a.window, d=dd)).loc[lambda x: x.d.abs() > 0.05]
print("\nall windows with |delta| > 0.05:\n", big.round(2).to_string(index=False))
