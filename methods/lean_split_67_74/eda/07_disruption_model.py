"""Walk-forward evaluation of the disruption model (material-level Brier skill vs the naive rate).

For each origin o: train on data released up to o, predict weeks o+1..o+26, score
  near = o+1..o+13  (like the public window from week 208)
  far  = o+14..o+26 (like the private window if we never refreshed)
against the naive baseline (each material's historical disruption rate up to o).
"""
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.models.disruption import DisruptionModel, material_probability

pd.set_option("display.width", 220)
data = load_data()
ORIGINS = list(range(65, data.last_week - 13 + 1, 13))
VARIANTS = [("logit", "supply+news", "0.2"), ("gbm", "supply+news", "0.2")]
if len(sys.argv) > 1:   # kind:features[:sigma_min], e.g. gbm:news+contagion:0.8
    VARIANTS = [tuple((v + ":0.2").split(":")[:3]) for v in sys.argv[1:]]

truth = data.outcomes[["week", "material", "disrupted"]]
rows, preds = [], []
for o in ORIGINS:
    info = data.until(o)
    feats = build_features(info)
    rate = info.outcomes.groupby("material").disrupted.mean()
    weeks = range(o + 1, min(o + 26, data.last_week) + 1)
    for kind, fs, sig in VARIANTS:
        m = DisruptionModel.fit(info, kind, fs, feats=feats, sigma_min=float(sig))
        p = material_probability(m.predict_suppliers(feats, o, weeks)).merge(truth, on=["week", "material"])
        p["naive"] = p.material.map(rate)
        p["h"] = p.week - o
        p = p.assign(origin=o, variant=f"{kind}:{fs}:{sig}")
        preds.append(p)
        for name, sel in [("near", p.h <= 13), ("far", p.h > 13)]:
            q = p[sel]
            if not len(q):
                continue
            bs, bs_n = ((q.p - q.disrupted) ** 2).mean(), ((q.naive - q.disrupted) ** 2).mean()
            rows.append(dict(origin=o, variant=f"{kind}:{fs}:{sig}", window=name, events=int(q.disrupted.sum()),
                             brier=bs, brier_naive=bs_n, skill=1 - bs / bs_n))
res = pd.DataFrame(rows)
P = pd.concat(preds)

print("== per-window Brier skill (mean over windows) and pooled skill")
summ = res.groupby(["variant", "window"]).agg(mean_skill=("skill", "mean"), median_skill=("skill", "median"),
                                              worst=("skill", "min"), n=("skill", "size"))
pooled = P.assign(win=np.where(P.h <= 13, "near", "far")).groupby(["variant", "win"]).apply(
    lambda q: 1 - ((q.p - q.disrupted) ** 2).mean() / ((q.naive - q.disrupted) ** 2).mean())
summ["pooled_skill"] = [pooled.loc[(v, w)] for v, w in summ.index]
print(summ.round(3).to_string())

best = summ.xs("near", level="window").pooled_skill.idxmax()
print(f"\n== per-origin near-window skill, best variant {best}")
print(res[(res.variant == best)].pivot(index="origin", columns="window", values=["skill", "events"]).round(3))

print(f"\n== calibration ({best}, all horizons): predicted vs observed")
b = P[P.variant == best].copy()
b["bin"] = pd.cut(b.p, [0, .02, .05, .1, .2, .4, .6, .8, .95, 1], include_lowest=True)
print(b.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("disrupted", "mean"), n=("p", "size")).round(3))
P.to_csv("reports/disruption_walkforward.csv", index=False)
