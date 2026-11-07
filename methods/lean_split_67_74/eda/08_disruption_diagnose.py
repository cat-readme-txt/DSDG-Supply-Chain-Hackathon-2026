"""Where does the disruption model lose Brier score? Supplier-level calibration split by
"hit now" (duration part) vs "not hit now" (onset part), and the material aggregation."""
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.labels import supplier_state
from dsdg_supply_chain.models import duration as dur
from dsdg_supply_chain.models.disruption import DisruptionModel, material_probability

pd.set_option("display.width", 200)
data = load_data()
kind = sys.argv[1] if len(sys.argv) > 1 else "logit"
truth = supplier_state(data)[["week", "material", "country", "hit"]].rename(columns={"hit": "y"})
out = data.outcomes[["week", "material", "disrupted"]]
rows = []
for o in range(65, data.last_week - 13 + 1, 13):
    info = data.until(o)
    feats = build_features(info)
    m = DisruptionModel.fit(info, kind, "supply+news", feats=feats)
    p = m.predict_suppliers(feats, o, range(o + 1, min(o + 26, data.last_week) + 1))
    rows.append(p.merge(truth, on=["week", "material", "country"]).assign(origin=o))
P = pd.concat(rows)
P["bin"] = pd.cut(P.p, [0, .02, .05, .1, .2, .4, .6, .8, .95, 1], include_lowest=True)
for name, sub in [("NOT hit at origin (onset classifier)", P[P.hit == 0]), ("HIT at origin (duration model)", P[P.hit == 1])]:
    print(f"== {name}: n={len(sub)}, Brier {((sub.p - sub.y) ** 2).mean():.4f}, "
          f"sum sq err {((sub.p - sub.y) ** 2).sum():.1f}")
    print(sub.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("p", "size")).round(3))
print("\n== HIT at origin: per origin, predicted vs actual persistence")
h = P[P.hit == 1]
print(h.groupby("origin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("p", "size")).round(2).T)

# material aggregation check: independence vs max
mp = material_probability(P).merge(out, on=["week", "material"])
mx = P.groupby(["week", "material", "origin"]).p.max().rename("pmax").reset_index().merge(out, on=["week", "material"])
for nm, col, d in [("independent union", "p", mp), ("max", "pmax", mx)]:
    print(f"material Brier with {nm}: {((d[col] - d.disrupted) ** 2).mean():.4f}")
