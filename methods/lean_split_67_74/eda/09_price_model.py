"""Walk-forward evaluation of the price model: MAPE skill vs carrying the last benchmark forward.

Variants
  carry           last benchmark price (the naive baseline, skill 0 by definition)
  uplift@T/keepK  normal level x shock uplift: new shocks when P > T; ongoing kept while P >= K
  gbm             a pure-ML comparator: gradient boosting (absolute-error loss) on material-level
                  features at the origin -> log(price[o+h] / price[o])
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.features.news import TYPES
from dsdg_supply_chain.models.disruption import supplier_forecast
from dsdg_supply_chain.models.price import PriceModel

pd.set_option("display.width", 220)
data = load_data()
ORIGINS = list(range(65, data.last_week - 13 + 1, 13))
THRESHOLDS = [0.3, 0.5, 0.7]
KEEPS = [0.0, 0.2, 0.5]
bench = data.outcomes[["week", "material", "benchmark_price"]]

MAT_COLS = ["hit", "news_stage", *[f"stage_{t}" for t in TYPES], "weeks_hit", "cap_ratio", "spot_chg4"]


def material_feats(feats):
    agg = {c: "max" for c in MAT_COLS if c != "cap_ratio"} | {"cap_ratio": "min"}
    return feats.groupby(["week", "material"]).agg(agg).reset_index()


def gbm_forecast(info, feats, weeks):
    mf = material_feats(feats)
    lp = np.log(info.outcomes.pivot(index="week", columns="material", values="benchmark_price"))
    rows = []
    for h in range(1, 27):
        tgt = (lp.shift(-h) - lp).stack().rename("y").reset_index()
        rows.append(mf.merge(tgt, on=["week", "material"]).assign(h=h))
    tr = pd.concat(rows).dropna()
    X = MAT_COLS + ["h"]
    m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, max_iter=200,
                                      min_samples_leaf=100, learning_rate=0.05, random_state=0)
    m.fit(tr[X], tr.y)
    at = mf[mf.week == info.last_week]
    te = pd.concat([at.assign(h=w - info.last_week, week=w) for w in weeks])
    last = lp.iloc[-1]
    te["price"] = np.exp(te.material.map(last) + m.predict(te[X]))
    return te[["week", "material", "price"]]


rows = []
for o in ORIGINS:
    info = data.until(o)
    feats = build_features(info)
    weeks = range(o + 1, min(o + 26, data.last_week) + 1)
    sup = supplier_forecast(info, weeks, feats=feats)
    last = info.outcomes[info.outcomes.week == o].set_index("material").benchmark_price
    preds = {"carry": bench[bench.week.isin(weeks)].assign(price=lambda d: d.material.map(last))[
        ["week", "material", "price"]]}
    pm = PriceModel.fit(info, feats)
    for t in THRESHOLDS:
        for k in KEEPS:
            pm.threshold, pm.keep = t, k
            preds[f"uplift@{t}/keep{k}"] = pm.predict(info, sup, weeks)
    preds["gbm"] = gbm_forecast(info, feats, weeks)
    for name, p in preds.items():
        q = p.merge(bench, on=["week", "material"])
        q["ape"] = (q.price - q.benchmark_price).abs() / q.benchmark_price
        q["win"] = np.where(q.week - o <= 13, "near", "far")
        for win, g in q.groupby("win"):
            rows.append(dict(origin=o, variant=name, window=win, mape=g.ape.mean()))
    if o == ORIGINS[-1]:
        print("learned log-uplifts by shock type (last origin):", {k: round(v, 3) for k, v in pm.uplift.items()})

res = pd.DataFrame(rows)
base = res[res.variant == "carry"].set_index(["origin", "window"]).mape
res["skill"] = 1 - res.mape / res.set_index(["origin", "window"]).index.map(base)
print(res.groupby(["variant", "window"]).skill.agg(["mean", "median", "min", "max"]).round(3).unstack(0).T
      .sort_index().to_string())
print("\nper-origin near-window skill:")
print(res[res.window == "near"].pivot(index="origin", columns="variant", values="skill").round(3).to_string())
