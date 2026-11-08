"""Are the supplier-state and market features predictive, and do they add to the news?

Two questions the disruption model has to answer, per supplier:
  Q1 new shock:   not hit now -> hit at some point in the next 4 weeks?
  Q2 persistence: hit now     -> still hit 1 / 4 weeks later?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features

pd.set_option("display.width", 220, "display.max_rows", 200)
f = build_features(load_data()).sort_values(["material", "country", "week"])
g = f.groupby(["material", "country"])
fwd_max = lambda s, h: s[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)  # noqa: E731
f["new_in_4"] = g.hit.transform(lambda s: fwd_max(s, 4))
f["hit_t1"] = g.hit.shift(-1)
f["hit_t4"] = g.hit.shift(-4)

num = [c for c in f.select_dtypes("number").columns
       if c not in {"week", "new_in_4", "hit_t1", "hit_t4", "hit"}]


def auc_table(df, target, cols):
    rows = []
    for c in cols:
        if df[c].nunique() < 2:
            continue
        a = roc_auc_score(df[target], df[c])
        rows.append((c, max(a, 1 - a), "+" if a >= 0.5 else "-"))
    return pd.DataFrame(rows, columns=["feature", "auc", "dir"]).sort_values("auc", ascending=False)


q1 = f[(f.hit == 0) & f.new_in_4.notna()]
print(f"== Q1 new shock within 4 weeks (not hit now): n={len(q1)}, base rate {q1.new_in_4.mean():.3f}")
print(auc_table(q1, "new_in_4", num).head(20).round(3).to_string(index=False))

print("\n-- does market data add to news? split by news_stage")
for name, sub in [("no news story", q1[q1.news_stage == 0]), ("open news story", q1[q1.news_stage > 0])]:
    print(f"\n{name}: n={len(sub)}, rate {sub.new_in_4.mean():.3f}")
    cols = ["tension_z", "tension_chg2", "congestion_z", "transit_z", "freight_z", "fx_chg4",
            "futures_premium", "spot_chg4", "tariff_announced", "lane_hit_share", "country_hit_share",
            "weeks_since_hit", "called_off_4w"]
    print(auc_table(sub, "new_in_4", cols).round(3).to_string(index=False))

print("\n-- binned: P(new shock in 4w | tension_z), stories open vs not")
q1 = q1.assign(tz=pd.cut(q1.tension_z, [-99, 0, 1, 2, 4, 99]), cz=pd.cut(q1.congestion_z, [-99, 0, 1, 2, 4, 99]))
print(q1.pivot_table(index="tz", columns=q1.news_stage > 0, values="new_in_4", aggfunc=["mean", "count"],
                     observed=False).round(3))
print(q1.pivot_table(index="cz", columns=q1.news_stage > 0, values="new_in_4", aggfunc=["mean", "count"],
                     observed=False).round(3))

q2 = f[(f.hit == 1) & f.hit_t4.notna()]
print(f"\n== Q2 persistence (hit now): n={len(q2)}, still hit +1w {q2.hit_t1.mean():.3f}, +4w {q2.hit_t4.mean():.3f}")
print(auc_table(q2, "hit_t4", num).head(12).round(3).to_string(index=False))
print("\nP(still hit in 4w) by hit type and weeks already hit:")
q2 = q2.assign(age=pd.cut(q2.weeks_hit, [0, 1, 3, 6, 12, 99]))
print(q2.pivot_table(index="hit_type", columns="age", values="hit_t4", aggfunc="mean", observed=False).round(2))
print(q2.pivot_table(index="hit_type", columns="age", values="hit_t4", aggfunc="count", observed=False))
print("\nP(still hit in 1w) by ongoing / recovered news:")
print(q2.groupby(["ongoing_2w", "recovered_4w"]).hit_t1.agg(["mean", "count"]).round(3))
