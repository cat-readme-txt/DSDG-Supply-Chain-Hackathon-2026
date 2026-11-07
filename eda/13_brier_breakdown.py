"""Where does the Brier score go? Squared error by situation, ours vs naive, on public windows.

Situations per (material, target week), from the truth at the origin and at the target:
  persist     disrupted at origin and still disrupted at target
  ended       disrupted at origin, not at target
  new_warned  not disrupted at origin, disrupted at target, a news story was open at origin
  new_unwarn  same, but no open story (e.g. reversals, the week-101 closure)
  quiet       not disrupted at origin nor at target
"""
import numpy as np
import pandas as pd

exec(open("eda/11_tune_policy.py").read().split("rng = np.random")[0])

out = data.outcomes.set_index(["week", "material"]).disrupted
rows = []
for j in jobs:
    if j["window"] != "public":
        continue
    o = j["info_week"]
    feats = j["feats"]
    story = feats.groupby("material").news_stage.max()
    sub = j["sub"][j["sub"].week.isin(j["win"])]
    base = j["base"].set_index(["week", "material"]).disruption_prob
    for r in sub.itertuples():
        y = out[(r.week, r.material)]
        y0 = out[(o, r.material)]
        sit = ("persist" if y else "ended") if y0 else (
            ("new_warned" if story[r.material] > 0 else "new_unwarn") if y else "quiet")
        rows.append(dict(origin=o, sit=sit, h=r.week - o, y=y, p=r.disruption_prob,
                         p_naive=base[(r.week, r.material)]))
D = pd.DataFrame(rows)
D["se"], D["se_naive"] = (D.p - D.y) ** 2, (D.p_naive - D.y) ** 2
g = D.groupby("sit").agg(n=("y", "size"), mean_p=("p", "mean"), obs=("y", "mean"), se=("se", "sum"),
                         se_naive=("se_naive", "sum"))
g["share_of_our_error"] = g.se / g.se.sum()
pd.set_option("display.width", 200)
print(g.round(3).to_string())
print(f"\ntotal SE ours {D.se.sum():.1f} vs naive {D.se_naive.sum():.1f}  -> pooled skill {1 - D.se.sum() / D.se_naive.sum():.3f}")
print("\npersist / ended by horizon (mean p vs obs):")
x = D[D.sit.isin(["persist", "ended"])].assign(hb=lambda d: pd.cut(d.h, [0, 3, 6, 9, 13]))
print(x.groupby("hb", observed=True).agg(p=("p", "mean"), obs=("y", "mean"), n=("y", "size")).round(3).to_string())
print("\nnew_warned by horizon (mean p for events vs non-events of open-story materials):")
w = D[(D.sit.isin(["new_warned", "quiet"]))]
print(w.groupby(["sit", pd.cut(w.h, [0, 2, 4, 8, 13])], observed=True).p.mean().round(3).unstack().to_string())
