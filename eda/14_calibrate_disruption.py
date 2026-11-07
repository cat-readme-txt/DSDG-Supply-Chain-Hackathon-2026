"""Post-hoc calibration of material disruption probabilities, using only history up to each origin.

  persist(h)  = P(material still disrupted h weeks later | disrupted now), from past pairs (t, t+h)
  surprise(h) = P(material disrupted h weeks later | not disrupted now and no open news story)
New probability:
  disrupted at origin:   (1 - a) p + a persist(h)
  not disrupted, no story: max(p, b x surprise(h))
Scored as mean per-window Brier skill on public + private windows (cached forecasts).
"""
import numpy as np
import pandas as pd

exec(open("eda/11_tune_policy.py").read().split("rng = np.random")[0])

OUT = data.outcomes.pivot(index="week", columns="material", values="disrupted")


def curves(info_week, story_by_week):
    d = OUT.loc[:info_week]
    pers, surp = {}, {}
    for h in range(1, 27):
        now, later = d.iloc[:-h], d.shift(-h).iloc[:-h]
        on = now == 1
        pers[h] = later[on].stack().mean() if on.values.sum() else 0.7
        quiet = (now == 0) & (story_by_week.reindex(now.index).fillna(0) == 0)
        surp[h] = later[quiet].stack().mean() if quiet.values.sum() else 0.1
    return pers, surp


feats_all = build_features(data) if "build_features" in dir() else None
from dsdg_supply_chain.features.build import build_features  # noqa: E402
FE = build_features(data)
STORY = FE.groupby(["week", "material"]).news_stage.max().unstack()

rows = []
for j in jobs:
    o = j["info_week"]
    pers, surp = curves(o, STORY.loc[:o])
    dis_now = OUT.loc[o]
    story_now = STORY.loc[o]
    m = j["mat_p"][j["mat_p"].week.isin(j["win"])].copy()
    m["h"] = m.week - o
    m["y"] = [OUT.loc[w, mt] for w, mt in zip(m.week, m.material)]
    base = j["base"].set_index(["week", "material"]).disruption_prob
    m["naive"] = [base[(w, mt)] for w, mt in zip(m.week, m.material)]
    on = m.material.map(dis_now).to_numpy() == 1
    quiet = (~on) & (m.material.map(story_now).to_numpy() == 0)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        for b in (0.0, 0.5, 1.0):
            p = m.p.to_numpy().copy()
            ph = m.h.map(pers).to_numpy()
            sh = m.h.map(surp).to_numpy()
            p = np.where(on, (1 - a) * p + a * ph, p)
            p = np.where(quiet, np.maximum(p, b * sh), p)
            bs, bn = ((p - m.y) ** 2).mean(), ((m.naive - m.y) ** 2).mean()
            rows.append(dict(origin=j["origin"], window=j["window"], a=a, b=b, skill=1 - bs / bn))
R = pd.DataFrame(rows)
t = R.groupby(["a", "b"]).skill.agg(["mean", "median", "min"]).round(3)
print(t.sort_values("mean", ascending=False).to_string())
