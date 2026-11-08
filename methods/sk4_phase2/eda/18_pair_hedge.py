"""Hedge the forecast columns across the two final submissions.

Kaggle keeps the better of our two final submissions. The ordering already splits into a calm bet
(lean_lp) and a shock bet (insured). Lean wins when no unwarned shock arrives, insured when one
does. So the disruption_prob column can be split the same way: lean gets calm-leaning
probabilities, insured gets shock-leaning ones. Each column is then sharper in the world where that
file is the one that counts.

P depends only on the forecast columns, so score = 30 P + (rest from the backtest CSVs).

usage: uv run python eda/18_pair_hedge.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.strategies import story_materials

data = load_data()
jobs = pickle.loads(Path("reports/policy_cache.pkl").read_bytes())
L = pd.read_csv("reports/backtest_lean_lp.csv").set_index(["origin", "window"])
I = pd.read_csv("reports/backtest_insured.csv").set_index(["origin", "window"])
rest_L, rest_I = L.score - 30 * L.P, I.score - 30 * I.P

key = ["week", "material"]
W = []
for j in jobs:
    out = data.outcomes[data.outcomes.week.isin(j["win"])][key + ["disrupted", "benchmark_price"]]
    m = (out.merge(j["sub"][key + ["disruption_prob", "price_forecast"]], on=key)
         .merge(j["base"][key + ["disruption_prob", "price_forecast"]], on=key, suffixes=("", "_n")))
    m["h"] = m.week - j["info_week"]
    m["bg"] = ~m.material.isin(story_materials(data.until(j["info_week"])))   # hedge background rows only
    rate = data.until(j["info_week"]).outcomes.groupby("material").disrupted.mean()
    m["w"] = m.material.map((rate / rate.mean()).clip(0, 2))
    W.append(((j["origin"], j["window"]), m))


def P_of(m, p):
    y, px = m.disrupted.to_numpy(float), m.benchmark_price.to_numpy(float)
    bs, bsn = np.mean((p - y) ** 2), np.mean((m.disruption_prob_n - y) ** 2)
    mp, mpn = np.mean(abs(m.price_forecast - px) / px), np.mean(abs(m.price_forecast_n - px) / px)
    b = 1 - bs / bsn if bsn > 0 else 0.0
    return float(np.clip(0.6 * b + 0.4 * (1 - mp / mpn), 0, 1))


def calm_bet(m, a, k):
    """Shrink probabilities beyond week k of the horizon by factor a (no news can be seen there)."""
    p = m.disruption_prob.to_numpy(float)
    return np.where(m.bg & (m.h.to_numpy() > k), a * p, p)


def shock_bet(m, b, k):
    """Raise probabilities beyond week k toward 1 by fraction b x relative material rate."""
    p = m.disruption_prob.to_numpy(float)
    return np.where(m.bg & (m.h.to_numpy() > k), p + b * m.w.to_numpy() * (1 - p), p)


def evaluate(a, kL, b, kI):
    rows = []
    for idx, m in W:
        sL = rest_L[idx] + 30 * P_of(m, calm_bet(m, a, kL))
        sI = rest_I[idx] + 30 * P_of(m, shock_bet(m, b, kI))
        rows.append(dict(idx=idx, lean=sL, ins=sI, best=max(sL, sI)))
    return pd.DataFrame(rows)


base = evaluate(1.0, 0, 0.0, 0)
print(f"no hedge: lean {base.lean.mean():.2f} insured {base.ins.mean():.2f} best-of-two {base.best.mean():.2f}")
res = []
for a in [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]:
    for kL in [0, 2, 4, 6]:
        for b in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
            for kI in [0, 2, 4]:
                r = evaluate(a, kL, b, kI)
                res.append(dict(a=a, kL=kL, b=b, kI=kI, best=r.best.mean(), worst=r.best.min(),
                                lean=r.lean.mean(), ins=r.ins.mean(),
                                pub=r.best[[i[1] == "public" for i in r.idx]].mean(),
                                priv=r.best[[i[1] == "private" for i in r.idx]].mean()))
R = pd.DataFrame(res).sort_values("best", ascending=False)
pd.set_option("display.width", 200)
print(R.head(15).round(2).to_string(index=False))
top = R.iloc[0]
r = evaluate(top.a, int(top.kL), top.b, int(top.kI))
print("\nper window, best hedge vs none:")
print(pd.DataFrame({"none": base.best.values, "hedged": r.best.values, "lean": r.lean.values, "ins": r.ins.values},
                   index=[f"{o}-{w}" for o, w in r.idx]).round(1).to_string())
