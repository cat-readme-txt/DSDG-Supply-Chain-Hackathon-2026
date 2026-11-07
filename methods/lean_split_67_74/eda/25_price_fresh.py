"""Fresh look at the price process: what predicts the benchmark's change over the next h weeks?

Candidates (all known at the origin): futures premium (futures_4w vs spot), supplier dispersion
(each supplier's spot vs the benchmark), anchor gap (log benchmark vs its 78-week mean), recent
momentum, yearly seasonality, fx moves. Quiet weeks only (no supplier hit at origin or target).
Out-of-sample: fit on weeks <= 130, test on later weeks.
usage: uv run python eda/25_price_fresh.py
"""
import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.labels import supplier_state

d = load_data()
pr = d.prices.copy()
o = d.outcomes.set_index(["material", "week"]).benchmark_price
st = supplier_state(d)
hit = st.groupby(["material", "week"]).hit.max()
pr["lfut"] = np.log(pr.futures_4w / pr.spot_price)
g = pr.groupby(["material", "week"])
X = pd.DataFrame({"fut_prem": g.lfut.mean(),
                  "fut_prem_max": g.lfut.max()})
X["lb"] = np.log(o)
X = X.reset_index().sort_values(["material", "week"])
G = X.groupby("material")
X["anchor_gap"] = X.lb - G.lb.transform(lambda s: s.rolling(78, min_periods=26).mean())
X["mom4"] = X.lb - G.lb.shift(4)
X["mom1"] = X.lb - G.lb.shift(1)
X["sin"], X["cos"] = np.sin(2 * np.pi * X.week / 52), np.cos(2 * np.pi * X.week / 52)
X["hit"] = hit.reindex(pd.MultiIndex.from_frame(X[["material", "week"]])).to_numpy()
fx = d.macro.pivot(index="week", columns="country", values="fx_rate")
for h in [1, 4, 8, 13]:
    X[f"y{h}"] = G.lb.shift(-h) - X.lb
    X[f"hit{h}"] = G.hit.shift(-h)
feats = ["fut_prem", "anchor_gap", "mom4", "mom1", "sin", "cos"]
rows = []
for h in [1, 4, 8, 13]:
    q = X[(X.hit == 0) & (X[f"hit{h}"] == 0)].dropna(subset=feats + [f"y{h}"])
    tr, te = q[q.week <= 130], q[q.week > 130]
    for f in feats:
        c = np.corrcoef(q[f], q[f"y{h}"])[0, 1]
        b = np.polyfit(tr[f], tr[f"y{h}"], 1)
        pred = np.polyval(b, te[f])
        r2 = 1 - ((te[f"y{h}"] - pred) ** 2).sum() / (te[f"y{h}"] ** 2).sum()
        rows.append(dict(h=h, feature=f, corr=c, slope=b[0], oos_r2_vs_carry=r2, n=len(q)))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(4).pivot(index="feature", columns="h", values=["corr", "oos_r2_vs_carry"]).to_string())
# futures as a direct 4-week forecast of each supplier's spot
pr = pr.sort_values(["material", "country", "week"])
pr["spot_t4"] = pr.groupby(["material", "country"]).spot_price.shift(-4)
q = pr.dropna(subset=["spot_t4"])
print("\nspot in 4 weeks: MAPE carry", (abs(q.spot_price - q.spot_t4) / q.spot_t4).mean().round(4),
      " futures", (abs(q.futures_4w - q.spot_t4) / q.spot_t4).mean().round(4))
print("corr(log fut/spot, log spot_t4/spot):", np.corrcoef(np.log(q.futures_4w / q.spot_price), np.log(q.spot_t4 / q.spot_price))[0, 1].round(3))
# supplier deviations from the benchmark: do they revert, and do they move the benchmark?
pr = pr.merge(d.outcomes[["material", "week", "benchmark_price"]], on=["material", "week"])
pr["dev"] = np.log(pr.spot_price / pr.benchmark_price)
pr["dev_t1"] = pr.groupby(["material", "country"]).dev.shift(-1)
q = pr.dropna(subset=["dev_t1"])
print("supplier deviation AR(1):", np.polyfit(q.dev, q.dev_t1, 1)[0].round(3), " std", q.dev.std().round(4))
