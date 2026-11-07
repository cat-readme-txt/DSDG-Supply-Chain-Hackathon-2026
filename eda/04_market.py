"""Tariff announcements, landed-cost formula, futures, macro tension, shipping vs shocks."""
import numpy as np
import pandas as pd

from common import all_data

d = all_data()
sup, sh, tar, pr, mac, shp, out = (d[k] for k in
                                   ["suppliers", "supply_history", "tariffs", "prices", "macro", "shipping", "outcomes"])
key = ["week", "material", "country"]
df = sh.merge(tar, on=key + ["date"]).merge(pr, on=key + ["date"]).merge(sup, on=["material", "country"])
df = df.merge(shp.rename(columns={"lane": "lane"}), on=["week", "date", "lane"])
pd.set_option("display.width", 250, "display.max_rows", 300)

print("== tariff announcements (rows with announced_rate)")
ann = tar.dropna(subset=["announced_rate"])
print(ann.groupby(["material", "country", "announced_rate", "announced_effective_week"])
         .week.agg(["min", "max", "count"]).reset_index().to_string(index=False))

print("\n== landed cost formula fit")
df["cost_guess"] = df.spot_price * (1 + df.tariff_rate)
df["resid"] = df.landed_cost - df.cost_guess
print(df.groupby("lane").apply(lambda g: pd.Series({
    "corr_resid_container": np.corrcoef(g.resid, g.container_rate)[0, 1],
    "resid_mean": g.resid.mean()})).round(3))
# per-material regression resid ~ container_rate
for m, g in df.groupby("material"):
    A = np.c_[g.container_rate, np.ones(len(g))]
    coef, *_ = np.linalg.lstsq(A, g.resid, rcond=None)
    err = g.resid - A @ coef
    print(f"{m:16s} resid = {coef[0]:.6f}*container + {coef[1]:.4f}  max|err|={np.abs(err).max():.4f}")

print("\n== futures_4w vs spot 4 weeks later (MAPE) vs naive spot carry")
pr = pr.sort_values(key)
pr["spot_t4"] = pr.groupby(["material", "country"]).spot_price.shift(-4)
v = pr.dropna(subset=["spot_t4"])
print("futures MAPE", (abs(v.futures_4w - v.spot_t4) / v.spot_t4).mean().round(4),
      " carry MAPE", (abs(v.spot_price - v.spot_t4) / v.spot_t4).mean().round(4))
print("futures/spot ratio quantiles", (pr.futures_4w / pr.spot_price).quantile([.01, .1, .5, .9, .99]).round(3).to_dict())

print("\n== benchmark price: weekly abs pct change, by disrupted")
o = out.sort_values(["material", "week"]).copy()
o["chg"] = o.groupby("material").benchmark_price.pct_change()
print(o.groupby("disrupted").chg.apply(lambda s: s.abs().describe()).unstack().round(4))

print("\n== futures/spot premium in weeks before shock onset (material-level, max over countries)")
pr["prem"] = pr.futures_4w / pr.spot_price - 1
prem = pr.groupby(["week", "material"]).prem.max().reset_index()
o = o.merge(prem, on=["week", "material"])
o["onset_in"] = np.nan
for m, g in o.groupby("material"):
    s = g.set_index("week").disrupted
    onsets = [w for w in s.index if s[w] == 1 and s.get(w - 1, 0) == 0]
    for w in g.week:
        nxt = [x - w for x in onsets if x >= w]
        if nxt and nxt[0] <= 8:
            o.loc[(o.material == m) & (o.week == w), "onset_in"] = nxt[0]
print(o.groupby(o.onset_in.fillna(-1)).prem.describe().round(4))

print("\n== macro tension index: mean by country, and around shocks")
print(mac.groupby("country").tension_index.describe().round(1))
print("\n== shipping congestion by lane")
print(shp.groupby("lane")[["container_rate", "transit_days", "congestion_index"]].describe().T.round(1))
