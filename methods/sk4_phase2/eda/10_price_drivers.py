"""Is there predictable structure in prices outside shocks?

Target: future log return of each supplier's spot price over h weeks (h = 1, 4, 13), in weeks with
no active shock on that material (shock uplifts are modelled separately).
Candidates: own lagged returns, deviation from trailing means (mean reversion), futures premium,
macro (fx, pmi, cpi, policy rate, tension) levels and changes, freight, demand changes,
cross-material/common-factor returns.
Test: correlation, and out-of-sample R^2 of a ridge regression trained on weeks < 130, tested on
weeks >= 130 (positive R^2 = genuinely predictive).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from dsdg_supply_chain.data import load_data

pd.set_option("display.width", 220, "display.max_rows", 100)
d = load_data()
key = ["week", "material", "country"]
pr = d.prices.merge(d.suppliers[["material", "country", "lane"]], on=["material", "country"]).sort_values(key)
pr["lp"] = np.log(pr.spot_price)
g = pr.groupby(["material", "country"]).lp
for k in (1, 4, 13):
    pr[f"ret_{k}"] = g.diff(k)
for k in (13, 26, 52):
    pr[f"dev_{k}"] = pr.lp - g.transform(lambda s, k=k: s.rolling(k, min_periods=k // 2).mean())
for h in (1, 4, 13):   # targets computed before any merge, so they stay aligned with their rows
    pr[f"y_{h}"] = g.shift(-h) - pr.lp
pr["fut_prem"] = np.log(pr.futures_4w / pr.spot_price)
# supplier premium vs the material's benchmark (cross-sectional mean reversion?)
pr["vs_bench"] = pr.lp - pr.groupby(["week", "material"]).lp.transform("mean")
# common factor: average return across all materials
common = pr.groupby("week").ret_1.mean().rename("common_ret_1")
common4 = pr.groupby("week").ret_4.mean().rename("common_ret_4")
pr = pr.merge(common, on="week").merge(common4, on="week")

mac = d.macro.sort_values(["country", "week"]).copy()
for c in ["fx_rate", "pmi", "cpi_yoy", "policy_rate", "tension_index"]:
    mac[f"{c}_chg4"] = mac.groupby("country")[c].diff(4)
mac["fx_ret4"] = np.log(mac.fx_rate).groupby(mac.country).diff(4)
mac["fx_lvl"] = np.log(mac.fx_rate)
pr = pr.merge(mac.drop(columns=["date"]), on=["week", "country"])
shp = d.shipping.sort_values(["lane", "week"]).copy()
shp["freight_chg4"] = np.log(shp.container_rate).groupby(shp.lane).diff(4)
pr = pr.merge(shp[["week", "lane", "container_rate", "freight_chg4"]], on=["week", "lane"])
dm = d.demand.sort_values(["material", "week"]).copy()
dm["dem_chg4"] = np.log(dm.demand_units).groupby(dm.material).diff(4)
dm["dem_fwd4"] = np.log(dm.demand_units).groupby(dm.material).shift(-4) - np.log(dm.demand_units)  # plan is known
pr = pr.merge(dm[["week", "material", "dem_chg4", "dem_fwd4"]], on=["week", "material"])

dis = d.outcomes.set_index(["week", "material"]).disrupted
CAND = ["ret_1", "ret_4", "ret_13", "dev_13", "dev_26", "dev_52", "fut_prem", "vs_bench", "common_ret_1",
        "common_ret_4", "pmi", "cpi_yoy", "policy_rate", "tension_index", "fx_lvl", "fx_ret4", "pmi_chg4",
        "cpi_yoy_chg4", "policy_rate_chg4", "tension_index_chg4", "freight_chg4", "dem_chg4", "dem_fwd4"]

for h in (1, 4, 13):
    # keep only windows with no shock on the material at the origin or the target
    quiet = [(dis.get((w, m), 1) == 0) and (dis.get((w + h, m), 1) == 0) for w, m in zip(pr.week, pr.material)]
    q = pr[quiet].dropna(subset=CAND + [f"y_{h}"])
    corr = q[CAND].corrwith(q[f"y_{h}"]).sort_values(key=abs, ascending=False)
    tr, te = q[q.week < 130], q[q.week >= 130]
    mu, sd = tr[CAND].mean(), tr[CAND].std()
    r = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit((tr[CAND] - mu) / sd, tr[f"y_{h}"])
    pred = r.predict((te[CAND] - mu) / sd)
    r2 = 1 - ((te[f"y_{h}"] - pred) ** 2).sum() / ((te[f"y_{h}"] - te[f"y_{h}"].mean()) ** 2).sum()
    r2_0 = 1 - (te[f"y_{h}"] ** 2).sum() / ((te[f"y_{h}"] - te[f"y_{h}"].mean()) ** 2).sum()
    print(f"\n== horizon {h}: n={len(q)}, out-of-sample R^2 ridge={r2:.4f} (zero-forecast R^2={r2_0:.4f})")
    print("top correlations with future return:")
    print(corr.head(8).round(3).to_string())
    coef = pd.Series(r.coef_, index=CAND).sort_values(key=abs, ascending=False)
    print("largest ridge coefficients:", coef.head(5).round(4).to_dict())
