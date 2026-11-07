"""Value of information: how much would the score rise if each forecast were perfect?

Variants (all on the same 22 backtest windows):
  current          our pipeline
  oracle_P         perfect disruption_prob (0/1 truth) and price_forecast (true price); policy unchanged
  oracle_policy    the planner given the TRUE supplier state (who is hit, true capacity, lead, cost)
  oracle_all       both of the above
  hindsight        hindsight-optimal orders (F = 1) with our forecasts; zero-order rows evenly split
The gaps say where improving the pipeline can pay, and how much.
"""
import numpy as np
import pandas as pd

exec(open("eda/11_tune_policy.py").read().split("rng = np.random")[0])  # jobs, evaluate helpers, data

from dsdg_supply_chain.data import SHARE_COLS  # noqa: E402
from dsdg_supply_chain.labels import supplier_state as true_state  # noqa: E402
from dsdg_supply_chain.simulator import hindsight_plan, truth_for  # noqa: E402

TRUE = true_state(data)
P = PolicyParams()


def true_planning_state(info, win):
    t = TRUE[TRUE.week.isin(win)].copy()
    t["p"] = t.hit.astype(float)
    t["severity"] = t.cap_ratio.where(t.hit_cap, 1.0)
    t["lead"] = t.lead_time_weeks
    t["cost"] = t.landed_cost
    return t[["week", "material", "country", "p", "severity", "lead", "cost", "normal_capacity_units"]]


def run(variant):
    rows = []
    for j in jobs:
        info = data.until(j["info_week"])
        sub = j["sub"].set_index(["week", "material"]).copy()
        if variant in ("oracle_P", "oracle_all"):
            o = data.outcomes.set_index(["week", "material"])
            idx = sub.index.intersection(o.index)
            sub.loc[idx, "disruption_prob"] = o.loc[idx, "disrupted"].astype(float)
            sub.loc[idx, "price_forecast"] = o.loc[idx, "benchmark_price"]
        if variant == "hindsight":
            for m in data.suppliers.material.unique():
                t = truth_for(data, m, j["win"])
                _, u = hindsight_plan(t)
                orders = u.sum(axis=0)
                for w_i, w in enumerate(j["win"]):
                    sub.loc[(w, m), "order_units"] = orders[w_i]
                    sub.loc[(w, m), SHARE_COLS] = 0.0
                    for s, c in enumerate(t.countries):
                        sub.loc[(w, m), f"share_{c}"] = u[s, w_i] / orders[w_i] if orders[w_i] > 0 else 1 / 3
        else:
            st = true_planning_state(info, j["win"]) if variant in ("oracle_policy", "oracle_all") else None
            mat_p = j["mat_p"]
            if st is not None:
                mat_p = data.outcomes.rename(columns={"disrupted": "p"})[["week", "material", "p"]]
            plan = plan_orders(info, j["win"], j["feats"], j["sup"], mat_p, P, state=st)
            plan = plan.set_index(["week", "material"])
            cols = ["order_units", *SHARE_COLS]
            sub.loc[plan.index, cols] = plan[cols]
        ws = score_window(sub.reset_index(), j["base"], data, j["win"])
        rows.append(dict(variant=variant, origin=j["origin"], window=j["window"], **ws.as_dict()))
    return pd.DataFrame(rows)


R = pd.concat([run(v) for v in ["current", "oracle_P", "oracle_policy", "oracle_all", "hindsight"]])
R["type"] = np.where(R.n_shock_weeks >= 30, "shock", "calm")
pd.set_option("display.width", 220)
print(R.groupby(["variant", "type"])[["score", "P", "F", "R", "fill_skill", "diversification"]].mean()
      .round(3).to_string())
print("\noverall mean / worst:")
print(R.groupby("variant").score.agg(["mean", "min"]).round(2).to_string())
R.to_csv("reports/value_of_information.csv", index=False)
