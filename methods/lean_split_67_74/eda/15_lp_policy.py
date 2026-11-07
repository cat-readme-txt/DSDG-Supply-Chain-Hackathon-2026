"""Evaluate the stochastic-programming (LP) policy against policy_v1 on the cached backtest jobs.

usage: uv run python eda/15_lp_policy.py [min_share,risk_mult,risk_floor,n_scenarios,hazard_mult ...]
"""
import sys
import time

import numpy as np
import pandas as pd

exec(open("eda/11_tune_policy.py").read().split("rng = np.random")[0])

from dsdg_supply_chain.data import SHARE_COLS  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, plan_orders_lp  # noqa: E402


def evaluate_lp(P: LPParams, state_fn=None) -> pd.DataFrame:
    rows = []
    for j in jobs:
        info = data.until(j["info_week"])
        st = state_fn(info, j) if state_fn else None
        plan = plan_orders_lp(info, j["win"], j["feats"], j["sup"], P, state=st).set_index(["week", "material"])
        sub = j["sub"].set_index(["week", "material"]).copy()
        cols = ["order_units", *SHARE_COLS]
        sub.loc[plan.index, cols] = plan[cols]
        ws = score_window(sub.reset_index(), j["base"], data, j["win"])
        rows.append(dict(origin=j["origin"], window=j["window"], **ws.as_dict()))
    r = pd.DataFrame(rows)
    r["type"] = np.where(r.n_shock_weeks >= 30, "shock", "calm")
    return r


def report(name, r):
    t = r.groupby("type")[["score", "F", "fill_skill", "diversification"]].mean()
    print(f"{name:42s} mean {r.score.mean():6.2f} worst {r.score.min():6.2f} | calm {t.loc['calm', 'score']:.1f} "
          f"(F {t.loc['calm', 'F']:.2f}) | shock {t.loc['shock', 'score']:.1f} (F {t.loc['shock', 'F']:.2f}, "
          f"fill {t.loc['shock', 'fill_skill']:.2f}) | div {r.diversification.mean():.2f}")


if __name__ == "__main__":
    report("policy_v1 (rule-based, tuned)", (lambda r: r.assign(type=np.where(r.n_shock_weeks >= 30, "shock", "calm")))(evaluate(PolicyParams())))
    configs = sys.argv[1:] or ["0.1,1,0,30"]
    for cfg in configs:
        ms, rm, rf, n, hz = (cfg.split(",") + ["0"])[:5]
        P = LPParams(n_scenarios=int(n), min_share=float(ms), risk_mult=float(rm), risk_floor=float(rf),
                     hazard_mult=float(hz))
        t = time.time()
        r = evaluate_lp(P)
        report(f"LP share={ms} risk={rm} floor={rf} N={n} hazard={hz}", r)
        print(f"   ({time.time() - t:.0f}s)")
