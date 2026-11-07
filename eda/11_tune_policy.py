"""Tune the ordering policy's knobs on the backtest windows.

Forecasts (steps 1-2) are computed once per information week and cached; each candidate set of
PolicyParams then only re-runs the planner and the scorer. Search: random search, then a
coordinate refinement around the best point. Objective: mean score over all public + private
windows; the worst window is reported too, to spot fragile settings.

usage: uv run python eda/11_tune_policy.py [n_random]
"""
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from dsdg_supply_chain.backtest import origins
from dsdg_supply_chain.data import WINDOW, load_data
from dsdg_supply_chain.policy import PolicyParams, plan_orders
from dsdg_supply_chain.scoring import score_window
from dsdg_supply_chain.strategies import _apply_forecasts, _forecasts, naive

CACHE = Path("reports/policy_cache.pkl")
RANGES = dict(safety_base=(0.0, 1.5), safety_risk=(0.0, 6.0), cost_temp=(0.0, 30.0),
              risk_aversion=(0.0, 8.0), diversify=(0.0, 1.0), overorder=(0.0, 1.0),
              diversify_risk=(0.0, 4.0))
data = load_data()


def build_cache():
    jobs = []
    for o in origins(data):
        weeks = range(o + 1, o + 2 * WINDOW + 1)
        base = naive(data.until(o), weeks)
        for name, info_week, win in [("public", o, range(o + 1, o + WINDOW + 1)),
                                     ("private", o + WINDOW, range(o + WINDOW + 1, o + 2 * WINDOW + 1))]:
            info = data.until(info_week)
            feats, sup, mat_p, px = _forecasts(info, weeks)
            sub = _apply_forecasts(naive(info, weeks), mat_p, px)
            jobs.append(dict(origin=o, window=name, win=win, info_week=info_week, base=base, sub=sub,
                             feats=feats[feats.week == info_week], sup=sup[sup.week.isin(win)], mat_p=mat_p))
    CACHE.write_bytes(pickle.dumps(jobs))
    return jobs


jobs = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build_cache()


def evaluate(P: PolicyParams) -> pd.DataFrame:
    rows = []
    for j in jobs:
        info = data.until(j["info_week"])
        plan = plan_orders(info, j["win"], j["feats"], j["sup"], j["mat_p"], P).set_index(["week", "material"])
        sub = j["sub"].set_index(["week", "material"]).copy()
        cols = ["order_units", *[c for c in sub if c.startswith("share_")]]
        sub.loc[plan.index, cols] = plan[cols]
        ws = score_window(sub.reset_index(), j["base"], data, j["win"])
        rows.append(dict(origin=j["origin"], window=j["window"], **ws.as_dict()))
    return pd.DataFrame(rows)


def summary(P):
    r = evaluate(P)
    return dict(mean=r.score.mean(), worst=r.score.min(), F=r.F.mean(), R=r.R.mean(), **asdict(P)), r


rng = np.random.default_rng(0)
n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 40
START = PolicyParams(safety_base=0.5, safety_risk=0.0, cost_temp=5.0, risk_aversion=4.0, diversify=0.0,
                     overorder=1.0)   # best from the first tuning run
results = [summary(PolicyParams())[0], summary(START)[0]]
for _ in range(n_random):
    P = PolicyParams(**{k: float(rng.uniform(*v)) for k, v in RANGES.items()})
    results.append(summary(P)[0])
res = pd.DataFrame(results).sort_values("mean", ascending=False)
pd.set_option("display.width", 220)
print("== random search (top 10)")
print(res.head(10).round(3).to_string(index=False))

# coordinate refinement around the best
best = PolicyParams(**{k: res.iloc[0][k] for k in RANGES})
best_score = res.iloc[0]["mean"]
for sweep in range(2):
    for k, (lo, hi) in RANGES.items():
        for v in np.linspace(lo, hi, 7):
            cand = replace(best, **{k: float(v)})
            s, _ = summary(cand)
            if s["mean"] > best_score + 1e-6:
                best, best_score = cand, s["mean"]
    print(f"after sweep {sweep + 1}: mean {best_score:.3f}  {asdict(best)}")

s, r = summary(best)
print("\n== best params:", {k: round(v, 3) for k, v in asdict(best).items()})
print(r.groupby("window")[["score", "P", "F", "R", "fill_skill", "diversification"]].mean().round(3).to_string())
print(r[["origin", "window", "start", "n_shock_weeks", "score", "F", "R"]].round(3).to_string(index=False))
Path("reports/policy_best.txt").write_text(repr(asdict(best)))
