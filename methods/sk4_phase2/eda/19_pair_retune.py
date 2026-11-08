"""Retune the insured specialist's ordering for the HEDGED pair (X6 in DECISIONS.md).

P depends only on the forecast columns, so the hedged P of each side is read from the end-to-end
backtests (reports/backtest_{lean,insured}_hedged.csv) and only the insured ORDERING is searched:
  insured score = rule-planner score - 30 P(unhedged) + 30 P(hedged, shock side)
Objective: mean over windows of max(lean_hedged, insured).

usage: uv run python eda/19_pair_retune.py [n_random]
"""
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

exec(open("eda/11_tune_policy.py").read().split("rng = np.random")[0])
from dsdg_supply_chain.strategies import INSURED_PARAMS  # noqa: E402

idx = ["origin", "window"]
lean = pd.read_csv("reports/backtest_lean_hedged.csv").set_index(idx).score
PI = pd.read_csv("reports/backtest_insured_hedged.csv").set_index(idx).P
RANGES = dict(safety_base=(0.0, 1.5), safety_risk=(0.0, 6.0), cost_temp=(0.0, 20.0), risk_aversion=(0.0, 8.0),
              diversify=(0.0, 1.0), overorder=(0.0, 1.0), diversify_risk=(0.0, 4.0), buffer_mid=(0.0, 6.0),
              taper=(1.0, 8.0))


def pair_score(P):
    r = evaluate(P).set_index(idx)
    s = r.score - 30 * r.P + 30 * PI.loc[r.index]
    m = np.maximum(s, lean.loc[r.index])
    return dict(pair=m.mean(), worst=m.min(), alone=s.mean(), wins=int((s > lean.loc[r.index]).sum())), s


start = PolicyParams(**INSURED_PARAMS)
best_s, s0 = pair_score(start)
print("current insured:", {k: round(v, 2) for k, v in best_s.items()})
best, best_v = start, best_s["pair"]
rng = np.random.default_rng(7)
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    cand = PolicyParams(**{k: float(rng.uniform(*v)) for k, v in RANGES.items()})
    v = pair_score(cand)[0]["pair"]
    if v > best_v:
        best, best_v = cand, v
print(f"after random search: {best_v:.2f}")
for sweep in range(2):
    for k, (lo, hi) in RANGES.items():
        for v in np.linspace(lo, hi, 7):
            c = replace(best, **{k: float(v)})
            sv = pair_score(c)[0]["pair"]
            if sv > best_v + 1e-6:
                best, best_v = c, sv
    print(f"sweep {sweep + 1}: pair {best_v:.2f}")
res, s = pair_score(best)
print("\n== retuned insured:", {k: round(v, 3) for k, v in asdict(best).items()})
print({k: round(v, 2) for k, v in res.items()})
t = pd.DataFrame({"lean": lean.loc[s.index], "insured_old": s0, "insured_new": s,
                  "best_old": np.maximum(s0, lean.loc[s.index]), "best_new": np.maximum(s, lean.loc[s.index])})
print(t.round(1).to_string())
print(t.groupby(level=1).mean().round(2))
