"""Two final submissions: Kaggle scores the better of the selected ones on the private board.

Fix the lean specialist (LP, no minimum share, no background shocks; best on calm windows) and
tune the insured specialist (rule-based planner with a tapering mid-window buffer) to maximise
mean over windows of max(lean, insured). Random search, then coordinate refinement.

usage: uv run python eda/16_pair_tune.py [n_random]
"""
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

exec(open("eda/15_lp_policy.py").read().split("if __name__")[0])

LEAN = LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0)
lean = evaluate_lp(LEAN).set_index(["origin", "window"]).score
RANGES = dict(safety_base=(0.0, 1.5), safety_risk=(0.0, 4.0), cost_temp=(0.0, 20.0), risk_aversion=(0.0, 8.0),
              diversify=(0.0, 1.0), overorder=(0.0, 1.0), diversify_risk=(0.0, 3.0), buffer_mid=(0.0, 6.0),
              taper=(1.0, 8.0))


def pair_score(P):
    r = evaluate(P).set_index(["origin", "window"])
    m = np.maximum(r.score, lean.loc[r.index])
    return dict(pair=m.mean(), pair_worst=m.min(), alone=r.score.mean(),
                insured_wins=int((r.score > lean.loc[r.index]).sum()), **asdict(P)), r


rng = np.random.default_rng(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
start = replace(PolicyParams(), buffer_mid=4.0, taper=4.0)
res = [pair_score(start)[0]]
for _ in range(n):
    res.append(pair_score(PolicyParams(**{k: float(rng.uniform(*v)) for k, v in RANGES.items()}))[0])
R = pd.DataFrame(res).sort_values("pair", ascending=False)
pd.set_option("display.width", 250)
print(f"lean alone: mean {lean.mean():.2f}")
print(R.head(8).round(2).to_string(index=False))
best = PolicyParams(**{k: R.iloc[0][k] for k in RANGES})
best_s = R.iloc[0]["pair"]
for sweep in range(2):
    for k, (lo, hi) in RANGES.items():
        for v in np.linspace(lo, hi, 7):
            s, _ = pair_score(replace(best, **{k: float(v)}))
            if s["pair"] > best_s + 1e-6:
                best, best_s = replace(best, **{k: float(v)}), s["pair"]
    print(f"sweep {sweep + 1}: pair {best_s:.3f}")
s, r = pair_score(best)
print("\n== insured specialist:", {k: round(v, 3) for k, v in asdict(best).items()})
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k in ("pair", "pair_worst", "alone", "insured_wins")})
both = pd.DataFrame({"lean": lean.loc[r.index], "insured": r.score, "best": np.maximum(r.score, lean.loc[r.index]),
                     "shock_weeks": r.n_shock_weeks})
print(both.round(1).to_string())
print(both.groupby(level=1)[["lean", "insured", "best"]].mean().round(2))
