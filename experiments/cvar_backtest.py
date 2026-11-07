"""Chronological direct-score check of explicit CVaR scenario LP variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, _with_plan, hedge_forecasts, naive  # noqa: E402

from cvar_lp import plan_orders_cvar  # noqa: E402

ORIGINS = (52, 78, 104, 130, 156, 182)
VARIANTS = ((.1, .8), (.3, .8), (.5, .8), (.3, .9))
PARAMS = LPParams(n_scenarios=30, min_share=0, hazard_mult=0,
                  cap_derate=.96, cheap_premium=.025, cheap_share=.3)


def cvar_strategy(info, weeks: range, weight: float, alpha: float):
    plan = _with_plan(info, weeks, lambda i, w, f, s, m:
                      plan_orders_cvar(i, w, f, s, PARAMS, weight, alpha))
    return hedge_forecasts(plan, info, "calm")


def main() -> None:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    records = []
    for origin in ORIGINS:
        info = data.until(origin)
        for offset in (1, 14):
            weeks = range(origin+offset, origin+offset+13)
            if weeks[-1] > data.last_week:
                continue
            # Each historical window is planned from its own information origin.
            local_origin = weeks[0]-1
            local_info = data.until(local_origin)
            naive_sub = naive(local_info, weeks)
            base = STRATEGIES["lean_split"](local_info, weeks)
            base_score = score_window(base, naive_sub, data, weeks)
            row = dict(origin=local_origin, window_start=weeks[0],
                       base=base_score.as_dict(), variants={})
            for weight, alpha in VARIANTS:
                variant = cvar_strategy(local_info, weeks, weight, alpha)
                score = score_window(variant, naive_sub, data, weeks)
                row["variants"][f"w{weight}_a{alpha}"] = dict(
                    score=score.score, delta=score.score-base_score.score,
                    cost_delta=score.cost-base_score.cost,
                    fill_delta=score.fill_rate-base_score.fill_rate,
                    diversity_delta=score.diversification-base_score.diversification)
            records.append(row)
            print("origin", local_origin, "base", round(base_score.score, 2),
                  "deltas", {name: round(v["delta"], 2)
                             for name, v in row["variants"].items()}, flush=True)
    output = ROOT / "reports" / "cvar_backtest_w208.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    for name in records[0]["variants"]:
        for label, subset in (("development", records[:8]), ("later_holdout", records[8:])):
            deltas = np.array([row["variants"][name]["delta"] for row in subset])
            print(name, label, "mean", round(deltas.mean(), 3),
                  "median", round(np.median(deltas), 3), "positive", int((deltas>0).sum()),
                  "of", len(deltas))
    print(output)


if __name__ == "__main__":
    main()
