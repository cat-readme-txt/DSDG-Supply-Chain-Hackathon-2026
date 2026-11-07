"""Chronological test of a portfolio buffer triggered by shared lane risk.

The scored LP minimizes expected cost material by material. This test instead
looks for correlated cobalt and natural-gas exposure to Orinoque's SOUTH lane,
then raises their early orders together to protect the pooled fill objective.
Only released data at each origin are used in the trigger and plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402

MATERIALS = ("cobalt", "natural_gas")
ORIGINS = tuple(range(52, 196, 4))
SCALES = (1.05, 1.1, 1.2, 1.3)


def trigger(sub: pd.DataFrame, weeks: range) -> tuple[str, ...]:
    early = sub[sub.week.isin(range(weeks[0], weeks[0] + 5))]
    risk = early[early.material.isin(MATERIALS)].groupby("material").disruption_prob.max()
    return tuple(m for m in MATERIALS if risk.get(m, 0) >= .25)


def modify(sub: pd.DataFrame, weeks: range, scale: float, active: tuple[str, ...]) -> pd.DataFrame:
    result = sub.copy()
    mask = result.material.isin(active) & result.week.isin(range(weeks[0], weeks[0] + 6))
    result.loc[mask, "order_units"] = result.loc[mask, "order_units"].astype(float) * scale
    return result


def main() -> None:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 14)
        info = data.until(origin)
        base = STRATEGIES["lean_split"](info, weeks)
        naive_sub = naive(info, weeks)
        active = trigger(base, weeks)
        base_score = score_window(base, naive_sub, data, weeks)
        record = dict(origin=origin, active=active, base=base_score.as_dict(),
                      variants={})
        if active:
            for scale in SCALES:
                variant = modify(base, weeks, scale, active)
                score = score_window(variant, naive_sub, data, weeks)
                record["variants"][str(scale)] = dict(score=score.score,
                                                        delta=score.score-base_score.score,
                                                        cost_delta=score.cost-base_score.cost,
                                                        fill_delta=score.fill_rate-base_score.fill_rate)
        rows.append(record)
        print(origin, "active", active, "base", round(base_score.score, 2),
              "variants", {k: round(v["delta"], 2) for k, v in record["variants"].items()},
              flush=True)
    output = ROOT / "reports" / "correlated_lane_buffer_w208.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
