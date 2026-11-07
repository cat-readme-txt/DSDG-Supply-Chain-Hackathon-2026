"""Cost-service-diversification frontier around the scored lean-split policy.

Move only the two cheapest supplier shares toward an equal pair, preserve any
third supplier share, and cap target allocations at normal supplier capacity.
This is a one-dimensional HHI reward frontier with every other CSV field fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402

ORIGINS = (52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195)
STRENGTHS = (.25, .5, 1.)


def pair_allocations(info) -> dict[str, tuple[str, str, float, float]]:
    last = info.supply_history[info.supply_history.week == info.last_week]
    costs = last.set_index(["material", "country"]).landed_cost
    result = {}
    for material, group in info.suppliers.groupby("material"):
        ranked = group.assign(cost=group.country.map(lambda c: costs.loc[(material, c)]))
        ranked = ranked.sort_values("cost")
        low, high = ranked.iloc[:2].itertuples(index=False)
        if high.cost <= low.cost*1.05:
            result[material] = (low.country, high.country,
                                float(low.normal_capacity_units), float(high.normal_capacity_units))
    return result


def apply_frontier(base: pd.DataFrame, info, strength: float) -> pd.DataFrame:
    output = base.copy()
    pairs = pair_allocations(info)
    for material, (first, second, cap1, cap2) in pairs.items():
        mask = (output.material == material) & (output.order_units > 0)
        first_col, second_col = f"share_{first}", f"share_{second}"
        for index in output.index[mask]:
            quantity = float(output.at[index, "order_units"])
            a = float(output.at[index, first_col])
            b = float(output.at[index, second_col])
            total = a+b
            target = (1-strength)*a + strength*total/2
            target = min(target, .96*cap1/quantity)
            target = max(target, total-.96*cap2/quantity)
            output.at[index, first_col] = target
            output.at[index, second_col] = total-target
    return output


def main() -> None:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    records = []
    for origin in ORIGINS:
        info = data.until(origin)
        weeks = range(origin+1, origin+14)
        base = STRATEGIES["lean_split"](info, weeks)
        naive_sub = naive(info, weeks)
        baseline = score_window(base, naive_sub, data, weeks)
        record = dict(origin=origin, base=baseline.as_dict(), variants={})
        for strength in STRENGTHS:
            candidate = apply_frontier(base, info, strength)
            score = score_window(candidate, naive_sub, data, weeks)
            record["variants"][str(strength)] = dict(
                delta=score.score-baseline.score,
                cost_delta=score.cost-baseline.cost,
                fill_delta=score.fill_rate-baseline.fill_rate,
                diversity_delta=score.diversification-baseline.diversification)
        records.append(record)
        print(origin, {k: round(v["delta"], 2) for k, v in record["variants"].items()}, flush=True)
    output = ROOT / "reports" / "diversification_frontier_w208.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    for strength in STRENGTHS:
        for label, subset in (("development", records[:8]), ("later_holdout", records[8:])):
            values = np.array([row["variants"][str(strength)]["delta"] for row in subset])
            print(strength, label, "mean", round(values.mean(), 3),
                  "median", round(np.median(values), 3), "positive", int((values>0).sum()))
    print(output)


if __name__ == "__main__":
    main()
