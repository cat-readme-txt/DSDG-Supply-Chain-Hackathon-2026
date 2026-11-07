"""Public-window perfect-information plan and week-221 private baseline.

Weeks 209–221 are released in Phase 2. Public probability/price columns use
their observed labels; their purchasing plan comes from a hindsight inventory
LP. Weeks 222–234 are forecast only from data through week 221.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.simulator import (HOLDING_RATE, SALVAGE_RATE, UNMET_MULT,
                                         hindsight_plan, truth_for)  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402
from dsdg_supply_chain.submission import write_submission  # noqa: E402


def hindsight_with_convention(truth, init: float, avg_cost: float) -> np.ndarray:
    """Exact cost-minimizing supplier units under a specified scorer convention."""
    suppliers, weeks = truth.capacity.shape
    orders = suppliers*weeks
    unmet, inventory = orders, orders+weeks
    total_vars = orders+2*weeks
    cost = np.zeros(total_vars)
    for s in range(suppliers):
        for t in range(weeks):
            late = t+int(truth.lead[s, t]) >= weeks
            cost[s*weeks+t] = truth.cost[s, t] - (SALVAGE_RATE*avg_cost if late else 0)
    cost[unmet:unmet+weeks] = UNMET_MULT*avg_cost
    cost[inventory:inventory+weeks] = HOLDING_RATE*avg_cost
    cost[inventory+weeks-1] -= SALVAGE_RATE*avg_cost
    a_eq = np.zeros((weeks, total_vars))
    rhs = -truth.demand.copy()
    rhs[0] += init
    for t in range(weeks):
        a_eq[t, inventory+t] = 1
        if t:
            a_eq[t, inventory+t-1] = -1
        a_eq[t, unmet+t] = -1
    for s in range(suppliers):
        for t in range(weeks):
            arrival = t+int(truth.lead[s, t])
            if arrival < weeks:
                a_eq[arrival, s*weeks+t] -= 1
    bounds = ([(0, truth.capacity[s, t]) for s in range(suppliers) for t in range(weeks)]
              + [(0, d) for d in truth.demand] + [(0, None)]*weeks)
    result = linprog(cost, A_eq=a_eq, b_eq=rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:orders].reshape(suppliers, weeks)


def public_hindsight(data, convention: str = "sum5") -> pd.DataFrame:
    weeks = range(209, 222)
    actual = data.outcomes[data.outcomes.week.isin(weeks)]
    rows = []
    for material in sorted(data.suppliers.material.unique()):
        truth = truth_for(data, material, weeks)
        if convention == "sum5":
            _, units = hindsight_plan(truth)
        elif convention == "5mean_spot":
            px = actual[actual.material == material].benchmark_price.mean()
            units = hindsight_with_convention(truth, 5*truth.demand.mean(), float(px))
        elif convention == "5mean_supweek":
            units = hindsight_with_convention(truth, 5*truth.demand.mean(), truth.avg_cost)
        else:
            raise ValueError(convention)
        total = units.sum(axis=0)
        for t, week in enumerate(weeks):
            row = dict(week=week, material=material,
                       id=f"w{week}_{material}",
                       disruption_prob=float(actual[(actual.material == material)
                                                    & (actual.week == week)].disrupted.iloc[0]),
                       price_forecast=float(actual[(actual.material == material)
                                                   & (actual.week == week)].benchmark_price.iloc[0]),
                       order_units=float(total[t]))
            row.update({col: 0. for col in SHARE_COLS})
            for s, country in enumerate(truth.countries):
                row[f"share_{country}"] = (float(units[s, t]/total[t])
                                           if total[t] > 1e-8 else 1/len(truth.countries))
            rows.append(row)
    return pd.DataFrame(rows)


def full_candidate(data, convention: str = "sum5") -> pd.DataFrame:
    private = STRATEGIES["lean_split"](data, range(209, 235))
    private = private[private.week >= 222].copy()
    public = public_hindsight(data, convention)
    full = pd.concat([public, private], ignore_index=True)
    full = full.sort_values(["week", "material"]).reset_index(drop=True)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "generated_candidates" / "phase2_hindsight_lean_w221.csv")
    parser.add_argument("--convention", choices=("sum5", "5mean_spot", "5mean_supweek"),
                        default="sum5")
    args = parser.parse_args()
    data = load_data(ROOT / "data" / "Phase 2 data")
    if data.last_week != 221:
        raise ValueError("Expected Phase 2 cutoff at week 221")
    candidate = full_candidate(data, args.convention)
    write_submission(candidate, args.output)
    from pipeline import CompetitionData
    CompetitionData(ROOT / "data" / "Phase 2 data").validate_file(args.output)
    public = candidate[candidate.week.between(209, 221)]
    sample = naive(data.until(208), range(209, 222))
    scored = score_window(public, sample, data, range(209, 222))
    report = dict(local_score=scored.as_dict(), output=str(args.output), convention=args.convention)
    (ROOT / "reports" / f"phase2_public_oracle_{args.convention}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(args.output)
    print("local public score", scored.score, "P", scored.P, "F", scored.F,
          "R", scored.R, "diversity", scored.diversification)


if __name__ == "__main__":
    main()
