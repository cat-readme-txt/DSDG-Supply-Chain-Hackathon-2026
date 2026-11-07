"""Improve released-window supplier shares with a convex quadratic program.

The scored 89.45 policy already knows public supply and demand. Keep its order
quantities, forecasts, and private half fixed. For each material, minimize true
landed plus holding cost and a score-scaled HHI penalty, subject to every public
demand week being fully supplied. This is a distinct optimization from the
cost-only hindsight LP and does not assume the official scorer is our replica.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import policy_arrays  # noqa: E402
from dsdg_supply_chain.simulator import truth_for  # noqa: E402
from dsdg_supply_chain.submission import VALUE_COLS, read_submission  # noqa: E402


def improve_material(frame: pd.DataFrame, truth, gap: float, diversity_scale: float,
                     avg_cost: str = "supweek", init: str = "sum5") -> tuple[pd.DataFrame, dict]:
    """Find the best share vector for fixed total orders and full demand fill."""
    weeks = truth.weeks
    orders, original = policy_arrays(frame, truth)
    active = np.flatnonzero(orders > 1e-6)
    if len(active) == 0:
        return frame, dict(material=truth.material, changed=0)
    s_count, t_count = truth.capacity.shape
    if avg_cost == "supweek":
        A = truth.avg_cost
    else:
        raise ValueError(avg_cost)
    init_stock = truth.demand[:5].sum() if init == "sum5" else 5 * truth.demand.mean()
    x0 = original[:, active].T.reshape(-1)
    # A filled order's exact marginal cost when all demand is met.
    marginal = np.zeros((len(active), s_count))
    for j, t in enumerate(active):
        for s in range(s_count):
            arrival = t + int(truth.lead[s, t])
            hold_weeks = max(t_count - arrival, 0)
            marginal[j, s] = orders[t] * (truth.cost[s, t] + .02 * A * hold_weeks - .70 * A)
    # A score point has the same scale as the public Financial and Diversity weights.
    cost_grad = 40 * marginal.ravel() / gap
    div_weight = diversity_scale * 9 / (130 * (2 / 3))
    def objective(x):
        return cost_grad @ x + div_weight * (x @ x)
    def gradient(x):
        return cost_grad + 2 * div_weight * x
    # Cumulative arrivals must cover cumulative demand in every week.
    cum_arrivals = np.zeros((t_count, len(x0)))
    for j, t in enumerate(active):
        for s in range(s_count):
            arrival = t + int(truth.lead[s, t])
            if arrival < t_count:
                cum_arrivals[arrival:, j*s_count+s] = orders[t]
    rhs = np.cumsum(truth.demand) - init_stock
    # The published CSV rounds quantities and shares. A few rows therefore
    # miss demand by under three units even though the underlying LP did not.
    # Preserve the baseline's cumulative delivery whenever that is tighter
    # than the ideal full-fill constraint.
    baseline_arrivals = np.zeros(t_count)
    for j, t in enumerate(active):
        for s in range(s_count):
            arrival = t + int(truth.lead[s, t])
            if arrival < t_count:
                baseline_arrivals[arrival] += min(orders[t] * original[s, t], truth.capacity[s, t])
    rhs = np.minimum(rhs, np.cumsum(baseline_arrivals) - 1e-4)
    row_sums = np.zeros((len(active), len(x0)))
    for j in range(len(active)):
        row_sums[j, j*s_count:(j+1)*s_count] = 1
    bounds = [(0., min(1., float(truth.capacity[s, t]/orders[t])))
              for t in active for s in range(s_count)]
    result = minimize(objective, x0, jac=gradient, method="SLSQP", bounds=bounds,
                      constraints=[dict(type="eq", fun=lambda x: row_sums @ x - 1,
                                        jac=lambda x: row_sums),
                                   dict(type="ineq", fun=lambda x: cum_arrivals @ x - rhs,
                                        jac=lambda x: cum_arrivals)],
                      options=dict(maxiter=2000, ftol=1e-11))
    if not result.success:
        raise RuntimeError(f"{truth.material}: {result.message}")
    x = result.x.reshape(len(active), s_count)
    out = frame.copy()
    for j, t in enumerate(active):
        mask = (out.material == truth.material) & (out.week == weeks[t])
        for s, country in enumerate(truth.countries):
            out.loc[mask, f"share_{country}"] = x[j, s]
    # Check feasibility with the real simulator, not only the QP constraints.
    new_orders, new_shares = policy_arrays(out, truth)
    from dsdg_supply_chain.simulator import simulate
    sim = simulate(truth, new_orders, new_shares)
    old_sim = simulate(truth, orders, original)
    if np.sum(sim.served) < np.sum(old_sim.served) - 1e-3:
        raise AssertionError(f"{truth.material}: QP reduced demand served")
    changed = int(np.count_nonzero(np.abs(x-x0.reshape(len(active), s_count)).max(axis=1) > 1e-6))
    return out, dict(material=truth.material, changed=changed,
                     objective_gain=float(objective(x0)-objective(result.x)),
                     cost=float(sim.total_cost))


def repair_stockouts(frame: pd.DataFrame, data) -> tuple[pd.DataFrame, list[dict]]:
    """Add a few units where CSV rounding made the hindsight plan just short."""
    from dsdg_supply_chain.simulator import simulate
    out = frame.copy()
    repairs = []
    for material in sorted(data.suppliers.material.unique()):
        truth = truth_for(data, material, range(209, 222))
        for _ in range(5):
            orders, shares = policy_arrays(out, truth)
            sim = simulate(truth, orders, shares)
            gaps = truth.demand - sim.served
            if gaps.max() < 1e-8:
                break
            missed = int(np.flatnonzero(gaps > 1e-8)[0])
            need = int(np.ceil(gaps[missed] + .1))
            options = []
            for t in np.flatnonzero(orders > 0):
                for s, country in enumerate(truth.countries):
                    if t + truth.lead[s, t] > missed:
                        continue
                    room = truth.capacity[s, t] - orders[t] * shares[s, t]
                    if room > need + .01:
                        marginal = truth.cost[s, t] + .02 * truth.avg_cost * (13 - t - truth.lead[s, t])
                        options.append((marginal, int(t), s, country))
            if not options:
                raise RuntimeError(f"No spare supplier capacity for {material} in week {truth.weeks[missed]}")
            _, t, s, country = min(options)
            units = orders[t] * shares[:, t]
            units[s] += need
            q = orders[t] + need
            mask = (out.material == material) & (out.week == truth.weeks[t])
            out.loc[mask, "order_units"] = q
            for k, c in enumerate(truth.countries):
                out.loc[mask, f"share_{c}"] = units[k] / q
            repairs.append(dict(material=material, order_week=int(truth.weeks[t]),
                                shortage_week=int(truth.weeks[missed]), added_units=need,
                                country=country))
        else:
            raise RuntimeError(f"Could not repair {material}")
    return out, repairs


def write_precise(frame: pd.DataFrame, path: Path) -> None:
    out = frame[["id", *VALUE_COLS]].copy()
    out["order_units"] = out.order_units.round().astype(int)
    out["disruption_prob"] = out.disruption_prob.round(4)
    out["price_forecast"] = out.price_forecast.round(4)
    shares = out[SHARE_COLS].to_numpy(float).round(8)
    top = shares.argmax(axis=1)
    shares[np.arange(len(shares)), top] += 1 - shares.sum(axis=1)
    out[SHARE_COLS] = shares.round(8)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "submissions" / "submission89.45.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "generated_candidates" / "phase2_public_diversification_qp_w221.csv")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--diversity-scale", type=float, default=.5)
    parser.add_argument("--init", choices=("sum5", "5mean"), default="sum5")
    args = parser.parse_args()
    data = load_data(ROOT / "data" / "Phase 2 data")
    frame = read_submission(args.base)
    public = frame[frame.week.between(209, 221)].copy()
    # The local scorer's denominator is a fixed scale. Its exact value matters
    # only for the cost/diversity tradeoff; we evaluate the resulting file under
    # multiple scorer conventions afterwards.
    gap = 5728480.540944307 - 4256386.797304769
    changes = []
    for material in sorted(data.suppliers.material.unique()):
        truth = truth_for(data, material, range(209, 222))
        public, summary = improve_material(public, truth, gap, args.diversity_scale, init=args.init)
        changes.append(summary)
    public, repairs = repair_stockouts(public, data)
    full = pd.concat([public, frame[frame.week >= 222]], ignore_index=True).sort_values(["week", "material"])
    write_precise(full, args.output)
    # Re-read the actual submission to catch shortages caused by serializing
    # fractional shares. A single extra unit is enough in these cases.
    for _ in range(3):
        serialized = read_submission(args.output)
        public, extra = repair_stockouts(serialized[serialized.week <= 221], data)
        if not extra:
            break
        repairs.extend(extra)
        full = pd.concat([public, serialized[serialized.week >= 222]], ignore_index=True).sort_values(["week", "material"])
        write_precise(full, args.output)
    else:
        raise RuntimeError("Serializing shares still creates shortages")
    from pipeline import CompetitionData
    CompetitionData(ROOT / "data" / "Phase 2 data").validate_file(args.output)
    from dsdg_supply_chain.scoring import score_window
    from dsdg_supply_chain.strategies import naive
    reference = naive(data.until(208), range(209, 222))
    before = score_window(frame, reference, data, range(209, 222))
    after = score_window(read_submission(args.output), reference, data, range(209, 222))
    report = dict(base=str(args.base), output=str(args.output), init=args.init,
                  diversity_scale=args.diversity_scale, materials=changes, repairs=repairs,
                  local_before=before.as_dict(), local_after=after.as_dict(),
                  local_delta=after.score-before.score)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
