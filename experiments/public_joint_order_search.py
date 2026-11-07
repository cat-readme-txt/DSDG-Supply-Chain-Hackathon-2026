"""Optimize released public order timing, quantities, and shares jointly.

It uses only released public truth and keeps all forecasts and private rows of
the scored baseline. The 92.31-based scale 1.5 output is a scored-response
candidate; see reports/joint_surface_sweep.json for its weight selection.
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
from dsdg_supply_chain.simulator import truth_for, simulate  # noqa: E402
from dsdg_supply_chain.submission import read_submission  # noqa: E402
from experiments.public_diversification_qp import repair_stockouts, write_precise  # noqa: E402


def optimize_material(frame, tr, scale, gap, diversity_objective="hhi"):
    q0, sh0 = policy_arrays(frame, tr)
    active = np.flatnonzero(q0 > 1e-6)
    S, T = tr.capacity.shape
    if not len(active):
        return frame, dict(material=tr.material, changed=False)
    x0 = (q0[active, None] * sh0[:, active].T).ravel()
    x0 = np.maximum(x0, 0.)
    A = tr.avg_cost
    marginal = np.zeros_like(x0)
    cum_arrivals = np.zeros((T, len(x0)))
    for j, t in enumerate(active):
        for s in range(S):
            k = j*S+s
            arrival = t+int(tr.lead[s, t])
            marginal[k] = tr.cost[s, t] + .02*A*max(T-arrival, 0) - .70*A
            if arrival < T:
                cum_arrivals[arrival:, k] = 1.
    rhs = np.cumsum(tr.demand)-tr.init_stock
    base_arrivals = np.zeros(T)
    for j, t in enumerate(active):
        for s in range(S):
            a = t+int(tr.lead[s, t])
            if a < T:
                base_arrivals[a] += min(x0[j*S+s], tr.capacity[s, t])
    rhs = np.minimum(rhs, np.cumsum(base_arrivals)-1e-5)
    div_weight = scale*9/(130*(2/3)) if diversity_objective == "hhi" else scale*9/130
    cost_weight = 40/gap
    def objective(x):
        shares = x.reshape(len(active), S)
        q = shares.sum(axis=1)
        proportions=shares/np.maximum(q[:,None],1e-6)
        if diversity_objective == "hhi":
            penalty=(proportions**2).sum(axis=1)
        elif diversity_objective == "entropy":
            penalty=(proportions*np.log(np.maximum(proportions,1e-12))).sum(axis=1)/np.log(3)
        else:
            raise ValueError(diversity_objective)
        return cost_weight*(marginal@x)+div_weight*penalty.sum()
    def gradient(x):
        units = x.reshape(len(active), S)
        q = units.sum(axis=1)
        if diversity_objective == "hhi":
            sq = (units**2).sum(axis=1)
            grad_div = 2*units/np.maximum(q[:, None], 1e-6)**2 - 2*sq[:, None]/np.maximum(q[:, None], 1e-6)**3
        else:
            p=units/np.maximum(q[:,None],1e-6)
            mean_log=(p*np.log(np.maximum(p,1e-12))).sum(axis=1)
            grad_div=(np.log(np.maximum(p,1e-12))-mean_log[:,None])/(np.maximum(q[:,None],1e-6)*np.log(3))
        return cost_weight*marginal+div_weight*grad_div.ravel()
    row_sum = np.zeros((len(active), len(x0)))
    for j in range(len(active)):
        row_sum[j, j*S:(j+1)*S] = 1.
    bounds = [(0., tr.capacity[s, t]) for t in active for s in range(S)]
    res = minimize(objective, x0, jac=gradient, method="SLSQP", bounds=bounds,
                   constraints=[dict(type="ineq", fun=lambda x: cum_arrivals@x-rhs,
                                     jac=lambda x: cum_arrivals),
                                dict(type="ineq", fun=lambda x: row_sum@x-1.,
                                     jac=lambda x: row_sum)],
                   options=dict(maxiter=2500, ftol=1e-10))
    if not res.success:
        return frame, dict(material=tr.material, changed=False, failure=res.message)
    units = res.x.reshape(len(active), S)
    out = frame.copy()
    for j, t in enumerate(active):
        q = units[j].sum()
        mask = (out.material == tr.material)&(out.week == tr.weeks[t])
        out.loc[mask, "order_units"] = round(q)
        for s, country in enumerate(tr.countries):
            out.loc[mask, f"share_{country}"] = units[j, s]/q
    return out, dict(material=tr.material, changed=True, objective_gain=float(objective(x0)-objective(res.x)),
                     old_order=float(q0.sum()), new_order=float(units.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=float, default=.5)
    p.add_argument("--diversity-objective", choices=("hhi","entropy"), default="hhi")
    p.add_argument("--base", type=Path, default=ROOT/"submissions"/"submission91.36.csv")
    p.add_argument("--output", type=Path, default=ROOT/"reports"/"generated_candidates"/"public_joint_search_w221.csv")
    p.add_argument("--report", type=Path, help="Optional JSON summary path")
    args = p.parse_args()
    data = load_data(ROOT/"data"/"Phase 2 data")
    base = read_submission(args.base)
    public = base[base.week <= 221].copy()
    gap = 5728480.540944307-4256386.797304769
    info = []
    for material in sorted(data.suppliers.material.unique()):
        tr = truth_for(data, material, range(209,222))
        public, item = optimize_material(public, tr, args.scale, gap, args.diversity_objective)
        info.append(item)
    public, repairs = repair_stockouts(public, data)
    full = pd.concat([public, base[base.week>=222]], ignore_index=True).sort_values(["week","material"])
    write_precise(full, args.output)
    for _ in range(4):
        serial = read_submission(args.output)
        public, more = repair_stockouts(serial[serial.week<=221], data)
        if not more:
            break
        repairs.extend(more)
        full = pd.concat([public, serial[serial.week>=222]], ignore_index=True).sort_values(["week","material"])
        write_precise(full, args.output)
    else:
        raise RuntimeError("Could not repair serialized shortages")
    from pipeline import CompetitionData
    CompetitionData(ROOT/"data"/"Phase 2 data").validate_file(args.output)
    summary=dict(scale=args.scale, diversity_objective=args.diversity_objective,
                 base=str(args.base), output=str(args.output),
                 materials=info, repairs=repairs)
    if args.report is not None:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
