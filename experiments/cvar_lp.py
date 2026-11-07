"""Scenario LP with an explicit conditional-value-at-risk cost term.

The archived planner minimizes mean simulated cost independently by material.
This variant minimizes (1-lambda) mean cost + lambda CVaR_alpha(cost), while
retaining the same fill, capacity, arrival, and near-cost-share constraints.
The extra variables eta and z[n] enforce z[n] >= cost[n] - eta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, WINDOW  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, draw_scenarios, shock_history  # noqa: E402
from dsdg_supply_chain.simulator import HOLDING_RATE, INIT_STOCK_WEEKS, SALVAGE_RATE, UNMET_MULT  # noqa: E402


def solve_window_cvar(demand: np.ndarray, sc: dict, params: LPParams,
                      tail_weight: float, alpha: float) -> np.ndarray:
    if not (0 <= tail_weight <= 1 and 0 < alpha < 1):
        raise ValueError("Invalid CVaR parameters")
    hit, capn, ratio, lead, cost = (sc[k] for k in ("hit", "cap", "cap_ratio", "lead", "cost"))
    n_scenarios, suppliers, weeks = hit.shape
    if cost.ndim == 2:
        cost = np.broadcast_to(cost, (n_scenarios, suppliers, weeks))
    avg_cost = float(cost.mean())
    n_orders = suppliers * weeks
    n_fills = n_scenarios * suppliers * weeks
    n_stock = n_scenarios * weeks
    n_unmet = n_scenarios * weeks
    eta = n_orders+n_fills+n_stock+n_unmet
    z_start = eta+1
    variables = z_start+n_scenarios

    def order_i(s, t): return s*weeks+t
    def fill_i(n, s, t): return n_orders+(n*suppliers+s)*weeks+t
    def stock_i(n, t): return n_orders+n_fills+n*weeks+t
    def unmet_i(n, t): return n_orders+n_fills+n_stock+n*weeks+t

    # Unscaled scenario loss: purchase, holding, shortage, and salvage.
    loss = np.zeros((n_scenarios, variables))
    for n in range(n_scenarios):
        for s in range(suppliers):
            for t in range(weeks):
                late = t+int(lead[n, s, t]) >= weeks
                loss[n, fill_i(n, s, t)] = cost[n, s, t] - (SALVAGE_RATE*avg_cost if late else 0)
        for t in range(weeks):
            loss[n, stock_i(n, t)] = HOLDING_RATE*avg_cost
            loss[n, unmet_i(n, t)] = UNMET_MULT*avg_cost
        loss[n, stock_i(n, weeks-1)] -= SALVAGE_RATE*avg_cost
    objective = (1-tail_weight)*loss.sum(axis=0)/n_scenarios
    objective[eta] = tail_weight
    objective[z_start:] = tail_weight / ((1-alpha)*n_scenarios)

    # Inventory flow and exact fills when no shock occurs.
    eq_row, eq_col, eq_val, eq_rhs = [], [], [], []
    row_i = 0
    init_stock = demand[:INIT_STOCK_WEEKS].sum()
    arrival = [[[] for _ in range(weeks)] for _ in range(n_scenarios)]
    for n in range(n_scenarios):
        for s in range(suppliers):
            for t in range(weeks):
                target = t+int(lead[n, s, t])
                if target < weeks:
                    arrival[n][target].append(fill_i(n, s, t))
    for n in range(n_scenarios):
        for t in range(weeks):
            eq_row.append(row_i); eq_col.append(stock_i(n, t)); eq_val.append(1.)
            if t:
                eq_row.append(row_i); eq_col.append(stock_i(n, t-1)); eq_val.append(-1.)
            eq_row.append(row_i); eq_col.append(unmet_i(n, t)); eq_val.append(-1.)
            for index in arrival[n][t]:
                eq_row.append(row_i); eq_col.append(index); eq_val.append(-1.)
            eq_rhs.append(-demand[t]+(init_stock if t == 0 else 0))
            row_i += 1
    for n in range(n_scenarios):
        for s in range(suppliers):
            for t in range(weeks):
                if not hit[n, s, t]:
                    eq_row.extend((row_i, row_i))
                    eq_col.extend((fill_i(n, s, t), order_i(s, t)))
                    eq_val.extend((1., -1.))
                    eq_rhs.append(0.)
                    row_i += 1
    a_eq = coo_matrix((eq_val, (eq_row, eq_col)), shape=(row_i, variables)).tocsr()

    ub_row, ub_col, ub_val, ub_rhs = [], [], [], []
    row_i = 0
    def upper(terms: list[tuple[int, float]], bound: float = 0.) -> None:
        nonlocal row_i
        for col, val in terms:
            ub_row.append(row_i); ub_col.append(col); ub_val.append(val)
        ub_rhs.append(bound)
        row_i += 1

    for n in range(n_scenarios):
        for s in range(suppliers):
            for t in range(weeks):
                if hit[n, s, t]:
                    upper([(fill_i(n, s, t), 1.), (order_i(s, t), -1.)])
    if params.min_share > 0:
        for t in range(weeks):
            for s in range(suppliers):
                upper([(order_i(k, t), params.min_share-(1. if k == s else 0.))
                       for k in range(suppliers)])
    if params.cheap_share > 0 and params.cheap_premium > 0:
        avg_supplier_cost = cost.mean(axis=(0, 2))
        for s in range(suppliers):
            if (avg_supplier_cost[s] > avg_supplier_cost.min()*(1+1e-9)
                    and avg_supplier_cost[s] <= avg_supplier_cost.min()*(1+params.cheap_premium)):
                for t in range(weeks):
                    upper([(order_i(k, t), params.cheap_share-(1. if k == s else 0.))
                           for k in range(suppliers)])
    # One epigraph inequality per scenario: loss[n] <= eta + z[n].
    for n in range(n_scenarios):
        terms = list(zip(np.flatnonzero(loss[n]), loss[n, np.flatnonzero(loss[n])]))
        terms.extend(((eta, -1.), (z_start+n, -1.)))
        upper(terms)
    a_ub = coo_matrix((ub_val, (ub_row, ub_col)), shape=(row_i, variables)).tocsr()
    bounds = ([(0, capn[s, t]) for s in range(suppliers) for t in range(weeks)]
              + [(0, capn[s, t]*ratio[n, s, t]) for n in range(n_scenarios)
                 for s in range(suppliers) for t in range(weeks)]
              + [(0, None)]*(n_stock)
              + [(0, d) for _ in range(n_scenarios) for d in demand]
              + [(None, None)] + [(0, None)]*n_scenarios)
    result = linprog(objective, A_ub=a_ub, b_ub=ub_rhs,
                     A_eq=a_eq, b_eq=eq_rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    orders = result.x[:n_orders].reshape(suppliers, weeks)
    return np.where(orders > 1e-6, orders, 0.)


def plan_orders_cvar(info, weeks: range, feats: pd.DataFrame, sup_p: pd.DataFrame,
                     params: LPParams, tail_weight: float, alpha: float) -> pd.DataFrame:
    state = supplier_state(info, feats, sup_p)
    state = state.merge(info.suppliers[["material", "country", "lane"]],
                        on=["material", "country"])
    demand = info.demand.set_index(["material", "week"]).demand_units
    rng = np.random.default_rng(params.seed)
    history = shock_history(info) if params.hazard_mult > 0 else None
    output = []
    for start in range(weeks[0], weeks[-1]+1, WINDOW):
        window = [w for w in range(start, start+WINDOW) if w in weeks]
        for material in sorted(state.material.unique()):
            st = state[(state.material == material) & state.week.isin(window)]
            d = np.array([float(demand.loc[(material, w)]) for w in window])
            scenarios = draw_scenarios(st, window, params, rng, history)
            x = solve_window_cvar(d, scenarios, params, tail_weight, alpha)
            total = x.sum(axis=0)
            rows = pd.DataFrame({"week": window, "material": material, "order_units": total})
            for col in SHARE_COLS:
                rows[col] = 0.
            for s, country in enumerate(scenarios["countries"]):
                rows[f"share_{country}"] = np.divide(
                    x[s], total, out=np.full(len(window), 1/len(scenarios["countries"])),
                    where=total > 1e-6)
            rows.loc[total <= 1e-6, "order_units"] = 0.
            output.append(rows)
    return pd.concat(output, ignore_index=True)
