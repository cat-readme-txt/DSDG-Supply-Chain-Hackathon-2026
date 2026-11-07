"""Evaluate fixed lithium orders under week-208 supplier scenarios.

This is a decision aid, not the official grader. It varies only public lithium
shares so an order-timing mistake cannot be mistaken for a supplier effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, draw_scenarios, shock_history  # noqa: E402


def simulate(sc: dict, orders: np.ndarray, shares: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized material-window cost and demand served for many scenarios."""
    n, suppliers, weeks = sc["hit"].shape
    capacity = sc["cap"][None, :, :] * sc["cap_ratio"]
    bought = np.minimum(orders[None, None, :] * shares[None, :, :], capacity)
    price = sc["cost"]
    if price.ndim == 2:
        price = np.broadcast_to(price, (n, suppliers, weeks))
    avg_price = price.mean(axis=(1, 2))
    arrivals = np.zeros((n, weeks + int(sc["lead"].max()) + 1))
    for supplier in range(suppliers):
        for week in range(weeks):
            dest = week + sc["lead"][:, supplier, week].astype(int)
            arrivals[np.arange(n), dest] += bought[:, supplier, week]
    stock = np.full(n, demand[:5].sum())
    served = np.zeros((n, weeks))
    hold = np.zeros(n)
    unmet = np.zeros(n)
    for week in range(weeks):
        stock += arrivals[:, week]
        served[:, week] = np.minimum(stock, demand[week])
        stock -= served[:, week]
        hold += 0.02 * avg_price * stock
        unmet += 5 * avg_price * (demand[week] - served[:, week])
    salvage = .7 * avg_price * (stock + arrivals[:, weeks:].sum(axis=1))
    return (bought * price).sum(axis=(1, 2)) + hold + unmet - salvage, served


def main() -> None:
    info = load_data(ROOT / "data" / "snapshots" / "w208")
    weeks = np.arange(209, 222)
    feats = build_features(info)
    sup = supplier_forecast(info, range(209, 222), feats=feats)
    state = supplier_state(info, feats, sup)
    state = state.merge(info.suppliers[["material", "country", "lane"]], on=["material", "country"])
    state = state[state.material == "lithium"]
    csv = pd.read_csv(ROOT / "submissions" / "submission67.74.csv")
    rows = csv[csv.id.str.contains("lithium") & csv.id.str.extract(r"w(\d+)")[0].astype(int).between(209, 221)]
    orders = rows.order_units.to_numpy(float)
    demand = info.demand.query("material == 'lithium' and 209 <= week <= 221").sort_values("week").demand_units.to_numpy(float)
    countries = state.country.drop_duplicates().tolist()
    baseline = rows[[f"share_{c}" for c in countries]].to_numpy(float).T
    variants = {"base67.74": baseline}
    for stop in (213, 215, 217):
        for val in (.1, .2, .3, .4):
            alternate = baseline.copy()
            for t, week in enumerate(weeks):
                if week <= stop and orders[t] > 0:
                    allocation = {"KES": .7 - val, "ORI": .3, "VAL": val}
                    alternate[:, t] = [allocation[c] for c in countries]
            variants[f"through{stop}_val{val:.1f}"] = alternate
    for hazard in (0., .25, 1.):
        params = LPParams(n_scenarios=5000, min_share=0, risk_mult=1,
                          cap_derate=.96, hazard_mult=hazard, seed=1523)
        scenarios = draw_scenarios(state, list(weeks), params, np.random.default_rng(1523),
                                   shock_history(info) if hazard else None)
        base_cost, base_served = simulate(scenarios, orders, baseline, demand)
        print("hazard", hazard, "baseline mean cost", round(base_cost.mean()),
              "fill", round(base_served.sum(axis=1).mean() / demand.sum(), 5))
        for name, shares in variants.items():
            if name == "base67.74":
                continue
            cost, served = simulate(scenarios, orders, shares, demand)
            delta_div = ((1-(shares**2).sum(axis=0))-(1-(baseline**2).sum(axis=0))).sum() / (13*10*(2/3))
            print(name, "cost", round((cost-base_cost).mean()),
                  "p90 cost", round(np.quantile(cost-base_cost, .9)),
                  "fill", round((served-base_served).sum(axis=1).mean(), 2),
                  "div score", round(9*delta_div, 3))


if __name__ == "__main__":
    main()
