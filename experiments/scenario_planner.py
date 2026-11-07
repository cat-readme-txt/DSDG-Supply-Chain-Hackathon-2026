"""Scenario-based order planning using only supplier history released at origin."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from pipeline import CompetitionData


@dataclass(frozen=True)
class Scenario:
    capacity: tuple[tuple[float, ...], ...]
    lead: tuple[tuple[int, ...], ...]
    cost: tuple[tuple[float, ...], ...]
    avg_cost: tuple[float, ...]


def supplier_weights(data: CompetitionData, material: str, origin: int, method: str) -> tuple[float, ...]:
    countries = tuple(data.suppliers[material])
    if method == "equal":
        return (1/3, 1/3, 1/3)
    if method == "cost_tilt":
        rank = data.ranked_suppliers(material, origin)
        by_country = dict(zip(rank, (0.45, 0.30, 0.25)))
        return tuple(by_country[country] for country in countries)
    if method == "strong_tilt":
        rank = data.ranked_suppliers(material, origin)
        by_country = dict(zip(rank, (0.55, 0.30, 0.15)))
        return tuple(by_country[country] for country in countries)
    raise ValueError(method)


def past_scenarios(data: CompetitionData, material: str, origin: int, count: int = 4,
                   stress_count: int = 0) -> list[Scenario]:
    """Use complete past 13-week supply blocks as plausible future conditions."""
    countries = tuple(data.suppliers[material])
    starts = list(range(max(1, origin - 13 * count + 1), origin - 11, 13))
    if stress_count:
        # Include earlier known shock regimes so quiet recent weeks cannot
        # erase the tail risk of a prolonged capacity reduction.
        stressed = sorted(
            range(1, origin - 11),
            key=lambda start: (
                sum(data.outcomes[week, material][1] for week in range(start, start + 13)),
                -start,
            ),
            reverse=True,
        )
        for start in stressed:
            if all(abs(start - chosen) >= 7 for chosen in starts):
                starts.append(start)
                stress_count -= 1
                if stress_count == 0:
                    break
    scenarios = []
    for start in starts:
        capacities = []
        leads = []
        costs = []
        averages = []
        for offset in range(13):
            week = start + offset
            values = [data.supply[week, material, country] for country in countries]
            capacities.append(tuple(value[0] for value in values))
            leads.append(tuple(value[1] for value in values))
            costs.append(tuple(value[2] for value in values))
            averages.append(statistics.mean(value[2] for value in values))
        scenarios.append(Scenario(tuple(capacities), tuple(leads), tuple(costs), tuple(averages)))
    return scenarios


def actual_scenario(data: CompetitionData, material: str, start: int) -> Scenario:
    countries = tuple(data.suppliers[material])
    capacities = []
    leads = []
    costs = []
    averages = []
    for offset in range(13):
        values = [data.supply[start+offset, material, country] for country in countries]
        capacities.append(tuple(value[0] for value in values))
        leads.append(tuple(value[1] for value in values))
        costs.append(tuple(value[2] for value in values))
        averages.append(statistics.mean(value[2] for value in values))
    return Scenario(tuple(capacities), tuple(leads), tuple(costs), tuple(averages))


def replay(demand: tuple[int, ...], orders: tuple[int, ...], weights: tuple[float, ...], scenario: Scenario,
           shortage_multiplier: float = 5.0) -> dict[str, float]:
    inventory = float(sum(demand[:5]))
    arrivals = [0.0] * 13
    purchase = holding = shortage_cost = shortage_units = in_transit = 0.0
    for week in range(13):
        for supplier, share in enumerate(weights):
            filled = min(orders[week] * share, scenario.capacity[week][supplier])
            purchase += filled * scenario.cost[week][supplier]
            arrival_week = week + scenario.lead[week][supplier]
            if arrival_week < 13:
                arrivals[arrival_week] += filled
            else:
                in_transit += filled
        inventory += arrivals[week]
        missed = max(0.0, demand[week] - inventory)
        inventory = max(0.0, inventory - demand[week])
        shortage_units += missed
        shortage_cost += shortage_multiplier * scenario.avg_cost[week] * missed
        holding += 0.02 * scenario.avg_cost[week] * inventory
    salvage = 0.7 * statistics.mean(scenario.avg_cost) * (inventory + in_transit)
    return {"cost": purchase + holding + shortage_cost - salvage, "shortage": shortage_units,
            "purchase": purchase, "holding": holding, "salvage": salvage}


def optimize_orders(demand: tuple[int, ...], scenarios: list[Scenario], weights: tuple[float, ...],
                    shortage_multiplier: float = 7.0, risk_aversion: float = 0.15) -> tuple[int, ...]:
    """Coordinate search over 13 weekly orders on past supply scenarios.

    A larger shortage penalty protects fill when the unknown future week is
    worse than the average scenario.  The simulator's actual cost still uses 5.
    """
    candidate = [0] * 13
    # A lead of about three weeks makes this a useful starting schedule.
    for week in range(2, 10):
        candidate[week] = demand[min(12, week + 3)]
    scale = max(1, round(statistics.mean(demand)))

    def objective(orders: list[int]) -> float:
        values = [replay(demand, tuple(orders), weights, scenario, shortage_multiplier) for scenario in scenarios]
        costs = [value["cost"] for value in values]
        return statistics.mean(costs) + risk_aversion * (max(costs) - statistics.mean(costs))

    for step_fraction in (0.5, 0.2, 0.05):
        step = max(1, round(step_fraction * scale))
        for _ in range(4):
            changed = False
            for week in range(13):
                base = candidate[week]
                current = objective(candidate)
                best = base
                for quantity in sorted({0, max(0, base-step), base+step, base+2*step}):
                    if quantity > 2.5 * max(demand):
                        continue
                    candidate[week] = quantity
                    score = objective(candidate)
                    if score < current - 1e-6:
                        best, current = quantity, score
                candidate[week] = best
                changed |= best != base
            if not changed:
                break
    return tuple(candidate)


def plan_window(data: CompetitionData, material: str, start: int, origin: int,
                shares: str = "equal", shortage_multiplier: float = 7.0,
                stress_count: int = 0, risk_aversion: float = 0.15) -> tuple[tuple[int, ...], tuple[float, ...]]:
    demand = tuple(data.demand[start+offset, material] for offset in range(13))
    weights = supplier_weights(data, material, origin, shares)
    scenarios = past_scenarios(data, material, origin, stress_count=stress_count)
    return optimize_orders(demand, scenarios, weights, shortage_multiplier, risk_aversion), weights


def backtest(data: CompetitionData, shares: str = "equal", shortage_multiplier: float = 7.0,
             stress_count: int = 0, risk_aversion: float = 0.15) -> dict[str, object]:
    folds = []
    for origin in data.complete_fold_origins():
        metrics = {"planner_cost": 0.0, "planner_shortage": 0.0, "primary_cost": 0.0,
                   "primary_shortage": 0.0, "demand": 0.0}
        for material in data.materials:
            start = origin + 1
            demand = tuple(data.demand[start+offset, material] for offset in range(13))
            orders, weights = plan_window(data, material, start, origin, shares, shortage_multiplier, stress_count, risk_aversion)
            actual = actual_scenario(data, material, start)
            outcome = replay(demand, orders, weights, actual)
            primary_orders = tuple(round(1.2*demand[i]) if i < 8 else 0 for i in range(13))
            primary = replay(demand, primary_orders, (1/3, 1/3, 1/3), actual)
            for key in ("cost", "shortage"):
                metrics["planner_"+key] += outcome[key]
                metrics["primary_"+key] += primary[key]
            metrics["demand"] += sum(demand)
        folds.append({"weeks": f"{origin+1}-{origin+13}", **metrics})
    def summarize(items: list[dict[str, float]]) -> dict[str, float]:
        return {key: sum(item[key] for item in items) for key in ("planner_cost", "planner_shortage", "primary_cost", "primary_shortage", "demand")}
    return {"shares": shares, "shortage_multiplier": shortage_multiplier, "stress_count": stress_count,
            "risk_aversion": risk_aversion, "folds": folds,
            "training": summarize(folds[:8]), "holdout": summarize(folds[8:])}
