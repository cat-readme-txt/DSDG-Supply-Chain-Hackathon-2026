"""Chronological supplier-share sweep for the risk-responsive order policy."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from pipeline import CompetitionData, PROFILES


@dataclass(frozen=True)
class Allocation:
    cheap_bias: float
    fast_bias: float
    late_start: int


def shares(data: CompetitionData, material: str, origin: int, offset: int,
           order: int, setting: Allocation) -> dict[str, float]:
    countries = tuple(data.suppliers[material])
    if order == 0:
        return {country: 1/3 for country in countries}
    costs = {}
    leads = {}
    for country in countries:
        rows = [data.supply[week, material, country]
                for week in range(max(1, origin-25), origin+1)]
        costs[country] = statistics.mean(row[2] for row in rows)
        leads[country] = statistics.mean(row[1] for row in rows)
    cost_rank = sorted(countries, key=lambda country: (costs[country], country))
    fast_rank = sorted(countries, key=lambda country: (leads[country], costs[country], country))
    cost_target = dict(zip(cost_rank, (.55, .30, .15)))
    fast_target = dict(zip(fast_rank, (.55, .30, .15)))
    fast_bias = setting.fast_bias if offset >= setting.late_start else 0.0
    return {
        country: (1-fast_bias)*((1-setting.cheap_bias)/3 + setting.cheap_bias*cost_target[country])
                 + fast_bias*fast_target[country]
        for country in countries
    }


def simulate(data: CompetitionData, origin: int, setting: Allocation) -> dict[str, float]:
    profile = PROFILES["long_memory"]
    start, end = origin+1, origin+13
    total = {name: 0.0 for name in ("cost", "met", "demand", "shock_met", "shock_demand", "diversity")}
    for material in data.materials:
        inventory = float(sum(data.demand[start+i, material] for i in range(5)))
        arrival = [0.0]*13
        in_transit = purchase = holding = shortage_cost = diversity = 0.0
        averages = []
        for offset in range(13):
            week = start+offset
            demand = data.demand[week, material]
            order = data.policy_units(material, week, start, end, origin, profile)
            allocation = shares(data, material, origin, offset, order, setting)
            diversity += 1.5*(1-sum(weight*weight for weight in allocation.values()))
            weekly_cost = statistics.mean(data.supply[week, material, c][2] for c in allocation)
            averages.append(weekly_cost)
            for country, share in allocation.items():
                capacity, lead, landed = data.supply[week, material, country]
                filled = min(order*share, capacity)
                purchase += filled*landed
                target = offset+lead
                if target < 13:
                    arrival[target] += filled
                else:
                    in_transit += filled
            inventory += arrival[offset]
            met = min(inventory, demand)
            inventory -= met
            shortage_cost += 5*weekly_cost*(demand-met)
            holding += .02*weekly_cost*inventory
            total["met"] += met
            total["demand"] += demand
            if any(data.outcomes[shock, material][1] for shock in range(max(1, week-4), week+1)):
                total["shock_met"] += met
                total["shock_demand"] += demand
        total["cost"] += purchase+holding+shortage_cost-.7*statistics.mean(averages)*(inventory+in_transit)
        total["diversity"] += diversity/13/len(data.materials)
    return total


def summarize(folds: list[dict[str, float]]) -> dict[str, float]:
    return {
        "cost": sum(row["cost"] for row in folds),
        "shock_fill": sum(row["shock_met"] for row in folds)/sum(row["shock_demand"] for row in folds),
        "fill": sum(row["met"] for row in folds)/sum(row["demand"] for row in folds),
        "diversity": statistics.mean(row["diversity"] for row in folds),
    }


def sweep(data: CompetitionData) -> list[dict[str, object]]:
    origins = data.complete_fold_origins()
    results = []
    for cheap in (0.0, .25, .5, .75, 1.0):
        for fast in (0.0, .25, .5, .75, 1.0):
            for late in (4, 6, 8):
                setting = Allocation(cheap, fast, late)
                folds = [simulate(data, origin, setting) for origin in origins]
                results.append({"allocation": setting.__dict__, "development": summarize(folds[:8]),
                                "later_diagnostic": summarize(folds[8:])})
    return results


if __name__ == "__main__":
    data = CompetitionData(Path("data"))
    rows = sweep(data)
    baseline = next(row for row in rows if row["allocation"] == Allocation(0.0, 0.0, 4).__dict__)
    eligible = [row for row in rows if row["development"]["shock_fill"] >= baseline["development"]["shock_fill"]-.001
                and row["development"]["diversity"] >= .98]
    print(json.dumps({"trial_count": len(rows), "baseline": baseline,
                      "best_development_cost": sorted(eligible, key=lambda row: row["development"]["cost"])[:5]}, indent=2))
