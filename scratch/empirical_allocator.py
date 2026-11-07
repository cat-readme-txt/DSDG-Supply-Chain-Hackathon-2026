"""Choose supplier shares by replaying released supply blocks, without a shock model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def share_grid(step: float = 0.1) -> np.ndarray:
    n = round(1 / step)
    return np.array([[i / n, j / n, (n - i - j) / n]
                     for i in range(n + 1) for j in range(n - i + 1)], dtype=float)


class EmpiricalAllocator:
    """Replay contiguous historical supply blocks against the future demand plan."""

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        self.suppliers = pd.read_csv(data_dir / "suppliers.csv")
        supply = pd.read_csv(data_dir / "supply_history.csv")
        self.supply = {
            (material, country): frame.set_index("week").sort_index()
            for (material, country), frame in supply.groupby(["material", "country"])
        }
        outcomes = pd.read_csv(data_dir / "outcomes.csv")
        self.outcomes = {
            material: frame.set_index("week").sort_index()
            for material, frame in outcomes.groupby("material")
        }
        demand = pd.read_csv(data_dir / "demand.csv")
        self.demand = demand.set_index(["material", "week"]).demand_units
        self.grid = share_grid()

    def scenarios(self, material: str, cutoff: int):
        eligible = self.suppliers[self.suppliers.material == material]
        countries = eligible.country.tolist()
        origins = range(max(1, cutoff - 104), cutoff - 11)
        blocks = []
        outcome = self.outcomes[material]
        current_price = float(outcome.loc[cutoff, "benchmark_price"])
        for start in origins:
            weeks = list(range(start, start + 13))
            scale = current_price / float(outcome.loc[weeks, "benchmark_price"].mean())
            cap = np.array([
                self.supply[material, country].loc[weeks, "capacity_units"].to_numpy(float)
                for country in countries])
            lead = np.array([
                self.supply[material, country].loc[weeks, "lead_time_weeks"].to_numpy(int)
                for country in countries])
            cost = np.array([
                self.supply[material, country].loc[weeks, "landed_cost"].to_numpy(float)
                for country in countries]) * scale
            disrupted = outcome.disrupted
            shock = np.array([int(disrupted.loc[max(1, week - 4):week].max() > 0)
                              for week in weeks], dtype=float)
            blocks.append((cap, lead, cost, shock))
        if not blocks:
            raise ValueError(f"No released 13-week blocks by cutoff {cutoff}")
        return countries, tuple(np.stack([block[i] for block in blocks]) for i in range(4))

    def costs(self, material: str, cutoff: int, orders: np.ndarray):
        countries, (capacity, lead, landed, shock) = self.scenarios(material, cutoff)
        demand = np.array([self.demand.loc[(material, cutoff + t + 1)]
                           for t in range(13)], dtype=float)
        C, B = len(self.grid), len(capacity)
        filled = np.minimum(
            orders[None, None, None, :] * self.grid[:, None, :, None],
            capacity[None, :, :, :])
        purchase = np.sum(filled * landed[None], axis=(2, 3))
        max_lead = int(lead.max())
        arrivals = np.zeros((C, B, 13 + max_lead + 1))
        for supplier in range(3):
            for t in range(13):
                indices = t + lead[:, supplier, t]
                arrivals[:, np.arange(B), indices] += filled[:, :, supplier, t]
        average = landed.mean(axis=(1, 2))
        inventory = np.full((C, B), demand[:5].sum(), dtype=float)
        holding = np.zeros((C, B))
        unmet = np.zeros((C, B))
        shock_unmet = np.zeros((C, B))
        for t in range(13):
            inventory += arrivals[:, :, t]
            served = np.minimum(inventory, demand[t])
            inventory -= served
            missed = demand[t] - served
            holding += 0.02 * average[None] * inventory
            unmet += 5 * average[None] * missed
            shock_unmet += missed * shock[None, :, t]
        salvage = 0.7 * average[None] * (inventory + arrivals[:, :, 13:].sum(axis=2))
        total = purchase + holding + unmet - salvage
        return countries, total, shock_unmet, float(average.mean())

    def choose(self, material: str, cutoff: int, orders: np.ndarray,
               mode: str = "mean") -> dict[str, float]:
        countries, costs, shock_unmet, average = self.costs(material, cutoff, orders)
        objective = costs.mean(axis=1)
        if mode == "robust":
            objective = objective + 0.3 * costs.std(axis=1) + 2 * average * shock_unmet.mean(axis=1)
        elif mode == "diverse":
            diversity = (1 - (self.grid ** 2).sum(axis=1)) / (2 / 3)
            objective = objective - 0.03 * average * orders.sum() * diversity
        elif mode != "mean":
            raise ValueError(mode)
        selected = self.grid[int(np.argmin(objective))]
        return dict(zip(countries, selected))
