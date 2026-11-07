"""Learn order plans from historical hindsight optima, with chronological training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.simulator import hindsight_plan, truth_for  # noqa: E402


class HindsightImitation:
    """A direct policy model trained on windows that ended before each cutoff."""

    def __init__(self, data_dir: Path):
        self.data = load_data(data_dir)
        self.materials = sorted(self.data.suppliers.material.unique())
        self.countries = sorted(self.data.suppliers.country.unique())
        self.suppliers = self.data.suppliers.set_index(["material", "country"])
        self.supply = self.data.supply_history.set_index(["material", "country", "week"]).sort_index()
        self.demand = self.data.demand.set_index(["material", "week"]).demand_units
        self.outcome = self.data.outcomes.set_index(["material", "week"]).disrupted
        self.macro = self.data.macro.set_index(["country", "week"])
        self.shipping = self.data.shipping.set_index(["lane", "week"])
        self.tariffs = self.data.tariffs.set_index(["material", "country", "week"])
        self.examples = self._historical_examples()

    def _window_features(self, origin: int, start: int, material: str) -> pd.DataFrame:
        countries = self.data.suppliers_of(material)
        demand = np.array([self.demand.loc[(material, week)]
                           for week in range(start, start + 13)], dtype=float)
        demand_avg = float(demand.mean())
        current_cost = {
            country: float(self.supply.loc[(material, country, origin), "landed_cost"])
            for country in countries
        }
        cheapest = min(current_cost.values())
        rows = []
        for country in countries:
            supplier = self.suppliers.loc[(material, country)]
            history = self.supply.loc[(material, country)].loc[max(1, origin - 25):origin]
            current = history.iloc[-1]
            base_lead = int(supplier.base_lead_weeks)
            normal = float(supplier.normal_capacity_units)
            cap_ratio = history.capacity_units.to_numpy(float) / normal
            lane = supplier.lane
            shipping = self.shipping.loc[(lane, origin)]
            macro = self.macro.loc[(country, origin)]
            tariff = self.tariffs.loc[(material, country, origin)]
            for t in range(13):
                arrival = t + base_lead
                row = {
                    "material_id": float(self.materials.index(material)),
                    "country_id": float(self.countries.index(country)),
                    "position": float(t), "arrival_position": float(arrival),
                    "normal_capacity_to_demand": normal / demand_avg,
                    "base_lead": float(base_lead),
                    "current_cap_ratio": float(current.capacity_units / normal),
                    "recent_cap_mean": float(cap_ratio[-4:].mean()),
                    "recent_cap_min": float(cap_ratio[-13:].min()),
                    "supplier_hit_rate": float((cap_ratio < 0.8).mean()),
                    "current_lead_delta": float(current.lead_time_weeks - base_lead),
                    "recent_lead_mean": float(history.lead_time_weeks.iloc[-4:].mean() - base_lead),
                    "cost_relative": current_cost[country] / cheapest,
                    "cost_level": np.log(current_cost[country]),
                    "recent_cost_change": float(
                        current_cost[country] / history.landed_cost.iloc[-4:].mean() - 1),
                    "material_disrupted_now": float(self.outcome.loc[(material, origin)]),
                    "demand_now_relative": demand[t] / demand_avg,
                    "demand_at_arrival": demand[arrival] / demand_avg if arrival < 13 else 0.0,
                    "demand_next_3": float(demand[t:min(13, t + 3)].mean() / demand_avg),
                    "demand_total": float(demand.sum() / (13 * demand_avg)),
                    "initial_stock_ratio": float(demand[:5].sum() / (5 * demand_avg)),
                    "remaining_demand": float(demand[t:].sum() / demand_avg),
                    "congestion": float(shipping.congestion_index),
                    "tension": float(macro.tension_index),
                    "tariff": float(tariff.tariff_rate),
                    "announced_delta": float(max(0, tariff.announced_rate - tariff.tariff_rate))
                    if pd.notna(tariff.announced_rate) else 0.0,
                }
                rows.append({"origin": origin, "start": start, "material": material,
                             "country": country, "week": start + t, **row})
        return pd.DataFrame(rows)

    def _historical_examples(self) -> pd.DataFrame:
        rows = []
        last = self.data.last_week
        for start in range(27, last - 11):
            origin = start - 1
            for material in self.materials:
                truth = truth_for(self.data, material, range(start, start + 13))
                _, optimum = hindsight_plan(truth)
                feature = self._window_features(origin, start, material)
                demand_avg = truth.demand.mean()
                target = {
                    (truth.countries[si], start + t): float(optimum[si, t] / demand_avg)
                    for si in range(len(truth.countries)) for t in range(13)
                }
                feature["target"] = [
                    target[(country, week)] for country, week in
                    zip(feature.country, feature.week)
                ]
                rows.append(feature)
            if start % 25 == 0:
                print(f"prepared hindsight windows through {start}", flush=True)
        return pd.concat(rows, ignore_index=True)

    def fit(self, cutoff: int) -> HistGradientBoostingRegressor:
        train = self.examples[self.examples.start + 12 <= cutoff]
        X = train.drop(columns=["origin", "start", "material", "country", "week", "target"])
        model = HistGradientBoostingRegressor(
            max_iter=140, max_leaf_nodes=12, min_samples_leaf=120,
            learning_rate=0.05, l2_regularization=10, random_state=0,
            categorical_features=["material_id", "country_id"])
        model.fit(X, train.target)
        return model

    def predict(self, cutoff: int, scale: float = 1.0, threshold: float = 0.05,
                model: HistGradientBoostingRegressor | None = None) -> pd.DataFrame:
        model = self.fit(cutoff) if model is None else model
        start = cutoff + 1
        out = []
        for material in self.materials:
            features = self._window_features(cutoff, start, material)
            X = features.drop(columns=["origin", "start", "material", "country", "week"])
            demand_avg = float(np.mean(
                [self.demand.loc[(material, week)] for week in range(start, start + 13)]))
            normalized = np.maximum(model.predict(X) * scale, 0)
            normalized[normalized < threshold] = 0
            features["units"] = normalized * demand_avg
            features["units"] = [
                min(units, 0.96 * self.suppliers.loc[(material, country), "normal_capacity_units"])
                for units, country in zip(features.units, features.country)
            ]
            for week, frame in features.groupby("week"):
                order = float(frame.units.sum())
                row = {"week": int(week), "material": material, "order_units": order}
                for share in SHARE_COLS:
                    row[share] = 0.0
                eligible = self.data.suppliers_of(material)
                for country in eligible:
                    units = float(frame.loc[frame.country == country, "units"].iloc[0])
                    row[f"share_{country}"] = units / order if order > 0 else 1 / len(eligible)
                out.append(row)
        return pd.DataFrame(out)
