"""Direct material forecasts trained only on released target weeks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

KEYWORDS = {
    "news_threat": ("strike", "sanction", "shortage", "export", "halt", "closure", "suspend"),
    "news_transport": ("port", "strait", "ship", "freight", "route", "congestion"),
    "news_policy": ("tariff", "duties", "license", "restriction", "trade dispute"),
    "news_resolution": ("averted", "eases", "returns", "deal reached", "recovered"),
}
LANE_NAMES = {
    "MERROW": "merrow strait", "NORTH": "northern sea", "SOUTH": "southern route",
    "ATLANTIC": "atlantic route", "RAIL": "domestic rail",
}


class DirectForecaster:
    """Build raw-table lag features, then fit a direct horizon model."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        def read(name: str) -> pd.DataFrame:
            return pd.read_csv(self.data_dir / f"{name}.csv")

        self.outcomes = read("outcomes")
        self.demand = read("demand")
        self.news = read("news")
        self.suppliers = read("suppliers")
        self.materials = sorted(self.suppliers.material.unique())
        self.out = {
            m: self.outcomes[self.outcomes.material == m].set_index("week").sort_index()
            for m in self.materials
        }
        self.dem = {
            m: self.demand[self.demand.material == m].set_index("week").demand_units.sort_index()
            for m in self.materials
        }

        supply = read("supply_history").merge(
            self.suppliers[["material", "country", "normal_capacity_units", "base_lead_weeks"]],
            on=["material", "country"], validate="many_to_one")
        supply["cap_ratio"] = supply.capacity_units / supply.normal_capacity_units
        supply["lead_delta"] = supply.lead_time_weeks - supply.base_lead_weeks
        self.supply = supply.groupby(["material", "week"]).agg(
            cap_min=("cap_ratio", "min"), cap_mean=("cap_ratio", "mean"),
            lead_max=("lead_delta", "max"), lead_mean=("lead_delta", "mean"),
            cost_min=("landed_cost", "min"), cost_max=("landed_cost", "max"))
        self.spot = read("prices").groupby(["material", "week"])[
            ["spot_price", "futures_4w"]].mean()
        macro = read("macro").merge(self.suppliers[["material", "country"]], on="country")
        self.macro = macro.groupby(["material", "week"]).agg(
            tension_max=("tension_index", "max"), pmi_mean=("pmi", "mean"),
            fx_mean=("fx_rate", "mean"))
        ship = read("shipping").merge(self.suppliers[["material", "lane"]], on="lane")
        self.ship = ship.groupby(["material", "week"]).agg(
            congestion_max=("congestion_index", "max"),
            transit_max=("transit_days", "max"), container_mean=("container_rate", "mean"))
        tariff = read("tariffs")
        tariff["announced_delta"] = (tariff.announced_rate - tariff.tariff_rate).clip(lower=0).fillna(0)
        tariff["announced_soon"] = tariff.announced_delta.where(
            tariff.announced_effective_week.between(tariff.week + 1, tariff.week + 13), 0)
        self.tariff = tariff.groupby(["material", "week"]).agg(
            tariff_mean=("tariff_rate", "mean"), announced_soon=("announced_soon", "max"))
        self.news_signals = self._news_signals()
        self.base = self._base_features()

    def _news_signals(self) -> dict[tuple[str, int], dict[str, float]]:
        names = {}
        for material in self.materials:
            suppliers = self.suppliers[self.suppliers.material == material]
            names[material] = {
                material.replace("_", " "),
                *suppliers.country_name.str.lower().tolist(),
                *(LANE_NAMES[lane] for lane in suppliers.lane.unique()),
            }
        signals = {}
        for article in self.news.itertuples():
            text = f"{article.headline} {article.body}".lower()
            for material in self.materials:
                if not any(name in text for name in names[material]):
                    continue
                entry = signals.setdefault(
                    (material, int(article.week)), {name: 0.0 for name in KEYWORDS})
                for name, words in KEYWORDS.items():
                    entry[name] += float(any(word in text for word in words))
        return signals

    def _base_features(self) -> dict[tuple[str, int], dict[str, float]]:
        base = {}
        last_week = int(self.outcomes.week.max())
        for mi, material in enumerate(self.materials):
            history = self.out[material]
            demand = self.dem[material]
            for week in range(26, last_week + 1):
                y = history.disrupted.loc[:week].to_numpy(float)
                px = history.benchmark_price.loc[:week].to_numpy(float)
                current_price = px[-1]
                hit_positions = np.flatnonzero(y)
                since_hit = week - (int(hit_positions[-1]) + 1) if len(hit_positions) else 52
                run = 0
                for value in y[::-1]:
                    if not value:
                        break
                    run += 1
                s = self.supply.loc[(material, week)]
                spot = self.spot.loc[(material, week)]
                macro = self.macro.loc[(material, week)]
                ship = self.ship.loc[(material, week)]
                tariff = self.tariff.loc[(material, week)]
                news = {name: 0.0 for name in KEYWORDS}
                news2 = {name: 0.0 for name in KEYWORDS}
                for prior in range(max(1, week - 7), week + 1):
                    event = self.news_signals.get((material, prior), {})
                    for name in KEYWORDS:
                        value = event.get(name, 0.0)
                        news[name] += value
                        if prior >= week - 1:
                            news2[name] += value
                row = {
                    "material_id": float(mi),
                    "week_sin": math.sin(2 * math.pi * week / 52),
                    "week_cos": math.cos(2 * math.pi * week / 52),
                    "disrupted_now": y[-1], "run_length": min(run, 20),
                    "weeks_since_hit": min(since_hit, 52),
                    "hit_rate_4": y[-4:].mean(), "hit_rate_13": y[-13:].mean(),
                    "hit_rate_26": y[-26:].mean(), "hit_rate_52": y[-52:].mean(),
                    "price_level": math.log(current_price),
                    "price_dev_4": math.log(current_price / px[-4:].mean()),
                    "price_dev_13": math.log(current_price / px[-13:].mean()),
                    "price_dev_52": math.log(current_price / px[-52:].mean()),
                    "price_ret_4": math.log(current_price / px[-4]),
                    "price_ret_13": math.log(current_price / px[-13]),
                    "price_vol_13": float(np.std(np.diff(np.log(px[-13:])))),
                    "cap_min": s.cap_min, "cap_mean": s.cap_mean,
                    "lead_max": s.lead_max, "lead_mean": s.lead_mean,
                    "cost_spread": s.cost_max / s.cost_min - 1,
                    "spot_ratio": spot.spot_price / current_price,
                    "futures_ratio": spot.futures_4w / spot.spot_price,
                    "tension_max": macro.tension_max, "pmi_mean": macro.pmi_mean,
                    "fx_mean": macro.fx_mean, "congestion_max": ship.congestion_max,
                    "transit_max": ship.transit_max, "container_mean": ship.container_mean,
                    "tariff_mean": tariff.tariff_mean,
                    "announced_soon": tariff.announced_soon,
                    "demand_now": float(demand.loc[week]),
                    "demand_next_5": float(demand.loc[week + 1:week + 5].mean()),
                    "demand_next_13": float(demand.loc[week + 1:week + 13].mean()),
                }
                row.update(news)
                row.update({f"{name}_2": value for name, value in news2.items()})
                base[material, week] = row
        return base

    def feature(self, material: str, origin: int, horizon: int) -> dict[str, float]:
        row = dict(self.base[material, origin])
        demand = self.dem[material]
        row["horizon"] = float(horizon)
        row["demand_target_ratio"] = float(demand.loc[origin + horizon] / demand.loc[origin])
        row["demand_through_ratio"] = float(
            demand.loc[origin + 1:origin + horizon].mean() / demand.loc[origin])
        return row

    def training(self, cutoff: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        features, disruption, price_change = [], [], []
        for origin in range(26, cutoff):
            for material in self.materials:
                current_price = float(self.out[material].loc[origin, "benchmark_price"])
                for horizon in range(1, min(26, cutoff - origin) + 1):
                    target = self.out[material].loc[origin + horizon]
                    features.append(self.feature(material, origin, horizon))
                    disruption.append(int(target.disrupted))
                    price_change.append(math.log(float(target.benchmark_price) / current_price))
        return pd.DataFrame(features), np.asarray(disruption), np.asarray(price_change)

    def fit(self, cutoff: int):
        features, disruption, change = self.training(cutoff)
        classifier = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=10, min_samples_leaf=120,
            learning_rate=0.05, l2_regularization=10, random_state=0,
            categorical_features=["material_id"])
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=120, max_leaf_nodes=8,
            min_samples_leaf=120, learning_rate=0.05, l2_regularization=10,
            random_state=0, categorical_features=["material_id"])
        classifier.fit(features, disruption)
        regressor.fit(features, change)
        return classifier, regressor

    def predict(self, cutoff: int, horizons: range, models=None) -> pd.DataFrame:
        classifier, regressor = self.fit(cutoff) if models is None else models
        rows = [(material, cutoff + h, self.feature(material, cutoff, h))
                for h in horizons for material in self.materials]
        features = pd.DataFrame([row[2] for row in rows])
        prob = classifier.predict_proba(features)[:, 1]
        ratio = np.exp(regressor.predict(features))
        return pd.DataFrame({
            "week": [r[1] for r in rows], "material": [r[0] for r in rows],
            "disruption_prob": prob,
            "price_forecast": [
                float(self.out[r[0]].loc[cutoff, "benchmark_price"]) * ratio[i]
                for i, r in enumerate(rows)
            ],
        })
