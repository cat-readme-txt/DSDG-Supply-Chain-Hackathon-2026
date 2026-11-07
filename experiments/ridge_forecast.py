"""Chronological, regularized forecasts for the supply-chain submission.

The models are fitted only to observations available at a forecast origin.  A
13-week training example is included only when all its labels are released.
"""

from __future__ import annotations

import math
import statistics
from functools import lru_cache

from pipeline import CompetitionData


def solve_ridge(rows: list[tuple[list[float], float]], penalty: float) -> list[float]:
    """Small dense ridge solve, with an unpenalized intercept."""
    size = len(rows[0][0])
    matrix = [[0.0] * (size + 1) for _ in range(size)]
    for features, target in rows:
        for i, value in enumerate(features):
            matrix[i][-1] += value * target
            for j in range(i, size):
                matrix[i][j] += value * features[j]
    for i in range(size):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
        matrix[i][i] += penalty if i else 1e-8
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(matrix[row][pivot]))
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        divisor = matrix[pivot][pivot]
        if abs(divisor) < 1e-12:
            raise ValueError("Singular regression matrix")
        for column in range(pivot, size + 1):
            matrix[pivot][column] /= divisor
        for row in range(size):
            if row == pivot:
                continue
            coefficient = matrix[row][pivot]
            for column in range(pivot, size + 1):
                matrix[row][column] -= coefficient * matrix[pivot][column]
    return [matrix[i][-1] for i in range(size)]


class ForecastModel:
    def __init__(self, data: CompetitionData, penalty: float = 10.0) -> None:
        self.data = data
        self.penalty = penalty
        self.risk_models: dict[int, list[float]] = {}
        self.price_models: dict[int, list[float]] = {}
        self.futures: dict[tuple[int, str], float] = {}
        import csv
        with (data.data_dir / "prices.csv").open(newline="") as stream:
            buckets: dict[tuple[int, str], list[float]] = {}
            for row in csv.DictReader(stream):
                key = int(row["week"]), row["material"]
                buckets.setdefault(key, []).append(float(row["futures_4w"]))
        self.futures = {key: statistics.mean(values) for key, values in buckets.items()}

    @lru_cache(maxsize=None)
    def features(self, material: str, origin: int, horizon: int, kind: str) -> tuple[float, ...]:
        history = [self.data.outcomes[week, material] for week in range(1, origin + 1)]
        last_price = history[-1][0]
        decay_fast = math.exp(-horizon / 3)
        decay_slow = math.exp(-horizon / 9)
        common = [1.0, horizon / 13]
        common += [1.0 if material == name else 0.0 for name in self.data.materials[:-1]]
        if kind == "risk":
            events = [row[1] for row in history]
            last = events[-1]
            recent4 = statistics.mean(events[-4:])
            recent13 = statistics.mean(events[-13:])
            recent52 = statistics.mean(events[-52:])
            full = statistics.mean(events)
            warning = self.data.warning_score(material, origin)
            return tuple(common + [
                full, recent52, recent13, recent4,
                last * decay_fast, last * decay_slow,
                recent4 * decay_fast, recent13 * decay_slow,
                warning * decay_fast, warning * decay_slow,
            ])
        if kind == "price":
            prices = [row[0] for row in history]
            trend4 = math.log(last_price / prices[-min(5, len(prices))])
            trend13 = math.log(last_price / prices[-min(14, len(prices))])
            trend52 = math.log(last_price / prices[-min(53, len(prices))])
            mean4_gap = math.log(statistics.mean(prices[-4:]) / last_price)
            future_gap = math.log(self.futures[origin, material] / last_price)
            return tuple(common + [
                trend4, trend13, trend52, mean4_gap, future_gap,
                trend4 * decay_fast, trend13 * decay_slow,
                future_gap * decay_fast,
                future_gap * decay_slow,
            ])
        raise ValueError(kind)

    def fit(self, cutoff: int) -> None:
        if cutoff in self.risk_models:
            return
        # Each example has all outcomes in the released data.  Striding reduces
        # the weight of many nearly identical origins in a long shock episode.
        risk_rows = []
        price_rows = []
        for origin in range(26, cutoff - 12, 3):
            for material in self.data.materials:
                last_price = self.data.outcomes[origin, material][0]
                for horizon in range(1, 14):
                    actual_price, event = self.data.outcomes[origin + horizon, material]
                    risk_rows.append((list(self.features(material, origin, horizon, "risk")), float(event)))
                    price_rows.append((list(self.features(material, origin, horizon, "price")), math.log(actual_price / last_price)))
        self.risk_models[cutoff] = solve_ridge(risk_rows, self.penalty)
        self.price_models[cutoff] = solve_ridge(price_rows, self.penalty * 0.001)

    def predict(self, material: str, origin: int, horizon: int, kind: str) -> float:
        self.fit(origin)
        if kind == "risk":
            model = self.risk_models[origin]
            features = self.features(material, origin, horizon, "risk")
            return min(1.0, max(0.0, sum(a*b for a, b in zip(model, features))))
        model = self.price_models[origin]
        features = self.features(material, origin, horizon, "price")
        last = self.data.outcomes[origin, material][0]
        return last * math.exp(sum(a*b for a, b in zip(model, features)))


def backtest(data: CompetitionData, penalty: float = 10.0) -> dict[str, object]:
    model = ForecastModel(data, penalty)
    folds = []
    for origin in data.complete_fold_origins():
        totals = {key: [] for key in ("risk_ml", "risk_news", "risk_state", "risk_full", "price_ml", "price_blend", "price_last")}
        for material in data.materials:
            for horizon in range(1, 14):
                actual_price, event = data.outcomes[origin + horizon, material]
                for key, prediction in (
                    ("risk_ml", model.predict(material, origin, horizon, "risk")),
                    ("risk_news", data.probability(material, origin, "news_decay", horizon)),
                    ("risk_state", data.probability(material, origin, "state_decay", horizon)),
                    ("risk_full", data.probability(material, origin, "full", horizon)),
                ):
                    totals[key].append((prediction-event)**2)
                for key, prediction in (
                    ("price_ml", model.predict(material, origin, horizon, "price")),
                    ("price_blend", data.price(material, origin, "price_blend")),
                    ("price_last", data.price(material, origin, "last")),
                ):
                    totals[key].append(abs(prediction-actual_price)/actual_price)
        folds.append({"weeks": f"{origin+1}-{origin+13}", **{key: statistics.mean(values) for key, values in totals.items()}})
    return {"penalty": penalty, "folds": folds, "train_mean": {key: statistics.mean(f[key] for f in folds[:8]) for key in totals}, "holdout_mean": {key: statistics.mean(f[key] for f in folds[8:]) for key in totals}}
