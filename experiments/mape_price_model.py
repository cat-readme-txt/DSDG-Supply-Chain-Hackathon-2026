"""Weighted-absolute-error price model with chronological validation.

Predict the future/current benchmark ratio. A sample with target y and current
price p gets weight p/y; weighted absolute error in the ratio is therefore
exactly absolute percentage error in the original price. Targets after each
validation origin are never used to train that origin's model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.models.price import predict_price  # noqa: E402

FEATURES = ["material_code", "h", "anchor_gap", "mom1", "mom4", "vol8",
            "futures_premium", "current_disrupted", "anchor_h", "mom4_h"]
ORIGINS = (130, 143, 156, 169, 182, 195)
MODELS = {
    "leaf80_l2_1": dict(min_samples_leaf=80, l2_regularization=1., max_iter=250),
    "leaf180_l2_10": dict(min_samples_leaf=180, l2_regularization=10., max_iter=250),
    "leaf350_l2_30": dict(min_samples_leaf=350, l2_regularization=30., max_iter=250),
}


def feature_table(data) -> pd.DataFrame:
    prices = data.outcomes[["week", "material", "benchmark_price", "disrupted"]].copy()
    prices = prices.sort_values(["material", "week"])
    material_codes = {m: k for k, m in enumerate(sorted(prices.material.unique()))}
    prices["material_code"] = prices.material.map(material_codes)
    g = prices.groupby("material", sort=False)
    logp = np.log(prices.benchmark_price)
    prices["anchor_gap"] = logp - g.benchmark_price.transform(
        lambda x: np.log(x).rolling(78, min_periods=26).mean())
    prices["mom1"] = logp - np.log(g.benchmark_price.shift(1))
    prices["mom4"] = logp - np.log(g.benchmark_price.shift(4))
    prices["vol8"] = g.benchmark_price.transform(
        lambda x: np.log(x).diff().rolling(8, min_periods=4).std())
    fut = data.prices.assign(premium=np.log(data.prices.futures_4w / data.prices.spot_price))
    fut = fut.groupby(["week", "material"], as_index=False).premium.mean()
    prices = prices.merge(fut, on=["week", "material"], how="left")
    prices = prices.rename(columns={"premium": "futures_premium", "disrupted": "current_disrupted"})
    parts = []
    for h in range(1, 14):
        frame = prices.copy()
        frame["h"] = h
        frame["target_week"] = frame.week + h
        frame["target_price"] = g.benchmark_price.shift(-h).to_numpy()
        frame["ratio"] = frame.target_price / frame.benchmark_price
        frame["weight"] = frame.benchmark_price / frame.target_price
        frame["anchor_h"] = frame.anchor_gap * h / 13
        frame["mom4_h"] = frame.mom4 * h / 13
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURES)


def source_predictions(data, origin: int) -> pd.DataFrame:
    info = data.until(origin)
    weeks = range(origin + 1, origin + 14)
    features = build_features(info)
    suppliers = supplier_forecast(info, weeks, feats=features)
    return predict_price(info, weeks, suppliers, feats=features).rename(columns={"price": "source"})


def model_prediction(rows: pd.DataFrame, origin: int, params: dict) -> pd.DataFrame:
    training = rows[(rows.target_week <= origin) & rows.ratio.notna()]
    current = rows[rows.week == origin].copy()
    model = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3,
                                          learning_rate=.05, random_state=27,
                                          **params)
    model.fit(training[FEATURES], training.ratio,
              sample_weight=training.weight)
    current["forecast"] = current.benchmark_price * model.predict(current[FEATURES])
    return current[["target_week", "material", "forecast"]]


def main() -> None:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    table = feature_table(data)
    outcomes = data.outcomes[["week", "material", "benchmark_price"]].rename(
        columns={"week": "target_week", "benchmark_price": "actual"})
    detail = []
    for origin in ORIGINS:
        source = source_predictions(data, origin).rename(columns={"week": "target_week"})
        known = table[table.week == origin][["target_week", "material", "benchmark_price"]]
        base = known.merge(source, on=["target_week", "material"]).merge(
            outcomes, on=["target_week", "material"])
        base["carry"] = base.benchmark_price
        for name, params in MODELS.items():
            pred = model_prediction(table, origin, params)
            scored = base.merge(pred, on=["target_week", "material"])
            for subset, mask in (("all", np.ones(len(scored), bool)),
                                 ("quiet", scored.material.isin(
                                     ["aluminum", "polymer_resin", "rare_earths",
                                      "semiconductors", "specialty_glass", "steel"]))):
                part = scored[mask]
                detail.append(dict(origin=origin, model=name, subset=subset,
                                   source_mape=float((abs(part.source-part.actual)/part.actual).mean()),
                                   model_mape=float((abs(part.forecast-part.actual)/part.actual).mean()),
                                   carry_mape=float((abs(part.carry-part.actual)/part.actual).mean())))
        print("finished", origin, flush=True)
    result = pd.DataFrame(detail)
    output = ROOT / "reports" / "mape_price_model_w208.json"
    output.write_text(json.dumps(result.to_dict(orient="records"), indent=2) + "\n")
    for subset in ("all", "quiet"):
        for name in MODELS:
            rows = result[(result.model == name) & (result.subset == subset)]
            for fold, mask in (("development", rows.origin <= 156),
                               ("later_holdout", rows.origin >= 169)):
                part = rows[mask]
                print(subset, name, fold, "source", round(part.source_mape.mean(), 5),
                      "weighted_mae", round(part.model_mape.mean(), 5),
                      "carry", round(part.carry_mape.mean(), 5))
    print(output)


if __name__ == "__main__":
    main()
