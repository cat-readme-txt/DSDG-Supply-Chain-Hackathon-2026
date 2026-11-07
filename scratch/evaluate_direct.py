"""Chronological score test of the raw-table direct forecast against 67.74."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402
from scratch.direct_forecast import DirectForecaster  # noqa: E402

ORIGINS = [78, 104, 130, 156, 182]
VARIANTS = {
    "scored_baseline": (0.0, 0.0),
    "direct_both": (1.0, 1.0),
    "direct_probability": (1.0, 0.0),
    "direct_price": (0.0, 1.0),
    "half_probability": (0.5, 0.0),
    "half_price": (0.0, 0.5),
    "half_both": (0.5, 0.5),
}


def main() -> None:
    data_dir = ROOT / "data" / "snapshots" / "w208"
    data = load_data(data_dir)
    direct = DirectForecaster(data_dir)
    cache = Path("/private/tmp/dsdg-scratch-folds")
    cache.mkdir(exist_ok=True)
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 27)
        base = naive(data.until(origin), weeks)
        for window, cutoff, win in (
            ("public", origin, range(origin + 1, origin + 14)),
            ("private", origin + 13, range(origin + 14, origin + 27)),
        ):
            cached = cache / f"lean_split_o{origin}_{window}.csv"
            if cached.exists():
                incumbent = pd.read_csv(cached)
            else:
                incumbent = STRATEGIES["lean_split"](data.until(cutoff), weeks)
                incumbent.to_csv(cached, index=False)
            prediction = direct.predict(cutoff, range(1, 14))
            keyed = prediction.set_index(["week", "material"])
            for name, (prob_weight, price_weight) in VARIANTS.items():
                sub = incumbent.copy()
                keys = pd.MultiIndex.from_frame(sub[["week", "material"]])
                mask = sub.week.isin(win)
                probability = keyed.reindex(keys[mask]).disruption_prob.to_numpy()
                price = keyed.reindex(keys[mask]).price_forecast.to_numpy()
                sub.loc[mask, "disruption_prob"] = (
                    (1 - prob_weight) * sub.loc[mask, "disruption_prob"].to_numpy()
                    + prob_weight * probability)
                sub.loc[mask, "price_forecast"] = (
                    (1 - price_weight) * sub.loc[mask, "price_forecast"].to_numpy()
                    + price_weight * price)
                score = score_window(sub, base, data, win)
                y = data.outcomes[data.outcomes.week.isin(win)][
                    ["week", "material", "disrupted", "benchmark_price"]]
                merged = sub[sub.week.isin(win)].merge(y, on=["week", "material"])
                brier = float(np.mean((merged.disruption_prob - merged.disrupted) ** 2))
                mape = float(np.mean(abs(merged.price_forecast - merged.benchmark_price)
                                     / merged.benchmark_price))
                rows.append({"origin": origin, "window": window, "cutoff": cutoff,
                             "variant": name, "brier": brier, "mape": mape,
                             **score.as_dict()})
            print(f"finished origin {origin} {window}", flush=True)
    frame = pd.DataFrame(rows)
    summary = {}
    for name in VARIANTS:
        variant = frame[frame.variant == name]
        summary[name] = {}
        for label, subset in (
            ("development", variant[variant.origin <= 130]),
            ("later_holdout", variant[variant.origin >= 156]),
            ("all", variant),
        ):
            summary[name][label] = {
                col: float(subset[col].mean())
                for col in ("score", "P", "F", "R", "brier", "mape")
            }
    report = ROOT / "reports" / "direct_forecast_w208.json"
    report.write_text(json.dumps(
        {"origins": ORIGINS, "variants": VARIANTS, "summary": summary,
         "folds": frame.to_dict(orient="records")}, indent=2) + "\n")
    print(report)
    for name in VARIANTS:
        print(name, "dev", round(summary[name]["development"]["score"], 2),
              "later", round(summary[name]["later_holdout"]["score"], 2),
              "devP", round(summary[name]["development"]["P"], 3),
              "laterP", round(summary[name]["later_holdout"]["P"], 3))


if __name__ == "__main__":
    main()
