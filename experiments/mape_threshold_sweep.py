"""Chronological check of the MAPE-optimal shock uplift threshold.

A two-point low/high price distribution favors the high forecast only when
p > high/(low+high), slightly above 0.5 for a positive shock uplift. Test that
decision threshold without changing any purchasing or disruption forecasts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.disruption import supplier_forecast  # noqa: E402
from dsdg_supply_chain.models.price import predict_price  # noqa: E402

ORIGINS = tuple(range(52, 209, 13))
THRESHOLDS = (.5, .53, .55, .6)


def main() -> None:
    data = load_data(ROOT / "data" / "Phase 2 data")
    rows = []
    for origin in ORIGINS:
        info = data.until(origin)
        weeks = range(origin+1, origin+14)
        if weeks[-1] > data.last_week:
            continue
        features = build_features(info)
        suppliers = supplier_forecast(info, weeks, feats=features)
        truth = data.outcomes[data.outcomes.week.isin(weeks)].set_index(["week", "material"]).benchmark_price
        for threshold in THRESHOLDS:
            pred = predict_price(info, weeks, suppliers, feats=features,
                                 threshold=threshold).set_index(["week", "material"]).price
            error = (pred.loc[truth.index]-truth).abs()/truth
            rows.append(dict(origin=origin, threshold=threshold,
                             mape=float(error.mean()),
                             by_material=error.groupby(level="material").mean().to_dict()))
        print("finished", origin, flush=True)
    output = ROOT / "reports" / "mape_threshold_sweep_w221.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for threshold in THRESHOLDS:
        chosen = [row for row in rows if row["threshold"] == threshold]
        for label, subset in (("development", [r for r in chosen if r["origin"] <= 156]),
                              ("later_holdout", [r for r in chosen if r["origin"] >= 169])):
            print(threshold, label, round(np.mean([r["mape"] for r in subset]), 5))
    print(output)


if __name__ == "__main__":
    main()
