"""Chronological score test for supervised hindsight imitation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402
from scratch.hindsight_imitation import HindsightImitation  # noqa: E402

ORIGINS = [78, 104, 130, 156, 182]
VARIANTS = {
    "baseline": None,
    "imitation_80": 0.8,
    "imitation_100": 1.0,
    "imitation_120": 1.2,
}


def main() -> None:
    data_dir = ROOT / "data" / "snapshots" / "w208"
    data = load_data(data_dir)
    imitation = HindsightImitation(data_dir)
    cache = Path("/private/tmp/dsdg-scratch-folds")
    cache.mkdir(exist_ok=True)
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 27)
        naive_base = naive(data.until(origin), weeks)
        for window, cutoff, win in (
            ("public", origin, range(origin + 1, origin + 14)),
            ("private", origin + 13, range(origin + 14, origin + 27)),
        ):
            cached = cache / f"lean_split_o{origin}_{window}.csv"
            if cached.exists():
                baseline = pd.read_csv(cached)
            else:
                baseline = STRATEGIES["lean_split"](data.until(cutoff), weeks)
                baseline.to_csv(cached, index=False)
            model = imitation.fit(cutoff)
            for name, scale in VARIANTS.items():
                sub = baseline.copy()
                if scale is not None:
                    plan = imitation.predict(cutoff, scale=scale, model=model)
                    key = ["week", "material"]
                    sub = sub.set_index(key)
                    p = plan.set_index(key)
                    sub.loc[p.index, ["order_units", *SHARE_COLS]] = p[
                        ["order_units", *SHARE_COLS]]
                    sub = sub.reset_index()
                score = score_window(sub, naive_base, data, win)
                rows.append({"origin": origin, "window": window, "cutoff": cutoff,
                             "variant": name, **score.as_dict()})
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
                for col in ("score", "P", "F", "R", "cost", "fill_rate", "diversification")
            }
    report = ROOT / "reports" / "hindsight_imitation_w208.json"
    report.write_text(json.dumps(
        {"origins": ORIGINS, "variants": VARIANTS, "summary": summary,
         "folds": frame.to_dict(orient="records")}, indent=2) + "\n")
    print(report)
    for name in VARIANTS:
        print(name, "dev", round(summary[name]["development"]["score"], 2),
              "later", round(summary[name]["later_holdout"]["score"], 2))


if __name__ == "__main__":
    main()
