"""Chronological comparison of empirical supplier allocation against 67.74."""

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
from scratch.empirical_allocator import EmpiricalAllocator  # noqa: E402

ORIGINS = [78, 104, 130, 156, 182]
VARIANTS = {
    "baseline": (None, 0.0),
    "empirical_mean": ("mean", 1.0),
    "empirical_robust": ("robust", 1.0),
    "empirical_diverse": ("diverse", 1.0),
    "half_robust": ("robust", 0.5),
}


def main() -> None:
    data_dir = ROOT / "data" / "snapshots" / "w208"
    data = load_data(data_dir)
    allocator = EmpiricalAllocator(data_dir)
    cache = Path("/private/tmp/dsdg-scratch-folds")
    cache.mkdir(exist_ok=True)
    rows = []
    allocations = []
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
            chosen = {}
            for material in allocator.suppliers.material.unique():
                subset = baseline[
                    (baseline.material == material) & baseline.week.isin(win)
                ].sort_values("week")
                orders = subset.order_units.to_numpy(float)
                chosen[material] = {
                    mode: allocator.choose(material, cutoff, orders, mode)
                    for mode in ("mean", "robust", "diverse")
                }
            allocations.append({"origin": origin, "window": window, "cutoff": cutoff,
                                "shares": chosen})
            for name, (mode, blend) in VARIANTS.items():
                sub = baseline.copy()
                if mode is not None:
                    for material, options in chosen.items():
                        mask = ((sub.material == material) & sub.week.isin(win)
                                & (sub.order_units > 0))
                        for country, target in options[mode].items():
                            col = f"share_{country}"
                            sub.loc[mask, col] = (
                                (1 - blend) * sub.loc[mask, col].to_numpy(float)
                                + blend * float(target))
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
    report = ROOT / "reports" / "empirical_allocator_w208.json"
    report.write_text(json.dumps(
        {"origins": ORIGINS, "variants": VARIANTS, "summary": summary,
         "folds": frame.to_dict(orient="records"), "allocations": allocations},
        indent=2) + "\n")
    print(report)
    for name in VARIANTS:
        print(name, "dev", round(summary[name]["development"]["score"], 2),
              "later", round(summary[name]["later_holdout"]["score"], 2))


if __name__ == "__main__":
    main()
