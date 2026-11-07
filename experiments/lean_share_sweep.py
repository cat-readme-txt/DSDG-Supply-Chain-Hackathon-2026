"""Chronological test of diversification changes to the reproduced lean policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods" / "lean_60_66" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import lean_hedged, naive  # noqa: E402


ORIGINS = [52, 78, 104, 130, 156, 182]
VARIANTS = {
    "original": ("constant", 0.0),
    "equal_10pct": ("constant", 0.1),
    "equal_20pct": ("constant", 0.2),
    "equal_40pct": ("constant", 0.4),
    "equal_60pct": ("constant", 0.6),
    "equal_100pct": ("constant", 1.0),
    "risk_1p5": ("risk", 1.5),
    "risk_3": ("risk", 3.0),
}


def diversify(sub: pd.DataFrame, info, weeks: range, mode: str, amount: float) -> pd.DataFrame:
    out = sub.copy()
    mask = out.week.isin(weeks) & (out.order_units > 0)
    if mode == "constant":
        delta = pd.Series(amount, index=out.index)
    else:
        delta = (amount * out.disruption_prob).clip(upper=0.6)
    for material in out.material.unique():
        rows = mask & (out.material == material)
        countries = info.suppliers_of(material)
        cols = [f"share_{country}" for country in countries]
        out.loc[rows, cols] = (
            out.loc[rows, cols].to_numpy(float) * (1 - delta[rows].to_numpy()[:, None])
            + delta[rows].to_numpy()[:, None] / len(countries)
        )
    return out


def main() -> None:
    data = load_data(Path("data"))
    cache = Path("/private/tmp/dsdg-lean-share-folds")
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 27)
        base = naive(data.until(origin), weeks)
        for window, info_week, win in (
            ("public", origin, range(origin + 1, origin + 14)),
            ("private", origin + 13, range(origin + 14, origin + 27)),
        ):
            cached = cache / f"lean_hedged_o{origin}_{window}.csv"
            info = data.until(info_week)
            if cached.exists():
                sub = pd.read_csv(cached)
            else:
                sub = lean_hedged(info, weeks)
                sub.to_csv(cached, index=False)
            for name, (mode, amount) in VARIANTS.items():
                choice = diversify(sub, info, win, mode, amount)
                score = score_window(choice, base, data, win)
                rows.append({"origin": origin, "window": window, "variant": name,
                             **score.as_dict()})
            print(f"finished origin {origin} {window}", flush=True)
    result = pd.DataFrame(rows)
    out = Path("reports/lean_share_sweep_w208.json")
    summary = {}
    for name in VARIANTS:
        variant = result[result.variant == name]
        summary[name] = {}
        for label, mask in (
            ("development", variant.origin <= 130),
            ("later_holdout", variant.origin >= 156),
            ("later_public", (variant.origin >= 156) & (variant.window == "public")),
            ("later_private", (variant.origin >= 156) & (variant.window == "private")),
        ):
            subset = variant[mask]
            summary[name][label] = {
                column: float(subset[column].mean())
                for column in ("score", "P", "F", "R", "cost", "fill_rate", "diversification")
            }
    out.write_text(json.dumps({"origins": ORIGINS, "variants": VARIANTS,
                               "summary": summary,
                               "folds": result.to_dict(orient="records")}, indent=2) + "\n")
    print(out)
    for name in VARIANTS:
        print(name, "development", round(summary[name]["development"]["score"], 2),
              "later_holdout", round(summary[name]["later_holdout"]["score"], 2))


if __name__ == "__main__":
    main()
