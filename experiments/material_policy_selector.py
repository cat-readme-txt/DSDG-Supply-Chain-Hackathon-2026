"""Select purchasing policy independently by material on early released folds.

For each material, swap only orders and shares from an alternative into the
lean-split plan and score the entire window. Select on the first eight
chronological windows and evaluate the resulting fixed choices on four later
untouched windows. Forecast columns always stay with lean-split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402

ORIGINS = (52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195)
ALTERNATIVES = ("lean_hedged", "insured_hedged")
POLICY_COLS = ["order_units", *SHARE_COLS]


def swap(base: pd.DataFrame, alt: pd.DataFrame, material: str) -> pd.DataFrame:
    result = base.copy().set_index(["week", "material"])
    other = alt.set_index(["week", "material"])
    index = result.index.get_level_values("material") == material
    result.loc[index, POLICY_COLS] = other.loc[result.index[index], POLICY_COLS].to_numpy()
    return result.reset_index()


def main() -> None:
    data = load_data(ROOT / "data" / "snapshots" / "w208")
    windows = []
    rows = []
    for origin in ORIGINS:
        info = data.until(origin)
        weeks = range(origin+1, origin+14)
        base = STRATEGIES["lean_split"](info, weeks)
        alternatives = {name: STRATEGIES[name](info, weeks) for name in ALTERNATIVES}
        naive_sub = naive(info, weeks)
        base_score = score_window(base, naive_sub, data, weeks)
        window = dict(origin=origin, base=base, alternatives=alternatives,
                      naive=naive_sub, weeks=weeks, base_score=base_score)
        windows.append(window)
        for material in sorted(base.material.unique()):
            for name, alt in alternatives.items():
                trial = swap(base, alt, material)
                score = score_window(trial, naive_sub, data, weeks)
                rows.append(dict(origin=origin, material=material, alternative=name,
                                 delta=score.score-base_score.score,
                                 cost_delta=score.cost-base_score.cost,
                                 fill_delta=score.fill_rate-base_score.fill_rate,
                                 div_delta=score.diversification-base_score.diversification))
        print("finished", origin, flush=True)
    table = pd.DataFrame(rows)
    choices = {}
    for material in sorted(table.material.unique()):
        early = table[(table.material == material) & (table.origin <= 143)]
        score = early.groupby("alternative").delta.mean()
        best = score.idxmax()
        if score[best] > .05:
            choices[material] = best
    composite = []
    for window in windows:
        trial = window["base"]
        for material, name in choices.items():
            trial = swap(trial, window["alternatives"][name], material)
        score = score_window(trial, window["naive"], data, window["weeks"])
        composite.append(dict(origin=window["origin"], delta=score.score-window["base_score"].score,
                              score=score.score, base_score=window["base_score"].score))
    output = {"selection_origins": list(ORIGINS[:8]),
              "holdout_origins": list(ORIGINS[8:]),
              "choices": choices, "individual": rows, "composite": composite}
    path = ROOT / "reports" / "material_policy_selector_w208.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print("choices", choices)
    for label, selected in (("development", composite[:8]), ("later_holdout", composite[8:])):
        delta = np.array([r["delta"] for r in selected])
        print(label, "mean", round(delta.mean(), 3), "median", round(np.median(delta), 3),
              "positive", int((delta>0).sum()), "of", len(delta))
    for material in sorted(table.material.unique()):
        t = table[table.material == material]
        print(material, {name: (round(t[(t.alternative==name)&(t.origin<=143)].delta.mean(), 2),
                               round(t[(t.alternative==name)&(t.origin>=156)].delta.mean(), 2))
                         for name in ALTERNATIVES})
    print(path)


if __name__ == "__main__":
    main()
