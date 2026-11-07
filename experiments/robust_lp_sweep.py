"""Chronological comparison of lean LP risk assumptions; no future labels in plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods" / "lean_60_66" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.policy import supplier_state  # noqa: E402
from dsdg_supply_chain.policy_lp import LPParams, plan_orders_lp  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import (  # noqa: E402
    _apply_forecasts, _forecasts, hedge_forecasts, naive, story_materials,
)
from experiments.lean_share_sweep import ORIGINS  # noqa: E402


PARAMS = {
    "original": LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0, cap_derate=0.96),
    "min_share_05": LPParams(n_scenarios=30, min_share=0.05, hazard_mult=0.0, cap_derate=0.96),
    "hazard_25": LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.25, cap_derate=0.96),
    "hazard_50": LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.50, cap_derate=0.96),
    "risk_mult_125": LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0,
                              cap_derate=0.96, risk_mult=1.25),
    "cap_derate_90": LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0, cap_derate=0.90),
}


def main() -> None:
    data = load_data(Path("data"))
    rows = []
    for origin in ORIGINS:
        weeks = range(origin + 1, origin + 27)
        base = naive(data.until(origin), weeks)
        for window, info_week, win in (
            ("public", origin, range(origin + 1, origin + 14)),
            ("private", origin + 13, range(origin + 14, origin + 27)),
        ):
            info = data.until(info_week)
            feats, sup, mat_p, px = _forecasts(info, weeks)
            pre = _apply_forecasts(naive(info, weeks), mat_p, px)
            future_sup = sup[sup.week > info_week]
            state = supplier_state(info, feats, future_sup)
            mats = story_materials(info)
            for name, params in PARAMS.items():
                plan = plan_orders_lp(info, win, feats, future_sup, params, state=state)
                cols = ["order_units", *SHARE_COLS]
                sub = pre.set_index(["week", "material"]).copy()
                p = plan.set_index(["week", "material"])
                sub.loc[p.index, cols] = p[cols]
                sub = sub.reset_index()
                sub = hedge_forecasts(sub, info, "calm", mats=mats)
                score = score_window(sub, base, data, win)
                rows.append({"origin": origin, "window": window, "variant": name,
                             **score.as_dict()})
            print(f"finished origin {origin} {window}", flush=True)
    result = pd.DataFrame(rows)
    summary = {}
    for name in PARAMS:
        variant = result[result.variant == name]
        summary[name] = {}
        for label, mask in (("development", variant.origin <= 130),
                            ("later_holdout", variant.origin >= 156),
                            ("later_public", (variant.origin >= 156) & (variant.window == "public")),
                            ("later_private", (variant.origin >= 156) & (variant.window == "private"))):
            subset = variant[mask]
            summary[name][label] = {
                column: float(subset[column].mean())
                for column in ("score", "P", "F", "R", "cost", "fill_rate", "diversification")
            }
    out = Path("reports/robust_lp_sweep_w208.json")
    out.write_text(json.dumps({"origins": ORIGINS, "params": {k: v.__dict__ for k, v in PARAMS.items()},
                               "summary": summary,
                               "folds": result.to_dict(orient="records")}, indent=2) + "\n")
    print(out)
    for name in PARAMS:
        print(name, "development", round(summary[name]["development"]["score"], 2),
              "later_holdout", round(summary[name]["later_holdout"]["score"], 2))


if __name__ == "__main__":
    main()
