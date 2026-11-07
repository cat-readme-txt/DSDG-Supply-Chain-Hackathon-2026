"""Chronological check of event-level scenario ordering against lean_split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import SHARE_COLS, load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402
from experiments.event_calendar import calendar_for  # noqa: E402
from experiments.event_saa_policy import event_plan  # noqa: E402

ORIGINS = (104, 117, 130, 143, 156, 169, 182, 195, 208)


def main():
    data = load_data(ROOT/"data"/"Phase 2 data")
    rows = []
    for origin in ORIGINS:
        weeks = range(origin+1, origin+14)
        info = data.until(origin)
        base = STRATEGIES["lean_split"](info, weeks)
        events = calendar_for(info)["active"]
        reference = naive(info, weeks)
        score0 = score_window(base, reference, data, weeks)
        for scale in (.5, 1.):
            plan = event_plan(info, weeks, count=80, event_scale=scale, seed=origin)
            out = base.set_index(["week", "material"])
            out["order_units"] = out.order_units.astype(float)
            plan = plan.set_index(["week", "material"])
            out.loc[plan.index, ["order_units", *SHARE_COLS]] = plan[["order_units", *SHARE_COLS]]
            out = out.reset_index()
            score1 = score_window(out, reference, data, weeks)
            rows.append(dict(origin=origin, event_scale=scale,
                             events=[x["entity"] for x in events if x["onset_weeks"]],
                             base=score0.score, new=score1.score,
                             delta=score1.score-score0.score,
                             cost_delta=score1.cost-score0.cost,
                             fill_delta=score1.fill_rate-score0.fill_rate,
                             diversity_delta=score1.diversification-score0.diversification))
        print("finished", origin, flush=True)
    report = ROOT/"reports"/"event_saa_backtest.json"
    report.write_text(json.dumps(rows,indent=2)+"\n")
    for row in rows:
        print(row['origin'],row['event_scale'],[x for x in row['events']],
              round(row['delta'],2),round(row['cost_delta']))
    print(report)


if __name__ == "__main__":
    main()
