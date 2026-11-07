"""Export the week-208 news state as an auditable event calendar.

The existing script forecaster already parses news. This exposes its latent
stories, cancellation state, possible onset weeks, affected suppliers, and
duration survival so correlated policy scenarios can be checked explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "lean_split_67_74" / "src"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.features.build import build_features  # noqa: E402
from dsdg_supply_chain.models.duration import DurationModel  # noqa: E402
from dsdg_supply_chain.models.scripts import ScriptModel, _targets, stories  # noqa: E402


def main() -> None:
    info = load_data(ROOT / "data" / "snapshots" / "w208")
    if info.last_week != 208:
        raise ValueError("This report must use the frozen week-208 snapshot")
    model = ScriptModel.fit(info)
    duration = DurationModel.fit(build_features(info), sigma_min=.6)
    active = []
    for r in model.open_stories.itertuples():
        stage_week = int(getattr(r, f"w{r.stage}"))
        pmf = model.onset_pmf(r.type, r.stage, info.last_week-stage_week, 13)
        target = _targets(info.suppliers, r.type, r.entity)
        survival = [float(duration.survival(
            pd.Series([r.type]), np.array([1.]), np.array([d-1.]))[0])
                    for d in range(1, 14)]
        active.append(dict(type=r.type, entity=r.entity, stage=int(r.stage),
                           first_week=int(r.start), last_news_week=int(r.last),
                           conversion_probability=model.conv.get((r.type, r.stage)),
                           onset_weeks={str(209+i): float(p) for i, p in enumerate(pmf) if p > 0},
                           duration_survival={str(d): p for d, p in enumerate(survival, 1)},
                           affected=target[["material", "country", "lane"]].to_dict(orient="records")))
    all_stories = stories(info)
    recent_cancellations = all_stories[(all_stories.called_off.notna())
                                       & (all_stories.called_off >= 200)]
    cancelled = recent_cancellations[["type", "entity", "called_off"]].to_dict(orient="records")
    output = dict(information_week=208, active=active, recently_cancelled=cancelled)
    path = ROOT / "reports" / "event_calendar_w208.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    for row in active:
        print(row["type"], row["entity"], "stage", row["stage"],
              "conversion", round(row["conversion_probability"], 3),
              "affected", len(row["affected"]))
    print("recently cancelled", cancelled)
    print(path)


if __name__ == "__main__":
    main()
