"""Export the latest released news state as an auditable event calendar.

The existing script forecaster already parses news. This exposes its latent
stories, cancellation state, possible onset weeks, affected suppliers, and
duration survival so correlated policy scenarios can be checked explicitly.
"""

from __future__ import annotations

import argparse
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


def calendar_for(info) -> dict:
    """Open-story onset and duration distributions at a chronological cutoff."""
    origin = info.last_week
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
                           onset_weeks={str(origin+1+i): float(p) for i, p in enumerate(pmf) if p > 0},
                           duration_survival={str(d): p for d, p in enumerate(survival, 1)},
                           affected=target[["material", "country", "lane"]].to_dict(orient="records")))
    all_stories = stories(info)
    recent_cancellations = all_stories[(all_stories.called_off.notna())
                                       & (all_stories.called_off >= origin-8)]
    cancelled = recent_cancellations[["type", "entity", "called_off"]].to_dict(orient="records")
    return dict(information_week=origin, active=active, recently_cancelled=cancelled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "Phase 2 data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "event_calendar_w221.json")
    args = parser.parse_args()
    output = calendar_for(load_data(args.data_dir))
    path = args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    for row in active:
        print(row["type"], row["entity"], "stage", row["stage"],
              "conversion", round(row["conversion_probability"], 3),
              "affected", len(row["affected"]))
    print("recently cancelled", cancelled)
    print(path)


if __name__ == "__main__":
    main()
