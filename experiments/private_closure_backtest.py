"""Chronological Brier check for closure-conditional forecasts.

Every information snapshot is restricted to its origin week; later outcomes
are read only after forecasts are formed for scoring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "sk4_phase2" / "src"))
os.environ.setdefault("DSDG_DATA_DIR", str(ROOT / "data" / "Phase 2 data"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.models import scripts  # noqa: E402
from dsdg_supply_chain.models.scripts import ScriptModel  # noqa: E402
from dsdg_supply_chain.strategies import _forecasts  # noqa: E402


def main() -> None:
    data = load_data()
    rows = []
    for origin in (147, 148, 149, 150, 171, 172, 203, 204, 205, 206):
        info = data.until(origin)
        live = ScriptModel.fit(info).open_stories
        closure = live[live.type == "closure"]
        if len(closure) != 1:
            continue
        entity = str(closure.entity.iloc[0])
        weeks = range(origin + 1, origin + 14)
        _, _, base, _ = _forecasts(info, weeks, market=True)
        old = dict(scripts.CONV_OVERRIDE)
        try:
            scripts.CONV_OVERRIDE["closure"] = 0.0
            _, _, absent, _ = _forecasts(info, weeks, market=True)
            scripts.CONV_OVERRIDE["closure"] = 1.0
            _, _, certain, _ = _forecasts(info, weeks, market=True)
        finally:
            scripts.CONV_OVERRIDE.clear()
            scripts.CONV_OVERRIDE.update(old)
        index = ["week", "material"]
        p = base.set_index(index).p
        delta = (certain.set_index(index).p - p).clip(lower=0)
        y = data.outcomes[data.outcomes.week.isin(weeks)].set_index(index).disrupted.reindex(p.index)
        assert y.notna().all()
        rate = info.outcomes.groupby("material").disrupted.mean().reindex(p.index.get_level_values("material"))
        denominator = ((rate.to_numpy() - y.to_numpy()) ** 2).mean()
        p_no = (p - 0.385 / 0.615 * delta).clip(0, 1)
        p_floor = absent.set_index(index).p
        p_yes = (p + delta).clip(0, 1)
        e0 = ((p - y) ** 2).mean()
        en = ((p_no - y) ** 2).mean()
        ef = ((p_floor - y) ** 2).mean()
        ey = ((p_yes - y) ** 2).mean()
        rows.append(dict(origin=origin, closure=entity, realized=entity == "MERROW",
                         changed=int((delta > 1e-6).sum()),
                         lean_gain_points=18 * (e0 - en) / denominator,
                         lean_floor_gain_points=18 * (e0 - ef) / denominator,
                         insured_gain_points=18 * (e0 - ey) / denominator,
                         base_brier=e0, no_brier=en, floor_brier=ef, yes_brier=ey))
    result = pd.DataFrame(rows)
    dest = ROOT / "reports" / "private_closure_backtest.csv"
    result.to_csv(dest, index=False)
    print(result.round(3).to_string(index=False))
    print("Saved", dest)


if __name__ == "__main__":
    main()
