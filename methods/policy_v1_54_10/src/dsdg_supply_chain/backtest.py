"""Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline.

For an origin week o (playing the role of week 208):
  * the naive baseline is built from data up to o, covering weeks o+1 .. o+26 (like the sample);
  * "public"  window o+1  .. o+13 is scored on a strategy run with data up to o      (Phase 1);
  * "private" window o+14 .. o+26 is scored on a strategy run with data up to o+13   (Phase 2),
    still against the stale naive baseline, exactly as the final private leaderboard is.
"""
from __future__ import annotations

import pandas as pd

from .data import WINDOW, Data
from .scoring import score_window
from .strategies import Strategy, naive

MIN_HISTORY = 52


def origins(data: Data, step: int = WINDOW) -> list[int]:
    last = data.last_week - 2 * WINDOW
    return list(range(last, MIN_HISTORY - 1, -step))[::-1]


def run_backtest(strategy: Strategy, data: Data, step: int = WINDOW) -> pd.DataFrame:
    rows = []
    for o in origins(data, step):
        weeks = range(o + 1, o + 2 * WINDOW + 1)
        base = naive(data.until(o), weeks)
        for name, info_week, win in [("public", o, range(o + 1, o + WINDOW + 1)),
                                     ("private", o + WINDOW, range(o + WINDOW + 1, o + 2 * WINDOW + 1))]:
            sub = strategy(data.until(info_week), weeks)
            ws = score_window(sub, base, data, win)
            rows.append({"origin": o, "window": name, **ws.as_dict()})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["score", "P", "F", "R", "brier_skill", "price_skill", "fill_skill", "diversification"]
    return results.groupby("window")[cols].mean()
