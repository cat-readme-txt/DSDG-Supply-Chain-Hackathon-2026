"""Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline.

For an origin week o (playing the role of week 208):
  * the naive baseline is built from data up to o, covering weeks o+1 .. o+26 (like the sample);
  * "public"  window o+1  .. o+13 is scored on a strategy run with data up to o      (Phase 1);
  * "private" window o+14 .. o+26 is scored on a strategy run with data up to o+13   (Phase 2),
    still against the stale naive baseline, exactly as the final private leaderboard is.
"""
from __future__ import annotations

import json

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
            rows.append({"origin": o, "window": name, **ws.as_dict(),
                         "per_material": json.dumps(ws.per_material)})
    return pd.DataFrame(rows)


def rescore(res: pd.DataFrame, F_agg: str = "pooled", fill_agg: str = "pooled", calm: str = "zero") -> pd.Series:
    """Recompute window scores under another reading of the metric from the saved per-material rows.
    F_agg / fill_agg: 'pooled' over materials or 'mean' of per-material values; calm: fill skill of a
    material (or pool) where naive never runs short: 'zero', 'one' (if we don't either) or 'skip'."""
    import numpy as np
    out = []
    for r in res.itertuples():
        m = pd.DataFrame(json.loads(r.per_material))
        if F_agg == "pooled":
            F = (m.naive_cost.sum() - m.cost.sum()) / max(m.naive_cost.sum() - m.hindsight_cost.sum(), 1e-9)
        else:
            F = ((m.naive_cost - m.cost) / (m.naive_cost - m.hindsight_cost).clip(lower=1e-9)).clip(0, 1).mean()

        def fs(sv, svn, ds):
            if ds <= 0 or svn >= ds - 1e-9:
                return {"zero": 0.0, "one": float(sv >= ds - 1e-9), "skip": np.nan}[calm]
            return (sv - svn) / (ds - svn)
        if fill_agg == "pooled":
            fill = fs(m.served.sum(), m.served_naive.sum(), m.shock_demand.sum())
            fill = 0.0 if np.isnan(fill) else fill
        else:
            v = [fs(a, b, c) for a, b, c in zip(m.served, m.served_naive, m.shock_demand)]
            fill = 0.0 if np.all(np.isnan(v)) else float(np.nanmean(v))
        R = np.clip(0.7 * fill + 0.3 * r.diversification, 0, 1)
        out.append(100 * (0.3 * r.P + 0.4 * np.clip(F, 0, 1) + 0.3 * R))
    return pd.Series(out, index=res.index)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["score", "P", "F", "R", "brier_skill", "price_skill", "fill_skill", "diversification"]
    return results.groupby("window")[cols].mean()
