"""Primary objective (Phase 2): run a strategy with the information available at week 208 (the
Phase 1 cut) and score weeks 209-221 against the released truth, with the local metric that
reproduces Kaggle (order-weighted diversification: 92.64 vs Kaggle 92.62 on a near-oracle file,
lean_split 67.83 vs 67.74). The same code then runs at week 221 for the private window.

A single 13-week window can be over-fit, so every change is also checked on the 72-window
walk-forward backtest (uv run dsdg backtest <strategy> --step 4) before it is kept.
usage: uv run python eda/32_holdout_208.py strategy [strategy ...]
"""
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.scoring import score_window
from dsdg_supply_chain.strategies import STRATEGIES
from dsdg_supply_chain.submission import read_submission

ORIGIN, WIN = 208, range(209, 222)


def weighted_score(sub, ws):
    s = sub[sub.week.isin(WIN)]
    sh = s[SHARE_COLS].to_numpy(float)
    dv = np.clip((1 - (sh ** 2).sum(1)) / (2 / 3), 0, 1)
    w = s.order_units.to_numpy(float)
    div = float((dv * w).sum() / w.sum()) if w.sum() else 0.0
    R = np.clip(0.7 * ws.fill_skill + 0.3 * div, 0, 1)
    return 100 * (0.3 * ws.P + 0.4 * ws.F + 0.3 * R), div


def holdout(name: str, data=None) -> dict:
    data = data or load_data()
    base = read_submission("data/comp/sample_submission.csv")
    sub = STRATEGIES[name](data.until(ORIGIN), range(209, 235))
    ws = score_window(sub, base, data, WIN)
    score, div = weighted_score(sub, ws)
    pm = pd.DataFrame(ws.per_material)
    return dict(strategy=name, score=score, P=ws.P, brier=ws.brier_skill, price=ws.price_skill, F=ws.F,
                fill=ws.fill_skill, div=div, regret=(pm.cost - pm.hindsight_cost).round(0).tolist())


if __name__ == "__main__":
    data = load_data()
    rows = [holdout(n, data) for n in sys.argv[1:]]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).drop(columns="regret").round(4).to_string(index=False))
