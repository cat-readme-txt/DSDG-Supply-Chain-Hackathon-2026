"""Backtest the ordering rule of the teammate's independent_w208.csv.

Rules recovered from the file: per 13-week window, order (1.08 + 0.6 x P(disrupted)) x demand for
the first 8 weeks and nothing for the last 5; even split across the 3 suppliers. Their probability
model is not available, so our disruption probabilities drive the multiplier (and fill the forecast
columns). Compared with the strategies' saved backtests, and as a best-of-two partner.
"""
import numpy as np
import pandas as pd

from dsdg_supply_chain.backtest import run_backtest
from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.strategies import _apply_forecasts, _forecasts, naive


def independent_like(info, weeks):
    sub = naive(info, weeks)
    if weeks[-1] <= info.last_week:
        return sub
    _, _, mat_p, px = _forecasts(info, weeks)
    sub = _apply_forecasts(sub, mat_p, px)
    pos = (sub.week - weeks[0]) % 13
    sub["order_units"] = np.where(pos < 8, (1.08 + 0.6 * sub.disruption_prob) * sub.order_units, 0.0)
    for m in sub.material.unique():
        idx = sub.material == m
        sub.loc[idx, [c for c in sub if c.startswith("share_")]] = 0.0
        for c in info.suppliers_of(m):
            sub.loc[idx, f"share_{c}"] = 1 / 3
    return sub


r = run_backtest(independent_like, load_data())
r.to_csv("reports/backtest_independent_like.csv", index=False)
S = {"independent_like": r.set_index(["origin", "window"]).score}
for k in ["teammate_v1", "teammate_hybrid", "policy_v1", "lean_lp", "insured"]:
    S[k] = pd.read_csv(f"reports/backtest_{k}.csv").set_index(["origin", "window"]).score
S = pd.DataFrame(S)
pd.set_option("display.width", 200)
print(S.groupby(level=1).mean().T.assign(mean=S.mean(), worst=S.min()).round(2).sort_values("mean", ascending=False))
for a, b in [("lean_lp", "insured"), ("lean_lp", "independent_like"), ("independent_like", "insured")]:
    m = S[[a, b]].max(axis=1)
    print(f"best-of-two {a} + {b}: mean {m.mean():.2f}  public {m.xs('public', level=1).mean():.2f}  "
          f"private {m.xs('private', level=1).mean():.2f}  worst {m.min():.2f}")
