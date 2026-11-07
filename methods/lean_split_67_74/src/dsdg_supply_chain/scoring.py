"""Local replica of the competition metric for one test window.

Score = 100 x (0.30 P + 0.40 F + 0.30 R), each part clipped to [0, 1] and scaled so the naive
submission scores 0.
  P = 0.6 Brier skill + 0.4 price (MAPE) skill, both relative to the naive submission's columns
  F = (naive cost - cost) / (naive cost - hindsight cost)
  R = 0.7 fill-rate skill in shock weeks + 0.3 diversification (1 - HHI, rescaled: single source 0,
      even three-way split 1)

Assumptions: costs and fill rates are pooled across materials; with no shock weeks in the window,
or a naive fill rate of 100%, fill-rate skill is 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SHARE_COLS, Data
from .simulator import Truth, hindsight_plan, simulate, truth_for


def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def policy_arrays(sub: pd.DataFrame, truth: Truth) -> tuple[np.ndarray, np.ndarray]:
    """Orders [T] and shares [S, T] for one material, ordered like the truth arrays."""
    rows = sub[sub.material == truth.material].set_index("week").loc[truth.weeks]
    orders = rows.order_units.to_numpy(float)
    shares = rows[[f"share_{c}" for c in truth.countries]].to_numpy(float).T
    return orders, shares


def diversification(sub: pd.DataFrame) -> float:
    sh =sub[SHARE_COLS].to_numpy(float)
    hhi = (sh ** 2).sum(axis=1)
    return float(np.clip((1 - hhi) / (1 - 1 / 3), 0, 1).mean())


@dataclass
class WindowScore:
    weeks: range
    brier_skill: float
    price_skill: float
    cost: float
    naive_cost: float
    hindsight_cost: float
    fill_rate: float
    naive_fill_rate: float
    fill_skill: float
    diversification: float
    n_shock_weeks: int
    P: float
    F: float
    R: float

    @property
    def score(self) -> float:
        return 100 * (0.30 * self.P + 0.40 * self.F + 0.30 * self.R)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "weeks"}
        return {"start": self.weeks[0], "end": self.weeks[-1], "score": self.score, **d}


def score_window(sub: pd.DataFrame, naive: pd.DataFrame, truth_data: Data, weeks: range) -> WindowScore:
    """Score the rows of `sub` that fall in `weeks` against released truth in `truth_data`."""
    sub = sub[sub.week.isin(weeks)]
    naive = naive[naive.week.isin(weeks)]
    key = ["week", "material"]
    out = truth_data.outcomes[truth_data.outcomes.week.isin(weeks)][key + ["disrupted", "benchmark_price"]]
    m = out.merge(sub, on=key).merge(naive[key + ["disruption_prob", "price_forecast"]], on=key,
                                     suffixes=("", "_naive"))
    if len(m) != len(out):
        raise ValueError("submission does not cover every (week, material) in the window")
    y, px = m.disrupted.to_numpy(float), m.benchmark_price.to_numpy(float)
    bs, bs_n = np.mean((m.disruption_prob - y) ** 2), np.mean((m.disruption_prob_naive - y) ** 2)
    mape, mape_n = (np.mean(np.abs(m.price_forecast - px) / px),
                    np.mean(np.abs(m.price_forecast_naive - px) / px))
    brier_skill = 1 - bs / bs_n if bs_n > 0 else 0.0
    price_skill = 1 - mape / mape_n if mape_n > 0 else 0.0

    cost = naive_cost = hind = 0.0
    served = served_n = shock_demand = 0.0
    n_shock = 0
    for mat in sub.material.unique():
        truth = truth_for(truth_data, mat, weeks)
        mine = simulate(truth, *policy_arrays(sub, truth))
        base = simulate(truth, *policy_arrays(naive, truth))
        cost += mine.total_cost
        naive_cost += base.total_cost
        hind += hindsight_plan(truth)[0]
        sw = truth.shock_week
        n_shock += int(sw.sum())
        served += mine.served[sw].sum()
        served_n += base.served[sw].sum()
        shock_demand += truth.demand[sw].sum()

    F = (naive_cost - cost) / (naive_cost - hind) if naive_cost - hind > 1e-9 else 0.0
    fr = served / shock_demand if shock_demand else 1.0
    fr_n = served_n / shock_demand if shock_demand else 1.0
    fill_skill = (fr - fr_n) / (1 - fr_n) if (shock_demand and 1 - fr_n > 1e-9) else 0.0
    div = diversification(sub)  # naive single-sources, so its diversification is 0
    return WindowScore(weeks=weeks, brier_skill=brier_skill, price_skill=price_skill, cost=cost,
                       naive_cost=naive_cost, hindsight_cost=hind, fill_rate=fr, naive_fill_rate=fr_n,
                       fill_skill=fill_skill, diversification=div, n_shock_weeks=n_shock,
                       P=_clip(0.6 * brier_skill + 0.4 * price_skill), F=_clip(F),
                       R=_clip(0.7 * fill_skill + 0.3 * div))
