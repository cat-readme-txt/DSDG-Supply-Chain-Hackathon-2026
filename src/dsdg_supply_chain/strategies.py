"""Strategies: functions mapping an information snapshot to a submission.

A strategy has the signature `(info: Data, weeks: range) -> DataFrame` and may only use data in
`info`, which `Data.until` has already cut off at the information week (demand stays complete).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .data import Data
from .submission import empty_submission

Strategy = Callable[[Data, range], pd.DataFrame]


def naive(info: Data, weeks: range) -> pd.DataFrame:
    """The sample submission's policy: historical disruption rate, last benchmark price, order
    exactly the demand, 100% from the primary (largest-capacity) supplier."""
    sub = empty_submission(weeks)
    rate = info.outcomes.groupby("material").disrupted.mean()
    last = info.outcomes[info.outcomes.week == info.last_week].set_index("material").benchmark_price
    dem = info.demand.set_index(["week", "material"]).demand_units
    sub["disruption_prob"] = sub.material.map(rate)
    sub["price_forecast"] = sub.material.map(last)
    sub["order_units"] = [float(dem.loc[(w, m)]) for w, m in zip(sub.week, sub.material)]
    for m in sub.material.unique():
        sub.loc[sub.material == m, f"share_{info.suppliers_of(m)[0]}"] = 1.0
    return sub


def disruption_v1(info: Data, weeks: range) -> pd.DataFrame:
    """Naive prices and ordering, but disruption probabilities from the disruption model."""
    from .models.disruption import predict_disruption
    sub = naive(info, weeks)
    p = predict_disruption(info, weeks).set_index(["week", "material"]).p
    sub["disruption_prob"] = [float(np.clip(p.loc[(w, m)], 0, 1)) for w, m in zip(sub.week, sub.material)]
    return sub


def _forecasts(info: Data, weeks: range):
    """Shared step 1-2 outputs: features, supplier forecast, material P(disrupted), prices."""
    from .features.build import build_features
    from .models.disruption import predict_disruption, supplier_forecast
    from .models.price import predict_price
    feats = build_features(info)
    sup = supplier_forecast(info, weeks, feats=feats)
    mat_p = predict_disruption(info, weeks, sup=sup)
    px = predict_price(info, weeks, sup, feats=feats)
    return feats, sup, mat_p, px


def _apply_forecasts(sub: pd.DataFrame, mat_p: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    key = list(zip(sub.week, sub.material))
    p = mat_p.set_index(["week", "material"]).p
    pr = px.set_index(["week", "material"]).price
    sub["disruption_prob"] = np.clip([p.loc[k] for k in key], 0, 1)
    sub["price_forecast"] = [pr.loc[k] for k in key]
    return sub


def forecast_v1(info: Data, weeks: range) -> pd.DataFrame:
    """Naive ordering; disruption probabilities and prices from the models (steps 1 and 2)."""
    sub = naive(info, weeks)
    if weeks[-1] <= info.last_week:
        return sub
    _, _, mat_p, px = _forecasts(info, weeks)
    return _apply_forecasts(sub, mat_p, px)


def _with_plan(info: Data, weeks: range, planner) -> pd.DataFrame:
    """Forecast columns from steps 1-2, orders/shares from `planner(info, plan_weeks, feats, sup, mat_p)`
    for whole windows that lie entirely in the future (past windows keep the naive rows)."""
    sub = naive(info, weeks)
    if weeks[-1] <= info.last_week:
        return sub
    feats, sup, mat_p, px = _forecasts(info, weeks)
    sub = _apply_forecasts(sub, mat_p, px)
    starts = [s for s in range(weeks[0], weeks[-1] + 1, 13) if s > info.last_week]
    if not starts:
        return sub
    plan_weeks = range(starts[0], weeks[-1] + 1)
    plan = planner(info, plan_weeks, feats, sup[sup.week > info.last_week], mat_p).set_index(["week", "material"])
    cols = ["order_units", *[c for c in sub if c.startswith("share_")]]
    sub = sub.set_index(["week", "material"])
    sub.loc[plan.index, cols] = plan[cols]
    return sub.reset_index()[["id", "week", "material", *[c for c in sub.columns if c != "id"]]]


def policy_v1(info: Data, weeks: range, params=None) -> pd.DataFrame:
    """Steps 1-3: model forecasts plus the lean, risk-aware rule-based ordering plan."""
    from .policy import PolicyParams, plan_orders
    return _with_plan(info, weeks, lambda i, w, f, s, m: plan_orders(i, w, f, s, m, params or PolicyParams()))


# --- the two final-submission specialists (eda/16_pair_tune.py) -----------------------------------
# Kaggle scores the better of the 2 selected submissions, so we submit two complementary policies:
#   lean:    stochastic-programming LP on the forecast scenarios; cheapest plan, best when no
#            unwarned shock hits (calm windows and warned shocks)
#   insured: rule-based planner carrying a tapering mid-window buffer and diversifying with risk;
#            best when an unwarned shock hits the primary supplier
# Backtest (22 windows): lean 46.3, insured 44.7, best-of-two 58.6 (public 56.7, private 60.5).
def lean_lp(info: Data, weeks: range) -> pd.DataFrame:
    from .policy_lp import LPParams, plan_orders_lp
    P = LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0, cap_derate=0.96)
    return _with_plan(info, weeks, lambda i, w, f, s, m: plan_orders_lp(i, w, f, s, P))


INSURED_PARAMS = dict(safety_base=0.25, safety_risk=4.0, cost_temp=10.0, risk_aversion=4.0, diversify=0.0,
                      overorder=0.333, diversify_risk=2.5, buffer_mid=2.251, taper=2.167)


def insured(info: Data, weeks: range) -> pd.DataFrame:
    from .policy import PolicyParams
    return policy_v1(info, weeks, PolicyParams(**INSURED_PARAMS))


def teammate_v1(info: Data, weeks: range) -> pd.DataFrame:
    """Replica of the teammate's primary_w208.csv rules, so the approach can be backtested.

    Per 13-week window: order 1.2 x demand for the first 8 weeks, nothing for the last 5;
    even split across the 3 suppliers; probability = historical rate x (1 - 0.8^h);
    price = EWMA(alpha 0.7) of the benchmark, flat.
    """
    sub = naive(info, weeks)
    h = sub.week - info.last_week
    sub["disruption_prob"] = sub.disruption_prob * (1 - 0.8 ** h.clip(lower=1))
    ewm = info.outcomes.sort_values("week").groupby("material").benchmark_price.apply(
        lambda s: s.ewm(alpha=0.7).mean().iloc[-1])
    sub["price_forecast"] = sub.material.map(ewm)
    pos = (sub.week - weeks[0]) % 13
    sub["order_units"] = np.where(pos < 8, 1.2 * sub.order_units, 0.0)
    for m in sub.material.unique():
        idx = sub.material == m
        sub.loc[idx, [c for c in sub if c.startswith("share_")]] = 0.0
        for c in info.suppliers_of(m):
            sub.loc[idx, f"share_{c}"] = 1 / 3
    return sub


def teammate_hybrid(info: Data, weeks: range) -> pd.DataFrame:
    """Teammate's ordering rules (teammate_v1) with our forecast columns (steps 1-2).
    Reproduces teammate_forecasts_primary_policy_w208.csv."""
    sub = teammate_v1(info, weeks)
    if weeks[-1] <= info.last_week:
        return sub
    _, _, mat_p, px = _forecasts(info, weeks)
    return _apply_forecasts(sub, mat_p, px)


STRATEGIES: dict[str, Strategy] = {"teammate_hybrid": teammate_hybrid,"naive": naive, "disruption_v1": disruption_v1, "forecast_v1": forecast_v1,
                                   "policy_v1": policy_v1, "lean_lp": lean_lp, "insured": insured,
                                   "teammate_v1": teammate_v1}
