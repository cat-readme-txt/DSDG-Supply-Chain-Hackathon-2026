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


def _forecasts(info: Data, weeks: range, market: bool = False):
    """Shared step 1-2 outputs: features, supplier forecast, material P(disrupted), prices.
    `market=True`: open news stories are confirmed or down-weighted by market signals (Round 6)."""
    from .features.build import build_features
    from .models.disruption import predict_disruption, supplier_forecast
    from .models.price import predict_price
    feats = build_features(info)
    sup = supplier_forecast(info, weeks, feats=feats, market=market)
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


def _with_plan(info: Data, weeks: range, planner, market: bool = False) -> pd.DataFrame:
    """Forecast columns from steps 1-2, orders/shares from `planner(info, plan_weeks, feats, sup, mat_p)`
    for whole windows that lie entirely in the future (past windows keep the naive rows)."""
    sub = naive(info, weeks)
    if weeks[-1] <= info.last_week:
        return sub
    feats, sup, mat_p, px = _forecasts(info, weeks, market)
    sub = _apply_forecasts(sub, mat_p, px)
    starts = [s for s in range(weeks[0], weeks[-1] + 1, 13) if s > info.last_week]
    if not starts:
        return sub
    plan_weeks = range(starts[0], weeks[-1] + 1)
    if "p_tariff" in sup:   # announced tariffs change cost, not supply: planners see the supply risk only
        from .models.disruption import predict_disruption
        mat_p = predict_disruption(info, weeks, sup=sup, tariff=False)
    plan = planner(info, plan_weeks, feats, sup[sup.week > info.last_week], mat_p).set_index(["week", "material"])
    cols = ["order_units", *[c for c in sub if c.startswith("share_")]]
    sub = sub.set_index(["week", "material"])
    sub.loc[plan.index, cols] = plan[cols]
    return sub.reset_index()[["id", "week", "material", *[c for c in sub.columns if c != "id"]]]


def policy_v1(info: Data, weeks: range, params=None, market: bool = False) -> pd.DataFrame:
    """Steps 1-3: model forecasts plus the lean, risk-aware rule-based ordering plan."""
    from .policy import PolicyParams, plan_orders
    return _with_plan(info, weeks, lambda i, w, f, s, m: plan_orders(i, w, f, s, m, params or PolicyParams()),
                      market)


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


def insured(info: Data, weeks: range, market: bool = False) -> pd.DataFrame:
    from .policy import PolicyParams
    return policy_v1(info, weeks, PolicyParams(**INSURED_PARAMS), market)


# --- forecast hedging across the pair (eda/18_pair_hedge.py) --------------------------------------
# Kaggle keeps the better of the two files, so each file's forecast columns should be sharp in the
# world where that file wins. Lean wins when no unwarned shock arrives, insured when one does.
# Beyond the first few weeks no news is visible (Brier skill ~0 past week 8), so:
#   calm side:  probabilities beyond week CALM_K are scaled by CALM_A
#   shock side: probabilities beyond week SHOCK_K move toward 1 by SHOCK_B x the material's
#               relative historical rate (materials that get hit more often move more); prices
#               ramp up to +SHOCK_PRICE by the end of the window (shock uplift)
# Backtest best-of-two: 59.5 -> 63.2.
CALM_A, CALM_K = 0.5, 2
SHOCK_B, SHOCK_K, SHOCK_PRICE = 0.5, 4, 0.05


def story_materials(info: Data, include_hits: bool = False) -> set[str]:
    """Materials with a live news story at the information week (a warned shock that has not started
    yet; optionally also materials hit now). Their probabilities come from evidence, so the hedge
    leaves them alone. Ongoing shocks are hedged by default: how long they last is the uncertain part."""
    from .labels import supplier_state
    from .models.scripts import ScriptModel, _targets
    st = supplier_state(info)
    mats = set(st[(st.week == info.last_week) & st.hit].material) if include_hits else set()
    sm = ScriptModel.fit(info)
    for r in sm.open_stories.itertuples():
        elapsed = info.last_week - int(getattr(r, f"w{r.stage}"))
        if sm.onset_pmf(r.type, r.stage, elapsed, 26).sum() > 0:      # dead (overdue) rumors don't count
            mats |= set(_targets(info.suppliers, r.type, r.entity).material)
    return mats


def hedge_forecasts(sub: pd.DataFrame, info: Data, side: str, mats: set[str] | None = None) -> pd.DataFrame:
    """Hedge the background (no-evidence) rows: calm side shrinks them, shock side raises them."""
    sub = sub.copy()
    mats = story_materials(info) if mats is None else mats
    bg = ~sub.material.isin(mats).to_numpy()
    h = (sub.week - info.last_week).to_numpy()
    p = sub.disruption_prob.to_numpy(float)
    if side == "calm":
        p = np.where(bg & (h > CALM_K), CALM_A * p, p)
    elif side == "shock":
        rate = info.outcomes.groupby("material").disrupted.mean()
        w = sub.material.map((rate / rate.mean()).clip(0, 2)).to_numpy(float)
        p = np.where(bg & (h > SHOCK_K), p + SHOCK_B * w * (1 - p), p)
        ramp = np.clip(h / 13, 0, 1) * (h > 0)
        sub["price_forecast"] = sub.price_forecast * (1 + SHOCK_PRICE * ramp)
    else:
        raise ValueError(side)
    sub["disruption_prob"] = np.clip(p, 0, 1)
    return sub


def lean_hedged(info: Data, weeks: range) -> pd.DataFrame:
    return hedge_forecasts(lean_lp(info, weeks), info, "calm")


def _lean_split(premium: float, share: float, kappa: float = 0.0, side: str = "calm", market: bool = False, **lp):
    """lean_hedged plus the near-free split: a supplier within `premium` of the cheapest gets at
    least `share` of every order. `market=True`: market-checked story probabilities (Round 6)."""
    def strat(info: Data, weeks: range) -> pd.DataFrame:
        from .policy_lp import LPParams, plan_orders_lp
        base = dict(n_scenarios=30, min_share=0.0, hazard_mult=0.0, cap_derate=0.96)
        P = LPParams(**{**base, **lp}, cheap_premium=premium, cheap_share=share, fill_kappa=kappa)
        return hedge_forecasts(_with_plan(info, weeks, lambda i, w, f, s, m: plan_orders_lp(i, w, f, s, P),
                                          market), info, side)
    return strat


def insured_hedged(info: Data, weeks: range) -> pd.DataFrame:
    return hedge_forecasts(insured(info, weeks), info, "shock")


def insured_hedged_mkt(info: Data, weeks: range) -> pd.DataFrame:
    """insured_hedged with market-checked story probabilities (Round 6)."""
    return hedge_forecasts(insured(info, weeks, market=True), info, "shock")


def _market_columns(base: Strategy, side: str) -> Strategy:
    """Orders and shares from `base`; forecast columns from the market-checked model, then hedged
    for `side`. Round 6: market-checked probabilities sharpen P (8 of 66 windows better, none
    worse), but fed into the planner they hurt orders, so the planner keeps its usual inputs."""
    def strat(info: Data, weeks: range) -> pd.DataFrame:
        sub = base(info, weeks)
        if weeks[-1] <= info.last_week:
            return sub
        _, _, mat_p, px = _forecasts(info, weeks, market=True)
        return hedge_forecasts(_apply_forecasts(sub, mat_p, px), info, side)
    return strat


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
                                   "teammate_v1": teammate_v1,
                                   "lean_hedged": lean_hedged, "insured_hedged": insured_hedged,
                                   "lean_split": _lean_split(0.025, 0.30),
                                   "lean_split_k10": _lean_split(0.025, 0.30, 10.0),
                                   "lean_split_d92": _lean_split(0.025, 0.30, 10.0, cap_derate=0.92),
                                   "lean_split_h25": _lean_split(0.025, 0.30, 10.0, hazard_mult=0.25),
                                   "lean_split_h50": _lean_split(0.025, 0.30, 10.0, hazard_mult=0.5),
                                   "insured_lp_h1": _lean_split(0.025, 0.30, 10.0, "shock", hazard_mult=1.0, cap_derate=0.92),
                                   "insured_lp_h2": _lean_split(0.025, 0.30, 10.0, "shock", hazard_mult=2.0, cap_derate=0.92),
                                   "insured_lp_h2k30": _lean_split(0.025, 0.30, 30.0, "shock", hazard_mult=2.0, cap_derate=0.92),
                                   "insured_lp_h4": _lean_split(0.025, 0.30, 10.0, "shock", hazard_mult=4.0, cap_derate=0.92),
                                   "lean_split_mkt": _lean_split(0.025, 0.30, market=True),
                                   "insured_hedged_mkt": insured_hedged_mkt,
                                   # lean_split whose LP scenarios follow the landed-cost path of their
                                   # shocks (models/cost_path.py): pre-buy before warned price jumps
                                   "lean_split_pxlp": _lean_split(0.025, 0.30, price_path=True),
                                   # ... plus freight x1.6 / x3 on lane shocks (hurt: DECISIONS)
                                   "lean_split_pxlp_fr": _lean_split(0.025, 0.30, price_path=True, path_freight=True)}
def _bet(base: Strategy, conv: dict) -> Strategy:
    """Run `base` with open stories of the given types forced to a conversion (e.g. a lane closure
    treated as real): the second file of the pair bets on the main open uncertainty."""
    def strat(info: Data, weeks: range) -> pd.DataFrame:
        from .models import scripts
        old = dict(scripts.CONV_OVERRIDE)
        scripts.CONV_OVERRIDE.update(conv)
        try:
            return base(info, weeks)
        finally:
            scripts.CONV_OVERRIDE.clear(); scripts.CONV_OVERRIDE.update(old)
    return strat


# Round 6 final candidates: market-checked forecast columns on top of each planner's own orders
STRATEGIES["lean_pxlp_mkt"] = _market_columns(STRATEGIES["lean_split_pxlp"], "calm")
STRATEGIES["insured_hedged_mktcol"] = _market_columns(insured_hedged, "shock")
# Phase 2 (DECISIONS P2-10): diversification is weighted by order units, so the high-unit materials
# get a minimum share per supplier where it is cheap
DIV_MIN_SHARE = (("steel", 0.2), ("copper", 0.1), ("aluminum", 0.1))
STRATEGIES["lean_pxlp_mkt_div"] = _market_columns(
    _lean_split(0.025, 0.30, price_path=True, min_share_mat=DIV_MIN_SHARE), "calm")
STRATEGIES["lean_pxlp_mkt_closure"] = _bet(_market_columns(_lean_split(0.025, 0.30, price_path=True), "calm"),
                                           {"closure": 0.95})
