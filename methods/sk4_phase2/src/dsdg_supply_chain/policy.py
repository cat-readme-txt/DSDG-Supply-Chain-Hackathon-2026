"""Ordering policy: a lean, risk-aware plan per material and per 13-week window.

The simulator gives each window 5 weeks of starting stock and charges 5x for unmet demand, 2%/week
for holding, and sells leftovers back at 70%. The hindsight optimum therefore buys only what the
window needs, times arrivals for when stock runs out, ends near zero, and avoids hit suppliers.
This policy approximates that with the forecasts:

  1. arrival plan: for each week, top projected end-of-week stock up to a buffer of
         (safety_base + safety_risk x P(material disrupted that week)) x demand;
     nothing is bought that cannot arrive inside the window;
  2. supplier weights: favour cheap suppliers (cost_temp), avoid suppliers likely to be hit
     (risk_aversion), blend toward an even split (diversify);
  3. each supplier's slice is ordered `lead` weeks before it is needed, using the lead time
     expected for that supplier; expected delivery = (1-p) min(q, cap) + p min(q, cap x severity);
  4. weeks with no order get an even split: diversification is scored on every row, for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SHARE_COLS, WINDOW, Data
from .simulator import INIT_STOCK_WEEKS

# capacity ratio a shock of each type typically leaves, and lead-time increase (eda/FINDINGS.md)
TYPE_CAP = {"ban": 0.05, "sanctions": 0.2, "shortage": 0.5, "reversal": 0.5, "closure": 0.75,
            "strike": 0.8, "tariff": 1.0}
DEFAULT_CAP = 0.5
TYPE_LEAD = {"closure": 3, "strike": 4, "sanctions": 3}


@dataclass(frozen=True)
class PolicyParams:
    """Defaults tuned on 22 backtest windows (eda/11_tune_policy.py): mean score 47.6."""
    safety_base: float = 0.5      # weeks of demand kept as buffer
    safety_risk: float = 0.0      # extra buffer weeks per unit of disruption probability (tuned to 0)
    cost_temp: float = 5.0        # weight factor exp(-cost_temp x cost premium)
    risk_aversion: float = 8.0    # weight factor (1 - p)^risk_aversion: steer away from at-risk suppliers
    diversify: float = 0.0        # base blend weight toward an even split
    overorder: float = 1.0        # inflate a slice by overorder x (1/expected_fill - 1)
    diversify_risk: float = 0.667 # extra even-split blend per unit of material disruption risk
    buffer_mid: float = 0.0       # extra insurance stock (weeks of demand) carried mid-window ...
    taper: float = 4.0            # ... and run down linearly over the last `taper` weeks, so it is
                                  # consumed (holding cost only) instead of sold back at 70%


def _buffer(P: "PolicyParams", tau: int, T: int) -> float:
    """Insurance stock target (weeks of demand) at window week tau."""
    if P.buffer_mid <= 0:
        return 0.0
    left = T - 1 - tau
    return P.buffer_mid * float(np.clip(left / max(P.taper, 1e-9), 0.0, 1.0))


def _blend(P: "PolicyParams", risk: float) -> float:
    return float(np.clip(P.diversify + P.diversify_risk * risk, 0.0, 1.0))


def supplier_state(info: Data, feats: pd.DataFrame, sup_p: pd.DataFrame) -> pd.DataFrame:
    """Per (week, supplier) planning inputs: P(hit), severity, expected lead, expected cost, capacity."""
    o = info.last_week
    base = info.suppliers[["material", "country", "base_lead_weeks", "normal_capacity_units"]]
    cost = info.supply_history[info.supply_history.week == o][["material", "country", "landed_cost"]]
    now = feats[feats.week == o][["material", "country", "cap_ratio", "lead_delta"]]
    tar = info.tariffs[info.tariffs.week == o][["material", "country", "tariff_rate", "announced_rate",
                                               "announced_effective_week"]]
    s = (sup_p.merge(base, on=["material", "country"]).merge(cost, on=["material", "country"])
         .merge(now, on=["material", "country"]).merge(tar, on=["material", "country"]))
    typ_cap = s.shock_type.map(TYPE_CAP).fillna(DEFAULT_CAP)
    s["severity"] = np.where(s.hit == 1, s.cap_ratio, typ_cap)          # capacity ratio if hit
    extra = np.where(s.hit == 1, s.lead_delta, s.shock_type.map(TYPE_LEAD).fillna(0))
    s["lead"] = (s.base_lead_weeks + np.where(s.p > 0.5, extra, 0)).astype(int)
    eff = s.announced_effective_week.fillna(np.inf)
    s["cost"] = s.landed_cost * np.where(s.week >= eff, (1 + s.announced_rate.fillna(0)) / (1 + s.tariff_rate), 1.0)
    return s


def _weights(cost: np.ndarray, p: np.ndarray, P: PolicyParams, risk: float = 0.0) -> np.ndarray:
    w = np.exp(-P.cost_temp * (cost / cost.min() - 1)) * (1 - p) ** P.risk_aversion
    w = w / w.sum() if w.sum() > 0 else np.full(len(cost), 1 / len(cost))
    b = _blend(P, risk)
    return (1 - b) * w + b / len(cost)


def plan_material_window(weeks: list[int], demand: np.ndarray, state: pd.DataFrame, risk: np.ndarray,
                         P: PolicyParams) -> tuple[list[str], np.ndarray]:
    """Units to order per supplier [S, T] for one material over one window."""
    T = len(weeks)
    countries = state.country.drop_duplicates().tolist()
    pos = {w: i for i, w in enumerate(weeks)}
    # arrays [S, T] indexed by order week
    arr = {k: np.zeros((len(countries), T)) for k in ["p", "sev", "lead", "cost", "cap"]}
    for i, c in enumerate(countries):
        g = state[state.country == c]
        for r in g.itertuples(index=False):
            if r.week in pos:
                t = pos[r.week]
                arr["p"][i, t], arr["sev"][i, t], arr["lead"][i, t] = r.p, r.severity, r.lead
                arr["cost"][i, t], arr["cap"][i, t] = r.cost, r.normal_capacity_units
    units = np.zeros((len(countries), T))
    expected_arrival = np.zeros(T)
    stock = float(demand[:INIT_STOCK_WEEKS].sum())
    for tau in range(T):
        avail = stock + expected_arrival[tau]
        need = demand[tau] * (1 + P.safety_base + P.safety_risk * risk[tau] + _buffer(P, tau, T)) - avail
        if need > 0:
            # each supplier can deliver at tau if some order week t has t + lead[t] == tau
            cand = [(i, t) for i in range(len(countries)) for t in range(tau)
                    if t + int(arr["lead"][i, t]) == tau]
            if cand:
                idx = np.array(cand)
                p = arr["p"][idx[:, 0], idx[:, 1]]
                w = _weights(arr["cost"][idx[:, 0], idx[:, 1]], p, P, risk[tau])
                for (i, t), wi in zip(cand, w):
                    sev, cap = arr["sev"][i, t], arr["cap"][i, t]
                    fill = 1 - arr["p"][i, t] * (1 - sev)
                    q = wi * need * (1 + P.overorder * (1 / max(fill, 0.05) - 1))
                    q = min(q, cap - units[i, t])          # never ask for more than normal capacity
                    if q <= 0:
                        continue
                    units[i, t] += q
                    pi = arr["p"][i, t]
                    expected_arrival[tau] += (1 - pi) * min(q, cap) + pi * min(q, cap * sev)
                avail = stock + expected_arrival[tau]
        stock = max(avail - demand[tau], 0.0)
    return countries, units


def plan_orders(info: Data, weeks: range, feats: pd.DataFrame, sup_p: pd.DataFrame, mat_p: pd.DataFrame,
                P: PolicyParams = PolicyParams(), state: pd.DataFrame | None = None) -> pd.DataFrame:
    """order_units and shares for every (week, material) in `weeks`, window by window.

    `state` (optional) replaces the forecast-based supplier state, e.g. with the true one for
    value-of-information experiments."""
    state = supplier_state(info, feats, sup_p) if state is None else state
    dem = info.demand.set_index(["material", "week"]).demand_units
    risk = mat_p.set_index(["material", "week"]).p
    out = []
    for start in range(weeks[0], weeks[-1] + 1, WINDOW):
        win = [w for w in range(start, start + WINDOW) if w in weeks]
        for m in state.material.unique():
            st = state[(state.material == m) & state.week.isin(win)]
            d = np.array([float(dem.loc[(m, w)]) for w in win])
            rk = np.array([float(risk.get((m, w), 0.0)) for w in win])
            countries, units = plan_material_window(win, d, st, rk, P)
            orders = units.sum(axis=0)
            rows = pd.DataFrame({"week": win, "material": m, "order_units": orders})
            for c in SHARE_COLS:
                rows[c] = 0.0
            for i, c in enumerate(countries):
                rows[f"share_{c}"] = np.divide(units[i], orders, out=np.full(len(win), 1 / len(countries)),
                                               where=orders > 0)
            out.append(rows)
    return pd.concat(out, ignore_index=True)
