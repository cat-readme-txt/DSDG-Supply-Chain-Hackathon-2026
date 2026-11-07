"""Stochastic-programming ordering policy (sample average approximation).

For one material and one window we draw N scenarios of the supply world from our forecasts, then
solve ONE linear program for the order plan x[c, t] (units asked of supplier c in week t) that
minimises the average simulated cost over the scenarios, under the competition's exact rules:
  fill = min(order slice, capacity), paid at landed cost, arriving after the lead time; stock
  starts at 5 weeks of demand; holding 2%/week; unmet demand 5x; leftovers sold back at 70%.

Scenarios. Each supplier draws one uniform u; it is hit in week t when p(t) > u, which reproduces
the forecast marginals and gives contiguous shock spells. Suppliers whose open story is a lane
event (strike/closure) share u across the lane; sanctions share u across the country. A hit
supplier gets the capacity ratio / lead increase / tariff of its shock type (or its current one).

Fill linearisation. In a scenario where a supplier is NOT hit, its capacity is the normal one,
which the plan never exceeds, so fill = x exactly. When hit, fill <= x and fill <= reduced
capacity; units are scarce then, so the optimum fills to the limit anyway.

Diversification: each order week, every supplier gets at least `min_share` of that week's order.

Price path (`price_path=True`). By default every scenario pays today's landed cost in every week
(announced tariffs aside). With the price path each scenario gets its own landed costs from
models/cost_path.py: a source shock (shortage, ban, reversal, sanctions) in the scenario lifts the
spot price of every supplier of the material from its onset week until it ends; an ongoing one loses
its uplift the week after the scenario's end; lane shocks multiply freight; announced tariffs last
while the hike does. The average landed cost A that scales holding, unmet and salvage is then the
scenario's own (the simulator computes it from the realised costs). So the plan can pre-buy before a
warned price jump and wait out an uplift, as the hindsight plan in the F denominator does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .data import SHARE_COLS, WINDOW, Data
from .policy import TYPE_CAP, TYPE_LEAD, supplier_state
from .simulator import HOLDING_RATE, INIT_STOCK_WEEKS, SALVAGE_RATE, UNMET_MULT

LANE_TYPES, COUNTRY_TYPES = {"strike", "closure"}, {"sanctions"}


@dataclass(frozen=True)
class LPParams:
    n_scenarios: int = 30
    min_share: float = 0.1     # each supplier's minimum share of an order week
    risk_mult: float = 1.0     # scale forecast probabilities (robustness to under-forecast risk)
    risk_floor: float = 0.0    # minimum weekly hit probability per supplier
    hazard_mult: float = 1.0   # scale of the background (unwarned) shock process; 0 = off
    cap_derate: float = 1.0    # plan against this fraction of normal capacity (capacity varies a few %
                               # week to week; ordering right at the limit gets shorted at 5x cost)
    seed: int = 0
    cheap_premium: float = 0.0  # suppliers within this cost premium of the cheapest get ...
    cheap_share: float = 0.0    # ... at least this share of every order week (near-free diversification)
    fill_kappa: float = 0.0     # score-aware: extra unmet cost (x avg landed cost) in shock weeks of
                                # scenarios where the primary is hit, i.e. where the naive baseline runs
                                # short and the fill-rate part of R pays for every unit served
    price_path: bool = False    # per-scenario landed costs from models/cost_path.py, and a per-scenario
                                # average cost A (see the module docstring); False = today's cost held flat
    path_freight: bool = False  # with price_path: also move freight on lane shocks (x1.6 strike, x3
                                # closure). Off by default: in the backtest it shifted orders off lanes
                                # with closure rumors and onto suppliers that were then banned (DECISIONS)


PRIOR_EXPOSURE = 400.0  # supplier-weeks of pooled-rate prior per country


def shock_history(info: Data) -> dict:
    """Background shock process from released history: onset hazard per country, run profiles."""
    from .labels import supplier_state as labels
    st = labels(info).sort_values(["material", "country", "week"])
    runs, exposure, onsets = [], {}, {}
    for (m, c), g in st.groupby(["material", "country"]):
        hit = g.hit.to_numpy()
        exposure[c] = exposure.get(c, 0) + int((~hit).sum())
        i = 0
        while i < len(hit):
            if hit[i]:
                j = i
                while j + 1 < len(hit) and hit[j + 1]:
                    j += 1
                seg = g.iloc[i:j + 1]
                runs.append(dict(length=j - i + 1, cap=float(seg.cap_ratio.where(seg.hit_cap, 1.0).min()),
                                 lead=int(seg.lead_delta.max()), tariff=float(seg.tariff_delta.max())))
                onsets[c] = onsets.get(c, 0) + 1
                i = j + 1
            else:
                i += 1
    pooled = sum(onsets.values()) / max(sum(exposure.values()), 1)
    hazard = {c: (onsets.get(c, 0) + PRIOR_EXPOSURE * pooled) / (exposure.get(c, 0) + PRIOR_EXPOSURE)
              for c in exposure}
    return dict(hazard=hazard, runs=pd.DataFrame(runs))


def draw_scenarios(state: pd.DataFrame, weeks: list[int], P: LPParams, rng, hist: dict | None = None,
                   cm=None) -> dict:
    """Arrays [N, S, T]: hit, capacity ratio, lead, cost; forecast risks plus background shocks.
    With `P.price_path` and a fitted CostPathModel `cm` (its components merged into `state`), the
    cost of each scenario follows its own shock path."""
    countries = state.country.drop_duplicates().tolist()
    S, T, N = len(countries), len(weeks), P.n_scenarios
    pos = {w: i for i, w in enumerate(weeks)}
    p = np.zeros((S, T)); sev = np.ones((S, T)); lead = np.zeros((S, T), int)
    extra = np.zeros((S, T), int); typ = [None] * S; lane = [None] * S; cost = np.zeros((S, T))
    cap = np.zeros((S, T))      # planning capacity per supplier and order week
    for i, c in enumerate(countries):
        g = state[state.country == c]
        for r in g.itertuples(index=False):
            if r.week in pos:
                t = pos[r.week]
                p[i, t] = min(1.0, max(P.risk_floor, P.risk_mult * r.p))
                sev[i, t] = r.severity
                lead[i, t] = r.base_lead_weeks
                extra[i, t] = (r.lead_delta if r.hit == 1 else TYPE_LEAD.get(r.shock_type, 0))
                cost[i, t] = r.cost
                cap[i, t] = r.normal_capacity_units * P.cap_derate
        typ[i] = g.shock_type.iloc[0] if len(g) else None
        lane[i] = g.lane.iloc[0] if "lane" in g and len(g) else None
    # shared uniforms: lane events across a lane, sanctions across a country (within this material
    # only one supplier per country, so the country share is per-supplier here)
    u = rng.random((N, S))
    for i in range(S):
        if typ[i] in LANE_TYPES:
            for k in range(i):
                if lane[k] == lane[i] and typ[k] in LANE_TYPES:
                    u[:, i] = u[:, k]
    hit = p[None, :, :] > u[:, :, None]
    cap_ratio = np.where(hit, sev[None], 1.0)
    L = lead[None] + np.where(hit, extra[None], 0)
    cost_s = np.repeat(cost[None], N, axis=0)
    haz = np.ones((N, S, T))    # background-shock tariff factor, reapplied on top of a price path
    if hist is not None and P.hazard_mult > 0 and len(hist["runs"]):
        runs = hist["runs"]
        for n in range(N):
            for i, c in enumerate(countries):
                lam = min(1.0, P.hazard_mult * hist["hazard"].get(c, 0.0))
                t = 0
                while t < T:
                    if rng.random() < lam:
                        r = runs.iloc[rng.integers(len(runs))]
                        span = range(t, min(T, t + int(r.length)))
                        # lane-type shocks (they add lead time) hit every supplier on the lane
                        targets = [k for k in range(S) if lane[k] == lane[i]] if r.lead > 0 else [i]
                        for k in targets:
                            for tt in span:
                                hit[n, k, tt] = True
                                cap_ratio[n, k, tt] = min(cap_ratio[n, k, tt], r.cap)
                                L[n, k, tt] = max(L[n, k, tt], lead[k, tt] + r.lead)
                                cost_s[n, k, tt] = cost[k, tt] * (1 + r.tariff)
                                haz[n, k, tt] = 1 + r.tariff
                        t = span[-1] + 1
                    else:
                        t += 1
    if P.price_path and cm is not None:
        path = _path_costs(state, countries, weeks, typ, hit, cm, P.path_freight)
        if path is not None:
            cost_s = path * haz
    return dict(countries=countries, hit=hit, cap=cap, cap_ratio=cap_ratio, lead=L, cost=cost_s)


def _path_costs(state: pd.DataFrame, countries: list[str], weeks: list[int], typ: list, hit: np.ndarray,
                cm, freight: bool = True) -> np.ndarray | None:
    """Per-scenario landed costs [N, S, T] (models/cost_path.scenario_costs), or None when the
    origin cost of a supplier can't be decomposed into spot, tariff and freight."""
    from .models.cost_path import scenario_costs
    S, T = len(countries), len(weeks)
    src = {"landed0": "landed_cost", "spot0": "spot0", "tariff0": "tariff_rate", "freight0": "freight0",
           "freight_base": "freight_base", "base_tariff": "base_tariff"}
    one = {k: np.zeros(S) for k in src}
    hit0, announced, hike = np.zeros(S, bool), np.zeros(S, bool), np.zeros((S, T))
    for i, c in enumerate(countries):
        g = state[state.country == c]
        if not len(g) or g[list(src.values())].isna().any(axis=None):
            return None
        r0 = g.iloc[0]                       # origin-week quantities: the same on every row
        for k, col in src.items():
            one[k][i] = r0[col]
        hit0[i] = r0.hit == 1
        eff = r0.announced_effective_week
        announced[i] = pd.notna(eff) and r0.announced_rate > r0.tariff_rate + 1e-6
        hike[i] = np.where(announced[i] & (np.array(weeks) >= (eff if announced[i] else np.inf)),
                           r0.announced_rate, r0.tariff_rate)
    return scenario_costs(one["landed0"], one["spot0"], one["tariff0"], one["freight0"], one["freight_base"],
                          one["base_tariff"], hike, announced, hit0, typ, hit, cm.steps, cm.freight_mult, freight)


def solve_window(demand: np.ndarray, sc: dict, P: LPParams) -> np.ndarray:
    """Optimal x[S, T] for one material-window over the scenarios."""
    hit, capn, ratio, L, cost = sc["hit"], sc["cap"], sc["cap_ratio"], sc["lead"], sc["cost"]
    N, S, T = hit.shape
    A = float(cost.mean())
    cost = cost if cost.ndim == 3 else np.repeat(cost[None], N, axis=0)
    # average landed cost per scenario (scales holding, unmet and salvage): the scenario's own when
    # costs follow a price path, else the common one
    An = cost.mean(axis=(1, 2)) if P.price_path else np.full(N, A)
    # shock weeks (any supplier hit now or in the 4 weeks before) in scenarios where the primary
    # (largest normal capacity, the naive policy's only supplier) is hit at some point
    prim = int(np.argmax(capn.mean(axis=1)))
    anyhit = hit.any(axis=1)                                     # [N, T]
    shock = np.array([[anyhit[n, max(0, t - 4):t + 1].any() for t in range(T)] for n in range(N)])
    fill_w = (shock & hit[:, prim, :].any(axis=1)[:, None]).astype(float)
    nx = S * T
    xi = lambda c, t: c * T + t  # noqa: E731
    fi = lambda n, c, t: nx + (n * S + c) * T + t  # noqa: E731
    Ii = lambda n, t: nx + N * S * T + n * T + t  # noqa: E731
    ui = lambda n, t: nx + N * S * T + N * T + n * T + t  # noqa: E731
    nv = nx + N * S * T + 2 * N * T
    c_obj = np.zeros(nv)
    for n in range(N):
        for c in range(S):
            for t in range(T):
                arrive_late = t + L[n, c, t] >= T
                c_obj[fi(n, c, t)] = (cost[n, c, t] - (SALVAGE_RATE * An[n] if arrive_late else 0.0)) / N
        for t in range(T):
            c_obj[Ii(n, t)] = HOLDING_RATE * An[n] / N
            c_obj[ui(n, t)] = (UNMET_MULT + P.fill_kappa * fill_w[n, t]) * An[n] / N
        c_obj[Ii(n, T - 1)] -= SALVAGE_RATE * An[n] / N
    rows, cols, vals, beq = [], [], [], []
    r = 0
    # inventory flow per scenario
    I0 = float(demand[:INIT_STOCK_WEEKS].sum())
    arrivals = [[[] for _ in range(T)] for _ in range(N)]
    for n in range(N):
        for c in range(S):
            for t in range(T):
                a = t + int(L[n, c, t])
                if a < T:
                    arrivals[n][a].append(fi(n, c, t))
    for n in range(N):
        for t in range(T):
            rows += [r]; cols += [Ii(n, t)]; vals += [1.0]
            if t:
                rows += [r]; cols += [Ii(n, t - 1)]; vals += [-1.0]
            rows += [r]; cols += [ui(n, t)]; vals += [-1.0]
            for v in arrivals[n][t]:
                rows += [r]; cols += [v]; vals += [-1.0]
            beq.append(-demand[t] + (I0 if t == 0 else 0.0))
            r += 1
    # fill = x when the supplier is not hit in the scenario
    for n in range(N):
        for c in range(S):
            for t in range(T):
                if not hit[n, c, t]:
                    rows += [r, r]; cols += [fi(n, c, t), xi(c, t)]; vals += [1.0, -1.0]; beq.append(0.0); r += 1
    Aeq = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    # inequalities: fill <= x when hit; min share: min_share * sum_c x - x_c <= 0
    rows, cols, vals, bub = [], [], [], []
    r = 0
    for n in range(N):
        for c in range(S):
            for t in range(T):
                if hit[n, c, t]:
                    rows += [r, r]; cols += [fi(n, c, t), xi(c, t)]; vals += [1.0, -1.0]; bub.append(0.0); r += 1
    if P.min_share > 0:
        for t in range(T):
            for c in range(S):
                for k in range(S):
                    rows += [r]; cols += [xi(k, t)]; vals += [P.min_share - (1.0 if k == c else 0.0)]
                bub.append(0.0); r += 1
    if P.cheap_share > 0 and P.cheap_premium > 0:
        cbar = cost.mean(axis=(0, 2))                      # mean planning cost per supplier
        for c in range(S):
            if cbar[c] > cbar.min() * (1 + 1e-9) and cbar[c] <= cbar.min() * (1 + P.cheap_premium):
                for t in range(T):      # cheap_share * sum_k x_k - x_c <= 0
                    for k in range(S):
                        rows += [r]; cols += [xi(k, t)]; vals += [P.cheap_share - (1.0 if k == c else 0.0)]
                    bub.append(0.0); r += 1
    Aub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr() if r else None
    bounds = [(0, capn[c, t]) for c in range(S) for t in range(T)]
    bounds += [(0, capn[c, t] * ratio[n, c, t]) for n in range(N) for c in range(S) for t in range(T)]
    bounds += [(0, None)] * (N * T) + [(0, d) for _ in range(N) for d in demand]
    res = linprog(c_obj, A_ub=Aub, b_ub=bub if r else None, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x[:nx].reshape(S, T)
    return np.where(x > 1e-6, x, 0.0)   # solver round-off can give tiny negatives (invalid shares)


def plan_orders_lp(info: Data, weeks: range, feats: pd.DataFrame, sup_p: pd.DataFrame,
                   P: LPParams = LPParams(), state: pd.DataFrame | None = None) -> pd.DataFrame:
    state = supplier_state(info, feats, sup_p) if state is None else state
    if "lane" not in state:
        state = state.merge(info.suppliers[["material", "country", "lane"]], on=["material", "country"])
    if "hit" not in state:
        state = state.assign(hit=0, lead_delta=0, shock_type=None)
    dem = info.demand.set_index(["material", "week"]).demand_units
    rng = np.random.default_rng(P.seed)
    hist = shock_history(info) if P.hazard_mult > 0 else None
    cm = None
    if P.price_path:
        from .models.cost_path import CostPathModel
        cm = CostPathModel.fit(info, feats)
        comp = cm.components[["material", "country", "spot0", "freight0", "freight_base", "base_tariff"]]
        state = state.drop(columns=[c for c in comp.columns[2:] if c in state]).merge(
            comp, on=["material", "country"], how="left")
    out = []
    for start in range(weeks[0], weeks[-1] + 1, WINDOW):
        win = [w for w in range(start, start + WINDOW) if w in weeks]
        for m in sorted(state.material.unique()):
            st = state[(state.material == m) & state.week.isin(win)]
            d = np.array([float(dem.loc[(m, w)]) for w in win])
            sc = draw_scenarios(st, win, P, rng, hist, cm)
            x = solve_window(d, sc, P)
            orders = x.sum(axis=0)
            rows = pd.DataFrame({"week": win, "material": m, "order_units": orders})
            for col in SHARE_COLS:
                rows[col] = 0.0
            S = len(sc["countries"])
            for i, c in enumerate(sc["countries"]):
                rows[f"share_{c}"] = np.divide(x[i], orders, out=np.full(len(win), 1 / S), where=orders > 1e-6)
            rows.loc[orders <= 1e-6, "order_units"] = 0.0
            out.append(rows)
    return pd.concat(out, ignore_index=True)
