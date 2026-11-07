"""Supply chain simulator reproducing the rules on the competition's Evaluation tab.

For one material and one test window:
  * start with INIT_STOCK_WEEKS of demand in stock and nothing in transit;
  * an order placed in week t is split by the shares; each country's slice is filled up to that
    country's capacity that week, paid at that week's landed cost, and arrives after that week's
    lead time (unfilled units are cancelled and not charged);
  * each week arrivals enter stock, then up to that week's demand is consumed;
  * end-of-week stock costs HOLDING_RATE x average landed cost per unit;
  * unmet demand costs UNMET_MULT x average landed cost per unit;
  * at the end, stock on hand and in transit is sold back at SALVAGE_RATE x average landed cost.

Assumptions where the rules are not explicit (kept as module constants so they are easy to revisit):
  * "average landed cost" = mean landed cost over the material's suppliers and the window's weeks;
  * the initial stock is the sum of demand over the window's first INIT_STOCK_WEEKS weeks;
  * holding is charged on end-of-week stock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .data import Data

INIT_STOCK_WEEKS = 5
HOLDING_RATE = 0.02
UNMET_MULT = 5.0
SALVAGE_RATE = 0.70


@dataclass(frozen=True)
class Truth:
    """What actually happened for one material over one window (arrays are [supplier, week])."""
    material: str
    weeks: np.ndarray        # [T]
    countries: list[str]     # [S], primary first
    demand: np.ndarray       # [T]
    capacity: np.ndarray     # [S, T]
    lead: np.ndarray         # [S, T] int
    cost: np.ndarray         # [S, T]
    disrupted: np.ndarray    # [T] 0/1
    shock_week: np.ndarray   # [T] bool: disrupted, or within 4 weeks after a disrupted week

    @property
    def avg_cost(self) -> float:
        return float(self.cost.mean())

    @property
    def init_stock(self) -> float:
        return float(self.demand[:INIT_STOCK_WEEKS].sum())


def truth_for(data: Data, material: str, weeks: range) -> Truth:
    """Extract ground truth from released data. Requires every week in `weeks` to be released."""
    if weeks[-1] > data.last_week:
        raise ValueError(f"week {weeks[-1]} not released (last is {data.last_week})")
    countries = data.suppliers_of(material)
    wk = np.array(weeks)
    sh = data.supply_history[data.supply_history.material == material]
    piv = lambda col: (sh.pivot(index="country", columns="week", values=col)  # noqa: E731
                       .loc[countries, wk].to_numpy())
    dem = data.demand[data.demand.material == material].set_index("week").demand_units.loc[wk].to_numpy()
    dis_all = data.outcomes[data.outcomes.material == material].set_index("week").disrupted
    dis = dis_all.loc[wk].to_numpy()
    # shock weeks = disrupted weeks plus 4 recovery weeks (lookback can reach before the window)
    shock = np.array([dis_all.loc[w - 4:w].max() == 1 for w in wk])
    return Truth(material, wk, countries, dem.astype(float), piv("capacity_units").astype(float),
                 piv("lead_time_weeks").astype(int), piv("landed_cost").astype(float), dis, shock)


@dataclass(frozen=True)
class SimResult:
    purchase: float
    holding: float
    unmet_penalty: float
    salvage: float
    served: np.ndarray       # [T]
    filled: np.ndarray       # [S, T] units actually bought

    @property
    def total_cost(self) -> float:
        return self.purchase + self.holding + self.unmet_penalty - self.salvage


def simulate(truth: Truth, orders: np.ndarray, shares: np.ndarray) -> SimResult:
    """Play out `orders` [T] split by `shares` [S, T] against the truth."""
    S, T = truth.capacity.shape
    filled = np.minimum(orders[None, :] * shares, truth.capacity)
    arrivals = np.zeros(T + truth.lead.max() + 1)
    for s in range(S):
        for t in range(T):
            arrivals[t + truth.lead[s, t]] += filled[s, t]
    stock, holding, unmet, served = truth.init_stock, 0.0, 0.0, np.zeros(T)
    A = truth.avg_cost
    for t in range(T):
        stock += arrivals[t]
        served[t] = min(stock, truth.demand[t])
        stock -= served[t]
        holding += HOLDING_RATE * A * stock
        unmet += UNMET_MULT * A * (truth.demand[t] - served[t])
    in_transit = arrivals[T:].sum()
    return SimResult(purchase=float((filled * truth.cost).sum()), holding=holding, unmet_penalty=unmet,
                     salvage=SALVAGE_RATE * A * (stock + in_transit), served=served, filled=filled)


def hindsight_plan(truth: Truth) -> tuple[float, np.ndarray]:
    """Cost-minimising plan with perfect knowledge, as an LP. Returns (cost, units [S, T])."""
    S, T = truth.capacity.shape
    A = truth.avg_cost
    nx = S * T                       # x[s, t] units bought, index s*T + t
    iu, ii = nx, nx + T              # u[t] unmet, I[t] end-of-week stock
    n = nx + 2 * T
    c = np.zeros(n)
    for s in range(S):
        for t in range(T):
            arrives = t + truth.lead[s, t]
            # bought units arriving after the window are sold back immediately at the salvage rate
            c[s * T + t] = truth.cost[s, t] - (SALVAGE_RATE * A if arrives >= T else 0.0)
    c[iu:iu + T] = UNMET_MULT * A
    c[ii:ii + T] = HOLDING_RATE * A
    c[ii + T - 1] -= SALVAGE_RATE * A          # final stock is sold back
    # flow: I[t] - I[t-1] - arrivals[t] - u[t] = -D[t]
    Aeq, beq = np.zeros((T, n)), -truth.demand.copy()
    for t in range(T):
        Aeq[t, ii + t] = 1
        if t:
            Aeq[t, ii + t - 1] = -1
        Aeq[t, iu + t] = -1
    beq[0] += truth.init_stock               # I[-1] = initial stock moves to the right-hand side
    for s in range(S):
        for t in range(T):
            if (a := t + truth.lead[s, t]) < T:
                Aeq[a, s * T + t] -= 1
    bounds = ([(0, truth.capacity[s, t]) for s in range(S) for t in range(T)]
              + [(0, d) for d in truth.demand] + [(0, None)] * T)
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"hindsight LP failed for {truth.material}: {res.message}")
    return float(res.fun), res.x[:nx].reshape(S, T)
