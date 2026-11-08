"""Landed-cost path model: how a shock moves every supplier's landed cost, week by week.

The simulator charges an order at that week's landed cost, and the F denominator (the hindsight
plan) pre-buys before price jumps and waits out uplifted prices. Holding today's landed cost flat
hides both moves from the planner. In the data, landed cost decomposes exactly as

    landed = spot x (1 + tariff) + freight

and each part moves in its own way (checked on weeks 1-208, eda/28_cost_path.py):
  * spot: a SOURCE shock (shortage, ban, reversal, sanctions) lifts the spot price of EVERY supplier
    of the material by a step that lands in the onset week and disappears the week after the
    shock ends. Measured steps (log): ban 0.22-0.32, reversal 0.16-0.18, shortage 0.14-0.16,
    sanctions 0.05-0.11; lane shocks and tariffs ~0;
  * freight: a constant per supplier, times a multiplier while its lane is shut by a strike
    (1.6x) or a closure (3.0x), exactly;
  * tariff: the announced rate from its effective week while the hike lasts, then the base rate.

`spot_steps` estimates the step per type from week-on-week changes of the material's mean log spot
price at onset (jump) and the week after the end (drop): step = (jump - drop) / 2, so the slow
random walk mostly cancels (weekly noise ~0.034). Each event counts once (one sanctions event that
hit 7 materials gives 7 measurements of weight 1/7). Types are shrunk toward the mean of all
source-shock events by STEP_PSEUDO pseudo-events, so a type never seen before (e.g. bans before
week 132) borrows the family's step instead of getting none.

`scenario_costs` turns per-scenario hit paths into landed costs [N, S, T]. It is written as today's
landed cost plus exact deltas, so a supplier-week that no shock touches keeps today's cost to the bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Data
from .duration import shock_class, shock_runs

SOURCE_TYPES = {"shortage", "ban", "reversal", "sanctions"}   # move the material's spot price
LANE_TYPES = {"strike", "closure"}                            # move the lane's freight
# Pseudo-events of the source-family mean added to each type. Between-type spread of the true
# steps (~0.08) is about twice the per-event noise (~0.04: within-type spread plus random walk), so
# a normal-normal model puts weight (0.04 / 0.08)^2 = 0.25 events on the family mean.
STEP_PSEUDO = 0.25


def _freight(info: Data) -> pd.DataFrame:
    """Per (week, material, country): spot, tariff, landed cost and freight = landed - spot (1 + tariff)."""
    key = ["week", "material", "country"]
    df = (info.supply_history[key + ["landed_cost"]]
          .merge(info.prices[key + ["spot_price"]], on=key)
          .merge(info.tariffs[key + ["tariff_rate"]], on=key))
    df["freight"] = df.landed_cost - df.spot_price * (1 + df.tariff_rate)
    return df


def spot_steps(info: Data, feats: pd.DataFrame) -> dict[str, float]:
    """Log spot step per source-shock type, from the onset jump and end drop of past events."""
    runs = shock_runs(feats)
    runs = runs[runs.type.isin(SOURCE_TYPES)].drop_duplicates(["material", "start", "type"])
    if not len(runs):
        return {}
    lsp = (info.prices.assign(log_spot=np.log(info.prices.spot_price))
           .pivot_table(index="week", columns="material", values="log_spot", aggfunc="mean"))
    dl = lsp.diff()                                         # weekly change of the mean log spot
    rows = []
    for r in runs.itertuples():
        if r.start not in dl.index or pd.isna(dl.at[r.start, r.material]):
            continue
        jump = float(dl.at[r.start, r.material])
        after = r.start + r.length                          # first week after the shock
        if r.ended and after in dl.index:
            step = 0.5 * (jump - float(dl.at[after, r.material]))
        else:                                               # still running: only the jump is seen
            step = jump
        rows.append((r.type, step, r.weight))
    if not rows:
        return {}
    ev = pd.DataFrame(rows, columns=["type", "step", "w"])
    family = float((ev.step * ev.w).sum() / ev.w.sum())
    g = ev.assign(ws=ev.step * ev.w).groupby("type")[["ws", "w"]].sum()
    steps = ((g.ws + STEP_PSEUDO * family) / (g.w + STEP_PSEUDO)).clip(lower=0.0).to_dict()
    return {t: float(steps.get(t, max(family, 0.0))) for t in SOURCE_TYPES}


def freight_multipliers(info: Data, feats: pd.DataFrame, comp: pd.DataFrame | None = None) -> dict[str, float]:
    """Freight / base freight while a supplier is hit by a lane shock of each type (median)."""
    fr = _freight(info) if comp is None else comp
    base = fr.groupby(["material", "country"]).freight.median().rename("base")
    typ = feats.assign(cls=shock_class(feats))[["week", "material", "country", "cls"]]
    m = fr.merge(typ, on=["week", "material", "country"]).join(base, on=["material", "country"])
    m = m[m.cls.isin(LANE_TYPES) & (m.base > 0)]
    return {t: float(g.median()) for t, g in (m.freight / m.base).groupby(m.cls)}


@dataclass
class CostPathModel:
    steps: dict             # source type -> log spot step (all suppliers of the material)
    freight_mult: dict      # lane type -> freight multiplier while the lane is hit
    components: pd.DataFrame  # per (material, country) at the origin: spot0, tariff0, freight0,
                              # freight_base (median over history), base_tariff

    @classmethod
    def fit(cls, info: Data, feats: pd.DataFrame | None = None) -> "CostPathModel":
        from ..features.build import build_features
        feats = build_features(info) if feats is None else feats
        fr = _freight(info)
        now = fr[fr.week == info.last_week].set_index(["material", "country"])
        comp = pd.DataFrame({"spot0": now.spot_price, "tariff0": now.tariff_rate, "landed0": now.landed_cost,
                             "freight0": now.freight,
                             "freight_base": fr.groupby(["material", "country"]).freight.median()}).dropna()
        comp = comp.join(info.suppliers.set_index(["material", "country"]).base_tariff).reset_index()
        return cls(spot_steps(info, feats), freight_multipliers(info, feats, fr), comp)


def scenario_costs(landed0: np.ndarray, spot0: np.ndarray, tariff0: np.ndarray, freight0: np.ndarray,
                   freight_base: np.ndarray, base_tariff: np.ndarray, tariff_hike: np.ndarray,
                   announced: np.ndarray, hit0: np.ndarray, typ: list, hit: np.ndarray,
                   steps: dict, freight_mult: dict, freight: bool = True,
                   tariff_on: np.ndarray | None = None) -> np.ndarray:
    """Landed cost [N, S, T] in every scenario week.

    Per supplier [S]: today's landed cost, spot, tariff, freight, base freight and base tariff,
    `announced` (a tariff hike is announced), `hit0` (hit today) and its shock type; `tariff_hike`
    [S, T] is the tariff while hit (the announced rate from its effective week, else today's);
    `hit` [N, S, T] are the scenario hit paths.

      spot    x exp(material step of the source shocks active in the scenario week - step in today's price)
      tariff  = tariff_hike while hit; the base tariff once a hike (ongoing or announced) is over
      freight = today's while an ongoing shock lasts, then the base; a new lane hit multiplies it
    """
    S = len(typ)
    src = np.array([steps.get(t, 0.0) if t in SOURCE_TYPES else 0.0 for t in typ])
    log_u = (hit * src[None, :, None]).max(axis=1)                    # [N, T] uplift in each week
    log_u0 = float((hit0 * src).max()) if S else 0.0                  # uplift already in today's spot
    g = np.exp(log_u - log_u0)[:, None, :]                            # [N, 1, T] spot factor
    calm_tariff = np.where(hit0 | announced, base_tariff, tariff0)
    on = hit if tariff_on is None else (hit | tariff_on[None])   # tariff_on [S, T]: announced hike in force
    tariff = np.where(on, tariff_hike[None], calm_tariff[None, :, None])
    mult = np.array([freight_mult.get(t, 1.0) if t in LANE_TYPES else 1.0 for t in typ])
    fr_hit = np.where(hit0, freight0, freight0 * mult)                # only new lane hits move freight
    fr_calm = np.where(hit0, freight_base, freight0)
    fr = np.where(hit, fr_hit[None, :, None], fr_calm[None, :, None])
    if not freight:          # freight channel off: today's freight in every scenario week
        fr = np.broadcast_to(freight0[None, :, None], hit.shape)
    # today's cost plus exact deltas: an untouched supplier-week keeps landed0 bit for bit
    delta = spot0[None, :, None] * (g * (1 + tariff) - (1 + tariff0)[None, :, None]) + (fr - freight0[None, :, None])
    return landed0[None, :, None] + delta
