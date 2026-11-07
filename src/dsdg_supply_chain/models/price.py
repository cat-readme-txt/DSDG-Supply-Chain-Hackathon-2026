"""Price model: benchmark price = normal level x shock uplift.

  * normal level: the last released benchmark price with the effect of any ongoing shock removed,
    then pulled part of the way back toward its long-run (78-week) average. Prices behave like a
    random walk tied loosely to a slow anchor: eda/10_price_drivers.py found no macro, currency,
    freight or demand driver, but a clear mean reversion (about half the gap closes in 13 weeks).
    The reversion speed per horizon is fitted on quiet history and applied at half strength;
  * shock uplift: how much a shock of each type lifts the benchmark, learned from past shocks
    (log-price during the shock vs the weeks just before it), shrunk toward 0 for rare types;
  * which uplift applies in a future week comes from the disruption model's per-supplier
    probabilities. The competition scores price with MAPE, which rewards the most likely outcome
    rather than the average, so an uplift is applied only when that supplier's shock is more likely
    than not (P > THRESHOLD).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Data
from ..features.build import build_features
from .duration import shock_runs

THRESHOLD = 0.5   # add a new shock's uplift when P(hit) > THRESHOLD
KEEP = 0.2        # keep an ongoing shock's uplift unless P(still hit) < KEEP: predicted shock ends
                  # are uncertain, so only drop the uplift when the end is quite likely
                  # (eda/09_price_model.py: KEEP 0.5 lost up to -0.52 skill, 0.2 at most -0.10)
SHRINK = 1.0      # pseudo-events of "no uplift" added to each type
PRE = (6, 3)      # baseline = mean log price over weeks start-6 .. start-3
ANCHOR_WEEKS = 78         # long-run anchor = trailing mean log price over this many weeks
REVERT_STRENGTH = 0.5     # apply half the fitted reversion (robust worst case in eda/10)
MAX_H = 26


def _anchor_dev(lp: pd.DataFrame) -> pd.DataFrame:
    return lp - lp.rolling(ANCHOR_WEEKS, min_periods=26).mean()


def reversion_speeds(info: Data) -> dict[int, float]:
    """k_h: fraction of the gap to the anchor that closes within h weeks (fitted on quiet weeks)."""
    lp = np.log(info.outcomes.pivot(index="week", columns="material", values="benchmark_price"))
    dis = info.outcomes.pivot(index="week", columns="material", values="disrupted")
    dev = _anchor_dev(lp)
    ks = {}
    for h in range(1, MAX_H + 1):
        quiet = (dis == 0) & (dis.shift(-h) == 0)
        x, y = dev[quiet].stack(), (lp.shift(-h) - lp)[quiet].stack()
        j = x.index.intersection(y.index)
        ks[h] = float(np.clip(-(x[j] * y[j]).sum() / (x[j] ** 2).sum(), 0, 1)) if len(j) > 20 else 0.0
    return ks


def shock_uplifts(info: Data, feats: pd.DataFrame) -> dict[str, float]:
    """Mean log uplift of the benchmark during shocks of each type (one value per event)."""
    runs = shock_runs(feats)
    if not len(runs):
        return {}
    lp = np.log(info.outcomes.pivot(index="week", columns="material", values="benchmark_price"))
    ev = runs.drop_duplicates(["material", "start", "type"])
    rows = []
    for r in ev.itertuples():
        base = lp.loc[r.start - PRE[0]:r.start - PRE[1], r.material]
        during = lp.loc[r.start:r.start + r.length - 1, r.material]
        if len(base) >= 2 and len(during):
            rows.append((r.type, during.mean() - base.mean()))
    u = pd.DataFrame(rows, columns=["type", "uplift"])
    g = u.groupby("type").uplift.agg(["sum", "size"])
    return (g["sum"] / (g["size"] + SHRINK)).clip(lower=0).to_dict()


@dataclass
class PriceModel:
    uplift: dict
    threshold: float = THRESHOLD
    keep: float = KEEP
    revert: dict | None = None
    revert_strength: float = REVERT_STRENGTH

    @classmethod
    def fit(cls, info: Data, feats: pd.DataFrame | None = None, threshold: float = THRESHOLD,
            keep: float = KEEP, revert_strength: float = REVERT_STRENGTH) -> "PriceModel":
        feats = build_features(info) if feats is None else feats
        return cls(shock_uplifts(info, feats), threshold, keep, reversion_speeds(info), revert_strength)

    def material_uplift(self, sup: pd.DataFrame) -> pd.DataFrame:
        """Log uplift per (week, material): the largest uplift among suppliers likely to be hit."""
        u = sup.shock_type.map(self.uplift).fillna(0.0)
        likely = np.where(sup.hit == 1, sup.p >= self.keep, sup.p > self.threshold)
        return sup.assign(u=u.where(likely, 0.0)).groupby(["week", "material"]).u.max().reset_index()

    def predict(self, info: Data, sup: pd.DataFrame, weeks: range) -> pd.DataFrame:
        """Benchmark price per (week, material). Released weeks get the observed price."""
        out = info.outcomes
        origin = info.last_week
        last = out[out.week == origin].set_index("material").benchmark_price
        # uplift already in the last price: ongoing shocks at the origin
        now = sup[sup.hit == 1].assign(p=1.0).drop_duplicates(["material", "country"])
        cur = self.material_uplift(now.assign(week=origin)).set_index("material").u if len(now) else pd.Series(dtype=float)
        cur = cur.reindex(last.index).fillna(0.0)
        level = np.log(last) - cur
        fut = self.material_uplift(sup)
        fut["price"] = np.exp(fut.material.map(level) + fut.u + self._reversion(info, fut, cur))
        past = out[out.week.isin(weeks)][["week", "material", "benchmark_price"]].rename(
            columns={"benchmark_price": "price"})
        return pd.concat([past, fut[["week", "material", "price"]]], ignore_index=True)

    def _reversion(self, info: Data, fut: pd.DataFrame, cur: pd.Series) -> np.ndarray:
        """Log adjustment pulling the level toward the long-run anchor (0 if not fitted).

        The gap is measured on the shock-free level (current uplift removed), so an ongoing shock's
        effect is not taken out twice."""
        if not self.revert or not self.revert_strength:
            return np.zeros(len(fut))
        lp = np.log(info.outcomes.pivot(index="week", columns="material", values="benchmark_price"))
        dev = (_anchor_dev(lp).iloc[-1] - cur).fillna(0.0)
        h = (fut.week - info.last_week).clip(1, MAX_H)
        k = h.map(self.revert).to_numpy()
        return -self.revert_strength * k * fut.material.map(dev).to_numpy()


def predict_price(info: Data, weeks: range, sup: pd.DataFrame, feats: pd.DataFrame | None = None,
                  threshold: float = THRESHOLD, keep: float = KEEP) -> pd.DataFrame:
    return PriceModel.fit(info, feats, threshold, keep).predict(info, sup, weeks)
