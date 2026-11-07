"""Disruption model: P(supplier is hit h weeks after the information week).

Two parts, because the two situations behave differently:
  * supplier NOT hit now -> a classifier on (origin features, horizon) learns when a new shock
    starts; this is where news warnings matter. Training rows are (origin o, supplier, h) with the
    label "hit at o + h", using only pairs where o + h is released.
  * supplier hit now -> a duration model (models/duration.py): how long shocks of this type last,
    fitted with ongoing shocks treated as censored, which avoids learning "long shocks never end"
    from a shock that is still in progress.
Supplier probabilities are combined into the material-level probability the competition scores:
P(any of the material's suppliers is hit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import Data
from ..features.build import build_features
from ..features.news import TYPES
from .duration import DurationModel, shock_class

KEY = ["week", "material", "country"]
MAX_H = 26

SUPPLY = ["weeks_since_hit", "country_hit_share", "lane_hit_share", "cap_ratio"]
NEWS = ["news_stage", *[f"stage_{t}" for t in TYPES], "weeks_since_signal", "story_age",
        "story_signals", "signals_8w", "called_off_4w", "recovered_4w", "ongoing_2w", "severity"]
MARKET = ["tension_z", "congestion_z", "tariff_announced", "weeks_to_tariff"]
CONTAGION = ["country_hit_share", "lane_hit_share"]   # is trouble already spreading nearby?
FEATURE_SETS = {"supply": SUPPLY, "news": NEWS, "news+contagion": NEWS + CONTAGION,
                "supply+news": SUPPLY + NEWS, "all": SUPPLY + NEWS + MARKET,
                "news+market": NEWS + CONTAGION + MARKET}


def training_rows(feats: pd.DataFrame, cols: list[str], max_h: int = MAX_H) -> pd.DataFrame:
    """Stack (origin, supplier, h) rows; label = supplier hit at origin + h (if released)."""
    label = feats[KEY + ["hit"]].rename(columns={"hit": "y"})
    base = feats[KEY + ["hit"] + [c for c in cols if c != "hit"]]
    parts = []
    for h in range(1, max_h + 1):
        lab = label.assign(week=label.week - h)          # label of week o+h, keyed by origin o
        parts.append(base.merge(lab, on=KEY).assign(h=h))
    return pd.concat(parts, ignore_index=True)


def _design(df: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    X = df[cols + ["h"]].to_numpy(float)
    if kind == "logit":   # let warning effects fade with the horizon
        h = X[:, -1:]
        X = np.hstack([X, X[:, :-1] * np.exp(-h / 4.0), np.log(h)])
    return X


def _estimator(kind: str):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    raise ValueError(kind)


@dataclass
class DisruptionModel:
    kind: str
    cols: list[str]
    onset: object
    duration: DurationModel

    @classmethod
    def fit(cls, info: Data, kind: str = "gbm", features: str = "supply+news",
            feats: pd.DataFrame | None = None, sigma_min: float = 0.2) -> "DisruptionModel":
        cols = FEATURE_SETS[features]
        feats = build_features(info) if feats is None else feats
        tr = training_rows(feats, cols)
        tr = tr[tr.hit == 0]                      # the onset classifier only models new shocks
        onset = _estimator(kind).fit(_design(tr, cols, kind), tr.y.to_numpy())
        return cls(kind, cols, onset, DurationModel.fit(feats, sigma_min=sigma_min))

    def predict_suppliers(self, feats: pd.DataFrame, origin: int, weeks: range) -> pd.DataFrame:
        """P(hit) per supplier for each target week > origin, from features at `origin`."""
        at = feats[feats.week == origin]
        targets = [w for w in weeks if w > origin]
        df = pd.concat([at.assign(target_week=w, h=min(w - origin, MAX_H)) for w in targets],
                       ignore_index=True)
        p_new = self.onset.predict_proba(_design(df, self.cols, self.kind))[:, 1]
        p_stay = self.duration.survival(shock_class(df), df.weeks_hit.to_numpy(float),
                                        (df.target_week - origin).to_numpy(float))
        df["p"] = np.where(df.hit.to_numpy() == 1, p_stay, p_new)
        return (df[["target_week", "material", "country", "h", "hit", "p"]]
                .rename(columns={"target_week": "week"}))


def material_probability(sup: pd.DataFrame) -> pd.DataFrame:
    """P(at least one supplier hit), treating suppliers as independent."""
    return (sup.assign(q=1 - sup.p).groupby(["week", "material"]).q.prod()
            .rsub(1).rename("p").reset_index())


# Defaults chosen by walk-forward evaluation (eda/07_disruption_model.py)
DEFAULT_KIND, DEFAULT_FEATURES, DEFAULT_SIGMA_MIN = "gbm", "news+contagion", 0.6
# Shrink toward each material's historical rate, more so further ahead: unwarned shocks exist.
BLEND_BASE, BLEND_PER_WEEK = 0.10, 0.01


def supplier_forecast(info: Data, weeks: range, feats: pd.DataFrame | None = None, kind: str = DEFAULT_KIND,
                      features: str = DEFAULT_FEATURES, sigma_min: float = DEFAULT_SIGMA_MIN) -> pd.DataFrame:
    """Per-supplier forecast for future weeks from the last released week.

    Columns: week, material, country, h, hit (hit at origin), p (P(hit) that week), shock_type
    (type of the ongoing shock if hit, else of the open news story, else None).
    """
    feats = build_features(info) if feats is None else feats
    origin = info.last_week
    future = range(max(weeks[0], origin + 1), weeks[-1] + 1)
    model = DisruptionModel.fit(info, kind, features, feats=feats, sigma_min=sigma_min)
    sup = model.predict_suppliers(feats, origin, future)
    at = feats[feats.week == origin].copy()
    at["shock_type"] = shock_class(at).where(at.hit == 1, at.news_type.where(at.news_stage > 0))
    at["shock_type"] = at.shock_type.replace({"none": None})
    return sup.merge(at[["material", "country", "shock_type"]], on=["material", "country"], how="left")


def predict_disruption(info: Data, weeks: range, sup: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """Material-level disruption probability for `weeks` (columns week, material, p).

    Weeks already released get the observed outcome; later weeks get the model's forecast from
    the last released week, blended toward the historical rate. `sup` reuses a supplier_forecast.
    """
    origin = info.last_week
    parts = [info.outcomes[info.outcomes.week.isin(weeks)][["week", "material", "disrupted"]]
             .rename(columns={"disrupted": "p"}).astype({"p": float})]
    if weeks[-1] > origin:
        sup = supplier_forecast(info, weeks, **kw) if sup is None else sup
        mp = material_probability(sup)
        rate = mp.material.map(info.outcomes.groupby("material").disrupted.mean())
        w = np.clip(BLEND_BASE + BLEND_PER_WEEK * (mp.week - origin - 1), 0, 1)
        mp["p"] = (1 - w) * mp.p + w * rate
        parts.append(mp)
    return pd.concat(parts, ignore_index=True)
