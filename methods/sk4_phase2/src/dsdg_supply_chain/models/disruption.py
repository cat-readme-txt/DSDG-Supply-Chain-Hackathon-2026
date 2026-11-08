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


def material_probability(sup: pd.DataFrame, tariff: bool = True) -> pd.DataFrame:
    """P(at least one supplier hit), treating suppliers as independent. `tariff=False` leaves out
    announced tariffs (column p_tariff), e.g. for supply planning."""
    q = 1 - sup.p
    if tariff and "p_tariff" in sup:
        q = q * (1 - sup.p_tariff)
    return (sup.assign(q=q).groupby(["week", "material"]).q.prod()
            .rsub(1).rename("p").reset_index())


# Defaults chosen by walk-forward evaluation (eda/07_disruption_model.py)
DEFAULT_KIND, DEFAULT_FEATURES, DEFAULT_SIGMA_MIN = "gbm", "news+contagion", 0.6
# Shrink toward each material's historical rate, more so further ahead: unwarned shocks exist.
BLEND_BASE, BLEND_PER_WEEK = 0.10, 0.01


def supplier_forecast(info: Data, weeks: range, feats: pd.DataFrame | None = None, kind: str = DEFAULT_KIND,
                      features: str = DEFAULT_FEATURES, sigma_min: float = DEFAULT_SIGMA_MIN,
                      market: bool = False) -> pd.DataFrame:
    """Per-supplier forecast for future weeks from the last released week.

    Columns: week, material, country, h, hit (hit at origin), p (P(hit) that week), shock_type
    (type of the ongoing shock if hit, else of the open news story, else None).
    `market=True` adds the Round 6 market check of open stories (models/scripts.py MarketCheck).
    """
    feats = build_features(info) if feats is None else feats
    origin = info.last_week
    future = range(max(weeks[0], origin + 1), weeks[-1] + 1)
    model = DisruptionModel.fit(info, kind, features, feats=feats, sigma_min=sigma_min)
    sup = model.predict_suppliers(feats, origin, future)
    at = feats[feats.week == origin].copy()
    at["shock_type"] = shock_class(at).where(at.hit == 1, at.news_type.where(at.news_stage > 0))
    at["shock_type"] = at.shock_type.replace({"none": None})
    sup = sup.merge(at[["material", "country", "shock_type"]], on=["material", "country"], how="left")
    if USE_SCRIPTS:
        sup = _add_scripts(info, future, sup, model.duration, market)
    return sup


# Story-script model (models/scripts.py) for suppliers with an open news story: max(GBM, script).
USE_SCRIPTS = True


def _add_scripts(info: Data, weeks: range, sup: pd.DataFrame, duration: DurationModel,
                 market: bool = False) -> pd.DataFrame:
    from .scripts import ScriptModel
    def survival(typ, d):
        return float(duration.survival(pd.Series([typ]), np.array([1.0]), np.array([d - 1.0]))[0])
    model = ScriptModel.fit(info, market=market)
    ann = _announced(info)
    if ANNOUNCED_TARIFF_OVERRIDES_STORY and len(ann) and len(model.open_stories):
        # an announced hike fixes the start week (7/7 started exactly then), so the tariff story's
        # own onset timing is dropped, and before the effective week the supplier only carries the
        # risk of other stories or, failing that, the background risk of its material's other suppliers
        keys = {f"{c}/{m}" for m, c in zip(ann.material, ann.country)}
        os_ = model.open_stories
        model.open_stories = os_[~((os_.type == "tariff") & os_.entity.isin(keys))]
    sc = model.predict(info, weeks, survival)
    if ANNOUNCED_TARIFF_OVERRIDES_STORY:
        for r in ann.itertuples():
            pre = (sup.material == r.material) & (sup.week < int(r.announced_effective_week))
            own = pre & (sup.country == r.country) & (sup.hit == 0) & (sup.shock_type == "tariff")
            others = sup[pre & (sup.country != r.country)].groupby("week").p.max()
            sup.loc[own, "p"] = sup.loc[own, "week"].map(others).fillna(0.0).to_numpy()
            sup.loc[own, "shock_type"] = None
    if not len(sc):
        return _add_tariff_announcements(info, sup, survival)
    sup = sup.merge(sc, on=["week", "material", "country"], how="left")
    new = (sup.hit == 0) & sup.p_script.notna()
    sup.loc[new, "p"] = np.maximum(sup.loc[new, "p"], sup.loc[new, "p_script"])
    sup.loc[new & sup.shock_type.isna(), "shock_type"] = sup.loc[new & sup.shock_type.isna(), "script_type"]
    sup = sup.drop(columns=["p_script", "script_type"])
    return _add_tariff_announcements(info, sup, survival)


# Phase 2: announced tariffs get their own duration curve (log-normal fitted to past tariff events
# only, sigma >= TARIFF_SIGMA_MIN) instead of the shared-sigma duration model, whose sigma (>= 0.6)
# is set by the much noisier shock types. Past tariffs lasted 10, 12 and 18 weeks.
ANNOUNCED_TARIFF_OVERRIDES_STORY = True
TARIFF_OWN_DURATION = True
TARIFF_SIGMA_MIN = 0.3


def _announced(info: Data) -> pd.DataFrame:
    t = info.tariffs[info.tariffs.week == info.last_week]
    return t[t.announced_effective_week.notna() & (t.announced_rate > t.tariff_rate + 1e-6)]


def tariff_runs(info: Data) -> list[int]:
    """Lengths of past tariff events (one per start week: a hike on several materials counts once)."""
    t = info.tariffs.merge(info.suppliers[["material", "country", "base_tariff"]], on=["material", "country"])
    t = t.assign(hit=t.tariff_rate > t.base_tariff + 1e-9).sort_values("week")
    runs = {}
    for _, g in t.groupby(["material", "country"]):
        h, w = g.hit.to_numpy(), g.week.to_numpy()
        i = 0
        while i < len(h):
            if h[i]:
                j = i
                while j < len(h) and h[j]:
                    j += 1
                if j < len(h):                      # ended before the information week
                    runs[w[i]] = max(runs.get(w[i], 0), j - i)
                i = j
            else:
                i += 1
    return list(runs.values())


def tariff_survival(info: Data):
    """P(an announced tariff is still in force in its d-th week), or None with < 2 past events."""
    L = np.array(tariff_runs(info), float)
    if len(L) < 2:
        return None
    from scipy.stats import norm
    mu, sd = np.log(L).mean(), max(np.log(L).std(ddof=1), TARIFF_SIGMA_MIN)
    return lambda d: float(norm.sf((np.log(d - 0.5) - mu) / sd)) if d > 1 else 1.0


ANNOUNCED_FLOOR = True
# Phase 2 bug fix: an announced tariff used to be written into the supplier's hit probability `p`,
# whose shock type (and so the planners' capacity/lead severity) could belong to another story: KES
# semiconductors got P = 1 tagged "closure", so the planners treated a MERROW closure as certain from
# week 224. The tariff only changes cost, which supplier_state already applies from the effective
# week, so it now lives in its own column `p_tariff`: it counts in the disruption forecast and is
# ignored by the planners' supply scenarios.
SEPARATE_TARIFF = True


def _announcement_floor(info: Data, mp: pd.DataFrame) -> np.ndarray:
    """Per (week, material) row of `mp`: P(some supplier is under an announced, not yet ended hike)."""
    t = _announced(info)
    floor = np.zeros(len(mp))
    surv = tariff_survival(info) if TARIFF_OWN_DURATION else None
    if not len(t) or surv is None:
        return floor
    for r in t.itertuples():
        eff = int(r.announced_effective_week)
        m = (mp.material == r.material).to_numpy() & (mp.week >= eff).to_numpy()
        d = (mp.week.to_numpy() - eff + 1)[m]
        floor[m] = np.maximum(floor[m], [surv(int(x)) for x in d])
    return floor


def _add_tariff_announcements(info: Data, sup: pd.DataFrame, survival) -> pd.DataFrame:
    """tariffs.csv publishes upcoming hikes (announced_rate, announced_effective_week). All 5 in
    weeks 1-208 took effect exactly as announced (eda/26), so a supplier with an announced hike is
    hit from that week with certainty, then follows the tariff duration curve."""
    t = _announced(info)
    own = tariff_survival(info) if TARIFF_OWN_DURATION else None
    surv = own or (lambda d: survival("tariff", d))
    if SEPARATE_TARIFF:
        sup["p_tariff"] = 0.0
    for r in t.itertuples():
        eff = int(r.announced_effective_week)
        rows = (sup.material == r.material) & (sup.country == r.country) & (sup.week >= eff) & (sup.hit == 0)
        d = (sup.loc[rows, "week"] - eff + 1).to_numpy()
        pt = np.array([surv(int(x)) for x in d])
        if SEPARATE_TARIFF:
            sup.loc[rows, "p_tariff"] = pt
            continue
        sup.loc[rows, "p"] = np.maximum(sup.loc[rows, "p"], pt)
        sup.loc[rows & sup.shock_type.isna(), "shock_type"] = "tariff"
    return sup


def predict_disruption(info: Data, weeks: range, sup: pd.DataFrame | None = None, tariff: bool = True,
                       **kw) -> pd.DataFrame:
    """Material-level disruption probability for `weeks` (columns week, material, p).

    Weeks already released get the observed outcome; later weeks get the model's forecast from
    the last released week, blended toward the historical rate. `sup` reuses a supplier_forecast.
    """
    origin = info.last_week
    parts = [info.outcomes[info.outcomes.week.isin(weeks)][["week", "material", "disrupted"]]
             .rename(columns={"disrupted": "p"}).astype({"p": float})]
    if weeks[-1] > origin:
        sup = supplier_forecast(info, weeks, **kw) if sup is None else sup
        mp = material_probability(sup, tariff)
        rate = mp.material.map(info.outcomes.groupby("material").disrupted.mean())
        w = np.clip(BLEND_BASE + BLEND_PER_WEEK * (mp.week - origin - 1), 0, 1)
        blended = (1 - w) * mp.p + w * rate
        if ANNOUNCED_FLOOR and tariff:
            # the blend stands for unwarned shocks; it must not pull a published tariff below its own odds
            floor = _announcement_floor(info, mp)
            blended = np.maximum(blended, floor)
        mp["p"] = blended
        parts.append(mp)
    return pd.concat(parts, ignore_index=True)
