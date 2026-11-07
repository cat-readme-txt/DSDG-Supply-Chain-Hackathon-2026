"""How long does an ongoing shock last? A censoring-aware duration model per shock type.

Durations are log-normal: log(duration) ~ Normal(mu_type, sigma). Each type's mu is shrunk toward a
pooled mu (a normal prior with scale TAU), so a type seen once or twice borrows strength from the
rest. Shocks still running at the information week are censored: they contribute P(duration > age)
rather than an exact length. Runs that start together (one event hitting several suppliers) are
down-weighted so each event counts once.

P(still hit h weeks later | hit for `age` weeks) = S(age + h) / S(age).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

TAU = 0.7            # prior spread of type log-medians around the pooled log-median
DEFAULT_MU, DEFAULT_SIGMA = np.log(8.0), 0.8
FALLBACK_BY_HIT = {"tariff": "tariff", "cap+lead": "closure", "lead": "strike", "cap": "shortage"}
PRIORITY = ["sanctions", "ban", "tariff", "shortage", "reversal", "closure", "strike"]


def shock_class(feats: pd.DataFrame) -> pd.Series:
    """Type of the shock a hit supplier is in: the news story that reached onset, else by hit type."""
    out = feats.hit_type.map(FALLBACK_BY_HIT)
    for t in reversed(PRIORITY):           # highest priority written last wins
        out = out.where(feats[f"stage_{t}"] < 4, t)
    return out.where(feats.hit == 1)


def shock_runs(feats: pd.DataFrame) -> pd.DataFrame:
    """One row per supplier hit run: type at start, length so far, whether it has ended."""
    f = feats.sort_values(["material", "country", "week"]).assign(cls=shock_class(feats))
    rows = []
    for (m, c), s in f.groupby(["material", "country"]):
        hit, wk, cls = s.hit.to_numpy(), s.week.to_numpy(), s.cls.to_numpy()
        i = 0
        while i < len(hit):
            if hit[i]:
                j = i
                while j + 1 < len(hit) and hit[j + 1]:
                    j += 1
                ended = j + 1 < len(hit)
                # type: first known onset story within the first 2 weeks of the run
                rows.append(dict(material=m, country=c, start=wk[i], length=j - i + 1, ended=ended,
                                 type=cls[min(i + 1, j)]))
                i = j + 1
            else:
                i += 1
    r = pd.DataFrame(rows, columns=["material", "country", "start", "length", "ended", "type"])
    if len(r):
        r["weight"] = 1.0 / r.groupby(["start", "type"]).start.transform("size")
    return r


@dataclass
class DurationModel:
    mu_pool: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA
    mu: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, feats: pd.DataFrame, sigma_min: float = 0.2) -> "DurationModel":
        r = shock_runs(feats)
        if r.ended.sum() < 2:
            return cls(sigma=max(DEFAULT_SIGMA, sigma_min))
        types = sorted(r.type.unique())
        idx = r.type.map({t: i for i, t in enumerate(types)}).to_numpy()
        L = r.length.to_numpy(float)
        log_lo, log_hi = np.log(np.maximum(L - 0.5, 0.5)), np.log(L + 0.5)
        ended, w = r.ended.to_numpy(), r.weight.to_numpy()

        def nll(theta):
            mu_pool, sig, mus = theta[0], np.exp(theta[1]), theta[2:]
            # discrete weeks: an ended run of length L means duration in (L-0.5, L+0.5]; ongoing: > L-0.5
            lo, hi = (log_lo - mus[idx]) / sig, (log_hi - mus[idx]) / sig
            ll = np.where(ended, np.log(np.clip(norm.cdf(hi) - norm.cdf(lo), 1e-12, None)), norm.logsf(lo))
            prior = ((mus - mu_pool) ** 2).sum() / (2 * TAU ** 2) + (mu_pool - DEFAULT_MU) ** 2 / 2
            return -(w * ll).sum() + prior

        x0 = np.r_[DEFAULT_MU, np.log(max(DEFAULT_SIGMA, sigma_min)), np.full(len(types), DEFAULT_MU)]
        res = minimize(nll, x0, method="L-BFGS-B",
                       bounds=[(0, 5), (np.log(sigma_min), np.log(max(2.0, sigma_min)))] + [(0, 5)] * len(types))
        mu_pool, log_sigma, mus = res.x[0], res.x[1], res.x[2:]
        return cls(float(mu_pool), float(np.exp(log_sigma)), dict(zip(types, map(float, mus))))

    def survival(self, typ: pd.Series, age: np.ndarray, h: np.ndarray) -> np.ndarray:
        """P(still hit h weeks later | hit for `age` weeks so far)."""
        mu = typ.map(self.mu).fillna(self.mu_pool).to_numpy()
        s_now = norm.sf((np.log(np.maximum(age - 0.5, 0.5)) - mu) / self.sigma)
        s_then = norm.sf((np.log(age + h - 0.5) - mu) / self.sigma)
        return np.clip(s_then / np.maximum(s_now, 1e-12), 0.0, 1.0)

    def median_weeks(self) -> dict:
        return {t: round(float(np.exp(m)), 1) for t, m in {**self.mu, "pooled": self.mu_pool}.items()}
