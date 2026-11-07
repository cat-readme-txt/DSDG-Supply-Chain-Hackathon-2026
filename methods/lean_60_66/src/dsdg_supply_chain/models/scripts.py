"""Story-script model: the news follows fixed scripts per shock type, so model the scripts directly.

eda/20_story_scripts.py groups signal articles into stories (one shock type about one entity: a
lane for strikes and closures, a country for sanctions, a country + material otherwise) and shows:
  * conversion depends on the type: shortages, tariffs, sanctions and bans almost always become
    real; closures and strikes are often rumors that get called off;
  * the lead from each stage to onset is short and regular (early -> onset ~2-8 weeks, escalating
    -> onset ~1-4, imminent -> onset ~1-4);
A gradient-boosted classifier with large leaves averages these few stories away, so here each
open story at the information week gets

  P(hit at week t) = conv(type, stage) x sum_k P(onset = k | not yet) x P(duration > t - k)

with conv and the lead distribution estimated only from stories already resolved at the origin
(Beta prior pooled over types), and the duration survival from models/duration.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Data
from ..features.news import COUNTRY_TYPES, LANE_TYPES, STAGE_LEVEL, STALE_WEEKS, parse_news

PRIOR_STRENGTH = 2.0     # pseudo-stories of the pooled conversion rate added to each type
LEAD_KERNEL = 1          # +-1 week smoothing of the empirical lead distribution
DEFAULT_LEADS = {1: [2, 3, 4, 5], 2: [1, 2, 3], 3: [1, 1, 2]}


def _entity(e, lane_of: dict) -> str | None:
    if e.type in LANE_TYPES:
        return e.lane or lane_of.get(e.country)
    if e.type in COUNTRY_TYPES:
        return e.country
    if e.country and e.material:
        return f"{e.country}/{e.material}"
    return None


def stories(info: Data) -> pd.DataFrame:
    """One row per story with the week each stage was first reached, onset week and resolution."""
    ev = parse_news(info.news)
    if not len(ev):
        return pd.DataFrame()
    ev = ev.astype(object).where(ev.notna(), None)
    lane_of = info.suppliers.groupby("country").lane.first().to_dict()
    typed = ev[ev.type.notna() & ev.stage.isin(list(STAGE_LEVEL) + ["called_off"])].copy()
    off = ev[ev.type.isna() & (ev.stage == "called_off")].copy()   # "Deal reached ... averted"
    off["type"] = "strike"
    typed = pd.concat([typed, off])
    typed["entity"] = [_entity(e, lane_of) for e in typed.itertuples()]
    typed = typed[typed.entity.notna()].sort_values("week")
    out = []
    for (t, ent), g in typed.groupby(["type", "entity"]):
        cur = None
        for e in g.itertuples():
            if cur is None or e.week - cur["last"] > STALE_WEEKS or cur["resolved"]:
                if cur:
                    out.append(cur)
                cur = dict(type=t, entity=ent, start=e.week, last=e.week, onset=None, called_off=None,
                           resolved=False, **{f"w{s}": None for s in (1, 2, 3)})
            cur["last"] = e.week
            if e.stage == "called_off":
                cur["called_off"], cur["resolved"] = e.week, True
            elif e.stage == "onset":
                cur["onset"], cur["resolved"] = e.week, True
            else:
                lvl = STAGE_LEVEL[e.stage]
                for s in range(1, lvl + 1):          # reaching stage s means reaching all lower ones
                    if cur[f"w{s}"] is None or pd.isna(cur[f"w{s}"]):
                        cur[f"w{s}"] = e.week
        out.append(cur)
    return pd.DataFrame(out)


def _targets(sup: pd.DataFrame, typ: str, entity: str) -> pd.DataFrame:
    """Supplier rows a story is about: a whole lane, a whole country, or one country + material."""
    if typ in LANE_TYPES:
        return sup[sup.lane == entity]
    if typ in COUNTRY_TYPES:
        return sup[sup.country == entity]
    c, m = entity.split("/")
    return sup[(sup.country == c) & (sup.material == m)]


@dataclass
class ScriptModel:
    conv: dict          # (type, stage) -> P(real | reached stage)
    leads: dict         # (type, stage) -> list of observed leads (weeks from stage to onset)
    open_stories: pd.DataFrame

    @classmethod
    def fit(cls, info: Data) -> "ScriptModel":
        S = stories(info)
        origin = info.last_week
        if not len(S):
            return cls({}, {}, S)
        stale = (~S.resolved) & (origin - S["last"] > STALE_WEEKS)
        done = S[S.resolved | stale].copy()
        done["real"] = done.onset.notna()
        conv, leads = {}, {}
        for s in (1, 2, 3):
            reached = done[done[f"w{s}"].notna()]
            pooled = (reached.real.sum() + 1) / (len(reached) + 2)
            for t in S.type.unique():
                r = reached[reached.type == t]
                conv[(t, s)] = float((r.real.sum() + PRIOR_STRENGTH * pooled) / (len(r) + PRIOR_STRENGTH))
                L = (r.onset - r[f"w{s}"]).dropna().astype(int).tolist()
                Lp = (reached.onset - reached[f"w{s}"]).dropna().astype(int).tolist() or DEFAULT_LEADS[s]
                leads[(t, s)] = L if len(L) >= 2 else Lp
        open_ = S[(~S.resolved) & (origin - S["last"] <= STALE_WEEKS)].copy()
        open_ = open_[open_[["w1", "w2", "w3"]].notna().any(axis=1)]
        open_["stage"] = [max(s for s in (1, 2, 3) if pd.notna(r[f"w{s}"])) for _, r in open_.iterrows()]
        return cls(conv, leads, open_)

    def onset_pmf(self, typ: str, stage: int, elapsed: int, horizon: int) -> np.ndarray:
        """P(onset at origin + k), k = 1..horizon, given the story reached `stage` `elapsed` weeks ago."""
        L = np.array(self.leads.get((typ, stage), DEFAULT_LEADS[stage]))
        grid = np.arange(0, 60)
        w = np.zeros(len(grid))
        for l in L:
            for d in range(-LEAD_KERNEL, LEAD_KERNEL + 1):
                if 0 <= l + d < len(grid):
                    w[l + d] += 1.0 if d == 0 else 0.5
        w[: elapsed + 1] = 0.0                  # it has not happened yet
        pmf = np.zeros(horizon)
        if w.sum() > 0:
            w /= w.sum()
            for k in range(1, horizon + 1):
                if elapsed + k < len(w):
                    pmf[k - 1] = w[elapsed + k]
        # else overdue: every real story of this kind had hit by now, so it is a dead rumor -> 0
        return pmf

    def predict(self, info: Data, weeks: range, survival) -> pd.DataFrame:
        """P(hit) per (week, material, country) from open stories. `survival(type, d)` = P(duration >= d)."""
        origin = info.last_week
        targets = [w for w in weeks if w > origin]
        cols = ["week", "material", "country", "p_script", "script_type"]
        if not targets or not len(self.open_stories):
            return pd.DataFrame(columns=cols)
        H = targets[-1] - origin
        sup = info.suppliers
        rows = []
        for r in self.open_stories.itertuples():
            tgt = _targets(sup, r.type, r.entity)
            stage_week = getattr(r, f"w{r.stage}")
            pmf = self.onset_pmf(r.type, r.stage, origin - int(stage_week), H)
            conv = self.conv.get((r.type, r.stage), 0.5)
            surv = np.array([survival(r.type, d) for d in range(1, H + 1)])
            # P(hit at origin + h) = conv * sum_{k <= h} pmf[k] * P(duration >= h - k + 1)
            p = np.array([conv * sum(pmf[k - 1] * surv[h - k] for k in range(1, h + 1)) for h in range(1, H + 1)])
            for t in tgt.itertuples():
                for w in targets:
                    rows.append((w, t.material, t.country, float(p[w - origin - 1]), r.type))
        out = pd.DataFrame(rows, columns=cols)
        return out.sort_values("p_script").groupby(["week", "material", "country"], as_index=False).last()
