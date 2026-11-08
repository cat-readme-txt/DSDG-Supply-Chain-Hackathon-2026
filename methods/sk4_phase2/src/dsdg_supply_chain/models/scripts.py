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

CONV_OVERRIDE: dict = {}  # story type -> forced conversion (scenario bets; empty = off)
PRIOR_STRENGTH = 2.0     # pseudo-stories of the pooled conversion rate added to each type
LEAD_KERNEL = 1          # +-1 week smoothing of the empirical lead distribution (planners)
# Phase 2: the forecast columns (market check on) use a wider +-3 smoothing: dense Brier skill
# 0.3232 -> 0.3249 (newest origins 0.314 -> 0.328), but fed to the planners it cost 0.1 on the
# 72-window backtest (-6.1 at 159 private, -4.7 at 171 public), so the planners keep +-1.
LEAD_KERNEL_FORECAST = 3
DEFAULT_LEADS = {1: [2, 3, 4, 5], 2: [1, 2, 3], 3: [1, 1, 2]}

# Stage gate (Phase 2): a warned shock passes through its last pre-onset stage at least one week
# before it starts. In weeks 1-221, 15 of 16 real warned stories did (bans 3/3, strikes 3/3,
# sanctions 2/2, tariffs 3/3, shortages 3/4, closures 1/1; the parser has no imminent closure
# headline, so a closure's last stage is "escalating"). An open story that has not reached that
# stage yet can start next week only with the pooled skip rate q = (skips + 1) / (real + 2); the
# rest of next week's onset mass moves to the week after. (Shifting the whole pmf instead double
# counted: the lead from a stage already includes the passage through the later stages; it lost
# 4 points at origin 171, the MERROW closure that went early -> escalating -> onset in 3 weeks.)
STAGE_GATE = True
LAST_STAGE = {"ban": 3, "strike": 3, "sanctions": 3, "tariff": 3, "shortage": 3, "closure": 2}


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


def _resolved(S: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Stories whose outcome is known at the origin (onset, called off, or stale = a rumor)."""
    stale = (~S.resolved) & (origin - S["last"] > STALE_WEEKS)
    done = S[S.resolved | stale].copy()
    done["real"] = done.onset.notna()
    return done


# --- market confirmation of open stories (Round 6) -------------------------------------------------
# Market series see a REAL upcoming event before it starts but ignore rumors (features/market.py
# market_signals; eda/28_market_signals.py). So, optionally, each open story gets a market check:
#   supplier stories (shortage, ban: the types that move prices): the supplier's futures basis
#     * confirmed: basis >= Z_CONFIRM sd of calm noise in some story week, or >= Z_HINT sd in two
#       weeks running -> conversion CONFIRMED_CONV; onset pmf reweighted by how far the ramp has grown
#       compared with past ramps (the basis reaches ~25% of its pre-onset peak 4 weeks ahead,
#       ~45% at 3, ~65% at 2 and the peak the week before onset)
#     * no hint: >= 2 story weeks observed and basis < Z_HINT sd throughout -> odds x P(miss | real)
#   lane stories (strike, closure) from the escalating stage on: lane congestion >= CONG_CONFIRM
#     (closures also: lane-country tension >= TENSION_CONFIRM that jumped >= TENSION_JUMP above its
#     6-month median) -> conversion CONFIRMED_CONV and onset next week (CONFIRMED_NEXT); at the final
#     pre-onset stage (strike: imminent, closure: escalating) without it -> odds x P(miss | real)
#   P(miss | real) = (real stories that lacked the signal + 1) / (real stories + 2), counted on stories
#   resolved before the origin (Laplace), taking rumors to never show it (none of the resolved did).
#   Guard (OFF, USE_GUARD): a basis ramp on a supplier with no open shortage/ban story treated as an
#   unannounced shortage with conversion GUARD_CONV. In weeks 1-208 every real ramp came with news and
#   the guard's only firing (ORI natural gas, week 118: 2.2 then 3.3 sd) was noise, so it stays off.
RAMP_TYPES = {"shortage", "ban"}
FINAL_STAGE = {"strike": 3, "closure": 2}   # the parser has no imminent-stage closure headline
Z_CONFIRM, Z_HINT = 3.0, 2.0                # in sd of calm basis noise: ~0.1% / ~2% of calm weeks
CONG_CONFIRM = 50.0                          # real strikes/closures 52-57 the week before onset; rumors <= 46
TENSION_CONFIRM, TENSION_JUMP = 47.0, 10.0   # real closures >= 47; rumors <= 43
CONFIRMED_CONV = 0.95
CONFIRMED_NEXT = 0.9                         # 3/3 confirmed lane stories started the next week
RAMP_TAU = 0.4                               # log-sd of basis / peak around the average ramp profile
GUARD_CONV = 0.3
USE_GUARD = False


def _odds_scale(p: float, lr: float) -> float:
    if p >= 1.0:
        return 1.0
    o = p / (1 - p) * lr
    return float(o / (1 + o))


@dataclass
class MarketCheck:
    bz: pd.DataFrame        # basis z-score, week x (material, country)
    basis: pd.DataFrame     # basis, week x (material, country)
    lane_sig: pd.DataFrame  # per (lane, week): congestion, lane_tension, lane_tension_jump
    lr_miss: dict           # "supplier" / "lane" -> P(no market signal | real story)
    peak: dict              # story type (and "pooled") -> mean basis the week before onset
    profile: dict           # k -> mean of basis(onset - k) / basis(onset - 1) over past real ramps
    counts: dict            # evidence behind lr_miss (for reports)
    tighten: bool = True

    @classmethod
    def fit(cls, info: Data, S: pd.DataFrame, tighten: bool = True) -> "MarketCheck":
        from ..features.market import market_signals
        sig = market_signals(info)
        bz = sig.pivot_table(index="week", columns=["material", "country"], values="basis_z")
        basis = sig.pivot_table(index="week", columns=["material", "country"], values="basis")
        lane_sig = sig.groupby(["lane", "week"])[["congestion", "lane_tension", "lane_tension_jump"]].first()
        mk = cls(bz, basis, lane_sig, {}, {}, {}, {}, tighten)
        done = _resolved(S, info.last_week) if len(S) else S
        n = dict(supplier_real=0, supplier_miss=0, supplier_rumor_alarm=0,
                 lane_real=0, lane_miss=0, lane_rumor_alarm=0)
        peaks, prof = {}, {}
        for r in done.itertuples() if len(done) else []:
            end = int(r.onset) - 1 if r.real else int(r.last)
            if r.type in RAMP_TYPES:
                v = mk.ramp(r.entity, int(r.start), min(int(r.start) + 1, end))
                if not r.real:
                    n["supplier_rumor_alarm"] += mk.ramp(r.entity, int(r.start), end) is True
                elif end - int(r.start) >= 1:       # two pre-onset weeks: the "no hint" rule could apply
                    n["supplier_real"] += 1
                    n["supplier_miss"] += v is False
                if r.real and mk.ramp(r.entity, int(r.start), end) is True:
                    c, m = r.entity.split("/")
                    b = basis[(m, c)]
                    top = b.get(end, np.nan)
                    if np.isfinite(top) and top > 0:
                        peaks.setdefault(r.type, []).append(top)
                        for k in range(1, 5):
                            if end - k + 1 >= r.start and np.isfinite(b.get(end - k + 1, np.nan)):
                                prof.setdefault(k, []).append(b[end - k + 1] / top)
            elif r.type in FINAL_STAGE:
                wf = getattr(r, f"w{FINAL_STAGE[r.type]}")
                if wf is None or pd.isna(wf):
                    continue
                hit = any(mk.lane_signal(r.type, r.entity, w) for w in range(int(wf), end + 1))
                if r.real:
                    n["lane_real"] += 1
                    n["lane_miss"] += not hit
                else:
                    n["lane_rumor_alarm"] += hit
        mk.lr_miss = {k: (n[f"{k}_miss"] + 1) / (n[f"{k}_real"] + 2) for k in ("supplier", "lane")}
        mk.peak = {t: float(np.mean(v)) for t, v in peaks.items()}
        if peaks:
            mk.peak["pooled"] = float(np.mean([x for v in peaks.values() for x in v]))
        mk.profile = {k: float(np.mean(v)) for k, v in sorted(prof.items()) if len(v) >= 2}
        mk.counts = n
        return mk

    def ramp(self, entity: str, w0: int, w1: int) -> bool | None:
        """Basis verdict for a supplier story over weeks w0..w1: True = ramp (confirmed),
        False = no hint at all over >= 2 weeks, None = too early or unclear."""
        c, m = entity.split("/")
        if (m, c) not in self.bz:
            return None
        z = self.bz[(m, c)].loc[w0:w1].to_numpy(float)
        if not len(z):
            return None
        if z.max() >= Z_CONFIRM or (len(z) >= 2 and (np.minimum(z[1:], z[:-1]) >= Z_HINT).any()):
            return True
        if len(z) >= 2 and z.max() < Z_HINT:
            return False
        return None

    def lane_signal(self, typ: str, lane: str, week: int) -> bool:
        if (lane, week) not in self.lane_sig.index:
            return False
        s = self.lane_sig.loc[(lane, week)]
        ok = s.congestion >= CONG_CONFIRM
        if typ == "closure":
            ok = ok or (s.lane_tension >= TENSION_CONFIRM and s.lane_tension_jump >= TENSION_JUMP)
        return bool(ok)

    def _rho(self, k: np.ndarray) -> np.ndarray:
        """Average basis / pre-onset peak k weeks before onset; beyond the profile, decay geometrically."""
        K = max(self.profile)
        last, prev = self.profile[K], self.profile.get(K - 1, 1.0)
        tail = last * (last / prev) ** np.maximum(k - K, 0)
        return np.clip([self.profile.get(int(x), t) for x, t in zip(k, tail)], 0.01, None)

    def tighten_pmf(self, typ: str, entity: str, pmf: np.ndarray, origin: int) -> np.ndarray:
        """Reweight the onset pmf by how far the ramp has grown (basis now / typical pre-onset peak)."""
        peak = self.peak.get(typ, self.peak.get("pooled"))
        if not self.tighten or not peak or len(self.profile) < 2 or pmf.sum() <= 0:
            return pmf
        c, m = entity.split("/")
        b = self.basis[(m, c)].get(origin, np.nan) if (m, c) in self.basis else np.nan
        if not np.isfinite(b) or b <= 0:
            return pmf
        k = np.arange(1, len(pmf) + 1)
        like = np.exp(-0.5 * ((np.log(b / peak) - np.log(self._rho(k))) / RAMP_TAU) ** 2)
        post = pmf * like
        return post * (pmf.sum() / post.sum()) if post.sum() > 0 else pmf

    def adjust(self, r, conv: float, pmf: np.ndarray, origin: int) -> tuple[float, np.ndarray, str]:
        """Market-updated (conversion, onset pmf, verdict) for an open story `r` at the origin."""
        if r.type in RAMP_TYPES and "/" in str(r.entity):
            v = self.ramp(r.entity, int(r.start), origin)
            if v is True:
                if pmf.sum() <= 0:     # overdue by the script, but the market says it is coming
                    pmf = self.onset_pmf_default(len(pmf))
                return max(conv, CONFIRMED_CONV), self.tighten_pmf(r.type, r.entity, pmf, origin), "ramp"
            if v is False:
                return _odds_scale(conv, self.lr_miss["supplier"]), pmf, "no ramp"
            return conv, pmf, "unclear"
        if r.type in FINAL_STAGE and r.stage >= 2:
            if self.lane_signal(r.type, r.entity, origin):
                nxt = np.zeros(len(pmf))
                nxt[0] = 1.0
                rest = pmf / pmf.sum() if pmf.sum() > 0 else nxt
                return max(conv, CONFIRMED_CONV), CONFIRMED_NEXT * nxt + (1 - CONFIRMED_NEXT) * rest, "lane signal"
            if r.stage >= FINAL_STAGE[r.type]:
                return _odds_scale(conv, self.lr_miss["lane"]), pmf, "no lane signal"
        return conv, pmf, "n/a"

    @staticmethod
    def onset_pmf_default(horizon: int) -> np.ndarray:
        pmf = np.zeros(horizon)
        pmf[: min(2, horizon)] = [0.6, 0.4][: min(2, horizon)]
        return pmf

    def unannounced(self, open_stories: pd.DataFrame, origin: int) -> list[dict]:
        """Suppliers whose basis ramps (>= Z_CONFIRM sd now, >= Z_HINT sd last week) with no open
        shortage/ban story on them: pseudo-stories of an unannounced shortage."""
        covered = set(open_stories.entity) if len(open_stories) else set()
        out = []
        if origin not in self.bz.index or origin - 1 not in self.bz.index:
            return out
        for (m, c) in self.bz.columns:
            z = self.bz[(m, c)]
            if z[origin] >= Z_CONFIRM and z[origin - 1] >= Z_HINT and f"{c}/{m}" not in covered:
                start = origin - 1
                while start - 1 in z.index and z[start - 1] >= Z_HINT:
                    start -= 1
                out.append(dict(type="shortage", entity=f"{c}/{m}", start=start, stage=1, w1=start))
        return out


def gate_skip_rate(done: pd.DataFrame) -> float:
    """Pooled P(a real warned story starts without first reaching its last pre-onset stage)."""
    real = done[done.real & done[["w1", "w2", "w3"]].notna().any(axis=1) & done.type.isin(list(LAST_STAGE))]
    if not len(real):
        return 1.0
    reached = [pd.notna(r[f"w{LAST_STAGE[r.type]}"]) and r[f"w{LAST_STAGE[r.type]}"] < r.onset
               for _, r in real.iterrows()]
    return float((len(reached) - sum(reached) + 1) / (len(reached) + 2))


@dataclass
class ScriptModel:
    conv: dict          # (type, stage) -> P(real | reached stage)
    leads: dict         # (type, stage) -> list of observed leads (weeks from stage to onset)
    open_stories: pd.DataFrame
    market: MarketCheck | None = None   # Round 6 market confirmation (off by default)
    skip: float = 1.0                   # P(onset without the last pre-onset stage); 1 = gate off
    lead_kernel: int = LEAD_KERNEL

    @classmethod
    def fit(cls, info: Data, market: bool = False, tighten: bool = True) -> "ScriptModel":
        S = stories(info)
        origin = info.last_week
        mk = MarketCheck.fit(info, S, tighten) if market else None
        if not len(S):
            return cls({}, {}, S, mk)
        done = _resolved(S, origin)
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
        return cls(conv, leads, open_, mk, gate_skip_rate(done) if STAGE_GATE else 1.0,
                   LEAD_KERNEL_FORECAST if market else LEAD_KERNEL)

    def onset_pmf(self, typ: str, stage: int, elapsed: int, horizon: int) -> np.ndarray:
        """P(onset at origin + k), k = 1..horizon, given the story reached `stage` `elapsed` weeks ago."""
        L = np.array(self.leads.get((typ, stage), DEFAULT_LEADS[stage]))
        grid = np.arange(0, 60)
        w = np.zeros(len(grid))
        for l in L:
            for d in range(-self.lead_kernel, self.lead_kernel + 1):
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

    def story_terms(self, r, origin: int, horizon: int) -> tuple[float, np.ndarray, str]:
        """(conversion, onset pmf over origin+1..origin+horizon, market verdict) for one open story."""
        stage_week = getattr(r, f"w{r.stage}")
        pmf = self.onset_pmf(r.type, r.stage, origin - int(stage_week), horizon)
        conv = self.conv.get((r.type, r.stage), 0.5)
        if self.market is None:
            out = conv, pmf, "off"
        else:
            out = self.market.adjust(r, conv, pmf, origin)
        if self.skip < 1.0 and r.stage < LAST_STAGE.get(r.type, 0) and len(out[1]) > 1:
            p = out[1].copy()
            p[1] += (1 - self.skip) * p[0]     # it cannot start next week, so that mass moves a week later
            p[0] *= self.skip
            out = (out[0], p, out[2])
        if r.type in CONV_OVERRIDE:        # scenario bet for a pair file (strategies._bet)
            return CONV_OVERRIDE[r.type], out[1], "bet"
        return out

    def live_stories(self, origin: int) -> pd.DataFrame:
        """Open stories, plus (market check on) pseudo-stories for unannounced basis ramps."""
        if self.market is None or not USE_GUARD:
            return self.open_stories
        extra = self.market.unannounced(self.open_stories, origin)
        if not extra:
            return self.open_stories
        return pd.concat([self.open_stories, pd.DataFrame(extra).assign(unannounced=True)], ignore_index=True)

    def explain(self, info: Data, horizon: int = 13) -> pd.DataFrame:
        """One row per live story: conversion and onset pmf without and with the market check."""
        origin = info.last_week
        rows = []
        base = ScriptModel(self.conv, self.leads, self.open_stories, None, self.skip, LEAD_KERNEL)
        for r in self.live_stories(origin).itertuples():
            guard = bool(getattr(r, "unannounced", False) is True)
            c0, pmf0, _ = (GUARD_CONV, base.onset_pmf(r.type, r.stage, origin - int(r.w1), horizon), "") \
                if guard else base.story_terms(r, origin, horizon)
            c1, pmf1, v = (c0, pmf0, "unannounced ramp") if guard else self.story_terms(r, origin, horizon)
            rows.append(dict(type=r.type, entity=r.entity, start=int(r.start), stage=int(r.stage), verdict=v,
                             conv=round(c0, 3), conv_market=round(c1, 3),
                             pmf=np.round(pmf0, 3).tolist(), pmf_market=np.round(pmf1, 3).tolist()))
        return pd.DataFrame(rows)

    def predict(self, info: Data, weeks: range, survival) -> pd.DataFrame:
        """P(hit) per (week, material, country) from open stories. `survival(type, d)` = P(duration >= d)."""
        origin = info.last_week
        targets = [w for w in weeks if w > origin]
        cols = ["week", "material", "country", "p_script", "script_type"]
        live = self.live_stories(origin)
        if not targets or not len(live):
            return pd.DataFrame(columns=cols)
        H = targets[-1] - origin
        sup = info.suppliers
        rows = []
        for r in live.itertuples():
            tgt = _targets(sup, r.type, r.entity)
            if getattr(r, "unannounced", False) is True:
                conv, pmf = GUARD_CONV, self.onset_pmf(r.type, r.stage, origin - int(r.w1), H)
            else:
                conv, pmf, _ = self.story_terms(r, origin, H)
            surv = np.array([survival(r.type, d) for d in range(1, H + 1)])
            # P(hit at origin + h) = conv * sum_{k <= h} pmf[k] * P(duration >= h - k + 1)
            p = np.array([conv * sum(pmf[k - 1] * surv[h - k] for k in range(1, h + 1)) for h in range(1, H + 1)])
            for t in tgt.itertuples():
                for w in targets:
                    rows.append((w, t.material, t.country, float(p[w - origin - 1]), r.type))
        out = pd.DataFrame(rows, columns=cols)
        return out.sort_values("p_script").groupby(["week", "material", "country"], as_index=False).last()
