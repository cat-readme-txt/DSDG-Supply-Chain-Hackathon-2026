"""News → structured shock signals → per-supplier weekly features.

Pipeline
  1. classify: match each headline against a small rule table to get a shock type and a story stage;
     headlines matching no rule are routine noise.
  2. resolve:  find the country / lane / material the article is about (headline first, then body)
     and the severity number in the body ("about 67 percent").
  3. expand:   map each event onto the supplier rows it concerns (lane → every country on it,
     country → all its materials, country + material → one row).
  4. track:    walk forward week by week keeping an open story per (supplier, shock type), and emit
     features that use only articles published up to that week.

Stages: early (1) → escalating (2) → imminent (3) → onset (4); plus ongoing, called_off, recovered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Data

TYPES = ["tariff", "shortage", "ban", "strike", "closure", "sanctions", "reversal"]
LANE_TYPES = {"strike", "closure"}
COUNTRY_TYPES = {"sanctions"}
EARLY, ESCALATING, IMMINENT, ONSET = 1, 2, 3, 4
STAGE_LEVEL = {"early": EARLY, "escalating": ESCALATING, "imminent": IMMINENT, "onset": ONSET}
STALE_WEEKS = 8   # a story with no new (or "ongoing") article for this long is dropped

# (headline regex, shock type or None for generic, stage)
RULES: list[tuple[str, str | None, str]] = [
    # tariff
    (r"trade dispute", "tariff", "early"),
    (r"officials review .* trade practices", "tariff", "early"),
    (r"confirms new duties|Tariff schedule on .* published|front-load .* ahead of tariff", "tariff", "imminent"),
    (r"New duties on .* take effect", "tariff", "onset"),
    # resource shortage
    (r"flag maintenance delays|threaten walkout", "shortage", "early"),
    (r"rations water|Accident halts", "shortage", "escalating"),
    (r"output forecast cut", "shortage", "imminent"),
    (r"production falls amid shortage", "shortage", "onset"),
    # export ban
    (r"debate limits on raw material exports|calls .* a strategic priority", "ban", "early"),
    (r"slower license approvals|drafts restrictions", "ban", "escalating"),
    (r"export curbs within weeks|Traders brace for .* export halt", "ban", "imminent"),
    (r"halts .* exports", "ban", "onset"),
    (r"officials rule out new restrictions", "ban", "called_off"),
    # port strike
    (r"vote on contract offer|Port labor talks drag on", "strike", "early"),
    (r"Congestion builds at ports", "strike", "escalating"),
    (r"Union sets strike date|Strike on the .* looks likely|Shippers rush cargo ahead", "strike", "imminent"),
    (r"Strike shuts ports", "strike", "onset"),
    # strait closure
    (r"Tensions rise between|Port waiting times creep", "closure", "early"),
    (r"war risk premiums|Naval patrols increase|Warning shots fired", "closure", "escalating"),
    (r"Traffic through the .* suspended", "closure", "onset"),
    (r"Calm returns near", "closure", "called_off"),
    # sanctions
    (r"Diplomatic talks between .* break down|currency slips|lawmakers criticize", "sanctions", "early"),
    (r"prepare for .* sanctions", "sanctions", "escalating"),
    (r"Sanctions vote on .* scheduled", "sanctions", "imminent"),
    (r"Sanctions on .* take effect", "sanctions", "onset"),
    # policy reversal: no warning, onset only
    (r"abruptly suspends .* export licenses", "reversal", "onset"),
    # generic story updates
    (r"Deal reached, .* averted", None, "called_off"),
    (r"disruption continues|No quick fix for|disruption drags on", None, "ongoing"),
    (r"supply eases|flows return to normal|Traffic returns to", None, "recovered"),
]
_RULES = [(re.compile(p, re.I), t, s) for p, t, s in RULES]

COUNTRY_NAMES = {"Kestria": "KES", "Valdmar": "VAL", "Sunda Coast": "SUN", "Orinoque": "ORI",
                 "Halvard": "HAL", "Aurelia": "AUR"}
LANE_NAMES = {"Merrow Strait": "MERROW", "Northern Sea route": "NORTH", "Southern route": "SOUTH",
              "Atlantic route": "ATLANTIC", "domestic rail": "RAIL"}
MATERIAL_NAMES = {"lithium": "lithium", "cobalt": "cobalt", "copper": "copper", "aluminum": "aluminum",
                  "steel": "steel", "rare earth": "rare_earths", "semiconductor": "semiconductors",
                  "polymer resin": "polymer_resin", "specialty glass": "specialty_glass",
                  "natural gas": "natural_gas"}
_SEVERITY = re.compile(r"about (\d+(?:\.\d+)?) percent")


def _find(text: str, names: dict[str, str]) -> list[str]:
    hits = sorted((text.find(k), v) for k, v in names.items() if k in text)
    return list(dict.fromkeys(v for _, v in hits))


def classify(headline: str) -> tuple[str | None, str] | None:
    for rx, typ, stage in _RULES:
        if rx.search(headline):
            return typ, stage
    return None


def parse_news(news: pd.DataFrame) -> pd.DataFrame:
    """Steps 1–2: one row per signal article with type, stage, entities and severity."""
    rows = []
    for a in news.itertuples(index=False):
        c = classify(a.headline)
        if c is None:
            continue
        typ, stage = c
        text = f"{a.headline} {a.body}"
        countries = _find(a.headline, COUNTRY_NAMES) or _find(a.body, COUNTRY_NAMES)
        if len(countries) > 1:          # Aurelia is the importer when another country is named
            countries = [x for x in countries if x != "AUR"]
        lanes = _find(text, LANE_NAMES)
        mats = _find(a.headline, MATERIAL_NAMES) or _find(a.body, MATERIAL_NAMES)
        sev = _SEVERITY.search(a.body)
        rows.append(dict(article_id=a.article_id, week=a.week, type=typ, stage=stage,
                         country=countries[0] if countries else None,
                         lane=lanes[0] if lanes else None,
                         material=mats[0] if mats else None,
                         severity=float(sev.group(1)) / 100 if sev else np.nan,
                         headline=a.headline))
    return pd.DataFrame(rows)


def expand_to_suppliers(events: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Step 3: one row per (event, supplier) the event concerns."""
    sup = suppliers[["material", "country", "lane"]]
    lane_of = suppliers.groupby("country").lane.first().to_dict()
    out = []
    events = events.astype(object).where(events.notna(), None)   # NaN → None so truth tests work
    for e in events.itertuples(index=False):
        lane_scope = e.type in LANE_TYPES or (e.type is None and e.lane and not e.material)
        if lane_scope:
            lane = e.lane or lane_of.get(e.country)
            tgt = sup[sup.lane == lane]
        elif e.type in COUNTRY_TYPES or (e.country and not e.material):
            tgt = sup[sup.country == e.country]
        else:
            tgt = sup[(sup.country == e.country) & (sup.material == e.material)]
        for t in tgt.itertuples(index=False):
            out.append({**e._asdict(), "material": t.material, "country": t.country, "lane": t.lane})
    out = pd.DataFrame(out)
    return out.astype(object).where(out.notna(), None)


@dataclass
class _Story:
    stage: int = 0
    start: int = 0
    last: int = 0
    n: int = 0
    severity: float = np.nan


def news_features(data: Data, weeks: range | None = None) -> pd.DataFrame:
    """Step 4: per (week, material, country) features from articles published up to that week."""
    events = expand_to_suppliers(parse_news(data.news), data.suppliers)
    weeks = weeks or range(1, data.last_week + 1)
    by_week = {w: g for w, g in events.groupby("week")}
    keys = list(zip(data.suppliers.material, data.suppliers.country))
    stories = {(k, t): _Story() for k in keys for t in TYPES}
    last_called_off = {k: -99 for k in keys}
    last_recovered = {k: -99 for k in keys}
    last_ongoing = {k: -99 for k in keys}
    recent: dict[tuple, list[int]] = {k: [] for k in keys}
    rows = []
    for w in range(1, weeks[-1] + 1):
        for e in (by_week[w].itertuples(index=False) if w in by_week else []):
            k = (e.material, e.country)
            if e.stage == "ongoing":
                last_ongoing[k] = w
                for t in TYPES:
                    if stories[(k, t)].stage == ONSET:
                        stories[(k, t)].last = w
            elif e.stage == "recovered":
                last_recovered[k] = w
                for t in TYPES:
                    stories[(k, t)] = _Story()
            elif e.stage == "called_off":
                last_called_off[k] = w
                for t in ([e.type] if e.type else LANE_TYPES):
                    stories[(k, t)] = _Story()
            else:
                s = stories[(k, e.type)]
                if s.stage == 0:
                    s.start = w
                s.stage = max(s.stage, STAGE_LEVEL[e.stage])
                s.last, s.n = w, s.n + 1
                if e.severity is not None:
                    s.severity = e.severity
                recent[k].append(w)
        if w not in weeks:
            continue
        for k in keys:
            row = {"week": w, "material": k[0], "country": k[1]}
            best_t, best = None, _Story()
            for t in TYPES:
                s = stories[(k, t)]
                if s.stage and w - s.last > STALE_WEEKS:
                    stories[(k, t)] = s = _Story()
                row[f"stage_{t}"] = s.stage
                if s.stage > best.stage or (s.stage == best.stage and s.stage and s.last > best.last):
                    best_t, best = t, s
            row.update(news_stage=best.stage, news_type=best_t,
                       weeks_since_signal=(w - best.last) if best.stage else np.nan,
                       story_age=(w - best.start) if best.stage else np.nan,
                       story_signals=best.n, severity=best.severity,
                       signals_8w=sum(1 for x in recent[k] if w - x < 8),
                       called_off_4w=int(w - last_called_off[k] < 4),
                       recovered_4w=int(w - last_recovered[k] < 4),
                       ongoing_2w=int(w - last_ongoing[k] < 2))
            rows.append(row)
    return pd.DataFrame(rows)
