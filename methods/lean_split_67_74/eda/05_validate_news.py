"""Validate the news parser and features against supplier-level ground truth.

Checks
  A. coverage      every non-routine headline is classified
  B. resolution    every signal event maps to at least one supplier
  C. onsets        onset articles line up with real hit starts (precision / recall)
  D. lead time     how early the features flag real shocks
  E. relevance     P(hit soon | news features) vs the base rate: are the features predictive?
  F. severity      extracted severity numbers vs actual capacity / tariff change
  G. false alarms  warning stories that never became a shock, and whether they were called off
  H. open stories  what the features say at the last released week
"""
import re

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.news import (classify, expand_to_suppliers, news_features,
                                             parse_news)
from dsdg_supply_chain.labels import supplier_state

pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 70)
ROUTINE = re.compile("|".join([
    r"holds rates steady", r"Weather stays mild", r"trade surplus widens", r"tracking technology",
    r"Mining stocks mixed", r"container terminal opens", r"unemployment rate dips",
    r"infrastructure spending", r"little changed in quiet trading", r"hosts regional trade fair",
    r"manufacturers report steady orders", r"criticizes budget", r"cooperation agreement",
    r"expect stable .* demand", r"Recycling of", r"factory activity edges", r"Freight rates steady"]))

data = load_data()
news = data.news
st = supplier_state(data)
key = ["week", "material", "country"]

print("== A. coverage")
unclassified = news[~news.headline.str.contains(ROUTINE) & news.headline.map(lambda h: classify(h) is None)]
print(f"non-routine headlines without a rule: {len(unclassified)}")
print(unclassified.headline.value_counts().to_string() if len(unclassified) else "  none")
routine_hit = news[news.headline.str.contains(ROUTINE) & news.headline.map(lambda h: classify(h) is not None)]
print(f"routine headlines wrongly classified as signal: {len(routine_hit)}")

ev = parse_news(news)
exp = expand_to_suppliers(ev, data.suppliers)
print(f"\n== B. resolution: {len(ev)} signal articles -> {len(exp)} supplier events")
orphans = ev[~ev.article_id.isin(exp.article_id)]
print(f"articles mapped to no supplier: {len(orphans)}")
if len(orphans):
    print(orphans[["week", "type", "stage", "country", "lane", "material", "headline"]].to_string(index=False))

print("\n== C. onset articles vs real hit starts")
st["start"] = st.hit & ~st.groupby(["material", "country"]).hit.shift(1, fill_value=False)
starts = st[st.start][key + ["lane", "cap_ratio", "lead_delta", "tariff_delta"]]
ons = exp[exp.stage == "onset"]
hit_set = set(map(tuple, st[st.hit][key].to_numpy()))
ons_ok = ons.apply(lambda r: any((r.week + d, r.material, r.country) in hit_set for d in (0, 1)), axis=1)
print(f"onset supplier-events that are really hit (same or next week): {ons_ok.sum()}/{len(ons)}")
if (~ons_ok).any():
    print(ons[~ons_ok][["week", "material", "country", "type", "headline"]].to_string(index=False))
ons_keys = {(r.week + d, r.material, r.country) for r in ons.itertuples() for d in (-1, 0, 1)}
starts["onset_news"] = [tuple(x) in ons_keys for x in starts[key].to_numpy()]
print(f"real hit starts with an onset article within ±1 week: {starts.onset_news.sum()}/{len(starts)}")
print(starts[~starts.onset_news].to_string(index=False))

feat = news_features(data)
f = feat.merge(st[key + ["hit", "start"]], on=key)
f = f.sort_values(key)
g = f.groupby(["material", "country"])
for h in (1, 2, 4):
    f[f"hit_in_{h}"] = g.hit.transform(lambda s, h=h: s[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1))
f["start_in_4"] = g.start.transform(lambda s: s[::-1].rolling(4, min_periods=1).max()[::-1].shift(-1))

print("\n== D. lead time: news_stage / type the week(s) before each real hit start")
rows = []
for r in starts.itertuples():
    row = {"start": r.week, "material": r.material, "country": r.country}
    for lag in (1, 2, 4):
        x = f[(f.week == r.week - lag) & (f.material == r.material) & (f.country == r.country)]
        row[f"stage_t-{lag}"] = int(x.news_stage.iloc[0]) if len(x) else None
        if lag == 1:
            row["type_t-1"] = x.news_type.iloc[0] if len(x) else None
    rows.append(row)
lead = pd.DataFrame(rows)
print(lead.to_string(index=False))
print(f"\nhit starts flagged (stage>=1) 1 week ahead: {(lead['stage_t-1'] >= 1).mean():.0%}, "
      f"2 weeks ahead: {(lead['stage_t-2'] >= 1).mean():.0%}, 4 weeks ahead: {(lead['stage_t-4'] >= 1).mean():.0%}")

print("\n== E. relevance: P(new hit within 4 weeks | features), suppliers not currently hit")
nh = f[(f.hit == 0) & f.start_in_4.notna()]
base = nh.start_in_4.mean()
print(f"base rate: {base:.4f}  (n={len(nh)})")
t = nh.groupby("news_stage").start_in_4.agg(["mean", "sum", "count"])
t["lift"] = t["mean"] / base
print(t.round(3).to_string())
print("\nby called_off_4w (stage >= 1 only):")
print(nh[nh.news_stage >= 1].groupby("called_off_4w").start_in_4.agg(["mean", "count"]).round(3).to_string())
print("\nby type (stage >= 1):")
print(nh[nh.news_stage >= 1].groupby("news_type").start_in_4.agg(["mean", "sum", "count"]).round(3).to_string())
print("\nP(still hit next week | currently hit), by ongoing_2w / recovered_4w:")
cur = f[(f.hit == 1) & f.hit_in_1.notna()]
print(cur.groupby(["ongoing_2w", "recovered_4w"]).hit_in_1.agg(["mean", "count"]).round(3).to_string())

print("\n== F. severity: extracted number vs actual effect at the hit")
sev = exp.dropna(subset=["severity"]).drop_duplicates(["article_id", "material", "country"])
for r in sev.itertuples():
    x = st[(st.material == r.material) & (st.country == r.country) & st.week.between(r.week, r.week + 6) & st.hit]
    act = (f"tariff +{x.tariff_delta.max():.3f}" if r.type == "tariff"
           else f"cap ratio {x.cap_ratio.min():.2f} (1-sev={1 - r.severity:.2f})") if len(x) else "no hit"
    print(f"  w{r.week} {r.type:9s} {r.material}/{r.country} severity={r.severity:.2f} -> {act}")

print("\n== G. warning stories that never became a hit (rumours)")
stories = []
for (m, c), s in f.groupby(["material", "country"]):
    s = s.set_index("week")
    open_ = (s.news_stage.between(1, 3)).astype(int)
    ends = open_.diff().fillna(0)
    starts_w = list(s.index[ends == 1]) + ([s.index[0]] if open_.iloc[0] else [])
    for a in starts_w:
        seg = s.loc[a:]
        b = seg.index[seg.news_stage.between(1, 3) == 0].min() if (seg.news_stage.between(1, 3) == 0).any() else s.index[-1]
        became = s.loc[a:b + 4].hit.any()
        stories.append(dict(material=m, country=c, start=a, end=b, type=s.loc[a, "news_type"],
                            max_stage=int(s.loc[a:b].news_stage.max()), became_hit=bool(became),
                            called_off=bool(s.loc[a:b + 1].called_off_4w.any())))
sto = pd.DataFrame(stories)
print(pd.crosstab([sto.type, sto.called_off], sto.became_hit, margins=True))

print(f"\n== H. open stories at week {data.last_week}")
last = f[(f.week == data.last_week) & ((f.news_stage > 0) | (f.called_off_4w == 1))]
print(last[key + ["news_stage", "news_type", "weeks_since_signal", "story_signals", "called_off_4w"]]
      .to_string(index=False))
