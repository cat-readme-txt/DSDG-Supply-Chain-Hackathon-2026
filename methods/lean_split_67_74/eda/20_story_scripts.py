"""Story-level statistics of the news scripts: how often each story type becomes a real shock,
how many weeks from each stage to onset, and how long the shock then lasts.

A story = consecutive signal articles of one shock type about one entity (lane for strikes and
closures, country for sanctions, country+material otherwise) with gaps of at most 8 weeks.
Real = an onset article appears; also checked against supply_history (the label).
usage: uv run python eda/20_story_scripts.py
"""
import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.news import LANE_TYPES, COUNTRY_TYPES, STAGE_LEVEL, parse_news
from dsdg_supply_chain.labels import supplier_state

data = load_data()
ev = parse_news(data.news)
ev = ev.astype(object).where(ev.notna(), None)
lane_of = data.suppliers.groupby("country").lane.first().to_dict()


def entity(e):
    if e.type in LANE_TYPES:
        return e.lane or lane_of.get(e.country)
    if e.type in COUNTRY_TYPES:
        return e.country
    return f"{e.country}/{e.material}"


typed = ev[ev.type.notna() & ev.stage.isin(list(STAGE_LEVEL) + ["called_off"])].copy()
typed["entity"] = [entity(e) for e in typed.itertuples()]
# 'Deal reached ... averted' has no type; attach to the strike story on its lane
generic_off = ev[ev.type.isna() & (ev.stage == "called_off")].copy()
generic_off["type"] = "strike"
generic_off["entity"] = [e.lane or lane_of.get(e.country) for e in generic_off.itertuples()]
typed = pd.concat([typed, generic_off]).sort_values("week")

stories = []
for (t, ent), g in typed.groupby(["type", "entity"]):
    cur = None
    for e in g.itertuples():
        if cur is None or e.week - cur["last"] > 8 or cur["end"] is not None:
            if cur:
                stories.append(cur)
            cur = dict(type=t, entity=ent, start=e.week, last=e.week, onset=None, end=None, stages=[],
                       first_stage_week={})
        cur["last"] = e.week
        cur["stages"].append(e.stage)
        cur["first_stage_week"].setdefault(e.stage, e.week)
        if e.stage == "onset" and cur["onset"] is None:
            cur["onset"] = e.week
        if e.stage == "called_off":
            cur["end"] = "called_off"
    stories.append(cur)
S = pd.DataFrame(stories)
S["real"] = S.onset.notna()
fs = lambda st: S.first_stage_week.apply(lambda d: d.get(st))  # noqa: E731
for st in ["early", "escalating", "imminent"]:
    S[f"lead_from_{st}"] = S.onset - fs(st)
S["max_stage"] = S.stages.apply(lambda s: max((STAGE_LEVEL.get(x, 0) for x in s), default=0))

# duration of the real shock from supply_history: weeks the targeted suppliers stay hit after onset
st = supplier_state(data)
def duration(r):
    if not r.real:
        return np.nan
    if r.type in LANE_TYPES:
        sup = data.suppliers[data.suppliers.lane == r.entity]
    elif r.type in COUNTRY_TYPES:
        sup = data.suppliers[data.suppliers.country == r.entity]
    else:
        c, m = r.entity.split("/")
        sup = data.suppliers[(data.suppliers.country == c) & (data.suppliers.material == m)]
    h = st.merge(sup[["material", "country"]]).groupby("week").hit.max()
    w, n = int(r.onset), 0
    while w in h.index and h.loc[w]:
        n, w = n + 1, w + 1
    return n if w in h.index else f">={n} (censored)"
S["duration"] = S.apply(duration, axis=1)
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
cols = ["type", "entity", "start", "onset", "end", "real", "max_stage", "lead_from_early",
        "lead_from_escalating", "lead_from_imminent", "duration"]
print(S.sort_values("start")[cols].to_string(index=False))
print("\nconversion by type and highest stage reached (excluding stories still open at the end):")
done = S[(S.last < data.last_week - 8) | S.real | S.end.notna()]
print(done.groupby(["type", "max_stage"]).real.agg(["sum", "count"]).to_string())
S.to_csv("eda/stories.csv", index=False)
