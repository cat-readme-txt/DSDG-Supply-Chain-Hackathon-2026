"""State of the world at an information week: what the final submission has to plan for.

Prints, at origin o (default: the last released week):
  1. ongoing hits: supplier, shock type, age, capacity / lead / tariff effect, P(still hit) by horizon;
  2. open news stories: type, entity, stage, age, conversion, onset distribution, and the market
     check (futures basis on the story's supplier for shortage/ban; congestion and lane-country
     tension for strike/closure) against the thresholds in eda/28_market_signals.py;
  3. tariff announcements (certain, eda/26);
  4. futures ramps with no news story (basis above the calm 99.5th percentile);
  5. material-level P(disrupted) for the next 13 weeks from the current disruption model.

usage: uv run python eda/29_state_report.py [origin]
"""
import sys

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.labels import supplier_state
from dsdg_supply_chain.models.disruption import predict_disruption, supplier_forecast
from dsdg_supply_chain.models.duration import DurationModel, shock_class
from dsdg_supply_chain.models.scripts import ScriptModel, _targets

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
BASIS_CALM_P995 = 0.032    # eda/28: calm basis 99.5th percentile (sd 1.2%)
CONGESTION_REAL = 50       # eda/28: real lane shocks 52-57 the week before onset, rumors <= 46
TENSION_REAL = 47          # eda/28: real closures >= 54 the week before onset, rumors <= 38 (open 43)

full = load_data()
o = int(sys.argv[1]) if len(sys.argv) > 1 else full.last_week
d = full.until(o)
H = 13
print(f"=== state at week {o} (planning weeks {o + 1}-{o + H}) ===")

st = supplier_state(d).merge(d.prices, on=["week", "material", "country"])
st["basis"] = st.futures_4w / st.spot_price - 1
feats = build_features(d)
dm = DurationModel.fit(feats, sigma_min=0.6)

# 1. ongoing hits
now = feats[(feats.week == o) & (feats.hit == 1)].copy()
print("\n1. ongoing hits")
if len(now):
    now["type"] = shock_class(now)
    s0 = st[st.week == o].set_index(["material", "country"])
    rows = []
    for r in now.itertuples():
        x = s0.loc[(r.material, r.country)]
        surv = dm.survival(pd.Series([r.type] * H), np.full(H, float(r.weeks_hit)), np.arange(1, H + 1, dtype=float))
        rows.append(dict(material=r.material, country=r.country, type=r.type, age=int(r.weeks_hit),
                         cap=round(x.cap_ratio, 2), lead=int(x.lead_delta), tariff=round(x.tariff_delta, 3),
                         **{f"h{h}": round(surv[h - 1], 2) for h in (1, 2, 4, 6, 8, 13)}))
    print(pd.DataFrame(rows).to_string(index=False))
    print("   duration medians (weeks):", dm.median_weeks())
else:
    print("   none")

# 2. open stories with market check
sm = ScriptModel.fit(d)
cong = d.shipping.pivot(index="week", columns="lane", values="congestion_index")
ten = d.macro.pivot(index="week", columns="country", values="tension_index")
B = st.pivot_table(index="week", columns=["material", "country"], values="basis")
lane_countries = d.suppliers.groupby("lane").country.unique().to_dict()
print("\n2. open news stories")
if not len(sm.open_stories):
    print("   none")
for r in sm.open_stories.itertuples():
    stage_week = int(getattr(r, f"w{r.stage}"))
    pmf = sm.onset_pmf(r.type, r.stage, o - stage_week, H)
    conv = sm.conv.get((r.type, r.stage), np.nan)
    print(f"   {r.type:9s} {r.entity:20s} stage {r.stage} since wk {stage_week} (story from {int(r.start)}), "
          f"conv {conv:.2f}, P(onset within {H}w) {pmf.sum():.2f}, pmf {np.round(pmf[:6], 2).tolist()}")
    wk = list(range(max(int(r.start) - 1, o - 5), o + 1))
    if r.type in ("shortage", "ban"):
        for t in _targets(d.suppliers, r.type, r.entity).itertuples():
            path = [float(round(100 * B[(t.material, t.country)].get(w, np.nan), 1)) for w in wk]
            flag = "RAMP (real-like)" if max(path) / 100 > BASIS_CALM_P995 else "no ramp yet"
            print(f"      basis {t.material}/{t.country} weeks {wk[0]}-{o}: {path} -> {flag}")
    elif r.type in ("strike", "closure"):
        cs = [int(round(cong.loc[w, r.entity])) for w in wk]
        ts = [int(round(ten.loc[w, list(lane_countries[r.entity])].max())) for w in wk]
        flag = "real-like" if cs[-1] >= CONGESTION_REAL or (r.type == "closure" and ts[-1] >= TENSION_REAL) else "rumor-like so far"
        print(f"      congestion {r.entity} weeks {wk[0]}-{o}: {cs}; max lane-country tension {ts} -> {flag}")

# 3. tariff announcements
t = d.tariffs[(d.tariffs.week == o) & d.tariffs.announced_effective_week.notna()]
print("\n3. tariff announcements at the origin")
print(t[["material", "country", "tariff_rate", "announced_rate", "announced_effective_week"]].to_string(index=False)
      if len(t) else "   none")

# 4. futures ramps with no story
story_sup = set()
for r in sm.open_stories.itertuples():
    story_sup |= set(map(tuple, _targets(d.suppliers, r.type, r.entity)[["material", "country"]].to_numpy()))
recent = st[(st.week >= o - 1) & (st.basis > BASIS_CALM_P995)]
recent = recent[[(m, c) not in story_sup for m, c in zip(recent.material, recent.country)]]
print("\n4. futures ramps without a news story (basis > calm 99.5th pct in the last 2 weeks)")
print(recent[["week", "material", "country", "basis", "hit"]].round(3).to_string(index=False) if len(recent) else "   none")

# 5. material-level probabilities from the current model
weeks = range(o + 1, o + H + 1)
sup = supplier_forecast(d, weeks, feats=feats)
mp = predict_disruption(d, weeks, sup=sup)
print("\n5. P(disrupted) by material, current model (rows: material, columns: weeks ahead)")
piv = mp[mp.week > o].assign(h=lambda x: x.week - o).pivot(index="material", columns="h", values="p")
rate = d.outcomes.groupby("material").disrupted.mean()
print(piv.round(2).assign(hist_rate=rate.round(3)).to_string())
