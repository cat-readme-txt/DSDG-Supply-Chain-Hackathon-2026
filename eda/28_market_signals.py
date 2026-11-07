"""Fresh-eyes re-analysis (Round 6): which market data move ahead of shocks, and which don't.

1. Futures basis (futures_4w / spot - 1) around every supplier-level shock onset and around every
   news story (real vs rumor): does the market anticipate real shocks, and rumors?
2. Lane stories: congestion and lane-country tension by story stage, real vs rumor.
3. Landed cost decomposition: landed = spot x (1 + tariff) + freight; freight vs container rate.
4. Negative checks: basis in calm weeks vs future returns, demand during shocks.

usage: uv run python eda/28_market_signals.py        (writes eda/market_signals_onsets.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.labels import supplier_state
from dsdg_supply_chain.models.scripts import stories

pd.set_option("display.width", 200)
OUT = Path(__file__).resolve().parent
d = load_data()
st = supplier_state(d).merge(d.prices, on=["week", "material", "country"])
st["basis"] = st.futures_4w / st.spot_price - 1
B = st.pivot_table(index="week", columns=["material", "country"], values="basis")


def runs(df):
    rows = []
    for (m, c), g in df.sort_values("week").groupby(["material", "country"]):
        h, w = g.hit.to_numpy(), g.week.to_numpy()
        starts = [i for i in range(len(h)) if h[i] and (i == 0 or not h[i - 1])]
        for i in starts:
            j = i
            while j + 1 < len(h) and h[j + 1]:
                j += 1
            rows.append(dict(material=m, country=c, start=int(w[i]), end=int(w[j])))
    return pd.DataFrame(rows)


# ---- 1. futures basis around onsets ---------------------------------------------------------------
R = runs(st)
ev = R.drop_duplicates(["material", "start"]).copy()
for k in range(-6, 3):
    ev[f"k{k:+d}"] = [round(100 * B[(r.material, r.country)].get(r.start + k, np.nan), 1) for r in ev.itertuples()]
ev = ev.sort_values("start")
print("1a. hit-supplier basis (%) by weeks relative to onset (k=0 is the onset week)")
print(ev.to_string(index=False))
ev.to_csv(OUT / "market_signals_onsets.csv", index=False)

near = np.zeros(B.shape, bool)
for r in R.itertuples():
    j = B.columns.get_loc((r.material, r.country))
    near[(B.index >= r.start - 6) & (B.index <= r.end + 1), j] = True
calm = pd.Series(B.where(~near).to_numpy().ravel()).dropna()
print(f"\n1b. calm basis: sd {100 * calm.std():.2f}%, 99.5th pct {100 * calm.quantile(.995):.2f}%, "
      f"max {100 * calm.max():.2f}%, share > 4%: {(calm > 0.04).mean():.4f}")

S = stories(d)
print("\n1c. shortage/ban stories: max basis on the story's supplier from story start to onset (or end)")
for s in S[S.type.isin(["shortage", "ban"])].itertuples():
    c, m = s.entity.split("/")
    stop = int(s.onset) - 1 if pd.notna(s.onset) else int(s.last) + 1
    seg = B[(m, c)].loc[int(s.start):stop]
    kind = "REAL" if pd.notna(s.onset) else ("rumor" if pd.notna(s.called_off) else "open")
    print(f"   {kind:5s} {s.type:8s} {s.entity:20s} weeks {int(s.start)}-{stop}: "
          f"max {100 * seg.max():5.1f}%  path {[round(100 * v, 1) for v in seg]}")

# ---- 2. lane stories: congestion and tension by stage ---------------------------------------------
cong = d.shipping.pivot(index="week", columns="lane", values="congestion_index")
ten = d.macro.pivot(index="week", columns="country", values="tension_index")
lane_countries = d.suppliers.groupby("lane").country.unique().to_dict()
print("\n2. lane stories: congestion / max lane-country tension per week from story start to onset or end")
for s in S[S.type.isin(["strike", "closure"])].itertuples():
    stop = int(s.onset) if pd.notna(s.onset) else int(s.last)
    wk = range(int(s.start), min(stop, d.last_week) + 1)
    cs = [round(cong.loc[w, s.entity]) for w in wk]
    ts = [round(ten.loc[w, list(lane_countries[s.entity])].max()) for w in wk]
    kind = "REAL" if pd.notna(s.onset) else ("rumor" if pd.notna(s.called_off) else "open/stale")
    print(f"   {kind:10s} {s.type:7s} {s.entity:8s} weeks {wk[0]}-{wk[-1]}: congestion {cs}  tension {ts}")

# ---- 3. landed cost decomposition -----------------------------------------------------------------
x = (st.merge(d.shipping[["week", "lane", "container_rate"]], on=["week", "lane"]))
x["freight"] = x.landed_cost - x.spot_price * (1 + x.tariff_rate)
rows = []
for (m, c, lane), g in x.groupby(["material", "country", "lane"]):
    slope, const = np.polyfit(g.container_rate, g.freight, 1)
    resid = g.freight - (slope * g.container_rate + const)
    rows.append(dict(material=m, country=c, lane=lane, const=round(const, 4), slope_per_1000=round(1000 * slope, 4),
                     resid_sd=round(resid.std(), 5), freight_share=round((g.freight / g.landed_cost).mean(), 3)))
print("\n3. freight = landed - spot x (1 + tariff) vs container rate (by supplier)")
print(pd.DataFrame(rows).to_string(index=False))
print("   container rate by lane, quantiles:", d.shipping.groupby("lane").container_rate.quantile([.5, .99, 1]).unstack().round(0).to_dict("index"))

# ---- 4. negative checks ---------------------------------------------------------------------------
lsp = np.log(st.pivot_table(index="week", columns=["material", "country"], values="spot_price"))
r4 = (lsp.shift(-4) - lsp).where(~near)
both = pd.DataFrame({"b": B.where(~near).to_numpy().ravel(), "r": r4[B.columns].to_numpy().ravel()}).dropna()
print(f"\n4a. calm weeks: corr(basis, next-4-week log return) = {both.b.corr(both.r):+.3f} (n={len(both)})")
dem = d.demand.pivot(index="week", columns="material", values="demand_units")
ratio = (dem / dem.rolling(9, center=True, min_periods=3).median()).loc[: d.last_week]
dis = d.outcomes.pivot(index="week", columns="material", values="disrupted")
print(f"4b. demand / local median: disrupted weeks {np.nanmean(ratio[dis == 1].to_numpy()):.4f}, "
      f"calm weeks {np.nanmean(ratio[dis == 0].to_numpy()):.4f}  (no link)")
