"""Do tariff announcements in tariffs.csv come true? (they feed models/disruption.py)
usage: uv run python eda/26_tariff_announcements.py
"""
import pandas as pd

from dsdg_supply_chain.data import load_data

d = load_data()
t = d.tariffs.merge(d.suppliers[["material", "country", "base_tariff"]])
a = t[t.announced_rate.notna()]
g = (a.groupby(["material", "country", "announced_effective_week"])
     .agg(first_seen=("week", "min"), rate=("announced_rate", "first")).reset_index())
for r in g.itertuples():
    x = t[(t.material == r.material) & (t.country == r.country)].set_index("week")
    w = int(r.announced_effective_week)
    if w > d.last_week:
        print(f"{r.material}/{r.country}: effective {w} (not released yet)")
        continue
    hit = x.tariff_rate > x.base_tariff + 1e-6
    start = hit[(hit.index >= r.first_seen) & hit].index.min()
    end = hit[(hit.index >= w) & ~hit].index.min()
    print(f"{r.material}/{r.country}: announced wk {r.first_seen} for wk {w} at {r.rate:.3f} -> "
          f"took effect wk {start} at {x.tariff_rate.loc[w]:.3f}, lasted {end - w if pd.notna(end) else '>='} weeks")
