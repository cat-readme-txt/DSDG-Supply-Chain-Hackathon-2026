"""Disruption episodes: when, how long, and which supplier/lane shows the damage."""
import pandas as pd

from common import all_data, episodes

d = all_data()
out, sh, sup = d["outcomes"], d["supply_history"], d["suppliers"]

print("== base disruption rate per material")
print(out.groupby("material")["disrupted"].agg(["mean", "sum"]).round(3))

# Normalise supply history against base values
sh = sh.merge(sup[["material", "country", "lane", "base_lead_weeks", "normal_capacity_units"]],
              on=["material", "country"])
sh["cap_ratio"] = sh["capacity_units"] / sh["normal_capacity_units"]
sh["lead_delta"] = sh["lead_time_weeks"] - sh["base_lead_weeks"]

print("\n== episodes per material, with per-country min capacity ratio / max lead delta inside episode")
rows = []
for m, g in out.groupby("material"):
    s = g.set_index("week")["disrupted"]
    for a, b in episodes(s):
        seg = sh[(sh.material == m) & sh.week.between(a, b)]
        stats = seg.groupby("country").agg(cap=("cap_ratio", "min"), lead=("lead_delta", "max"),
                                           lane=("lane", "first"))
        hit = stats[(stats.cap < 0.7) | (stats.lead > 0)]
        rows.append(dict(material=m, start=a, end=b, length=b - a + 1,
                         hit=";".join(f"{c}({r.lane}) cap{r.cap:.2f} lead+{r.lead}"
                                      for c, r in hit.iterrows()) or "-"))
ep = pd.DataFrame(rows).sort_values("start")
pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 500)
print(ep.to_string(index=False))
print("\nlength distribution:\n", ep.length.describe())
ep.to_csv("eda/episodes.csv", index=False)
