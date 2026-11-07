"""Per material-country anomaly signals and whether their union reproduces `disrupted`."""
import pandas as pd

from common import all_data, episodes

d = all_data()
out, sh, sup, tar, pr = d["outcomes"], d["supply_history"], d["suppliers"], d["tariffs"], d["prices"]
key = ["week", "material", "country"]
df = (sh.merge(tar, on=key + ["date"]).merge(pr, on=key + ["date"])
        .merge(sup, on=["material", "country"]))
df["cap_ratio"] = df.capacity_units / df.normal_capacity_units
df["lead_delta"] = df.lead_time_weeks - df.base_lead_weeks
df["tar_delta"] = (df.tariff_rate - df.base_tariff).round(4)
# spot price relative to rolling median of same material-country
df = df.sort_values(key)
df["px_rel"] = df.spot_price / df.groupby(["material", "country"]).spot_price.transform(
    lambda s: s.rolling(26, min_periods=4).median().shift(1))

print("== capacity ratio quantiles", df.cap_ratio.quantile([.001, .01, .05, .5, .95, .99]).round(3).to_dict())
print("== lead_delta counts", df.lead_delta.value_counts().sort_index().to_dict())
print("== tariff delta counts", df.tar_delta.value_counts().sort_index().to_dict())

# Candidate per-row shock flag
df["flag_cap"] = df.cap_ratio < 0.85
df["flag_lead"] = df.lead_delta > 0
df["flag_tar"] = df.tar_delta > 0.001
df["any"] = df.flag_cap | df.flag_lead | df.flag_tar
agg = df.groupby(["week", "material"])[["flag_cap", "flag_lead", "flag_tar", "any"]].any().reset_index()
m = out.merge(agg, on=["week", "material"])
print("\n== crosstab disrupted vs union(cap<0.85 | lead>base | tariff>base)")
print(pd.crosstab(m.disrupted, m["any"]))
for f in ["flag_cap", "flag_lead", "flag_tar"]:
    print(f, pd.crosstab(m.disrupted, m[f]).values.tolist())

# list per-country shock runs
print("\n== per material-country runs of `any`")
rows = []
for (mat, c), g in df.groupby(["material", "country"]):
    s = g.set_index("week")["any"].astype(int)
    for a, b in episodes(s):
        seg = g[g.week.between(a, b)]
        rows.append(dict(mat=mat, c=c, lane=seg.lane.iloc[0], a=a, b=b, n=b - a + 1,
                         cap=round(seg.cap_ratio.min(), 2), lead=seg.lead_delta.max(),
                         tar=seg.tar_delta.max(), px=round(seg.px_rel.max(), 2)))
runs = pd.DataFrame(rows).sort_values(["a", "lane", "c"])
pd.set_option("display.width", 250, "display.max_rows", 500)
print(runs.to_string(index=False))
runs.to_csv("eda/country_runs.csv", index=False)

# mismatches
print("\n== disrupted=1 but no flag:")
print(m[(m.disrupted == 1) & ~m["any"]][["week", "material"]].to_string(index=False))
print("== flag but disrupted=0 (first 40):")
print(m[(m.disrupted == 0) & m["any"]][["week", "material", "flag_cap", "flag_lead", "flag_tar"]].head(40).to_string(index=False))
