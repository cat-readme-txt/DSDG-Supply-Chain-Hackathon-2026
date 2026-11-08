"""Non-routine headlines in chronological order, annotated with shocks starting soon / active."""
import re

import pandas as pd

from common import load

ROUTINE = [
    r"holds rates steady", r"Weather stays mild", r"trade surplus widens", r"tracking technology",
    r"Mining stocks mixed", r"container terminal opens", r"unemployment rate dips",
    r"infrastructure spending", r"little changed in quiet trading", r"hosts regional trade fair",
    r"manufacturers report steady orders", r"criticizes budget", r"cooperation agreement",
    r"expect stable .* demand", r"Recycling of", r"factory activity edges", r"Freight rates steady",
]
rx = re.compile("|".join(ROUTINE))

n = load("news")
sig = n[~n.headline.str.contains(rx)].copy()
print(f"{len(sig)} non-routine of {len(n)} articles")

runs = pd.read_csv("eda/country_runs.csv")
# collapse runs into events: same start week & lane/country
ev = (runs.groupby(["a", "b", "c", "lane"])
          .agg(mats=("mat", lambda s: ",".join(sorted(s))), cap=("cap", "min"), lead=("lead", "max"),
               tar=("tar", "max")).reset_index().sort_values("a"))


def ctx(w):
    tags = []
    for r in ev.itertuples():
        if r.a <= w <= r.b:
            tags.append(f"ACTIVE {r.c}/{r.lane}[{r.a}-{r.b}]")
        elif 0 < r.a - w <= 12:
            tags.append(f"in{r.a - w}w {r.c}/{r.lane}[{r.a}-{r.b}]")
        elif 0 < w - r.b <= 6:
            tags.append(f"ended{w - r.b}w {r.c}/{r.lane}")
    return " | ".join(tags)


sig["ctx"] = sig.week.map(ctx)
pd.set_option("display.width", 300, "display.max_colwidth", 110, "display.max_rows", 1000)
print(sig[["week", "source", "headline", "ctx"]].to_string(index=False))
sig.to_csv("eda/signal_news.csv", index=False)
