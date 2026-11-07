"""Rank every strategy (alone and as a best-of-two pair) under a chosen reading of the metric, from
the saved step-4 backtests (reports/backtest_*_step4.csv, which store per-material ingredients).
Use it right after eda/17 has shown which reading matches Kaggle.

usage: uv run python eda/27_pick_pair.py [F_agg fill_agg calm]     e.g.  mean mean one
       (defaults: pooled pooled zero = our scorer)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dsdg_supply_chain.backtest import rescore

F_agg, fill_agg, calm = (sys.argv[1:4] + ["pooled", "pooled", "zero"][len(sys.argv[1:4]):])
S = {}
for f in sorted(Path("reports").glob("backtest_*_step4.csv")):
    d = pd.read_csv(f)
    if "per_material" in d:
        S[f.stem.removeprefix("backtest_").removesuffix("_step4")] = rescore(d, F_agg, fill_agg, calm)
S = pd.DataFrame(S)
print(f"reading: F {F_agg}, fill {fill_agg}, calm-window fill {calm}; {len(S)} windows")
print(S.agg(["mean", "median", "min"]).T.sort_values("mean", ascending=False).round(2).to_string())
pairs = sorted(((np.maximum(S[a], S[b]).mean(), np.maximum(S[a], S[b]).min(), a, b)
                for a, b in itertools.combinations(S, 2)), reverse=True)
print("\nbest pairs (mean of max, worst window):")
for m, w, a, b in pairs[:8]:
    print(f"  {m:6.2f}  {w:6.1f}  {a} + {b}")
