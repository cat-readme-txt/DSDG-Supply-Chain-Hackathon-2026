"""Fill the released public weeks (209-221) of a week-221 submission with the hindsight plan.

The week-221 files plan only the private window (222-234); their 209-221 rows carry the observed
labels and prices but naive orders, so Kaggle's public board shows 30.00 (P = 1, F = R = 0). The
private board scores 222-234 on its own (each window is simulated separately), so those rows are
untouched here. Public rows get the hindsight LP plan on the released truth (the plan the F
denominator uses), split by its units; weeks with no order get an even split.
usage: uv run python eda/33_fill_public.py in.csv out.csv
"""
import sys

import numpy as np

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.simulator import hindsight_plan, truth_for
from dsdg_supply_chain.submission import read_submission, validate

d = load_data()
WIN = range(209, 222)
sub = read_submission(sys.argv[1])
before = sub[sub.week >= 222].copy()
for mat in sub.material.unique():
    tr = truth_for(d, mat, WIN)
    _, x = hindsight_plan(tr)
    for j, w in enumerate(WIN):
        i = sub.index[(sub.week == w) & (sub.material == mat)][0]
        tot = float(x[:, j].sum())
        sh = np.round(x[:, j] / tot, 6) if tot > 1e-6 else np.full(len(tr.countries), 1 / 3)
        sh = np.round(sh, 4); sh[0] = round(1 - sh[1:].sum(), 4)
        sub.loc[i, SHARE_COLS] = 0.0
        for s, c in enumerate(tr.countries):
            sub.at[i, f"share_{c}"] = float(sh[s])
        sub.at[i, "order_units"] = round(tot, 3) if tot > 1e-6 else 0.0
assert sub[sub.week >= 222].equals(before), "private rows changed"
errs = validate(sub, d)
assert not errs, errs[:3]
cols = ["id", "disruption_prob", "price_forecast", "order_units", *SHARE_COLS]
sub[cols].to_csv(sys.argv[2], index=False)
print("wrote", sys.argv[2])
