"""Does the story-script model (models/scripts.py) improve the Brier skill out of sample?
Walk-forward over the backtest jobs (info weeks o and o+13), GBM alone vs max(GBM, script).
usage: uv run python eda/21_script_backtest.py
"""
import numpy as np
import pandas as pd

from dsdg_supply_chain.backtest import origins
from dsdg_supply_chain.data import WINDOW, load_data
from dsdg_supply_chain.models import disruption as D
from dsdg_supply_chain.strategies import naive

data = load_data()
rows = []
for o in origins(data):
    weeks = range(o + 1, o + 2 * WINDOW + 1)
    base = naive(data.until(o), weeks).set_index(["week", "material"]).disruption_prob
    for name, iw in [("public", o), ("private", o + WINDOW)]:
        win = range(iw + 1, iw + WINDOW + 1)
        info = data.until(iw)
        y = data.outcomes[data.outcomes.week.isin(win)].set_index(["week", "material"]).disrupted
        r = dict(origin=o, window=name)
        for flag in [False, True]:
            D.USE_SCRIPTS = flag
            p = D.predict_disruption(info, win).set_index(["week", "material"]).p.loc[y.index]
            bs, bsn = ((p - y) ** 2).mean(), ((base.loc[y.index] - y) ** 2).mean()
            r["script" if flag else "gbm"] = 1 - bs / bsn
        rows.append(r)
        print(r, flush=True)
R = pd.DataFrame(rows)
R["gain"] = R.script - R.gbm
print(R.round(3).to_string())
print(R.groupby("window")[["gbm", "script", "gain"]].mean().round(3))
print("mean", R[["gbm", "script", "gain"]].mean().round(3).to_dict())
R.to_csv("reports/script_backtest.csv", index=False)
