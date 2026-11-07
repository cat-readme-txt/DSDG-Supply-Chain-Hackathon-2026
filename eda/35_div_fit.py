"""Phase 2: which diversification formula does Kaggle use? P, F and fill come from the local scorer
(which matches every slack file); only the diversification definition varies. Known Kaggle scores
are in eda/34_metric_fit.KAGGLE.
usage: uv run python eda/35_div_fit.py
"""
import importlib.util

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.scoring import score_window
from dsdg_supply_chain.submission import read_submission

spec = importlib.util.spec_from_file_location("m34", "eda/34_metric_fit.py")
m34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m34)
W = range(209, 222)


def hhi_div(sh):
    return np.clip((1 - (sh ** 2).sum(-1)) / (2 / 3), 0, 1)


def variants(s, cost):
    sh = s[SHARE_COLS].to_numpy(float)
    d = hhi_div(sh)
    u = s.order_units.to_numpy(float)
    val = u * s.material.map(cost).to_numpy()
    pos = u > 0
    out = {"w_units": (d * u).sum() / u.sum(), "row_mean": d.mean(),
           "row_mean_ordering": d[pos].mean() if pos.any() else 0.0,
           "w_value": (d * val).sum() / val.sum()}
    per_w, per_agg, per_row = [], [], []
    for m, g in s.groupby("material"):
        gu = g.order_units.to_numpy(float); gs = g[SHARE_COLS].to_numpy(float); gd = hhi_div(gs)
        per_w.append((gd * gu).sum() / gu.sum() if gu.sum() else 0.0)
        tot = (gs * gu[:, None]).sum(0)
        per_agg.append(float(hhi_div(tot / tot.sum())) if tot.sum() else 0.0)
        per_row.append(gd[gu > 0].mean() if (gu > 0).any() else 0.0)
    out |= {"mat_mean_w_units": np.mean(per_w), "mat_mean_window_shares": np.mean(per_agg),
            "mat_mean_row_ordering": np.mean(per_row)}
    return out


d = load_data()
naive = read_submission("data/comp/sample_submission.csv")
cost = d.outcomes[d.outcomes.week.isin(W)].groupby("material").benchmark_price.mean()
rows = []
for f, k in m34.KAGGLE.items():
    sub = read_submission(f)
    ws = score_window(sub, naive, d, W)
    s = sub[sub.week.isin(W)]
    for name, dv in variants(s, cost).items():
        R = np.clip(0.7 * ws.fill_skill + 0.3 * dv, 0, 1)
        rows.append(dict(file=f.split("/")[-1], div=name, value=dv,
                         score=100 * (0.3 * ws.P + 0.4 * ws.F + 0.3 * R), kaggle=k))
R = pd.DataFrame(rows)
R["err"] = R.score - R.kaggle
pd.set_option("display.width", 250)
print(R.groupby("div").err.agg(rmse=lambda e: np.sqrt((e ** 2).mean()), maxabs=lambda e: e.abs().max())
      .sort_values("rmse").round(3).to_string())
print(R.pivot(index="file", columns="div", values="err").round(2).to_string())
