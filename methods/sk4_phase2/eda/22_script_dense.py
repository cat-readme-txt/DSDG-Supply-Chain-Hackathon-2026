"""Dense check of the story-script model: an origin every 2 weeks (52..195), 13-week horizon,
Brier skill vs the historical-rate baseline, GBM alone vs max(GBM, script). Rows where a live story
is open at the origin are reported separately (that is where the script model acts).
usage: uv run python eda/22_script_dense.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data


def one(o):
    from dsdg_supply_chain.models import disruption as D
    from dsdg_supply_chain.strategies import story_materials
    data = load_data()
    info = data.until(o)
    win = range(o + 1, o + 14)
    y = data.outcomes[data.outcomes.week.isin(win)].set_index(["week", "material"]).disrupted
    rate = info.outcomes.groupby("material").disrupted.mean()
    out = pd.DataFrame({"y": y})
    out["naive"] = out.index.get_level_values(1).map(rate)
    for flag in [False, True]:
        D.USE_SCRIPTS = flag
        out["script" if flag else "gbm"] = D.predict_disruption(info, win).set_index(["week", "material"]).p.loc[y.index]
    out["story"] = out.index.get_level_values(1).isin(story_materials(info))
    return out.assign(origin=o).reset_index()


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        R = pd.concat(ex.map(one, range(52, 196, 2)))
    R.to_csv("reports/script_dense.csv", index=False)
    def skill(g):
        b = lambda c: ((g[c] - g.y) ** 2).mean()  # noqa: E731
        return pd.Series({"n": len(g), "gbm": 1 - b("gbm") / b("naive"), "script": 1 - b("script") / b("naive"),
                          "bs_gbm": b("gbm"), "bs_script": b("script")})
    print("all rows:\n", skill(R).round(4).to_dict())
    print(R.groupby("story").apply(skill).round(4))
