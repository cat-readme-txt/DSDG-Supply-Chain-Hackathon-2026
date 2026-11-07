"""Round 6: does the market check of open stories (models/scripts.py MarketCheck) sharpen the
disruption probabilities? Dense test like eda/22: an origin every STEP weeks (52..195), 13-week
horizon, Brier skill vs the historical-rate baseline, story-script model without / with the market
check, on all rows and on rows with a live story. Also lists the market verdict per live story and
origin, whether the unannounced-ramp guard ever fires, and the readout at the last released week.
usage: uv run python eda/28_market_dense.py [STEP]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from dsdg_supply_chain.data import load_data


def one(o):
    from dsdg_supply_chain.models import disruption as D
    from dsdg_supply_chain.models.scripts import ScriptModel
    from dsdg_supply_chain.strategies import story_materials
    data = load_data()
    info = data.until(o)
    win = range(o + 1, o + 14)
    y = data.outcomes[data.outcomes.week.isin(win)].set_index(["week", "material"]).disrupted
    rate = info.outcomes.groupby("material").disrupted.mean()
    out = pd.DataFrame({"y": y})
    out["naive"] = out.index.get_level_values(1).map(rate)
    for flag in [False, True]:
        p = D.predict_disruption(info, win, market=flag).set_index(["week", "material"]).p
        out["market" if flag else "script"] = p.loc[y.index]
    out["story"] = out.index.get_level_values(1).isin(story_materials(info))
    ex = ScriptModel.fit(info, market=True).explain(info).assign(origin=o)
    return out.assign(origin=o).reset_index(), ex


def skill(g):
    b = lambda c: ((g[c] - g.y) ** 2).mean()  # noqa: E731
    return pd.Series({"n": len(g), "script": 1 - b("script") / b("naive"), "market": 1 - b("market") / b("naive"),
                      "bs_script": b("script"), "bs_market": b("market")})


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ProcessPoolExecutor(8) as ex:
        parts = list(ex.map(one, range(52, 196, step)))
    R = pd.concat([p[0] for p in parts])
    E = pd.concat([p[1] for p in parts])
    R.to_csv(f"reports/market_dense_step{step}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 70)
    print("all rows:", skill(R).round(4).to_dict())
    print(R.groupby("story").apply(skill).round(4))
    d = R.assign(gain=(R.script - R.y) ** 2 - (R.market - R.y) ** 2)
    print("\nBrier gain (script - market, summed over rows) by origin, nonzero only:")
    print(d.groupby("origin").gain.sum().loc[lambda s: s.abs() > 1e-4].round(3).to_string())
    print("\nmarket verdicts on live stories:")
    print(E[E.verdict != "n/a"][["origin", "type", "entity", "stage", "verdict", "conv", "conv_market"]]
          .to_string(index=False))
    print("\nunannounced-ramp guard fired at:", sorted(E[E.verdict == "unannounced ramp"].origin.unique()))
