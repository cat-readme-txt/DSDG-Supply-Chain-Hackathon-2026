"""Phase 2: do the stage gate (models/scripts.py STAGE_GATE) and the announced-tariff fixes
(models/disruption.py ANNOUNCED_TARIFF_OVERRIDES_STORY, TARIFF_OWN_DURATION, ANNOUNCED_FLOOR) sharpen the
disruption probabilities? Dense walk-forward test like eda/28: an origin every STEP weeks from 52 to
the last origin with a full 13-week window, Brier skill vs the historical-rate baseline, market check
on. Variants: old (all off), gate only, tariff only, both.
usage: uv run python eda/31_stage_gate.py [STEP]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from dsdg_supply_chain.data import load_data

VARIANTS = {"old": (False, False), "gate": (True, False), "tariff": (False, True), "both": (True, True)}


def one(o):
    from dsdg_supply_chain.models import disruption as D
    from dsdg_supply_chain.models import scripts as S
    data = load_data()
    info = data.until(o)
    win = range(o + 1, o + 14)
    y = data.outcomes[data.outcomes.week.isin(win)].set_index(["week", "material"]).disrupted
    rate = info.outcomes.groupby("material").disrupted.mean()
    out = pd.DataFrame({"y": y})
    out["naive"] = out.index.get_level_values(1).map(rate)
    for name, (gate, tar) in VARIANTS.items():
        S.STAGE_GATE = gate
        D.ANNOUNCED_TARIFF_OVERRIDES_STORY = D.TARIFF_OWN_DURATION = D.ANNOUNCED_FLOOR = tar
        p = D.predict_disruption(info, win, market=True).set_index(["week", "material"]).p
        out[name] = p.loc[y.index]
    return out.assign(origin=o).reset_index()


def skill(g):
    b = lambda c: ((g[c] - g.y) ** 2).mean()  # noqa: E731
    return pd.Series({"n": len(g), **{v: 1 - b(v) / b("naive") for v in VARIANTS}})


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    last = load_data().last_week - 13
    with ProcessPoolExecutor(8) as ex:
        R = pd.concat(list(ex.map(one, range(52, last + 1, step))))
    R.to_csv(f"reports/stage_gate_step{step}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print("Brier skill, all rows:", skill(R).round(4).to_dict())
    print("Brier skill, origins >= 196 (new data):", skill(R[R.origin >= 196]).round(4).to_dict())
    for v in ["gate", "tariff", "both"]:
        g = R.assign(gain=(R.old - R.y) ** 2 - (R[v] - R.y) ** 2).groupby("origin").gain.sum()
        print(f"\n{v}: Brier gain vs old by origin (summed over rows), nonzero:")
        print(g[g.abs() > 1e-3].round(3).to_string())
