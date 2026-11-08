"""Evaluate the closure forecast hedge on past Phase-2-style windows.

The naive comparator is frozen 13 weeks before each information cutoff, while
candidate policies only see observations through that cutoff.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "sk4_phase2" / "src"))
os.environ.setdefault("DSDG_DATA_DIR", str(ROOT / "data" / "Phase 2 data"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.scoring import score_window  # noqa: E402
from dsdg_supply_chain.strategies import STRATEGIES, naive  # noqa: E402
from private_closure_conditional import closure_branches  # noqa: E402


def main() -> None:
    data = load_data()
    rows = []
    # Every historical information cutoff with a live closure story and a
    # complete 13-week outcome window in the Phase 2 release.
    for origin in (147, 148, 149, 150, 171, 172, 203, 204, 205, 206):
        info = data.until(origin)
        weeks = range(origin + 1, origin + 14)
        base = naive(data.until(origin - 13), range(origin - 12, origin + 14))
        down, up = closure_branches(info)
        pair = {}
        for side, strategy in (("lean", "lean_pxlp_mkt_div"),
                               ("insured", "insured_hedged_mktcol")):
            sub = STRATEGIES[strategy](info, weeks)
            old = score_window(sub, base, data, weeks).score
            key = pd.MultiIndex.from_frame(sub[["week", "material"]])
            delta = (down if side == "lean" else up).reindex(key).to_numpy(float)
            assert np.isfinite(delta).all()
            mod = sub.copy()
            mod["disruption_prob"] = np.clip(sub.disruption_prob.to_numpy(float) +
                                              (-delta if side == "lean" else delta), 0, 1)
            new = score_window(mod, base, data, weeks).score
            pair[side] = (old, new)
            rows.append(dict(origin=origin, side=side, baseline=old, conditional=new, delta=new-old))
        rows.append(dict(origin=origin, side="best_of_two",
                         baseline=max(x[0] for x in pair.values()),
                         conditional=max(x[1] for x in pair.values()),
                         delta=max(x[1] for x in pair.values()) - max(x[0] for x in pair.values())))
    result = pd.DataFrame(rows)
    dest = ROOT / "reports" / "private_closure_pair_backtest.csv"
    result.to_csv(dest, index=False)
    print(result.round(3).to_string(index=False))
    print("Saved", dest)


if __name__ == "__main__":
    main()
