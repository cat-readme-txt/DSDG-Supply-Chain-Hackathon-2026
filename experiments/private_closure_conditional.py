"""Build the selected final pair conditioned on the open MERROW closure.

The SK4 lean and insured policies already favor different closure outcomes. This
experiment changes only weeks 222-234 disruption probabilities: the lean member
forecasts the no-closure branch, and the insured member the closure branch. The
event-specific conditional values are calculated from the week-221 snapshot;
public rows, prices, orders, and shares remain exactly those in the SK4 files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SK4 = ROOT / "methods" / "sk4_phase2"
sys.path.insert(0, str(SK4 / "src"))
os.environ.setdefault("DSDG_DATA_DIR", str(ROOT / "data" / "Phase 2 data"))

from dsdg_supply_chain.data import load_data  # noqa: E402
from dsdg_supply_chain.models import scripts  # noqa: E402
from dsdg_supply_chain.strategies import _forecasts  # noqa: E402

FILES = {
    "lean": "lean_pxlp_mkt_div_w221_pub.csv",
    "insured": "insured_hedged_mktcol_w221_pub.csv",
}


def closure_branches(info) -> tuple[pd.Series, pd.Series]:
    weeks = range(info.last_week + 1, info.last_week + 14)
    _, _, base, _ = _forecasts(info, weeks, market=True)
    old = dict(scripts.CONV_OVERRIDE)
    try:
        scripts.CONV_OVERRIDE["closure"] = 0.0
        _, _, absent, _ = _forecasts(info, weeks, market=True)
        scripts.CONV_OVERRIDE["closure"] = 1.0
        _, _, certain, _ = _forecasts(info, weeks, market=True)
    finally:
        scripts.CONV_OVERRIDE.clear()
        scripts.CONV_OVERRIDE.update(old)
    key = ["week", "material"]
    p = base.set_index(key).p
    p_yes = certain.set_index(key).p
    p_zero = absent.set_index(key).p
    # Keep the GBM and other-risk floor in the no-closure branch. Some false
    # closure warnings preceded unrelated disruptions on the same materials.
    return (p - p_zero).clip(lower=0.0), (p_yes - p).clip(lower=0.0)


def build(alpha: float, output: Path) -> None:
    info = load_data().until(221)
    down, up = closure_branches(info)
    output.mkdir(parents=True, exist_ok=True)
    for side, name in FILES.items():
        source = SK4 / "submissions" / name
        frame = pd.read_csv(source)
        key = pd.MultiIndex.from_tuples(
            [(int(w), m) for w, m in frame.id.str.extract(r"w(\d+)_(.+)").itertuples(index=False, name=None)],
            names=["week", "material"],
        )
        private = key.get_level_values("week") > 221
        event_delta = (down if side == "lean" else up).reindex(key[private]).to_numpy(float)
        assert np.isfinite(event_delta).all()
        old = frame.loc[private, "disruption_prob"].to_numpy(float)
        move = (-1.0 if side == "lean" else 1.0) * event_delta
        frame.loc[private, "disruption_prob"] = np.clip(old + alpha * move, 0.0, 1.0)
        suffix = "" if alpha == 1.0 else f"_a{alpha:g}"
        dest = output / f"private_closure_conditional_{side}{suffix}_w221.csv"
        frame.to_csv(dest, index=False)
        changed = np.count_nonzero(abs(frame.loc[private, "disruption_prob"].to_numpy(float) - old) > 1e-9)
        print(f"{dest}: {changed} private probabilities changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "submissions" / "candidates")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be in [0, 1]")
    build(args.alpha, args.output)
