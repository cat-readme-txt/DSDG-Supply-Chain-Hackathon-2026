"""Score submission CSVs on every released test week with the local scorer (run after Phase 2).

The naive baseline is the sample submission (built from week 208), as on Kaggle. Only the weeks
already released can be scored: after Phase 2 that is the public window 209-221.

usage: uv run python eda/24_score_files.py file1.csv [file2.csv ...]
       (no arguments: every CSV in submissions/ and submissions/teammate/)
"""
import sys
from pathlib import Path

import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.scoring import score_window
from dsdg_supply_chain.submission import read_submission, validate

data = load_data()
win = range(209, min(221, data.last_week) + 1)
if data.last_week < 209:
    sys.exit(f"Only weeks up to {data.last_week} are released, so there is nothing to score yet. "
             "Download the Phase 2 data first (eda/phase2.sh or the HANDOFF steps).")
if len(win) < 13:
    print(f"warning: only weeks {win[0]}-{win[-1]} released; scoring a partial window")

base = read_submission(Path(__file__).resolve().parents[1] / "data" / "comp" / "sample_submission.csv")
files = sys.argv[1:] or sorted(map(str, Path("submissions").glob("*.csv"))) + sorted(
    map(str, Path("submissions/teammate").glob("*.csv")))
rows = []
for f in files:
    sub = read_submission(f)
    errs = validate(sub, data)
    if errs:
        print(f"{f}: INVALID ({errs[0]})")
        continue
    ws = score_window(sub, base, data, win)
    rows.append(dict(file=f, **{k: v for k, v in ws.as_dict().items()
                                if k in ("score", "P", "F", "R", "brier_skill", "price_skill",
                                         "fill_skill", "diversification")}))
pd.set_option("display.width", 220, "display.max_colwidth", 60)
print(f"local score on weeks {win[0]}-{win[-1]}:")
print(pd.DataFrame(rows).sort_values("score", ascending=False).round(3).to_string(index=False))
