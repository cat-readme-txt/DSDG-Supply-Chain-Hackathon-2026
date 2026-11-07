"""Command line: `uv run dsdg {backtest,validate,submit} ...`."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, summarize
from .data import load_data
from .strategies import STRATEGIES
from .submission import FIRST_TEST_WEEK, LAST_TEST_WEEK, read_submission, validate, write_submission

REPORTS = Path(os.environ.get("DSDG_REPORTS_DIR", Path(__file__).resolve().parents[2] / "reports"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dsdg")
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("backtest", help="walk-forward backtest of a strategy on released weeks")
    b.add_argument("strategy", choices=STRATEGIES)
    v = sp.add_parser("validate", help="check a submission CSV against Kaggle's rules")
    v.add_argument("path")
    s = sp.add_parser("submit", help="build the 209-234 submission CSV from all released data")
    s.add_argument("strategy", choices=STRATEGIES)
    s.add_argument("-o", "--out", default=None)
    args = p.parse_args(argv)
    data = load_data()

    if args.cmd == "backtest":
        res = run_backtest(STRATEGIES[args.strategy], data)
        REPORTS.mkdir(exist_ok=True)
        out = REPORTS / f"backtest_{args.strategy}.csv"
        res.to_csv(out, index=False)
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(res[["origin", "window", "start", "end", "n_shock_weeks", "score", "P", "F", "R"]]
                  .to_string(index=False))
            print("\nmean by window:\n", summarize(res))
        print(f"\nwrote {out}")
    elif args.cmd == "validate":
        errs = validate(read_submission(args.path), data)
        print("valid" if not errs else "\n".join(errs))
    elif args.cmd == "submit":
        weeks = range(FIRST_TEST_WEEK, LAST_TEST_WEEK + 1)
        sub = STRATEGIES[args.strategy](data, weeks)
        if errs := validate(sub, data, weeks):
            raise SystemExit("invalid submission:\n" + "\n".join(errs))
        out = args.out or f"submissions/{args.strategy}_w{data.last_week}.csv"
        write_submission(sub, out)
        print(f"wrote {out}")
