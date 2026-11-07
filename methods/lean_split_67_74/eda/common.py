"""Shared loaders for EDA."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "comp"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def all_data() -> dict[str, pd.DataFrame]:
    names = ["demand", "macro", "news", "outcomes", "prices", "sample_submission",
             "shipping", "suppliers", "supply_history", "tariffs"]
    return {n: load(n) for n in names}


def episodes(series: pd.Series) -> list[tuple[int, int]]:
    """Return (start_week, end_week) inclusive runs where series==1. Index must be week."""
    runs, start, prev = [], None, None
    for w, v in series.items():
        if v == 1 and start is None:
            start = w
        if v == 0 and start is not None:
            runs.append((start, prev))
            start = None
        prev = w
    if start is not None:
        runs.append((start, prev))
    return runs
