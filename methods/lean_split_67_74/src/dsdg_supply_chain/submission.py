"""Submission format: building, validating, reading and writing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import COUNTRIES, MATERIALS, SHARE_COLS, Data

VALUE_COLS = ["disruption_prob", "price_forecast", "order_units", *SHARE_COLS]
FIRST_TEST_WEEK, LAST_TEST_WEEK = 209, 234


def make_id(week: int, material: str) -> str:
    return f"w{week}_{material}"


def empty_submission(weeks: range) -> pd.DataFrame:
    """Skeleton with one row per (week, material) and every value column zeroed."""
    rows = [(make_id(w, m), w, m) for w in weeks for m in MATERIALS]
    df = pd.DataFrame(rows, columns=["id", "week", "material"])
    for c in VALUE_COLS:
        df[c] = 0.0
    return df


def read_submission(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    parts = df.id.str.extract(r"^w(\d+)_(.+)$")
    df.insert(1, "week", parts[0].astype(int))
    df.insert(2, "material", parts[1])
    return df


def write_submission(df: pd.DataFrame, path: Path | str) -> None:
    out = df[["id", *VALUE_COLS]].copy()
    out["order_units"] = out.order_units.round().astype(int)
    out["disruption_prob"] = out.disruption_prob.round(4)
    out["price_forecast"] = out.price_forecast.round(4)
    # rounding can break the sum-to-1 rule: push the residual onto the largest share
    sh = out[SHARE_COLS].to_numpy().round(4)
    top = sh.argmax(axis=1)
    sh[np.arange(len(sh)), top] += 1.0 - sh.sum(axis=1)
    out[SHARE_COLS] = sh.round(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def validate(df: pd.DataFrame, data: Data, weeks: range | None = None) -> list[str]:
    """Return a list of problems (empty == valid), mirroring Kaggle's rejection rules."""
    weeks = weeks or range(FIRST_TEST_WEEK, LAST_TEST_WEEK + 1)
    errs: list[str] = []
    expected = {make_id(w, m) for w in weeks for m in MATERIALS}
    ids = df.id.tolist()
    if len(ids) != len(expected):
        errs.append(f"expected {len(expected)} rows, got {len(ids)}")
    if dup := sorted({i for i in ids if ids.count(i) > 1}):
        errs.append(f"duplicate ids: {dup[:5]}")
    if missing := sorted(expected - set(ids)):
        errs.append(f"missing ids: {missing[:5]}")
    if extra := sorted(set(ids) - expected):
        errs.append(f"unexpected ids: {extra[:5]}")
    if df[VALUE_COLS].isna().any().any():
        errs.append("blank values present")
    if not df.disruption_prob.between(0, 1).all():
        errs.append("disruption_prob outside [0, 1]")
    if not (df.price_forecast > 0).all():
        errs.append("price_forecast must be > 0")
    if not (df.order_units >= 0).all():
        errs.append("order_units must be >= 0")
    sh = df[SHARE_COLS].to_numpy()
    if ((sh < 0) | (sh > 1)).any():
        errs.append("shares outside [0, 1]")
    if not np.allclose(sh.sum(axis=1), 1.0, atol=1e-6):
        errs.append("shares do not sum to 1")
    sells = set(zip(data.suppliers.material, data.suppliers.country))
    for c in COUNTRIES:
        bad = df[(df[f"share_{c}"] > 0) & ~df.material.map(lambda m, c=c: (m, c) in sells)]
        if len(bad):
            errs.append(f"share_{c} > 0 for non-supplier, first row {bad.id.iloc[0]}")
    return errs
