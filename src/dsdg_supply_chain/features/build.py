"""Combine news, supplier-state and market features into one supplier-week table, with checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Data
from .market import market_features
from .news import news_features
from .supply import supply_features

KEY = ["week", "material", "country"]
# columns where NaN is meaningful ("no open story") and gets an explicit fill value
NEWS_FILL = {"weeks_since_signal": 99, "story_age": 0, "severity": 0.0, "news_type": "none"}


def build_features(data: Data) -> pd.DataFrame:
    news = news_features(data).fillna(NEWS_FILL)
    sup = supply_features(data)
    mkt = market_features(data)
    df = sup.merge(news, on=KEY, how="left", validate="1:1").merge(mkt, on=KEY, how="left", validate="1:1")
    check_features(df, data)
    return df


def check_features(df: pd.DataFrame, data: Data) -> None:
    """Fail loudly on the problems that silently corrupt a model."""
    n_expected = data.last_week * len(data.suppliers)
    if len(df) != n_expected:
        raise ValueError(f"feature table has {len(df)} rows, expected {n_expected}")
    if df.duplicated(KEY).any():
        raise ValueError("duplicate (week, material, country) rows")
    if bad := [c for c in df.columns if df[c].isna().any()]:
        raise ValueError(f"NaN in feature columns: {bad}")
    num = df.select_dtypes("number")
    if bad := [c for c in num.columns if not np.isfinite(num[c]).all()]:
        raise ValueError(f"non-finite values in: {bad}")
    if df.week.max() > data.last_week:
        raise ValueError("features extend past the last released week")
