"""Market features per supplier: country tension and currency, lane congestion, tariff announcements,
and the futures premium.

Levels are expressed against each series' own trailing history (z-scores over the previous
BASELINE weeks, excluding the current one) so the features mean the same thing for every country
and lane, and use no future data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Data

KEY = ["week", "material", "country"]
BASELINE = 26


def _trailing_z(df: pd.DataFrame, by: str, col: str) -> pd.Series:
    g = df.groupby(by, sort=False)[col]
    mu = g.transform(lambda s: s.shift(1).rolling(BASELINE, min_periods=8).median())
    sd = g.transform(lambda s: s.shift(1).rolling(BASELINE, min_periods=8).std())
    return ((df[col] - mu) / sd.where(sd > 1e-9)).fillna(0.0)


def market_features(data: Data) -> pd.DataFrame:
    sup = data.suppliers[["material", "country", "lane"]]

    mac = data.macro.sort_values(["country", "week"]).copy()
    mac["tension_z"] = _trailing_z(mac, "country", "tension_index")
    mac["tension_chg2"] = mac.groupby("country").tension_index.diff(2).fillna(0.0)
    mac["fx_chg4"] = mac.groupby("country").fx_rate.pct_change(4).fillna(0.0)
    mac = mac[["week", "country", "tension_index", "tension_z", "tension_chg2", "fx_chg4"]]

    shp = data.shipping.sort_values(["lane", "week"]).copy()
    shp["congestion_z"] = _trailing_z(shp, "lane", "congestion_index")
    shp["transit_z"] = _trailing_z(shp, "lane", "transit_days")
    shp["freight_z"] = _trailing_z(shp, "lane", "container_rate")
    shp = shp[["week", "lane", "congestion_index", "congestion_z", "transit_z", "freight_z"]]

    tar = data.tariffs.copy()
    # blank announced_* means "no announcement" — turn it into explicit, NaN-free columns
    tar["tariff_announced"] = tar.announced_rate.notna().astype(int)
    tar["announced_increase"] = (tar.announced_rate - tar.tariff_rate).fillna(0.0)
    tar["weeks_to_tariff"] = (tar.announced_effective_week - tar.week).fillna(99).clip(upper=99)
    tar = tar[KEY + ["tariff_announced", "announced_increase", "weeks_to_tariff"]]

    pr = data.prices.sort_values(KEY).copy()
    pr["futures_premium"] = pr.futures_4w / pr.spot_price - 1
    pr["spot_chg4"] = pr.groupby(["material", "country"]).spot_price.pct_change(4).fillna(0.0)
    pr = pr[KEY + ["futures_premium", "spot_chg4"]]

    weeks = pd.DataFrame({"week": sorted(data.outcomes.week.unique())})
    out = (weeks.merge(sup, how="cross")
           .merge(mac, on=["week", "country"], how="left")
           .merge(shp, on=["week", "lane"], how="left")
           .merge(tar, on=KEY, how="left")
           .merge(pr, on=KEY, how="left"))
    return out.drop(columns="lane").sort_values(KEY).reset_index(drop=True)
