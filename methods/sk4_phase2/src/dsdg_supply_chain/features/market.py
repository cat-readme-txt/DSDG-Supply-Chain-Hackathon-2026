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


# --- market signals for confirming news stories (Round 6, see models/scripts.py MarketCheck) ------
# Market series that react to REAL upcoming supply events but not to rumors (eda/28_market_signals.py):
#   * the futures basis (futures_4w / spot - 1) of the supplier a shortage or export ban will hit
#     ramps up from the week of the story's first article and collapses at onset (calm noise sd ~1.2%);
#   * lane congestion reaches >= 52 the week before a real strike or closure (rumors peak at 40-46);
#   * tension in the lane's countries jumps to >= 47 the week before a real closure (rumors <= 43).
# Lane shocks, tariffs, sanctions and reversals show no basis ramp; nothing signals a shock's end.
NOISE_MIN_WEEKS = 8   # weeks of history before the basis noise level is trusted


def robust_sd(x) -> float:
    """1.4826 x median absolute deviation: the sd of the bulk of the values, ignoring rare ramps."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _expanding_robust_sd(df: pd.DataFrame, col: str) -> pd.Series:
    """Per week: robust sd of `col` pooled over all rows up to and including that week."""
    vals = {w: g[col].to_numpy(float) for w, g in df.groupby("week")}
    acc, out = [], {}
    for w in sorted(vals):
        acc.append(vals[w])
        out[w] = robust_sd(np.concatenate(acc)) if len(acc) >= NOISE_MIN_WEEKS else np.nan
    return df.week.map(out)


def market_signals(data: Data) -> pd.DataFrame:
    """Per (week, material, country): the market's read of an upcoming supply event.

    basis              futures_4w / spot - 1 of this supplier
    basis_max2         max of this week's and last week's basis
    basis_chg          week-on-week change of the basis
    basis_z            basis / robust sd of every supplier's basis up to this week (calm noise ~1.2%)
    congestion         lane congestion_index; congestion_z against its trailing 26-week baseline
    lane_tension       max tension_index over the countries on this supplier's lane, and
    lane_tension_jump  the same minus its trailing 26-week median (a jump, not a long crisis level)
    n_imminent         imminent-stage news articles about this supplier published this week
    """
    from .news import expand_to_suppliers, parse_news
    sup = data.suppliers[["material", "country", "lane"]]

    pr = data.prices.sort_values(KEY).copy()
    pr["basis"] = pr.futures_4w / pr.spot_price - 1
    prev = pr.groupby(["material", "country"]).basis.shift(1)
    pr["basis_max2"] = np.fmax(pr.basis, prev)
    pr["basis_chg"] = (pr.basis - prev).fillna(0.0)
    pr["basis_z"] = (pr.basis / _expanding_robust_sd(pr, "basis")).fillna(0.0)
    pr = pr[KEY + ["basis", "basis_max2", "basis_chg", "basis_z"]]

    shp = data.shipping.sort_values(["lane", "week"]).copy()
    shp["congestion_z"] = _trailing_z(shp, "lane", "congestion_index")
    shp = shp.rename(columns={"congestion_index": "congestion"})[["week", "lane", "congestion", "congestion_z"]]

    lanes = sup[["country", "lane"]].drop_duplicates()
    ten = (data.macro.merge(lanes, on="country").groupby(["lane", "week"]).tension_index.max()
           .rename("lane_tension").reset_index().sort_values(["lane", "week"]))
    base = ten.groupby("lane").lane_tension.transform(
        lambda s: s.shift(1).rolling(BASELINE, min_periods=8).median())
    ten["lane_tension_jump"] = (ten.lane_tension - base).fillna(0.0)

    ev = parse_news(data.news)
    imm = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                        [("week", int), ("material", object), ("country", object), ("n_imminent", int)]})
    if len(ev) and (ev.stage == "imminent").any():
        imm = (expand_to_suppliers(ev[ev.stage == "imminent"], data.suppliers)
               .groupby(KEY).size().rename("n_imminent").reset_index().astype({"week": int}))

    weeks = pd.DataFrame({"week": sorted(data.outcomes.week.unique())})
    out = (weeks.merge(sup, how="cross")
           .merge(pr, on=KEY, how="left")
           .merge(shp, on=["week", "lane"], how="left")
           .merge(ten, on=["week", "lane"], how="left")
           .merge(imm, on=KEY, how="left"))
    out["n_imminent"] = out.n_imminent.fillna(0).astype(int)
    return out.sort_values(KEY).reset_index(drop=True)
