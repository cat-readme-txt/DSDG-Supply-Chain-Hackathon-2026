"""Supplier state features: is the supplier hit right now, how, for how long, and how widespread.

Everything at week w uses supply_history / tariffs up to and including week w only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Data
from ..labels import supplier_state

KEY = ["week", "material", "country"]


def _run_length(flag: pd.Series) -> pd.Series:
    """Consecutive weeks the flag has been true, ending at each row (0 when false)."""
    grp = (~flag).cumsum()
    return flag.astype(int).groupby(grp).cumsum()


def supply_features(data: Data) -> pd.DataFrame:
    st = supplier_state(data)
    g = st.groupby(["material", "country"], sort=False)
    st["weeks_hit"] = g.hit.transform(_run_length)
    # weeks since the supplier was last hit (capped; 52 = not in the past year)
    last_hit = st.week.where(st.hit).groupby([st.material, st.country]).ffill()
    st["weeks_since_hit"] = (st.week - last_hit).where(~st.hit).clip(upper=52).fillna(0)
    st.loc[st.hit, "weeks_since_hit"] = 0
    st.loc[last_hit.isna(), "weeks_since_hit"] = 52
    # how widespread: share of the same country's / lane's supplier rows hit this week
    st["country_hit_share"] = st.groupby(["week", "country"]).hit.transform("mean")
    st["lane_hit_share"] = st.groupby(["week", "lane"]).hit.transform("mean")
    st["hit_type"] = np.select([st.hit_cap & st.hit_lead, st.hit_cap, st.hit_lead, st.hit_tariff],
                               ["cap+lead", "cap", "lead", "tariff"], "none")
    cols = ["hit", "hit_cap", "hit_lead", "hit_tariff", "hit_type", "cap_ratio", "lead_delta",
            "tariff_delta", "weeks_hit", "weeks_since_hit", "country_hit_share", "lane_hit_share"]
    out = st[KEY + cols].copy()
    for c in ["hit", "hit_cap", "hit_lead", "hit_tariff"]:
        out[c] = out[c].astype(int)
    return out
