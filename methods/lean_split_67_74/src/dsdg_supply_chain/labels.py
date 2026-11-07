"""Supplier-level ground truth: which (material, country) is hit by an active shock each week.

EDA showed that `outcomes.disrupted` equals, with no exceptions, the union over a material's
suppliers of: capacity < 0.85 x normal, lead time > base, or tariff > base.
"""
from __future__ import annotations

import pandas as pd

from .data import Data

CAP_HIT = 0.85


def supplier_state(data: Data) -> pd.DataFrame:
    """One row per (week, material, country) with the hit flag and its components."""
    key = ["week", "material", "country"]
    df = (data.supply_history.merge(data.tariffs[key + ["tariff_rate"]], on=key)
          .merge(data.suppliers[["material", "country", "lane", "base_lead_weeks",
                                 "normal_capacity_units", "base_tariff"]], on=["material", "country"]))
    df["cap_ratio"] = df.capacity_units / df.normal_capacity_units
    df["lead_delta"] = df.lead_time_weeks - df.base_lead_weeks
    df["tariff_delta"] = (df.tariff_rate - df.base_tariff).round(6)
    df["hit_cap"] = df.cap_ratio < CAP_HIT
    df["hit_lead"] = df.lead_delta > 0
    df["hit_tariff"] = df.tariff_delta > 1e-6
    df["hit"] = df.hit_cap | df.hit_lead | df.hit_tariff
    return df.sort_values(key).reset_index(drop=True)
