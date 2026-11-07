"""Loading the competition files and producing leakage-free snapshots of them."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pandas as pd

# Override with DSDG_DATA_DIR (e.g. a Databricks Volume: /Volumes/<catalog>/<schema>/<volume>/comp)
DATA_DIR = Path(os.environ.get("DSDG_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "comp"))

COUNTRIES = ["KES", "VAL", "SUN", "ORI", "HAL", "AUR"]
SHARE_COLS = [f"share_{c}" for c in COUNTRIES]
MATERIALS = ["lithium", "cobalt", "copper", "aluminum", "steel", "rare_earths",
             "semiconductors", "polymer_resin", "specialty_glass", "natural_gas"]
WINDOW = 13  # weeks per leaderboard window

# Tables indexed by week that must be cut off at the information week.
_TIMED = ["macro", "news", "outcomes", "prices", "shipping", "supply_history", "tariffs"]


@dataclass(frozen=True)
class Data:
    demand: pd.DataFrame          # all weeks (the production plan is known in advance)
    macro: pd.DataFrame
    news: pd.DataFrame
    outcomes: pd.DataFrame
    prices: pd.DataFrame
    shipping: pd.DataFrame
    suppliers: pd.DataFrame
    supply_history: pd.DataFrame
    tariffs: pd.DataFrame

    @property
    def last_week(self) -> int:
        return int(self.outcomes.week.max())

    def until(self, week: int) -> "Data":
        """Snapshot as if only weeks <= `week` had been released (demand stays complete)."""
        cut = {name: getattr(self, name).loc[lambda df: df.week <= week].reset_index(drop=True)
               for name in _TIMED}
        return replace(self, **cut)

    def suppliers_of(self, material: str) -> list[str]:
        """Supplier countries for a material, primary (largest normal capacity) first."""
        s = self.suppliers[self.suppliers.material == material]
        return s.sort_values("normal_capacity_units", ascending=False).country.tolist()


def load_data(path: Path | str = DATA_DIR) -> Data:
    path = Path(path)
    return Data(**{f.name: pd.read_csv(path / f"{f.name}.csv") for f in fields(Data)})
