"""Evidence for models/cost_path.py: landed = spot x (1 + tariff) + freight; freight multipliers on
lane shocks; spot steps at onset / end of source shocks; the fitted model at the last week.
usage: uv run python eda/28_cost_path.py
"""
import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.models.cost_path import _freight, CostPathModel
from dsdg_supply_chain.models.duration import shock_runs

d = load_data()
fr = _freight(d)
base = fr.groupby(["material", "country"]).freight.transform("median")
mult = (fr.freight / base).round(2)
lane = fr.merge(d.suppliers[["material", "country", "lane"]]).lane
print("freight / base freight by lane (counts):")
print(pd.crosstab(lane, mult).to_string())

feats = build_features(d)
lsp = (d.prices.assign(l=np.log(d.prices.spot_price))
       .pivot_table(index="week", columns="material", values="l", aggfunc="mean").diff())
runs = shock_runs(feats).drop_duplicates(["material", "start", "type"])
rows = [(r.type, r.material, r.start, r.length, lsp.at[r.start, r.material],
         lsp.at[r.start + r.length, r.material] if r.ended else np.nan) for r in runs.itertuples()]
ev = pd.DataFrame(rows, columns=["type", "material", "start", "length", "jump", "drop"])
ev["step"] = 0.5 * (ev.jump - ev["drop"])
print("\nlog spot change (mean over suppliers) at onset and the week after the end:")
print(ev.sort_values(["type", "start"]).round(3).to_string(index=False))
cm = CostPathModel.fit(d, feats)
print("\nfitted steps:", {k: round(v, 3) for k, v in cm.steps.items()},
      "freight multipliers:", {k: round(v, 2) for k, v in cm.freight_mult.items()})
