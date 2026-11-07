"""Generates notebooks/lean_hedged.ipynb (run: uv run python notebooks/make_lean_hedged.py).

The notebook builds the lean_hedged submission stage by stage with the same functions as
`uv run dsdg submit lean_hedged`, so the CSV it writes is identical to the CLI's.
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# lean_hedged: end-to-end submission

Builds the `lean_hedged` submission (Kaggle public **60.66**) stage by stage:

1. **Load and validate** the competition files
2. **Features**: news-story parser, supplier state, market z-scores
3. **Disruption model**: GBM onset classifier + story-script model + duration model
4. **Price model**: shock-free level, mean reversion, shock uplift
5. **Ordering**: stochastic-programming LP (30 scenarios, exact simulator rules)
6. **Hedge**: calm-bet forecast columns (this file is the calm half of the final pair)
7. **Validate and write** the CSV

Paths come from `DSDG_DATA_DIR` (default `data/comp`), so this also runs on Databricks with the data in a Volume.
Every choice below is logged in `DECISIONS.md`."""),
    code("""import os, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))           # use the repo's package even without `uv sync`
os.environ.setdefault("DSDG_DATA_DIR", str(ROOT / "data" / "comp"))

import numpy as np
import pandas as pd
pd.set_option("display.width", 200, "display.max_columns", 20)

from dsdg_supply_chain.data import load_data, SHARE_COLS
from dsdg_supply_chain.submission import FIRST_TEST_WEEK, LAST_TEST_WEEK, validate, write_submission, read_submission
print("data dir:", os.environ["DSDG_DATA_DIR"])"""),
    md("## 1. Load and validate the data"),
    code("""data = load_data()
WEEKS = range(FIRST_TEST_WEEK, LAST_TEST_WEEK + 1)      # 209..234, always the submission window
ORIGIN = data.last_week                                 # information week: 208 in Phase 1, 221 in Phase 2
for name in ["demand", "news", "outcomes", "prices", "supply_history", "tariffs", "suppliers"]:
    df = getattr(data, name)
    print(f"{name:15s} rows {len(df):6d}  nulls {int(df.isna().sum().sum()):5d}  dup rows {int(df.duplicated().sum())}")
print("information week:", ORIGIN)"""),
    md("""## 2. Features
One row per (week, material, supplier): news-story stage per shock type, current hit state, contagion
(share of the lane / country already hit) and market z-scores."""),
    code("""from dsdg_supply_chain.features.build import build_features
feats = build_features(data)
print(feats.shape)
feats[feats.week == ORIGIN].query("news_stage > 0 or hit == 1")[["material", "country", "hit", "news_type", "news_stage", "story_age"]]"""),
    md("""## 3. Disruption model
* **Onset**: GBM classifier for suppliers not hit now (news + contagion features).
* **Story scripts** (`models/scripts.py`): for every live news story,
  P(hit) = conversion(type, stage) × lead distribution × duration, learned only from stories resolved before the origin.
  Combined with the GBM as max(GBM, script).
* **Duration**: censoring-aware log-normal per shock type for suppliers hit now.
* Material probability = 1 − Π(1 − p_supplier), blended toward the historical rate."""),
    code("""from dsdg_supply_chain.models.scripts import ScriptModel
sm = ScriptModel.fit(data)
print("conversion rates (type, stage):", {k: round(v, 2) for k, v in sm.conv.items() if k[1] == 1})
sm.open_stories[["type", "entity", "start", "last", "stage"]]"""),
    code("""from dsdg_supply_chain.models.disruption import supplier_forecast, predict_disruption
sup = supplier_forecast(data, WEEKS, feats=feats)
mat_p = predict_disruption(data, WEEKS, sup=sup)
mat_p[mat_p.week > ORIGIN].pivot(index="material", columns="week", values="p").round(2).iloc[:, :13]"""),
    md("## 4. Price model"),
    code("""from dsdg_supply_chain.models.price import predict_price
px = predict_price(data, WEEKS, sup, feats=feats)
px[px.week > ORIGIN].pivot(index="material", columns="week", values="price").round(2).iloc[:, [0, 4, 8, 12]]"""),
    md("""## 5. Ordering: stochastic-programming LP
Per material and 13-week window: 30 scenarios drawn from the supplier forecasts, one LP that minimises
the average simulated cost under the competition's rules (fill up to capacity, lead times, 5-week starting
stock, 2% holding, 5× unmet penalty, 70% salvage). Lean settings: no minimum share, no background shocks,
plan at 96% of normal capacity."""),
    code("""from dsdg_supply_chain.strategies import naive, _apply_forecasts
from dsdg_supply_chain.policy_lp import LPParams, plan_orders_lp

sub = _apply_forecasts(naive(data, WEEKS), mat_p, px)            # skeleton + forecast columns
starts = [s for s in range(WEEKS[0], WEEKS[-1] + 1, 13) if s > ORIGIN]
plan_weeks = range(starts[0], WEEKS[-1] + 1)                     # only windows fully in the future
LEAN = LPParams(n_scenarios=30, min_share=0.0, hazard_mult=0.0, cap_derate=0.96)
plan = plan_orders_lp(data, plan_weeks, feats, sup[sup.week > ORIGIN], LEAN).set_index(["week", "material"])

cols = ["order_units", *SHARE_COLS]
sub = sub.set_index(["week", "material"])
sub.loc[plan.index, cols] = plan[cols]
sub = sub.reset_index()[["id", "week", "material", *[c for c in sub.columns if c not in ("id",)]]]
print("plan covers weeks", plan_weeks[0], "-", plan_weeks[-1])
sub[sub.material == "lithium"].set_index("week")[["order_units", "share_KES", "share_VAL", "share_ORI"]].round(2).T"""),
    md("""## 6. Hedge (calm bet)
Kaggle keeps the better of the two final files. This is the calm half: background and ongoing-shock
probabilities are scaled by 0.5 beyond week 2. Live warned stories are left alone."""),
    code("""from dsdg_supply_chain.strategies import hedge_forecasts, story_materials, CALM_A, CALM_K
print("live-story materials (not hedged):", sorted(story_materials(data)), "| CALM_A", CALM_A, "CALM_K", CALM_K)
sub = hedge_forecasts(sub, data, "calm")"""),
    md("## 7. Validate and write"),
    code("""errs = validate(sub, data, WEEKS)
assert not errs, errs
out = ROOT / "submissions" / f"lean_hedged_w{ORIGIN}_nb.csv"
write_submission(sub, out)
print("wrote", out)

ref = ROOT / "submissions" / f"lean_hedged_w{ORIGIN}.csv"   # file built by `dsdg submit lean_hedged`
if ref.exists():
    a, b = pd.read_csv(out), pd.read_csv(ref)
    num = [c for c in a if c != "id"]
    print("identical to", ref.name, ":", a.id.equals(b.id) and np.allclose(a[num], b[num]))"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/lean_hedged.ipynb")
print("wrote notebooks/lean_hedged.ipynb")
