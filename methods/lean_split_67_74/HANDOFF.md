# Handoff: DSDG Hackathon 2026 "Supply Chain Disruptor"

Read this first, then `DECISIONS.md` (every modelling choice and its evidence) and `eda/FINDINGS.md` (what the data
is). Work continues on branch `SK3` (branched from `SK2` at commit 1158114; all pushed to origin). The data in
`data/comp/` is gitignored: re-download it with the Setup steps below.

## Competition in one paragraph
Kaggle `dsdg-hackathon-2026-supply-chain-disruptor`. 10 materials × 3 supplier countries (6 countries, 5 lanes),
synthetic weekly data. Submit 260 rows (weeks 209–234 × 10 materials): `disruption_prob`, `price_forecast`,
`order_units`, `share_KES … share_AUR`. Score = 100 × (0.30 P + 0.40 F + 0.30 R), each part clipped to [0, 1], and
the sample submission scores 0.
- **P:** 0.6 Brier skill + 0.4 MAPE skill.
- **F:** cost vs the naive and hindsight plans, from a supply-chain simulator.
- **R:** 0.7 fill-rate skill in shock weeks + 0.3 diversification.

Public leaderboard = weeks 209–221; **private = 222–234 (final)**. Phase 2 data (weeks 1–221) lands **Sat Nov 7, 12 PM CT**.
The final deadline is **Sun Nov 8, 12 PM CT**: select 2 submissions and upload a 3-minute video. 3 submissions per day,
resetting at 00:00 UTC (6 PM CT).

## Setup
```
uv sync                                   # Python ≥ 3.11; deps in pyproject.toml
set -a; . ./.env; set +a                  # KAGGLE_API_TOKEN (gitignored); the CLI does not read .env itself
uv run kaggle competitions download -c dsdg-hackathon-2026-supply-chain-disruptor -p data/raw
unzip -o data/raw/*.zip -d data/comp      # the code reads data/comp (override with DSDG_DATA_DIR)
uv run pytest -q                          # 27 tests, all passing
```

## Pipeline (`src/dsdg_supply_chain/`)
| Stage | Module | What it does |
|---|---|---|
| data | `data.py` | Loads the 10 CSVs. `Data.until(week)` gives a leakage-free snapshot (demand stays complete) |
| labels | `labels.py` | Supplier hit = capacity < 0.85 × normal, OR lead > base, OR tariff > base. Reproduces `disrupted` exactly |
| features | `features/news.py` | Rule-table news parser: headline → (shock type, stage, country/lane/material, severity) → per-supplier weekly story features. 238/238 articles mapped |
| | `features/supply.py`, `market.py`, `build.py` | Supplier state, market z-scores, and a combined NaN-checked table (6,240 × 45) |
| step 1 | `models/disruption.py`, `models/duration.py` | GBM onset classifier (news + contagion features) for suppliers not hit now, plus a censoring-aware log-normal duration model per shock type for hit suppliers. Material P = 1 − Π(1 − p), blended toward the historical rate. Brier skill ≈ 0.33 |
| step 2 | `models/price.py` | Last price with the current shock removed, mean reversion toward the 78-week average (half strength), plus a shock uplift by type when P > 0.5. Price skill ≈ 0.04 public / 0.41 private |
| step 3a | `policy.py` | Rule-based lean planner (supplier weights by cost and risk, tapering mid-window buffer) |
| step 3b | `policy_lp.py` | **Stochastic-programming LP**: 30 scenarios from the forecasts (+ an optional background shock process), one LP per material-window under the exact simulator rules. Given the truth it equals the hindsight optimum (tested) |
| scorer | `simulator.py`, `scoring.py`, `backtest.py` | Local copy of the Kaggle metric plus 11 walk-forward origins × (public, private) windows. Tracks Kaggle (see below) |
| glue | `strategies.py`, `submission.py`, `cli.py` | Strategies (info, weeks) → submission; validation; `uv run dsdg {backtest,validate,submit} <strategy>` |

## Current best approach: two hedged specialist submissions (Round 3, Sat early morning)
Kaggle keeps the better of the 2 selected finals (rules §3b confirm 2 finals; 5 submissions/day). Unwarned shocks
make a single policy trade calm-window cost against shock-window resilience, so we submit two complementary files
**and hedge their forecast columns the same way** (DECISIONS X6–X8):

| Strategy | Ordering | Forecast columns | Backtest mean |
|---|---|---|---|
| `lean_split` (**Kaggle 67.74**) | as lean_hedged + a supplier within 2.5% of the cheapest gets ≥30% of each order | same as lean_hedged | 53.2 (66 windows) |
| `lean_hedged` | LP, no min share, no background shocks, plans at 96% of capacity | calm bet: background and ongoing-shock probabilities ×0.5 beyond week 2 | 48.0 |
| `insured_hedged` | `policy.py` with `INSURED_PARAMS` (tapering mid-window buffer) | shock bet: those probabilities move toward 1 (0.5 × relative material rate) beyond week 4; prices ramp to +5% | 40.6 |
| **best of the two** | | | **63.8** (public 61.8, private 65.8, worst 43.3) — was 59.5 |

Both files use the new **story-script model** (`models/scripts.py`): per open news story, P(hit) = conversion(type,
stage) × lead distribution × duration, combined as max(GBM, script). Live warned stories are NOT hedged.
Files from week-208 data: `submissions/lean_hedged_w208.csv`, `submissions/insured_hedged_w208.csv` (validated).
Older pair: `lean_lp_w208.csv`, `insured_w208.csv`.

## Kaggle results so far (public, weeks 209–221)
| File | Kaggle | Note |
|---|---|---|
| `submissions/lean_split_w208.csv` | **67.74** | best so far (3rd on public; leader 68.65); submitted on team "Shengzhe Pan". lean_hedged + near-free second supplier (DECISIONS Y5). Notebook: `notebooks/lean_split.ipynb` |
| `submissions/lean_hedged_w208.csv` | 60.66 | best so far; submitted on team "Pi W", not "Team 1". The calm-bet file won public (+6.6 over policy_v1) |
| `submissions/insured_hedged_w208.csv` | 48.55 | shock-bet file; lower when no surprise shock hits, as designed |
| `submissions/policy_v1_w208.csv` | 54.10 | "Team 1" |
| `submissions/teammate/primary_w208.csv` | 44.95 | teammate: 1.2× demand, even split, no orders in the last 5 weeks (= strategy `teammate_v1`) |
| `submissions/teammate/independent_w208.csv` | not submitted | same ordering style as primary, news-aware probabilities |

All 4 scored files feed the metric check in `eda/phase2.sh`.

## Key findings to remember
- **Shocks:** news is templated, and every warned shock follows a fixed stage sequence. Rumors end in "Deal reached",
  "Calm returns" or "rule out". Policy reversals have no warning.
- **Value of information:** perfect forecast columns would add +21, but most of that is unwarned shocks, which can't
  be predicted. Planner structure was worth +15, and the LP and best-of-2 captured most of it.
- **Where the Brier error comes from:** 52% from shocks unwarned at the origin, 37% from under-predicted persistence
  (the sanctions). Post-hoc calibration didn't help.
- **Price:** macro, currency, freight and demand don't drive prices (out-of-sample R² ≤ 0). Only mean reversion and
  the shock uplift do.
- **Calm windows:** about 21 points are unattainable for everyone, because fill skill is 0 when naive already fills 100%.

## Next steps
1. **Phase 2 (Sat 12 PM CT): `bash eda/phase2.sh`.** It keeps the Phase 1 data in `data/comp_p1`, downloads, checks
   for new news templates, runs the **metric-variant check** (`eda/17_metric_variants.py`: which of 144 readings of the
   rules reproduces Kaggle's 44.95 / 54.10 on weeks 209–221), rebuilds the caches and backtests, and writes the `_w221` pair.
   Add any other files scored on Kaggle (e.g. the hedged pair) as `path=score` arguments to sharpen the metric check.
2. If the matching metric reading differs from ours (fill pooled vs per material, calm-window fill skill, order-weighted
   diversification), switch `scoring.py` to it and retune (`eda/16`, `eda/18`, `eda/19`), because it changes which
   materials matter (steel = 42% of units).
3. Open stories at 208 for the public window: KES lithium shortage (3/3 shortage stories were real, onset ~211, ~9
   weeks) and a SOUTH strike (about 1 in 3). At 221, check `ScriptModel.fit(data).open_stories` for the private window.
4. Optional: a Databricks notebook (track opt-in closes Sat 6 PM CT); the 3-minute video.

## Working preferences (user)
- Explain simply and briefly.
- Keep `DECISIONS.md` updated with every change.
- Keep the pipeline structured and presentable (cleaning, EDA, features, models, policy, submission), not
  reverse-engineered shortcuts.
- Keep it portable to Databricks: paths via `DSDG_DATA_DIR` / `DSDG_REPORTS_DIR`, plain pandas.
- Never submit to Kaggle: the user or teammate does that.
- Useful evidence scripts are `eda/01…16_*.py`. The tuning scripts read a cache at `reports/policy_cache.pkl`, which
  must be deleted whenever the forecasts or data change.
