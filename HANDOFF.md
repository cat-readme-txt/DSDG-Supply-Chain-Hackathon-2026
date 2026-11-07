# Handoff: DSDG Hackathon 2026 "Supply Chain Disruptor"

Read this first, then `DECISIONS.md` (every modelling choice and its evidence) and `eda/FINDINGS.md` (what the data
is). Branch `SK`. Nothing is committed yet; everything is in the working tree.

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

## Current best approach: two specialist submissions
Kaggle scores the better of the 2 selected final submissions (**standard Kaggle rule, not yet confirmed on this
competition's page; check it**). Unwarned shocks make a single policy trade calm-window cost against shock-window
resilience, so we submit two complementary policies:

| Strategy | Idea | Backtest mean |
|---|---|---|
| `lean_lp` | LP, no minimum share, no background shocks, plans against 96% of capacity. Cheapest plan; wins calm windows and windows with warned shocks | 47.3 |
| `insured` | `policy.py` with `INSURED_PARAMS` (tapering 2.25-week mid-window buffer, diversifies with risk). Wins when an unwarned shock hits the primary supplier | 44.7 |
| **best of the two** | | **59.5** (public 57.7, private 61.4, worst 43.4) |

Files built from week 208 data: `submissions/lean_lp_w208.csv`, `submissions/insured_w208.csv` (both validated).

## Kaggle results so far (public, weeks 209–221)
| File | Kaggle | Note |
|---|---|---|
| `submissions/teammate/primary_w208.csv` (teammate: 1.2× demand, even split, no orders in the last 5 weeks) | 44.95 | = strategy `teammate_v1` |
| `submissions/policy_v1_w208.csv` | **54.5** | backtest analogue +6.1 vs Kaggle +9.5, so the local scorer tracks Kaggle |
| teammate's `teammate_forecasts_primary_policy_w208.csv` (teammate ordering + our forecasts) | not submitted | = strategy `teammate_hybrid`; backtest 38.9/42.8, expected ~50; worse partner for `lean_lp` than `insured` (58.4 vs 59.5) |
| `lean_lp_w208.csv`, `insured_w208.csv` | not yet submitted | — |

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
1. **Phase 2 (Sat 12 PM CT):**
   - Re-download the data.
   - Score the teammate's and our files on 209–221 locally against their Kaggle scores (44.95 / 54.5) to confirm
     scorer assumption E2.
   - Rerun `eda/05_validate_news.py` for new templates.
   - Rebuild `lean_lp` and `insured` with data through 221: `uv run dsdg submit lean_lp` / `insured`. Files are named
     `_w221`.
   - Retune: `rm reports/policy_cache.pkl` and rerun `eda/16_pair_tune.py`. Backtest origins extend automatically.
2. Confirm the best-of-2 rule on the competition page before choosing the final pair.
3. Optional improvements: a better lean specialist in warned-shock windows; a data-quality (silver) stage; turning
   the EDA scripts into a pipeline stage; a Databricks notebook (track opt-in closes Sat 6 PM CT); the 3-minute video.

## Working preferences (user)
- Explain simply and briefly.
- Keep `DECISIONS.md` updated with every change.
- Keep the pipeline structured and presentable (cleaning, EDA, features, models, policy, submission), not
  reverse-engineered shortcuts.
- Keep it portable to Databricks: paths via `DSDG_DATA_DIR` / `DSDG_REPORTS_DIR`, plain pandas.
- Never submit to Kaggle: the user or teammate does that.
- Useful evidence scripts are `eda/01…16_*.py`. The tuning scripts read a cache at `reports/policy_cache.pkl`, which
  must be deleted whenever the forecasts or data change.
