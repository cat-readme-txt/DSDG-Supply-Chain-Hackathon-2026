# Handoff: the lean_split approach

What the pipeline does, what we explored, what we assumed, and where each piece lives in the code.
Branch `SK4`. Full decision log with evidence: `DECISIONS.md`. Data facts: `eda/FINDINGS.md`.

## 1. Problem in one paragraph

A manufacturer buys 10 materials, each from 3 supplier countries, over weekly periods. For each future week and
material we output a disruption probability, a benchmark-price forecast, an order quantity, and how that order is
split across suppliers. The score is 100 × (0.30 P + 0.40 F + 0.30 R), each part clipped to [0, 1], scaled so a naive
submission scores 0:
- **P (prediction):** 0.6 × Brier skill on `disruption_prob` (vs each material's historical rate) + 0.4 × MAPE skill on
  `price_forecast` (vs the last known benchmark carried forward).
- **F (cost):** (naive cost − our cost) / (naive cost − hindsight cost), from a supply simulator.
- **R (resilience):** 0.7 × fill-rate skill in shock weeks (disrupted + 4 recovery weeks, vs naive) + 0.3 × diversification
  (1 − HHI of the shares; single source = 0, even 3-way split = 1).

Naive = order exactly the demand each week, 100% from the largest-capacity supplier, historical disruption rate,
last benchmark price.

## 2. Setup and commands

```
uv sync                                         # Python >= 3.11
uv run pytest -q                                # 31 tests
uv run dsdg backtest <strategy> --step 4        # 66 walk-forward windows, ~1 min
uv run dsdg submit <strategy>                   # -> submissions/<strategy>_w<last week>.csv
uv run dsdg validate <csv>
```
Data is read from `data/comp/` (override with `DSDG_DATA_DIR`); reports go to `reports/` (`DSDG_REPORTS_DIR`).
Notebook walkthrough: `notebooks/lean_split.ipynb` (regenerate with `uv run python notebooks/make_notebooks.py`).

## 3. Pipeline (src/dsdg_supply_chain/)

| Stage | Module | What it does |
|---|---|---|
| Load / snapshot | `data.py` | `load_data()`; `Data.until(week)` cuts every time-indexed table at an information week (demand stays complete: it is a known production plan) |
| Labels | `labels.py` | supplier-level hit = capacity < 0.85 × normal, or lead > base, or tariff > base. Their union per material equals `disrupted` exactly (0 mismatches) |
| News parsing | `features/news.py` | 90 headline templates → (shock type, stage, country/lane/material, severity from the body text). Stages: early → escalating → imminent → onset; plus ongoing, called_off, recovered |
| Features | `features/build.py`, `supply.py`, `market.py` | per (week, material, supplier) news, contagion and supply features; `market_signals()` adds futures basis, lane congestion and tension (Round 6) |
| Disruption | `models/disruption.py` | GBM onset classifier for suppliers not hit now + duration model for suppliers hit now; combined per material as 1 − Π(1 − p); blended toward the historical rate (10% + 1%/week ahead) |
| Story scripts | `models/scripts.py` | open news stories → P(hit at t) = conversion(type, stage) × onset-lead distribution × duration survival, from stories resolved before the origin; overdue rumors → 0; optional market confirmation |
| Duration | `models/duration.py` | log-normal duration per shock type, censoring-aware, shrunk to a pooled mean, σ ≥ 0.6 |
| Tariffs | `models/disruption._add_tariff_announcements` | announced hikes in `tariffs.csv` are treated as certain from the effective week |
| Price | `models/price.py` | shock-free level × type uplift (added only when P > 0.5; an ongoing uplift kept unless P(still hit) < 0.2) + half-strength pull to the 78-week mean |
| Cost path | `models/cost_path.py` | splits landed cost into spot, tariff and freight; per-type spot steps for price-moving shocks (Round 6) |
| Ordering (LP) | `policy_lp.py` | stochastic-programming LP over 30 forecast scenarios using the exact simulator rules |
| Ordering (rules) | `policy.py` | rule planner used by the insured file (tapering mid-window buffer) |
| Strategies | `strategies.py` | maps (info snapshot, weeks) → submission; forecast hedging for the pair |
| Scoring | `simulator.py`, `scoring.py`, `backtest.py` | local replica of the metric, hindsight LP, walk-forward backtest, `rescore()` under alternative metric readings |

## 4. The lean_split approach

`lean_split = _lean_split(0.025, 0.30)` in `strategies.py`. Three layers:

**4.1 Forecast columns**
- Disruption: GBM + duration model, max'ed with the story-script model, plus certain tariff announcements, then blended
  toward the historical rate.
- Price: last benchmark with any ongoing shock uplift removed, pulled halfway toward the 78-week average, plus a type
  uplift when a shock is more likely than not.

**4.2 Orders: stochastic LP (`plan_orders_lp`)**
- Per material and 13-week window, draw 30 scenarios: each supplier gets one uniform u and is hit in week t when
  p(t) > u. This gives contiguous shock spells that match the forecast marginals. Lane events share u across a lane.
- A hit supplier gets its shock type's capacity ratio, lead increase and tariff (or the current values if already hit).
- One LP minimises the average simulated cost over the scenarios: purchases at landed cost, 2%/week holding, 5× unmet
  demand, 70% sell-back of leftover and in-transit stock, 5 weeks of starting stock.
- Fill is linear: when a supplier isn't hit, fill = order; when hit, fill ≤ order and fill ≤ reduced capacity.
- Planning capacity = 96% of normal (the 5th percentile of weekly capacity noise). Ordering right at the limit got
  shorted at 5× cost.
- No background hazard (`hazard_mult = 0`): unwarned shocks cannot be planned for and always cost calm windows.
- **Near-free split:** a supplier within 2.5% of the cheapest landed cost gets ≥ 30% of every order week
  (`cheap_premium=0.025, cheap_share=0.30`). This applies to lithium/ORI, natural gas, copper/VAL, polymer/SUN and
  semis/SUN. It buys diversification almost for free and puts a second lane in play for lithium, gas and copper.
- Weeks with no order get an even split (diversification is scored on every row).

**4.3 Pair hedge (`hedge_forecasts`)**
- Two files are produced: a lean (calm-bet) file and an insured (shock-bet) file.
- Calm side: background and ongoing probabilities × 0.5 beyond week 2.
- Shock side: background probabilities move toward 1 by 0.5 × relative material rate beyond week 4, and prices ramp
  to +5% by week 13.
- Live news stories (warned shocks that haven't started, excluding overdue rumors) are left alone in both files.
- Insured file = `insured_hedged`: rule planner with a 2.25-week tapering mid-window buffer (`INSURED_PARAMS`).

**4.4 Round 6 additions (latest candidates)**
- `lean_pxlp_mkt`: price-aware LP orders (`lean_split_pxlp`) + market-checked forecast columns, calm hedge.
- `insured_hedged_mktcol`: insured orders + market-checked forecast columns, shock hedge.
- The market-checked probabilities are deliberately **not** fed to the planners: in one window that made the LP order
  from a supplier it was ~90% sure would be banned.

## 5. Assumptions

**Simulator (rules are silent; constants in `simulator.py`):**
- "Average landed cost" (scale of holding, unmet and sell-back) = mean landed cost over the material's suppliers and
  the window's weeks.
- Starting stock = the sum of the first 5 window weeks' demand.
- Holding is charged on end-of-week stock.
- Hindsight cost = an LP with perfect knowledge.

**Metric (alternatives can be recomputed with `backtest.rescore`; `eda/17_metric_variants.py` covers 144 readings):**
- F pooled over materials in dollars.
- Fill rate pooled over materials in units.
- Fill skill = 0 when naive never runs short ("calm = zero"). The alternative, "calm = one", gives full credit if we
  don't run short either; it changes levels by ~10 points and makes safety stock worth more.
- Diversification = mean over all rows (order-weighted is the alternative).
- P pooled over all rows.

**Backtest (`backtest.py`):** origins every 4 weeks; for origin o the naive baseline is frozen at o. A first window
o+1..o+13 uses information up to o; a second window o+14..o+26 uses information up to o+13 but is still scored
against the naive frozen at o (its price and rate are stale). This gives 66 windows.

**Modelling:**
- Suppliers are independent within a material (better than max in tests), except that lane shocks share a lane.
- Shock severity by type: ban 0.05, sanctions 0.20, shortage ~0.33–0.61 (stated in the news body), reversal ~0.5,
  closure 0.75 + 3 weeks lead, strike 0.8 + 3–4 weeks lead, tariff: rate from the announcement.

## 6. What we explored (and the verdict)

| Area | Finding | Verdict |
|---|---|---|
| Labels | `disrupted` decodes exactly from supply_history + tariffs | used |
| News | templated scripts per shock type; rumors end in "called off"/"calm returns"; severity in body text; ends never announced | used (parser + script model) |
| Story conversion | shortage 3/3 real (onset ~4 weeks after first "maintenance delays"), tariff 3/3, sanctions 2/2, ban 3/4, closure 2/6, strike 2/6 | used |
| Tariff announcements | 5/5 took effect exactly as announced, 2–8 weeks ahead | used (certain) |
| Futures basis | ramps 2–4 weeks before every real shortage/ban (to 12–25%), flat for the one rumor; calm sd 1.2%; useless for calm-week price prediction | used for story confirmation (dense Brier skill 0.306 → 0.322) |
| Lane congestion / tension | separate real from rumor lane stories only the week before onset (≥ 50 vs ≤ 46) | used for story confirmation |
| Prices | random walk + weak mean reversion (variance ratio 0.60 at 13 weeks); no macro, freight, demand, cross-supplier or cross-material signal; uplift by shock type, all 3 suppliers, onset week to end + 1 | used; full-strength reversion tested only in a scratch walk-forward (+0.04 skill), not adopted |
| Landed cost | spot × (1 + tariff) + freight; freight ∝ container rate on MERROW/ATLANTIC | in cost path; freight scenarios hurt (closure rumors moved orders badly), off |
| Durations | reversal 2–3, strike 2–5, closure 3–8, ban 5–11, shortage 9, tariff 10–18, sanctions ~18 (the 35-week run is two overlapping events) | used |
| Demand | independent of shocks | — |
| Background hazard in LP | always costs calm windows (−1) | off in lean; insurance lives in the second file |
| Calibration of P | post-hoc recalibration +0.004; errors come from unwarned shocks | rejected |
| Score-aware LP (`fill_kappa`), 92% capacity (`d92`) | +0.15–0.3, within noise | kept as variants |
| LP-based insured partner | better alone, worse as a partner | rejected; rule-based insured kept |
| Price-aware LP | +2.3 where a price jump is warned; −1.1 where long shocks are bet to end early | `lean_split_pxlp`, net +0.03 |

## 7. Backtest reference (66 windows, our metric reading)

| Strategy | Mean | Pair with insured |
|---|---|---|
| lean_hedged | 52.7 | 59.7 |
| lean_split | 53.22 | 60.21 |
| lean_pxlp_mkt (orders + market columns) | 53.40 | 60.39 (with insured_hedged_mktcol) |
| insured_hedged | 39.4 | — |

Where the points go for lean in normal windows: P ≈ 9/30 (information-limited), F ≈ 34/40 (0.93–0.97 in calm
windows), R ≈ 13/30. A calm window caps near 60 under "calm = zero". High scores come from windows with a warned or
ongoing shock.

## 8. Open items / next steps

- Confirm the metric reading (§5) against known scores, then re-pick the pair with `eda/27_pick_pair.py` under it.
- At a new information week, run `uv run python eda/29_state_report.py` (ongoing shocks with survival odds, open stories
  with the market check, tariff announcements, unexplained futures ramps). Then rebuild with
  `uv run dsdg submit lean_pxlp_mkt` and `uv run dsdg submit insured_hedged_mktcol`.
- If a long shock is ongoing at the information week, compare with plain `lean_split` orders: the price-aware LP bets
  on it ending.
- Not done: full-strength price reversion in `models/price.py`; "share sinks" (large shares to suppliers with near-zero
  capacity, since unfilled units are not charged); consolidating order weeks for diversification; the cost path in
  the insured rule planner.
