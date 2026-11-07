# Decision log

Every modelling choice so far, the evidence behind it, and what to revisit. Newest last. Evidence scripts are in `eda/`
(run them with `uv run python eda/<file>`); backtests with `uv run dsdg backtest <strategy>`.

## Setup
| # | Decision | Why | Revisit |
|---|---|---|---|
| S1 | uv project, `src/dsdg_supply_chain/`, pandas and no Spark | the data is 1.7 MB | — |
| S2 | Paths come from `DSDG_DATA_DIR` / `DSDG_REPORTS_DIR`; Python ≥ 3.11, pandas ≥ 2.1 | to move to Databricks later (bronze/silver/gold) | when building the Databricks notebook |
| S3 | Kaggle token lives in a gitignored `.env`; the CLI needs `set -a; . ./.env` first | security | use Databricks secrets later |

## Scorer (`simulator.py`, `scoring.py`, `backtest.py`)
| # | Decision | Why | Revisit |
|---|---|---|---|
| E1 | A local copy of the Evaluation-tab simulator, plus a hindsight LP for the F denominator | lets us score strategies without spending Kaggle submissions | — |
| E2 | Assumptions: average landed cost = mean over suppliers × window weeks; initial stock = demand of the first 5 window weeks; holding charged on end-of-week stock; fill skill = 0 when there are no shock weeks or naive fills 100% | the rules don't say | **Check after Phase 2:** score `submissions/teammate/primary_w208.csv` on weeks 209–221 and compare with its Kaggle public score of 44.95 |
| E3 | Backtest: 11 origins o; "public" window o+1..o+13 uses info up to o; "private" window o+14..o+26 uses info up to o+13; the naive baseline is always built at o | mirrors Phase 1 / Phase 2, where the sample submission stays frozen at week 208 | — |
| E4 | Tests: naive scores 0, the hindsight plan scores F = 1, and our naive equals the sample submission | proves the scorer is self-consistent | — |

## EDA findings that shaped everything (`eda/FINDINGS.md`)
| # | Finding | Evidence |
|---|---|---|
| F1 | `disrupted` = any supplier with capacity < 0.85× normal, or lead > base, or tariff > base (0 mismatches) | `02_country_shocks.py` |
| F2 | News is templated (90 templates); signal stories follow fixed stage sequences, and rumors end in "called off" | `03_news_timeline.py` |
| F3 | Same 3-tier supplier structure for every material; the naive policy buys 100% from the primary | `04_market.py` |
| F4 | Prices are a random walk with mean reversion to a 78-week anchor; macro, currency, freight and demand don't drive them | `10_price_drivers.py` |

## Step 1: disruption model
| # | Decision | Why | Revisit |
|---|---|---|---|
| D1 | Rule-table news parser → (type, stage, entity, severity) → features per supplier per week | 238/238 articles mapped, 61/61 onset matches, severity exact | new templates in Phase 2? Rerun `05_validate_news.py` |
| D2 | NaN → None normalisation inside the news expansion | NaN is truthy, which stopped "Deal reached" from closing stories (bug found in validation) | data-quality stage |
| D3 | Model at the supplier level, combined into a material probability as 1 − Π(1 − p) | the policy needs to know *which* supplier is at risk; independence beat max (`08_…`) | correlated lane events |
| D4 | Two parts: an onset classifier (not hit now) plus a duration model (hit now) | a single classifier learned "long shocks never end" from the censored sanctions (skill −7 at origin 130) | — |
| D5 | Duration: log-normal per shock type, censoring-aware, shrunk to pooled, σ ≥ 0.6 | a free σ (0.24) was overconfident: predicted 0.2% where 49% happened | tune σ / prior after Phase 2 |
| D6 | Onset features = news + contagion only (no `weeks_since_hit`, `cap_ratio`) | history features made never-hit suppliers look safe; near skill 0.19 → 0.28 | — |
| D7 | Market features (tension, congestion, tariff announcements) left out | weak on their own and no added value once news is known (`06_…`, `07_…`) | re-test with more data |
| D8 | GBM (depth 3, min leaf 200) over logistic | similar mean, better median/pooled | — |
| D9 | Blend toward the historical rate: 10% + 1% per week ahead | covers unwarned shocks; worst window −0.43 → −0.35 | — |
| | **Result:** Brier skill ≈ 0.33 (backtest) | | |

## Step 2: price model
| # | Decision | Why | Revisit |
|---|---|---|---|
| P1 | Price = shock-free level × shock uplift by type | uplift depends on type (ban ~23%, shortage/reversal/sanctions ~12–14%, lanes ~0), not on capacity lost (`09_…` event study) | — |
| P2 | Add a new shock's uplift only when P > 0.5 | MAPE rewards the most likely outcome, not the mean | — |
| P3 | Keep an ongoing shock's uplift unless P(still hit) < 0.2 | removing it at P < 0.5 lost up to −0.52 skill during the sanctions | — |
| P4 | Mean reversion toward the 78-week average, at half the fitted speed, measured on the shock-free level | the only out-of-sample signal found (+0.05 near / +0.09 far); measuring on shocked prices double-counted | — |
| P5 | Rejected: GBM on price, linear trend, EWMA, futures, macro regression | each was ≤ carry-forward out of sample | — |
| | **Result:** price skill 0.04 public / 0.41 private (the 0.38 is mostly from refreshing after Phase 2) | | |

## Step 3: ordering policy (`policy.py`, strategy `policy_v1`)
| # | Decision | Why | Revisit |
|---|---|---|---|
| O1 | A lean planner: buy only what each window needs on top of the 5 weeks of starting stock, timed so slices arrive when stock runs out | the cost breakdown showed naive and teammate overbuying in calm windows (about 9 weeks of leftover sold at 70%), so F = 0 there | — |
| O2 | Plan per supplier slice using that supplier's expected lead; expected delivery = (1−p)·min(q, cap) + p·min(q, cap·severity) | open-loop plan; capacity and lead depend on the shock | scenario-based optimisation |
| O3 | Zero-order weeks get an even split | diversification is scored on every row, so this is free | — |
| O4 | Weights: exp(−5 × cost premium) × (1−p)^8, blended toward even by 0.667 × material risk; buffer 0.5 week; over-order on at-risk slices | tuned on 22 windows (`11_tune_policy.py`); mean 45.9 → 47.6 | retune after Phase 2 |
| O5 | The tuner set the risk-driven buffer (`safety_risk`) to 0 | extra stock before predicted shocks didn't pay off; steering away from the at-risk supplier did | re-test with Phase 2 |
| O6 | Diversification vs cost is a real trade-off: cheapest-only lean scores 45.7 calm but 24.5 shock; tuned mix 40–46 calm, 50–78 shock | unwarned shocks (window 170) need baseline insurance | — |
| | **Result:** backtest 45.8 public / 49.4 private (teammate 33.7 / 37.7; naive 0 / 5.4) | | |

## Round 2: end-to-end re-analysis
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| R1 | Value of information: perfect forecast columns +21, planner given the true state +4, hindsight orders +15. Even perfect inputs with the v1 planner only reach 72.5, so "70 average locally" isn't a realistic target; shock windows are where the high scores are | `12_value_of_information.py` | — |
| R2 | Brier error in public windows: 52% from shocks unwarned at the origin, 37% from under-predicted persistence | `13_brier_breakdown.py` | — |
| R3 | Post-hoc calibration (empirical persistence, surprise floor) adds only +0.004 Brier skill, so it was **rejected**. The errors come from unprecedented long shocks (sanctions) | `14_calibrate_disruption.py` | more data in Phase 2 |
| R4 | A stochastic-programming LP planner (`policy_lp.py`): 30 scenarios drawn from the forecasts; exact simulator rules; fill linearised exactly when not hit. Given the truth it reproduces the hindsight optimum (test) | `15_lp_policy.py`, `test_lp_plan_matches_hindsight…` | — |
| R5 | The lean LP wins calm windows (54 vs 41) but collapses on unwarned primary-supplier shocks (windows 92 and 170 → 0). Naive's constant ~5-week stock is an accidental buffer | `15_…` per-window table | — |
| R6 | A background shock process (country hazards plus bootstrapped historical runs, lane-correlated) makes the LP safer, but there is still a calm/shock trade-off: every policy sits on one frontier, because unwarned shocks can't be predicted | `15_…` grid | — |
| R7 | A tapering mid-window buffer (consumed before the window ends, so holding cost only) is the cheap insurance; it fixes window 170 (27 → 62) but costs calm windows | buffer grid | — |
| R8 | **Two final submissions = best of two.** A lean LP plus an insured rule planner covers both sides of the frontier. The insured specialist was tuned for mean max(lean, insured) | `16_pair_tune.py` | **confirm the best-of-2 rule on Kaggle** |
| R9 | The LP plans against 96% of normal capacity (the 5th percentile of normal weekly capacity); ordering at the limit got shorted at 5× | capacity quantiles; lean calm 52.7 → 54.4 | — |
| R10 | The LP insured specialist (heavy hazard) is close to the rule-based one (59.4 vs 59.5); keep the rule-based one | `16_…`, comparison | — |
| | **Result:** lean 47.3, insured 44.7, **best-of-two 59.5** (public 57.7, private 61.4, worst 43.4), up from 47.6 | | |
| | Files: `submissions/lean_lp_w208.csv`, `submissions/insured_w208.csv` | | |

## Kaggle results (public window 209–221)
| Submission | Kaggle public | Backtest analogue |
|---|---|---|
| teammate `primary_w208.csv` | 44.95 | window 92–104: teammate 43.7 |
| ours `lean_split_w208.csv` | **67.74** | +7.1 over lean_hedged on Kaggle vs +0.5 in the 66-window backtest. Likely cause (unverified until Phase 2): the split moved lithium to ORI on another lane before the expected KES lithium shortage; confirm with `eda/24_score_files.py` |
| ours `lean_split_d92_w208.csv` | 67.40 | backtest +0.3 over lean_split, Kaggle −0.34: within noise; keep lean_split |
| ours `lean_hedged_w208.csv` | 60.66 | backtest alone 48.0; the calm-bet file won public |
| ours `insured_hedged_w208.csv` | 48.55 | shock-bet file |
| ours `policy_v1_w208.csv` | **54.10** (Kaggle listing; 54.5 was a typo) | same window: ours 49.8 (+6.1 vs Kaggle +9.5) → the local scorer tracks Kaggle |

Risk spotted: in window 170–182 (an unwarned Merrow closure) policy_v1 scored 26.7 against the teammate's 62.7. The lean, cheap
sourcing is exposed to surprise lane shocks. The next version must be robust to these, for the private window.

## Round 3: fresh re-analysis (Sat Nov 7, early morning)
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| X1 | Official rules fetched with `kaggle competitions pages list --content --format json -- <comp>` → `reports/kaggle_pages/pages.md`. **5 submissions/day** (not 3); select 2 finals (none selected → 2 best public); hand-labelling future weeks or reverse-engineering the hidden solution is forbidden | rules §3, §5b | — |
| X2 | Where the points are: in calm windows even an oracle scores ~58–60 under our metric reading (fill skill = 0 when naive fills 100%), so R is diversification only; shock windows reach 81–96. 70+ needs a shock window, a better P, or a different metric reading | `reports/value_of_information.csv` per window | E2 check at Phase 2 |
| X3 | Brier skill by horizon: ~0.9 at weeks 1–2, ~0.5 at 3–4, ~0 beyond 8. Nothing beats the historical rate once no news is visible; that's an information limit, not a model gap | `reports/disruption_walkforward.csv` | — |
| X4 | Units are very unequal across materials (steel 42%, polymer 21%, aluminum 16%) while dollars are balanced. If Kaggle pools fill rate over units, R is mostly steel/polymer/aluminum | per-material demand × landed cost | decide after E2 check |
| X5 | `eda/17_metric_variants.py` scores a file under 144 readings of the rules (F/fill pooled vs per-material, calm-window fill 0/1/skip, diversification row vs order-weighted, P pooled vs per-material, average-cost and initial-stock definitions). Variants differ by ±3 points and can flip rankings | demo on window 105–117 | run at Phase 2 with the 2 known Kaggle scores (44.95, 54.10) |
| X6 | **Forecast hedge across the pair.** Kaggle keeps the better file, so each file's forecasts should be sharp in the world where it wins. Lean (wins calm) gets probabilities ×0.5 beyond week 2; insured (wins shocks) gets probabilities moved toward 1 by 0.5 × relative material rate beyond week 4, and prices ramp to +5% by week 13. Orders unchanged | `eda/18_pair_hedge.py`: the grid is flat around the choice (a 0.3–0.7, b 0.5–0.7) | retune with Phase 2 data; there are only 12 independent windows, so moderate values were chosen |
| X7 | **The hedge leaves live news stories alone.** A warned shock that hasn't started yet (e.g. KES lithium at 208) follows the evidence in both files. Ongoing shocks and background risk are hedged, because how long a shock lasts is the uncertain part. Dead rumors (overdue past every observed lead) don't count as live | masking variants: none 63.2, **stories 63.4**, stories+hits 61.4 | — |
| X8 | **Story-script model** (`models/scripts.py`). The news follows fixed scripts per type, and the GBM (min leaf 200) can't learn rules backed by 3 stories. Per open story: P(hit at t) = conversion(type, stage) × Σ P(onset = k \| not yet) × P(duration ≥ t−k+1), with conversion and lead estimated only from stories resolved before the origin (Beta prior pooled over types), duration from `duration.py`; combined as max(GBM, script) | `eda/20_story_scripts.py` → `eda/stories.csv`: shortage 3/3 real (lead 4 from the first "maintenance delays"), tariff 3/3, sanctions 2/2, ban 3/4, closure 2/6 (rumors stall at stage ≤ 2), strike 2/6. At 208: KES lithium P 0.55 → 0.81 at week 212. Backtest Brier skill 0.326 → 0.328 (origins every 13 weeks rarely see an open story); dense test (`eda/22_script_dense.py`, an origin every 2 weeks, 9,360 rows): rows with a live story 0.294 → **0.328**, all rows 0.291 → 0.306, no-story rows unchanged | — |
| X9 | Bug fixed: an unresolved, overdue rumor (the NORTH strike from week 149) stayed open until it went stale at 8 weeks. Overdue stories now get 0 | origin 156 Brier gain −0.04 → 0 | — |
| X10 | The insured ordering retuned for the hedged pair is unchanged (63.55 → 63.56), so `INSURED_PARAMS` is kept | `eda/19_pair_retune.py` | — |
| | **Result (end to end, scripts + masked hedge):** best-of-two **59.5 → 63.8** (public 61.8, private 65.8), worst unchanged 43.3. Files: `submissions/lean_hedged_w208.csv`, `submissions/insured_hedged_w208.csv` | `reports/backtest_{lean,insured}_hedged.csv` | |

| X11 | Denser backtest (`dsdg backtest <s> --step 4`: 66 windows instead of 22). Means: policy_v1 48.0, lean_hedged 52.7, insured_hedged 39.4, **best of pair 59.7** (the 63.8 from 22 windows was optimistic). Windows without a surprise shock: policy_v1 49.0 vs Kaggle 54.10, lean_hedged 57.0 vs Kaggle 60.66, so the backtest ranks files correctly (+8 vs Kaggle's +6.6) but runs ~4–5 below Kaggle | `reports/backtest_*_step4.csv` | explain the gap at Phase 2: scorer readings (E2) vs a favourable public window |

## Round 4: "57 → 70?" from scratch (Sat ~3–4 AM)
Where lean_hedged's points go in the 61 normal windows (66-window backtest): **P 9.4/30, F 34.4/40, R 13.2/30** (fill 8.4/21, diversification 4.8/9).
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| Y1 | Persistence of ongoing shocks *looked* badly under-predicted (p 0.43 vs 0.65; 48% of Brier error), but it's entirely the 35-week sanctions. Outside 99–133 the model is calibrated (slightly over). Not recalibrated | `reports/script_dense.csv` by era | a long shock running at 221 |
| Y2 | Prices from scratch: only the pull toward the 78-week average predicts (out-of-sample R² +0.06 at h=13). Futures (worse than carry-forward), seasonality and momentum don't. P is information-limited on both halves | `eda/25_price_fresh.py` | — |
| Y3 | F loss vs hindsight: clean calm windows already reach F 0.93–0.97 (the rest is supplier choice under cost noise). The loss is concentrated in an unwarned ban (unmet $498k, origin 150) and a first-ever strike at origin 70 (duration unknown → overbought 10.7 vs 7.7 weeks of steel). By 221 strikes have history | `floss.py` / `o70.py` (scratch) | — |
| Y4 | **Tariff announcements are certain:** 5/5 took effect at the announced week and rate, lasting 10–18 weeks. An announced hike now means P = 1 from that week, then the tariff duration curve. Small backtest gain (rare), free certainty when live | `eda/26_tariff_announcements.py`; dense Brier 0.3055 → 0.3061 | check tariffs.csv at 221 |
| Y5 | **Near-free split:** 5 materials have a second supplier within 2.3% of the cheapest (lithium/ORI 0.3%, natural gas/ORI 1.0%, copper/VAL 1.4%, polymer/SUN 1.8%, semis/SUN 2.3%). The LP (`cheap_premium`, `cheap_share`) gives such suppliers ≥30% of each order; for lithium, gas and copper that is also another lane. Strategy `lean_split` | 66 windows: lean 52.7 → 53.2, pair 59.7 → 60.2, diversification 0.53 → 0.60; settings 2.5%/45% and 5%/30% within noise | — |
| Y6 | **Verdict:** under our metric reading, a calm window caps near 60 even for a perfect forecaster (fill skill = 0 when naive never runs short; P information-limited). 70+ only happens in windows with a warned or ongoing shock (lean averages 68.9 in windows where naive runs short). The next real levers are the noon metric check and planning for the specific state at week 221 | above | Phase 2 |

## Round 5: pushing past lean_split (Sat ~4–5 AM)
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| Z1 | lean_hedged → lean_split changed only orders (forecast columns identical), yet Kaggle gave +7.1. Diversification explains only ~+1 under any definition (row mean +1.1, order-weighted +0.9, per-material +0.8), so ~+6 came from cost/fill in what actually happened in 209–221 | diversification of the 5 scored files on 209–221 | confirm with eda/24 at noon |
| Z2 | Backtests now store per-material ingredients; `backtest.rescore` recomputes any metric reading without re-running strategies; `eda/27_pick_pair.py` ranks strategies and pairs under a reading | — | — |
| Z3 | No metric reading reproduces +7 on average (history has only 2 windows where 1–3 materials run short). But the **levels** fit "calm = one" (a material neither we nor naive run short on gets full fill credit) better: insured 49–50 vs Kaggle 48.55 (our reading 39.4). Under it, pairs score ~69.5–70.8 | rescore table | **decisive check at noon** |
| Z4 | Score-aware LP: `fill_kappa` adds κ × average cost to a short unit in shock weeks of scenarios where the primary is hit (where fill skill pays). κ = 10: lean +0.2 | 66 windows | — |
| Z5 | `lean_split_d92` (κ 10, plan at 92% of capacity) is the best lean under all 3 readings, but only +0.15–0.3; background hazard in lean always hurts (−1) | 66 windows × 3 readings | — |
| Z6 | An LP insured (split + κ + hazard 1–4) scores higher alone (44–56) but much lower where lean collapses (22–54 vs 65), so the pair is worse. The rule-based tapering buffer is the real insurance; `insured_hedged` stays | same | — |
| Z7 | Library ceiling: picking the best 2 of all our strategies per window *with hindsight* gives 60.8 vs 60.2 for the fixed pair. Under our scorer, planning is exhausted; the remaining gap to the oracle (~72) is information | oracle over library | — |

## Round 6: fresh-eyes re-analysis (Sat ~4 AM)
Market signals → sharper story probabilities (`features/market.py market_signals`, `models/scripts.py MarketCheck`,
option `market=True`, default off; `eda/28_market_dense.py`).
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| M1 | **Futures basis confirms shortage/ban stories.** basis = futures_4w/spot − 1; calm noise sd 1.21% (robust sd used: 1.4826 × MAD, expanding). Every real shortage/ban (6/6) ramped ≥ 3 sd by its 2nd story week, starting the week of the first article; the one ban rumor stayed at −1.3..0%. Rule: basis ≥ 3 sd in a story week (or ≥ 2 sd two weeks running) → conversion 0.95; ≥ 2 story weeks with basis < 2 sd throughout → odds × P(miss \| real) = (misses+1)/(real+2) on resolved stories (0.125 at 208) | weeks 1–208 sanity check; resolved counts at 208: supplier real 6, miss 0, rumor alarms 0 | Phase 2 stories |
| M2 | **Lane congestion confirms strikes/closures.** From the escalating stage: congestion ≥ 50 (closures also tension ≥ 47 and ≥ 10 above its 6-month median) → conversion 0.95, onset next week (90%); at the final pre-onset stage (strike imminent, closure escalating; the parser has no imminent closure headline) without it → odds × P(miss \| real) (0.2 at 208) | real 52–57 vs rumors ≤ 46; lane real 3, miss 0, rumor alarms 0 | — |
| M3 | Onset pmf of a confirmed supplier story reweighted by ramp size: basis / mean pre-onset peak of past real ramps vs the average profile (k=1: 1.0, 2: 0.67, 3: 0.45, 4: 0.24), log-sd 0.4 | profile and peaks learned walk-forward from resolved stories | small effect; not the cause of M6 (window 138 is 38.9 without it) |
| M4 | **Unannounced-ramp guard off.** Every real ramp came with news; the guard (ramp with no story → shortage, conv 0.3) fired once in 1–208, on noise (ORI natural gas wk 117–118: 2.2 then 3.3 sd, no shock) | full-data scan | — |
| M5 | **Dense test (origin every 2 weeks 52–194, 9,360 rows): Brier skill all rows 0.3061 → 0.3224; rows with a live story 0.330 → 0.370; no-story rows unchanged (0.2898). No origin gets worse.** Biggest gain origin 172 (MERROW closure confirmed by congestion 55); NORTH strike rumor at 150–156 down-weighted 0.59 → 0.26 | `reports/market_dense_step2.csv`; baseline reproduces Y4's 0.3061 exactly | — |
| M6 | **Backtest (66 windows): `lean_split_mkt` 52.96 (public 50.51, private 55.40) vs lean_split 53.22 (50.64 / 55.81); 4 wins, 4 losses. Pair with insured_hedged: 59.94 vs 60.21 (pooled/pooled/zero), 68.85 vs 69.44 (pooled/pooled/one), 70.73 vs 70.75 (mean/mean/one).** Brier skill rises in every changed window, but F falls in 3 of them. Window 138-private (VAL natural-gas ban at 153): P 0.364 → 0.414, F 0.925 → 0.557 (−13.1), because the LP sent 31–48% of the week 154–155 orders to VAL even though P(VAL hit) was 0.9. Window 150-public −4.0: the NORTH/SOUTH rumors were down-weighted correctly, so the LP leaned on VAL, which an unwarned ban then hit. Gains: 134-public +2.2, 122-private +0.8 | `reports/backtest_lean_split_mkt_step4.csv` | **Do not swap into the final pair yet.** Next: market-checked forecast columns only (P) with lean_split's orders, and check how the LP treats a confirmed ban on a secondary supplier (window 138) |
| M7 | Week 208: KES lithium confirmed (basis 2.6% → 4.1% = 3.4 sd). Conversion 0.86 → 0.95; onset pmf at 210/211/212: 0.25/0.50/0.25 → 0.08/0.60/0.32; supplier P at 212 0.855 → 0.947; material P peak 0.81 → 0.87. SOUTH strike is still escalating (congestion 39) → no market update (P unchanged) | `ScriptModel.fit(data, market=True).explain(data)` | re-read at 221 |
| # | Finding / decision | Evidence | Revisit |
|---|---|---|---|
| V1 | **Landed cost = spot × (1 + tariff) + freight, exactly.** Freight is a constant per supplier × {1.0, 1.6 in a strike, 3.0 in a closure} (MERROW, ATLANTIC; no lane event ever hit NORTH/SOUTH/RAIL). A source shock (shortage, ban, reversal, sanctions) lifts the spot of **every** supplier of the material by a step that lands in the onset week and leaves the week after the end. Steps (log, ½·(onset jump − end drop) of the mean log spot): ban 0.22–0.32, reversal 0.16–0.18, shortage 0.14–0.16, sanctions 0.05–0.11; strike, closure and tariff ≈ 0 | `eda/28_cost_path.py` | — |
| V2 | `models/cost_path.py`: spot step per type = event mean, shrunk toward the source-family mean by 0.25 pseudo-events (between-type spread ≈ 2× per-event noise), so an unseen type borrows the family step. At 208: ban 0.29, reversal 0.17, shortage 0.16, sanctions 0.11 | same | — |
| V3 | **LP price path** (`LPParams.price_path`, default off): every scenario gets its own landed costs. A source shock in the scenario lifts every supplier's spot from its onset week to its end; an ongoing one loses its uplift after the scenario's end; announced tariffs last only while the hike does. A (holding, unmet, salvage scale) is per scenario. Costs are built as today's cost plus exact deltas, so untouched supplier-weeks are bit-identical (calm windows unchanged). Strategy **`lean_split_pxlp`** | origin 154 (HAL steel shortage warned): pre-buys steel at capacity in 155–157, steel F 0.77 → 0.95 | — |
| V4 | 66 windows, `lean_split_pxlp` vs `lean_split`: +0.03 (public +0.10, private −0.03), 18 wins / 13 losses / 35 ties, F 0.792 → 0.793. In the 35 windows with a live source shock at the information week: +0.10, F over those materials 0.767 → 0.773. Gains where a shock is warned (134 public +2.3, 142p/154/158/162 +0.4–0.6); losses where an ongoing shock outlived the scenarios (94 private −1.1, rare-earth reversal 78 public −0.7 / 66 private −0.3) and on an announced tariff ending early (166 private −1.0, 178 public −0.5). Pair with insured_hedged: 60.21 → 60.25 (pooled/pooled/zero), 69.44 → 69.47 (pooled/pooled/one), 70.75 → 70.79 (mean/mean/one) | `eda/29_pxlp_compare.py`, `reports/backtest_lean_split_pxlp_step4.csv` | a long shock at 221: the price path bets on it ending |
| V5 | **Freight channel rejected** (`path_freight=True`, `lean_split_pxlp_fr`): 66-window −0.35, pair 60.21 → 59.90. A SOUTH closure rumor at 147 tripled ORI's freight in scenarios and moved natural gas to VAL just before VAL's gas ban (134 private −16.9); closure rumors convert only 2/6 | `reports/backtest_lean_split_pxlp_fr_step4.csv` | — |
| V6 | Week 208: only lithium changes. pxlp front-loads before the expected KES shortage (209–210: 2,213 / 2,116 units vs 1,152 / 1,238), buys little in 212–214 (781 / 357 / 837 vs 1,443 / 1,341 / 1,152), total +1.7%, ORI share 31% → 33% | `submissions/lean_split_pxlp_w208.csv` | rebuild at 221 |

## Teammate submission
| T2 | `submissions/teammate/independent_w208.csv`: same style of ordering as primary_w208 (1.2× demand for 8 weeks, then 0; even split), news-aware probabilities that decay, prices carried forward. Expected public ~45–52; score it exactly after Phase 2 |

| # | Note |
|---|---|
| T1 | `submissions/teammate/primary_w208.csv` is valid and scored 44.95 on Kaggle public. It's reproduced exactly by the `teammate_v1` strategy (backtest ~34 public / ~38 private). Its ordering rules (even split, 1.2× buffer, no orders in the last 5 weeks) are a baseline for step 3. |

## Open items
- Data-quality stage (silver layer) and EDA-as-pipeline stage.
- Phase 2 (Sat 12 PM CT): re-download, rerun the validations, check E2 against 44.95, and retune for the private window.
- Databricks notebook; the Databricks track opt-in closes Sat 6 PM CT.

| # | Round 6 combination | Evidence | Revisit |
|---|---|---|---|
| C1 | **Final candidates:** `lean_pxlp_mkt` = price-aware LP orders (lean_split_pxlp) + market-checked forecast columns (calm hedge); `insured_hedged_mktcol` = insured_hedged orders + market-checked columns (shock hedge). Market-checked probabilities are NOT fed to the planners (they hurt orders in window 138) | 66 windows: lean 53.22 → 53.40 (21 better / 11 worse), pair 60.21 → 60.39; calm=one 69.44 → 69.62; mean/mean/one 70.75 → 70.93. Files `submissions/{lean_pxlp_mkt,insured_hedged_mktcol}_w208.csv` | at 221: if a long shock is ongoing, the price path bets on it ending (lost −1.1 in sanctions windows) — compare with plain lean_split orders |

## Week-221 final pair (Sat ~1:30 PM)
| # | Decision | Evidence | Revisit |
|---|---|---|---|
| W1 | Metric reading: F and fill pooled; diversification order-weighted fits best (rmse 0.71 over 7 Kaggle scores) vs our row-mean reading (2.49); calm-window fill rule not identifiable (all files filled 100% in 209–221) | `reports/metric_check_p2.txt` | — |
| W2 | State at 221: KES tariff 0.35 on aluminum + semiconductors from 224 (announced, certain); VAL cobalt ban and KES rare-earths ban confirmed by futures ramps (11.6%, 9.3%; conversion 0.62 → 0.95); MERROW closure story at its early stage (conv 0.39; tension 25 → 41 → 54 like the real 173 closure) | `eda/29_state_report.py`, `ScriptModel.fit(d, market=True).explain(d)` | — |
| W3 | **Final pair: `lean_pxlp_mkt_w221.csv` + `insured_hedged_mktcol_w221.csv`.** On 48 simulated private windows: lean wins without the closure (74.1 vs 65.7), insured wins with it (79.8 vs 68.4; fill 1.00 vs 0.74); pair expected 76.4, best of all pairs. A "closure-real" lean variant was dominated by insured | `eda/30_scenario_pair.py` | scenarios assume our uplifts/durations; no unwarned shocks |
