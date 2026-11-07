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
| ours `policy_v1_w208.csv` | **54.5** | same window: ours 49.8 (+6.1 vs Kaggle +9.5) → the local scorer tracks Kaggle |

Risk spotted: in window 170–182 (an unwarned Merrow closure) policy_v1 scored 26.7 against the teammate's 62.7. The lean, cheap
sourcing is exposed to surprise lane shocks. The next version must be robust to these, for the private window.

## Teammate submission
| # | Note |
|---|---|
| T1 | `submissions/teammate/primary_w208.csv` is valid and scored 44.95 on Kaggle public. It's reproduced exactly by the `teammate_v1` strategy (backtest ~34 public / ~38 private). Its ordering rules (even split, 1.2× buffer, no orders in the last 5 weeks) are a baseline for step 3. |

## Open items
- Data-quality stage (silver layer) and EDA-as-pipeline stage.
- Phase 2 (Sat 12 PM CT): re-download, rerun the validations, check E2 against 44.95, and retune for the private window.
- Databricks notebook; the Databricks track opt-in closes Sat 6 PM CT.
