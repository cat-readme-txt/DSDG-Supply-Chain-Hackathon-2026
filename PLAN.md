# DSDG Supply Chain Disruptor — technical decision plan

**Status:** planning and data analysis only. Choose one complete strategy below, then edit any individual decision you want to change. No production model, submission CSV, or application has been built.

**Source and confidence.** The attached *Participant Cheat Sheet (share with teams).docx* supplies competition rules. The ten CSV files in `data/` supply the measured data facts below. The [Kaggle competition page](https://www.kaggle.com/competitions/dsdg-hackathon-2026-supply-chain-disruptor) was inaccessible here on November 6, 2026; no starter notebook, simulator, or scoring code was provided. Historical backtests and cost/capacity replays below are **proxies**, not official scores or future outcomes. The world is fictional and permits only competition data; do not use outside datasets.

## Choose a complete starting strategy

**Working recommendation to edit: `Balanced`.** It improves both measured forecast baselines and cuts single-supplier exposure for a 2.9% historical-cost premium over the sample. I would compare `Resilience-first` as the second candidate because it handled the worst historical capacity shocks better. This is a recommendation for **what to test first**, not a claim that either will win the official score.

**Your strategy:** `Balanced` ← replace with `Cost-first`, `Resilience-first`, or `Custom` if preferred. Choosing a bundle sets D2–D6 below; section 2 lets you override any part without designing a plan from scratch.

| Bundle | What choosing it would produce for all 260 rows | Pros / measured upside | Cons / measurable tradeoff |
| --- | --- | --- | --- |
| **Cost-first** | Full-history disruption rate; last benchmark price; order exactly known demand; 100% from the recent cheapest eligible supplier. | Simple, lowest future **spend proxy: 12.792 million** cost units. Historical forecast proxies: Brier **0.1520**, price MAPE **6.55%**. | One supplier outage could remove 100% of that order. Worst unfilled demand in four capacity-only holdouts: **59.65%**. Closely resembles the sample that the cheat sheet says scores 0. |
| **Balanced — suggested first test** | 50/50 full-history/recent-52 disruption rate; last-four benchmark mean; order known demand; 60% recent cheapest + 40% second cheapest. | Historical forecast proxies: Brier **0.1472**, price MAPE **6.26%**. A full outage of the 60% source leaves 40% of the planned order with the other source. | Future spend proxy **13.165 million**, **+2.9%** versus sample. Worst unfilled demand in four capacity-only holdouts: **26.82%**. Still concentrated and has no advance inventory buffer. |
| **Resilience-first** | Same blended disruption and four-week-mean price forecasts; order known demand; one-third from each eligible supplier. | A full outage of one source leaves two-thirds of the planned order with the other two. Worst unfilled demand in four capacity-only holdouts: **4.92%**; it had the lowest cost-plus-shortage proxy in the two earliest holdouts. | Future spend proxy **13.965 million**, **+9.1%** versus sample. It spent more and lost the capacity-only proxy test in the quiet 157–182 holdout. No advance inventory buffer. |

The outage fractions assume a **full failure of one supplier with no inventory or substitutions**. The future spend estimate uses weeks 183–208 average landed costs and known weeks 209–234 demand. The unfilled-demand percentages come from four historical capacity-only replays that omit shipping lead time, starting inventory, in-transit orders, and holding cost. None is an official score or a promise about future weeks. The same forecast methods are used in Balanced and Resilience-first so their difference isolates supplier shares.

**Fast choice:** choose `Cost-first` if lowest planned spend is the priority; `Balanced` if a modest premium for diversification is acceptable; `Resilience-first` if avoiding a severe supplier shock matters more than nominal spend. If the actual simulator rules become available, revisit the order quantity and allocation before submission.

## 1. Objective and fixed constraints

Produce **one CSV with exactly 260 rows**: 10 materials × 26 weeks (209–234), one row per material and week. Begin from `data/sample_submission.csv`, which the cheat sheet says is valid and scores 0. The CSV must contain:

| Field | Required value |
| --- | --- |
| `id` | `w{week}_{material}`, such as `w209_lithium`; every expected ID exactly once. |
| `disruption_prob` | Probability in `[0, 1]` that at least one supplier of that material is disrupted that week. |
| `price_forecast` | Positive benchmark price for that week: average spot price across the three suppliers. |
| `order_units` | Nonnegative units ordered that week. |
| `share_KES` through `share_AUR` | Fractions of that order assigned to each country. Each row sums to 1; a country that cannot sell that material gets 0. Obtain the complete column list and eligibility map from the sample and competition data. |

The stated score is `100 × (0.30 × Prediction + 0.40 × Financial + 0.30 × Resilience)`. **If** each component is on a 0–1 scale, a 0.01 gain in Prediction, Financial, or Resilience adds 0.30, 0.40, or 0.30 total-score points respectively. This arithmetic does **not** tell us how the components themselves are calculated. The cheat sheet says the simulator plays out orders against actual outcomes and that a stockout costs five times the material price. Exact simulator timing, initial inventory/pipeline, holding cost, budget, and scoring formulas still need confirmation.

The public leaderboard covers weeks 209–221 (**130 rows**); the private leaderboard covers weeks 222–234 (**130 rows**). The private leaderboard determines 80% of final placement after Saturday noon; the video is judged for the other 20%. There are five submissions per team per day. Up to two final submissions may be selected by Sunday noon; otherwise Kaggle selects the two best public scores. Treat the public score as a limited check, especially after the private window becomes decisive.

## 2. Decision register

The `Suggested` column defines the **Balanced** bundle. Leave `Your override` as `—` to accept it, or write an option code from section 4. Selecting `Cost-first` changes D2/D3/D4/D5/D6 to `A1/A1/A/A/A`; selecting `Resilience-first` changes them to `A2/A2/A/B/C`. D1, D7, and D8 are independent of the strategy bundle.

| ID | Decision | Suggested | Your override | Result of switching from the suggestion |
| --- | --- | --- | --- | --- |
| D1 | Execution environment | **B: local Python** | — | A puts the run on Kaggle; C uses both. Local data is already available. |
| D2 | Disruption probability | **A2: blended rate** | — | A1 repeats sample probabilities (Brier 0.1520 vs 0.1472); B/C require new model tests. |
| D3 | Benchmark price | **A2: four-week mean** | — | A1 repeats sample prices (MAPE 6.55% vs 6.26%); simple linear trends tested worse, while seasonality and C remain untested. |
| D4 | Order quantity | **A: known demand** | — | B buys ahead by a rule; C optimizes scenarios. Their score impact is unknown without inventory and arrival rules. |
| D5 | Supplier shares | **C1: 60/40** | — | A lowers future spend proxy by about 2.9% versus C1 but concentrates risk; B raises it about 6.1% versus C1 and diversifies further; C2 is untested. |
| D6 | Risk objective | **B: balance average/downside** | — | A favors average result; C favors worst shocks. Neither can be optimized against the official score yet. |
| D7 | Databricks track | **B: focus main track** | — | A opts in before Saturday 6 PM and may add track-specific work; rules remain unread. |
| D8 | Final candidates | **A: best + different backup** | — | B pairs mean and downside winners; C submits only one if all alternatives are worse. Choose after valid candidates exist. |

**Working rule:** preserve the sample as a valid zero-score reference and test more complex choices on the same historical windows. The top table is a decision aid; a bundle is not a validated final submission.

## 3. Imported data and what it permits

All imported files have unique natural keys, and required numeric fields are populated. The only blanks found are `announced_rate` and `announced_effective_week` on 6,210 of 6,240 tariff rows; these are sparse announcement fields, not missing target labels. Historical week 208 is dated December 22, 2025; the known-demand horizon ends at week 234, dated June 22, 2026.

| File | Rows and coverage | Decision use |
| --- | --- | --- |
| `data/demand.csv` | 2,340 material-weeks, weeks 1–234 | **Future demand is supplied exactly** for all 260 submission rows; no demand forecast is needed. |
| `data/outcomes.csv` | 2,080 material-weeks, weeks 1–208 | Historical `disrupted` labels and benchmark prices for D2/D3 backtests. |
| `data/prices.csv` | 6,240 supplier-weeks, weeks 1–208 | Three spot prices and `futures_4w` per material-week. Historical benchmark equals the three-spot mean to rounding (maximum absolute difference 0.0000334). |
| `data/suppliers.csv` | 30 material-country pairs | Exactly three eligible suppliers per material across six countries; base lead times 1–4 weeks and normal capacities. |
| `data/supply_history.csv` | 6,240 supplier-weeks, weeks 1–208 | Historical capacity, lead time, and landed cost for policy comparisons. |
| `data/tariffs.csv` | 6,240 supplier-weeks, weeks 1–208 | Current tariffs plus 30 nonblank rows of advance tariff announcements. |
| `data/macro.csv` | 1,248 country-weeks, weeks 1–208 | Historical economic and tension signals; no future macro rows yet. |
| `data/shipping.csv` | 1,040 lane-weeks, weeks 1–208 | Historical container rate, transit time, and congestion for five lanes. |
| `data/news.csv` | 4,317 articles, weeks 1–208 | Historical text signals; no news for the forecast horizon yet. |
| `data/sample_submission.csv` | 260 IDs, weeks 209–234 | Exact header and row template; six share columns: KES, VAL, SUN, ORI, HAL, AUR. |

**Sample behavior measured from the files:** all 260 orders equal that material's known future demand for that week; every row gives 100% of the order to one eligible country; each material's probability is its weeks 1–208 disruption frequency; each material's price forecast is its week 208 benchmark price. The sample has no horizon-varying probability or price. All sample suppliers have a 3- or 4-week base lead time. These facts explain what a decision changes relative to the baseline; they do not reveal how the official simulator scores it.

**Still missing:** official simulator/scoring rules; starting inventory and in-transit orders; holding and budget rules; whether future demand and all historical signals are known when each order is placed; exact use of lead time, capacity, tariffs, and shocks; Databricks track requirements. Phase 2 file changes must be inspected when released. Because every signal file except demand ends at week 208, models for weeks 209–234 cannot use same-week future macro, news, shipping, supplier cost/capacity, or realized spot price unless Phase 2 supplies them.

## 4. Path outcomes and decision evidence

### D1 — Execution environment

| Path | What choosing it produces | Cost / risk | Evidence to choose it |
| --- | --- | --- | --- |
| **A. Kaggle notebook** | Direct access to competition data under `/kaggle/input/`; short route to a reproducible CSV and submission. | Runtime/package limits and notebook execution time may constrain experiments. No starter notebook was supplied. | A new notebook loads all files and writes a valid template-based CSV. |
| **B. Local Python workflow** | The ten CSVs are already local, allowing quick iteration and debugging. | Local package setup and an extra final Kaggle upload step. | Local validation and a Kaggle upload of the exact validated CSV. |
| **C. Hybrid** | Local backtests plus a minimal Kaggle final run. | Two environments to reconcile; more operational work. | Use if local iteration is materially faster and final Kaggle run is reproducible. |

### D2 — Disruption probability

| Path | Expected result if chosen | Data and measurement |
| --- | --- | --- |
| **A1. Full-history material rate** | Reproduces the sample's probabilities. Stable, but slow to respond to a new regime. | `data/outcomes.csv`; combined historical Brier 0.1520. |
| **A2. 50/50 full-history and last-52 rate** | Adjusts to recent regimes without the extreme 0/1 estimates of a 26-week window. | Same labels; combined historical Brier 0.1472. |
| **B. Regularized tabular classifier** with lagged material/supplier signals | May use predictive signals while remaining interpretable; can underfit nonlinear events. | Needs valid features known before each predicted week. Time-forward Brier score, calibration curve, and feature availability check. |
| **C. Boosted trees plus calibration** | May capture interactions and nonlinear warnings; higher tuning and leakage risk on a short history. | Same features as B, multiple historical folds, calibration on out-of-fold predictions. Select only if improvement persists across folds. |

Predict the **material-week event** “any supplier hit,” not an average of supplier probabilities unless the event relationship is explicitly modeled. Clip only for numeric safety; do not turn all forecasts into 0/1 labels.

**Measured 26-week forecast test.** Fit only through the stated origin, then hold each material's probability fixed for the next 26 weeks. These two nonoverlapping folds contain 520 material-week predictions total. Brier score is lower when better; no classifier has been trained or scored yet.

| Probability method | Weeks 157–182 | Weeks 183–208 | Combined |
| --- | ---: | ---: | ---: |
| Full-history material rate, as in sample | 0.2221 | 0.0819 | 0.1520 |
| Last 52-week material rate | 0.2427 | 0.0795 | 0.1611 |
| **50/50 full-history + last-52 rate** | **0.2148** | **0.0795** | **0.1472** |
| Last 26-week material rate | 0.2580 | 0.1379 | 0.1980 |

The two folds contain unusually clustered events: eight of ten materials are disrupted in every week 173–180, and rare earths remain disrupted through week 197. This makes a short recent window unstable; it also makes the 0.0048 combined Brier advantage of the blend over full history provisional rather than a guaranteed future gain. The recent 26 weeks contain zero disruptions for eight materials, yet that does not prove their future risk is zero.

### D3 — Benchmark price

| Path | Expected result if chosen | Data and measurement |
| --- | --- | --- |
| **A1. Last observed benchmark** | Reproduces the sample's price forecasts. Low variance, but can be noisy at the origin. | `data/outcomes.csv` and `data/prices.csv`; combined historical MAPE 6.55%. |
| **A2. Mean of last four benchmarks** | Smooths origin-week noise; can lag a real price jump. | Same data; combined historical MAPE 6.26%. |
| **B. Trend and seasonality model** | Can track smooth drift or periodicity; extrapolation can fail after a shock. | Needs sufficient history per material. Compare horizon-specific MAE and inspect forecasts for positivity. |
| **C. Lagged-feature regression / boosted trees** | Can use competition signals and interactions; may overfit or rely on future information. | Needs data with known-at-forecast-time features. Compare on identical rolling-origin folds; inspect large-error weeks. |

The target is the **average spot price across three suppliers**, not a chosen supplier's purchase price. Keep forecasts positive. Do not use the actual price in the week being predicted as an input.

**Measured 26-week forecast test.** Predict each material's next 26 prices from the origin without seeing later prices. Mean absolute percentage error (MAPE) gives each material-week comparable weight; two folds supply 520 predictions. A straight trend means extrapolating the fitted slope of the specified recent window.

| Price method | Weeks 157–182 MAPE | Weeks 183–208 MAPE | Combined MAPE |
| --- | ---: | ---: | ---: |
| Last benchmark, as in sample | 6.34% | 6.76% | 6.55% |
| **Mean of last 4 benchmarks** | 6.64% | **5.89%** | **6.26%** |
| Mean of last 13 benchmarks | **6.31%** | 6.85% | 6.58% |
| Linear trend from last 13 weeks | 13.48% | 7.93% | 10.70% |
| Linear trend from last 26 weeks | 11.51% | 10.06% | 10.79% |

The 4-week mean has a small combined advantage over last price (0.29 percentage points); the two folds disagree about which simple window is best. On a separate rolling test of 2,040 four-week-ahead material forecasts, the average `futures_4w` quote had 4.68% MAPE versus 4.66% for the last benchmark. These tests do not establish that a complex price model will help, and no official prediction score formula is available.

**What the simple forecast paths would output now.** These are week-208-origin values repeated over the horizon, not observed future outcomes. The blend is `0.5 × full-history rate + 0.5 × last-52 rate`.

| Material | Sample disruption probability | 50/50 blend probability | Last-26 probability | Sample price | Last-4 mean price |
| --- | ---: | ---: | ---: | ---: | ---: |
| aluminum | 0.226 | 0.209 | 0.077 | 3.0774 | 2.9813 |
| cobalt | 0.264 | 0.209 | 0.000 | 63.6995 | 61.2704 |
| copper | 0.038 | 0.019 | 0.000 | 11.1055 | 10.9954 |
| lithium | 0.207 | 0.180 | 0.000 | 47.5186 | 48.8084 |
| natural_gas | 0.053 | 0.094 | 0.000 | 4.3339 | 4.3465 |
| polymer_resin | 0.207 | 0.180 | 0.000 | 1.7455 | 1.7194 |
| rare_earths | 0.337 | 0.409 | 0.577 | 109.0300 | 111.6072 |
| semiconductors | 0.216 | 0.185 | 0.000 | 591.2295 | 599.7382 |
| specialty_glass | 0.087 | 0.120 | 0.000 | 7.4711 | 7.5245 |
| steel | 0.303 | 0.315 | 0.000 | 1.2711 | 1.2325 |

**Feature design if D2-B/C or D3-C is selected.** Build training examples at an origin week `t` for material `m` and horizon `h ∈ 1…26`; label them with `data/outcomes.csv` at week `t+h`. Join supplier data by `(week, material, country)` and aggregate to material-level features using only rows at or before `t`. Candidate features are trailing disruption rates and time since last event; benchmark price level and volatility; supplier capacity/normal-capacity ratio, lead-time excess, landed-cost spread and futures quotes; tariff announcements whose issue week is at or before `t`; country PMI/tension; lane congestion; and counts of news headlines associated with the material/country. Known future demand may be included only if competition rules confirm it is available at the decision origin. Encode material and horizon explicitly. With only 2,080 labeled material-weeks and strong cross-material event clustering, keep regularization strong, evaluate by whole future weeks, and calibrate probabilities on held-out origins. The more elaborate route remains a **proposed experiment**, not a model result.

### D4 — Order quantity

| Path | Expected result if chosen | Data and measurement |
| --- | --- | --- |
| **A. Sample or fixed coverage baseline** | The sample orders exactly one week of **known** demand in every row. This is a valid reference but creates no intentional buffer or early purchase. | Preserve the sample as the cheat-sheet zero-score anchor; test 1.0×, 1.1×, and other coverage levels only after inventory and arrival rules are known. |
| **B. Risk-adjusted reorder rule** | Uses known demand, projected inventory, lead time, and disruption risk to buy ahead or cap spend. It can reduce stockouts but may increase inventory cost. | Demand, base supplier lead times, capacities, and historical landed costs are available. Initial inventory/pipeline and simulator state transitions are still required for a credible backtest. |
| **C. Scenario-based optimizer** | Coordinates orders across weeks under forecast uncertainty; can outperform a rule if the transition rules are known and forecasts are calibrated. | Needs exact simulator/constraints and enough time to test. A mathematical optimum of a guessed simulator has no known competition value. |

The five-times-material stockout cost makes shortage prevention potentially valuable. Future demand is known, but the optimal buffer **cannot** be calculated without starting inventory, in-transit orders, timing, holding cost, and simulator rules. The sample places all orders with 3- or 4-week-base-lead suppliers; how early weeks are covered is therefore especially sensitive to the unknown starting state. Any numeric break-even threshold would be an assumption, not a competition fact.

**Known quantities for the 26 submission weeks.** Maximum weekly demand is the minimum capacity a single source would need to fill an entire week's order under normal conditions; this says nothing about arrival timing or disrupted capacity.

| Material | Mean future weekly demand | Maximum weekly demand | Sample source / normal weekly capacity / base lead | Fastest eligible base lead |
| --- | ---: | ---: | --- | ---: |
| aluminum | 17,057 | 19,621 | KES / 25,200 / 4 weeks | 1 week |
| cobalt | 608 | 653 | KES / 840 / 4 weeks | 3 weeks |
| copper | 5,276 | 5,640 | ORI / 7,000 / 3 weeks | 3 weeks |
| lithium | 1,230 | 1,346 | KES / 1,680 / 4 weeks | 3 weeks |
| natural_gas | 9,550 | 10,465 | ORI / 14,000 / 3 weeks | 1 week |
| polymer_resin | 27,351 | 29,105 | KES / 36,400 / 4 weeks | 1 week |
| rare_earths | 416 | 447 | KES / 560 / 4 weeks | 3 weeks |
| semiconductors | 92 | 101 | KES / 125 / 4 weeks | 3 weeks |
| specialty_glass | 6,667 | 7,219 | SUN / 9,800 / 4 weeks | 1 week |
| steel | 51,963 | 56,211 | KES / 70,000 / 4 weeks | 1 week |

### D5 — Supplier shares

| Path | Expected result if chosen | Data and measurement |
| --- | --- | --- |
| **A. Cheapest eligible supplier** | Lowest recent landed-cost proxy and 100% exposure to that supplier's shock. The sample is single-source on all 260 rows and picks the recent cheapest for nine of ten materials; natural gas uses the slightly costlier ORI supplier. | All three eligible suppliers and their historical landed costs/capacities are known. |
| **B. Equal shares across three eligible suppliers** | 1/3 allocation can survive one supplier failure in a static full-outage thought experiment, at a higher cost proxy. Correlated shocks and lead times can erase that advantage. | Historical costs/capacities permit the replay below; exact simulator cost and arrivals remain unknown. |
| **C1. 60/40 two-cheapest split** | Limits exposure to one supplier at a smaller cost premium than equal shares. | Measured cost and capacity-only replay below. |
| **C2. Dynamic risk/lead-time allocation** | Could shift to a faster or safer supplier when inventory is at risk, with a variable cost premium. | Needs forecast supplier risk, inventory state, and exact arrival rules; no measured outcome yet. |

For `order_units = 0`, the row still needs valid shares summing to 1. Use an eligible-country allocation even when no units are ordered. Never allocate to an ineligible country.

**Historical landed-cost comparison for the future demand horizon.** Apply the average landed cost from weeks 183–208 to the supplied demand for weeks 209–234. This is a **price and volume proxy**, not a future price forecast or simulator result. One cheapest supplier costs 12.792 million cost units; the sample's chosen suppliers cost 12.796 million; 60/40 with the two cheapest costs 13.165 million (**+2.9% versus sample**); equal shares cost 13.965 million (**+9.1% versus sample**). All three planned allocations fit each supplier's normal capacity and the minimum capacity observed in weeks 183–208 when applied to future weekly demand **before** any new disruption. In a full outage of only the largest-share supplier, with no inventory or substitution, the remaining planned order fraction is 0% for single source, 40% for 60/40, and 67% for equal shares. This stress example is deliberately simpler than the real simulator.

**Optional material-level D5 overrides.** These premiums compare each path with that material's lowest average landed cost in weeks 183–208, applied to known future demand. `—` inherits the selected bundle's D5 choice; write `A`, `B`, `C1`, or `C2` in the last column only if that material should differ.

| Material | Recent cheapest / second | C1 60/40 premium | B equal-thirds premium | D5 override |
| --- | --- | ---: | ---: | --- |
| aluminum | KES / HAL | 7.66% | 15.71% | — |
| cobalt | KES / ORI | 1.78% | 3.65% | — |
| copper | ORI / VAL | 0.45% | 6.15% | — |
| lithium | KES / ORI | 0.54% | 2.20% | — |
| natural_gas | VAL / ORI | 0.14% | 7.78% | — |
| polymer_resin | KES / SUN | 0.63% | 10.68% | — |
| rare_earths | KES / VAL | 4.21% | 10.26% | — |
| semiconductors | KES / SUN | 0.83% | 7.50% | — |
| specialty_glass | SUN / HAL | 6.50% | 13.80% | — |
| steel | KES / HAL | 7.47% | 15.43% | — |

**Capacity-only historical replay.** At each fold origin, rank suppliers by their preceding 26-week mean landed cost; hold the ranking for the next 26 weeks; order that week's demand. Planned spend uses that week's realized landed cost. Delivered units are `min(order share, realized capacity)` per supplier, with **no inventory, pipeline, or lead-time model**. The proxy adds `5 × actual benchmark price × unfilled units` to planned spend. It uses future realized values only to *score* each holdout week, never to choose its supplier ranking. The large changes across folds are the main result: diversification helps during some shocks and costs more during quiet periods.

| Holdout weeks | Allocation | Planned spend (millions) | Unfilled demand | Spend + five-times-price shortage proxy (millions) |
| --- | --- | ---: | ---: | ---: |
| 105–130 | One cheapest | 13.19 | 59.65% | 46.65 |
| 105–130 | 60/40 two cheapest | 13.41 | 26.82% | 28.44 |
| 105–130 | Equal thirds | 14.10 | **4.92%** | **16.85** |
| 131–156 | One cheapest | 12.28 | 7.60% | 20.18 |
| 131–156 | 60/40 two cheapest | 12.60 | 3.68% | 15.90 |
| 131–156 | Equal thirds | 13.32 | **1.02%** | **14.79** |
| 157–182 | **One cheapest** | **12.56** | **0.07%** | **12.72** |
| 157–182 | 60/40 two cheapest | 12.84 | 0.78% | 13.50 |
| 157–182 | Equal thirds | 13.48 | 0.64% | 14.03 |
| 183–208 | One cheapest | 12.64 | 0.30% | 14.92 |
| 183–208 | **60/40 two cheapest** | 12.96 | **0.00%** | **12.96** |
| 183–208 | Equal thirds | 13.74 | 0.00% | 13.74 |

The 157–182 fold shows why blind diversification is not enough: 60/40 and equal splits bought from natural-gas suppliers whose observed capacity was lower in that period, while the cheapest supplier stayed available. Country, lane, material, and lead-time exposure need to be considered together. Aurelia (`AUR`) has a one-week base lead for aluminum, steel, polymer resin, specialty glass, and natural gas; in the last-26-week landed-cost sample it costs about 23–30% more than each material's cheapest supplier. Its value as a fast reserve cannot be scored without inventory and arrival rules.

### D6 — Risk objective

| Path | Expected result if chosen | Evidence to choose it |
| --- | --- | --- |
| **A. Best mean replay score** | Favors average performance; may fail badly in rare disruptions. | Compare mean score and worst-window stockouts across rolling folds. |
| **B. Balanced mean and downside** | Gives up some average score for fewer severe failures. | Choose a downside weight only after viewing the score distribution and cost tradeoff. |
| **C. Conservative worst-case target** | Prioritizes survival under stressed scenarios; can overspend when shocks do not occur. | Use only if stress cases resemble observed competition patterns and financial score remains competitive. |

If the official simulator is unavailable, compare **proxies**, not claimed official scores: purchase spend; stockout units and their five-times-price penalty where definable; inventory held; and outcome under isolated supplier disruptions. Clearly label any assumed holding or lead-time rule.

For a replay with known unit purchase prices and stockout quantity, a transparent partial cost is `purchase spend + 5 × material price × stockout units`. Add holding, shipping, or other terms only if the competition defines them. This expression is a **comparison proxy**, not the published Financial component or a complete simulator.

The capacity-only replay in D5 has a different winner in different folds: equal thirds in 105–130 and 131–156; one cheapest in 157–182; 60/40 in 183–208. Thus none of the three replayed allocations is a measured universal winner. D6 should be selected from a stated tolerance for extra spend and missed demand, then checked against the official simulator if it becomes available.

### D7 — Databricks track

| Path | Result | Decision gate |
| --- | --- | --- |
| **A. Opt in** | Eligibility for a separate track, subject to its rules; may require additional work. | Track requirements and remaining time justify it; submit form before Saturday 6 PM Central. |
| **B. Skip** | Focus all time on the main CSV and video. | Track requirements are unclear or divert effort from final validation. |

### D8 — Final submission pair

| Path | Result | Decision gate |
| --- | --- | --- |
| **A. Best local mean + diversified backup** | Two distinct policies can hedge model uncertainty. | Both CSVs pass validation and have a documented reason to differ. |
| **B. Best local mean + best downside** | One candidate pursues average score, one pursues resilience. | Backtests show a real tradeoff and neither has a severe financial penalty. |
| **C. One strongest candidate only** | Simplest final choice; no second policy to hedge private-week uncertainty. | Other candidates are invalid or consistently worse. |

Select final submissions explicitly by Sunday noon. Do not rely on Kaggle's automatic choice of the best **public** scores when the private window decides most of the placement.

## 5. Evaluation protocol and comparison sheet

1. **Audit before fitting.** The current file coverage and keys are summarized in section 3. Before implementation, confirm when the competition makes each signal available and whether future demand is available at every decision time. Record any rule or file changes after Phase 2.
2. **Use forward-time folds.** The forecast tables above use train through week 156 → validate 157–182 and train through 182 → validate 183–208. The allocation proxy also uses earlier 105–130 and 131–156 folds. Add a 13-week fold to resemble the private horizon before making a final choice. Never randomly mix future weeks into training.
3. **Compare forecasts independently.** Use Brier score (`mean((p-y)^2)`) and calibration for disruption; MAE plus a percentage metric for price. Report by material, horizon, and fold, not only one aggregate. Determine the official prediction subscore formula before treating a proxy as authoritative. Test B/C only after proving their forecast-time features are available.
4. **Compare policies under identical forecasts and starting state.** The D5 replay isolates capacity and cost but omits inventory and lead times. If simulator code/rules become available, use them to compare Prediction, Financial, Resilience, and total score. Otherwise retain the proxy label and assumptions.
5. **Stress test.** Include no-shock, one-supplier shock, correlated supplier shock, price spike, and sustained disruption **only if the competition rules make each scenario possible**. Vary forecast error and record worst-window stockout and spend.
6. **Reserve leaderboard submissions.** Submit a valid baseline first if useful, then only candidates with a distinct hypothesis. Maintain a log so the five-per-day limit is not exhausted by format fixes.

Fill this sheet as full candidates are tested; blank cells are intentionally **unknown**, not zeros. The baseline forecast proxies are historical tests of its rules, while the 0 is the cheat sheet's statement about the submitted sample:

| Candidate | D2 | D3 | D4 | D5 | D6 | Brier ↓ | Price MAPE ↓ | Official replay total ↑ | Financial ↑ | Resilience ↑ | Worst-window stockout ↓ | Public score | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline rules / sample | A1 | A1 | A: demand | Sample single source | TBD | 0.1520 proxy | 6.55% proxy | **0 per cheat sheet** | TBD | TBD | TBD | TBD | Reference |
| Candidate 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Candidate 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Candidate 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Fair comparison rule:** keep validation weeks, inputs, forecast origin, and simulator state identical when changing one decision. A leaderboard difference is suggestive, not a substitute for these checks.

## 6. Planned workflow and gates

| Time (Central) | Work product / gate |
| --- | --- |
| **Friday, Nov 6, after 6:30 PM** | Phase 1 weeks 1–208 available. Audit data, rules, and sample; create a valid baseline CSV; record first schema and simulator findings. |
| **Friday night / Saturday morning** | Run forward-time forecast baselines, compare order/share policies, and log component scores or labeled proxies. Keep one working candidate reproducible. |
| **Saturday, Nov 7, noon** | Phase 2 weeks 1–221 released. Inspect what changed; update forecasts for weeks 222–234 using only information then available. Continue emitting all 260 required rows. |
| **Saturday, by 6 PM** | Resolve D7 and submit the Databricks opt-in form if selected. |
| **Saturday evening** | Choose two distinct, validated finalists if the evidence supports them. Prepare a concise explanation of signals, forecast-to-order logic, shock survival, and limitations. |
| **Sunday, Nov 8, before noon** | Rerun from a clean input state; validate CSV row IDs, types, ranges, eligible shares, and sums; make final submissions and select up to two on Kaggle. Upload the approximately three-minute video to the team folder. |

The video should cover the four topics in the cheat sheet: signals found, how forecasts become orders, how the plan survives shocks, and what the approach misses. Use actual backtest numbers and examples once available; do not claim unmeasured resilience.

## 7. Required checks for every candidate CSV

- Exact ID set from `data/sample_submission.csv`; preserve its row order for convenience. Require 260 unique rows, weeks 209–234, and 10 materials per week.
- Exact required columns; numeric finite probabilities, prices, units, and shares.
- `0 ≤ disruption_prob ≤ 1`, `price_forecast > 0`, `order_units ≥ 0`.
- Each share is in `[0, 1]`, each row's shares sum to 1 within the accepted numeric tolerance, and all unavailable country shares are 0.
- No NaN, infinity, duplicate ID, missing row, unexpected country allocation, or accidental index column.
- The CSV submitted is the exact file whose checks and experiment results were recorded.

## 8. Open findings to fill in

| Finding | Value / source | Consequence for decisions |
| --- | --- | --- |
| Exact files and schemas | Ten CSVs audited in section 3; no duplicate natural keys found. | D1–D5 |
| Simulator and scoring formulas | TBD | D4–D6, D8 |
| Material / country eligibility and capacities | Three suppliers per material in `data/suppliers.csv`; historical capacities in `data/supply_history.csv`; future realized capacities unknown. | D5 |
| Lead times, demand, inventory, holding cost, budget | Future demand and base lead times known; initial inventory/pipeline, timing, holding, budget, and future arrivals unknown. | D4, D6 |
| Available forecast-time features and labels | Outcomes and signals end at week 208; demand reaches week 234. Exact decision-time availability needs rule confirmation. | D2, D3 |
| Phase 2 differences | TBD | Final retraining and private forecast |
| Databricks track requirements | TBD | D7 |

**Competition links from the cheat sheet:** [Kaggle invite](https://www.kaggle.com/t/8d5cc0ce805c4eda8db5d328011767c6), [competition page](https://www.kaggle.com/competitions/dsdg-hackathon-2026-supply-chain-disruptor), [Databricks opt-in](https://docs.google.com/forms/d/e/1FAIpQLSfst7f3Ay_CILsIJjm2fznPy41CplIDS_nBd-wCVVlPrsghow/viewform). The team video folder is shared with the team's Google email; search Drive for the team name.
