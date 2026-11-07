# Hackathon implementation progress

Times below are Central (America/Chicago). This file records completed work, measured results, and remaining dependencies. Historical replays are explicitly labeled as proxies; they are not Kaggle scores.

## Friday, November 6, 2026 — 7:21 PM

**Scope confirmed:** Implement the approved Balanced strategy as the primary candidate and Resilience-first as a distinct backup, aiming for the highest final placement. Use the ten competition CSVs only. Consolidate final decisions in `PLAN.md` and produce reproducible local candidate CSVs tonight. No Kaggle submission has been made.

**Existing evidence from data audit:** Ten CSVs are present in `data/`. Historical labels and prices cover weeks 1–208; demand covers weeks 1–234; the template has 260 IDs. The sample orders known demand, uses a single supplier, and fixes each material's probability and price across the horizon. Historical two-fold forecast proxies: blended probability Brier 0.1472 versus full-history 0.1520; four-week price mean MAPE 6.26% versus last price 6.55%. These figures will be reproduced by the implementation.

**Next work tonight:** finalize decisions; implement data/schema checks and backtests; generate Balanced, Resilience-first, and cost reference CSVs; validate every output against the sample template; record exact commands and measured results here.

**External dependency:** Exact simulator and evaluation rules, including initial inventory and order-arrival mechanics, were requested from the user. Those are needed before treating an inventory buffer or official Financial/Resilience score as validated.

## Friday, November 6, 2026 — 7:22 PM

**Plan finalized:** `PLAN.md` now consolidates D1–D8 as confirmed first-build decisions. Primary: local Python; 50/50 historical/recent-52 disruption probabilities; four-week mean benchmark price; known-demand order quantity; 60/40 cheapest/second-cheapest supplier allocation; mean/downside comparison. Backup: equal thirds. Databricks work is outside tonight's critical path. Changes to these choices require matched backtest evidence or newly supplied official rules.

**Next:** build a dependency-free pipeline so the CSVs are reproducible in the current Python 3.15 environment, then run schema checks and backtests. No candidate has been generated yet.

## Friday, November 6, 2026 — 7:24 PM

**Official evaluation details received from user:** P is 60% Brier skill relative to historical-rate baseline plus 40% price MAPE skill relative to last-price baseline. F compares our simulator cost with naive and hindsight costs. R is 70% shock-week fill improvement and 30% supplier diversification, where equal thirds earns maximum diversification. Each 13-week public/private window is simulated independently, starts with five weeks of stock and no orders in transit, and uses per-supplier capacity, landed cost, and lead time. Holding costs 2% of average landed cost per unit per week; shortages cost 5×; ending stock and transit have 70% salvage. Each shock week includes four recovery weeks for the fill metric.

**Decision impact:** The original 26-week capacity-only replay omits key economics and is no longer sufficient to select D4/D5. The first-build candidate remains a reproducible baseline, but the priority is now a 13-week simulator-grounded order and supplier policy comparison. The finalized plan will be amended with measured results before a final policy is chosen. Exact conventions for initial five-week stock and component normalization are still not stated in the pasted rules; the local replay will label its assumptions.

## Friday, November 6, 2026 — 7:25 PM

**13-week forecast check:** On twelve nonoverlapping 13-week holdouts (1,560 material-week predictions), the previously selected 50/50 historical/recent-52 probability blend scored Brier **0.2005**, worse than the official-style historical-rate baseline **0.1986**. The four-week mean price forecast scored MAPE **5.5922%**, worse than last-price **5.3681%**. These supersede the earlier two 26-week-fold comparisons for model selection because each official leaderboard window is 13 weeks.

**New candidate tested without future leakage:** A disruption-state decay forecast, `historical_rate + (last_disrupted - historical_rate) × 0.8^h`, improves Brier across both the full and recent 13-week folds (grid tests indicate approximately 0.15 overall versus 0.1986 baseline). A conservative price blend, `last_price + 0.25 × (last_four_mean - last_price)`, has MAPE **5.341%** overall versus 5.368% baseline. These will be reproduced exactly in the code before adoption. The stronger official-score opportunity is the simulator-grounded order policy.

## Friday, November 6, 2026 — 7:32 PM

**Pipeline implemented:** `pipeline.py` uses only Python 3.15's standard library. It audits the ten CSVs, verifies all 260 IDs and supplier eligibility, checks 2,080 historical benchmark prices against three-spot averages, runs 13-week forecast backtests, generates candidate CSVs, and validates their output. `python3 pipeline.py all` completed after fixing a JSON reporting issue on folds with no shock weeks. Three initial candidates were generated and validated in `submissions/`; an initial report is in `reports/backtest_w208.json`.

**Reproduced forecast results (12 disjoint 13-week folds, 1,560 material-week predictions):** full-history Brier **0.198628**; old 50/50 blend **0.200500**; new state-decay Brier **0.150651**, a **24.15%** skill improvement relative to the baseline. Last-price MAPE **5.3681%**; old four-week mean **5.5922%**; new 25% blend MAPE **5.3407%**, a **0.51%** relative skill gain. The official P formula would credit the Brier improvement far more than the small price gain.

**Local simulator assumptions:** five weeks of starting stock means the sum of the first five demands; orders arrive in `order_week + observed_lead_weeks`; holding is charged on inventory after demand; the average landed cost is the mean of three supplier costs in that week; ending stock and transit are salvaged at 70% of the window mean cost. These conventions are plausible readings of the page, not confirmed implementation details.

**Initial simulator proxy over 12 historical windows:** one-cheapest demand-order reference cost **78.187M** cost units, total fill **91.36%**, shock-week fill **76.85%**; 60/40 demand-order cost **64.186M**, fill **97.25%**, shock fill **92.33%**; equal-thirds demand-order cost **62.019M**, fill **99.91%**, shock fill **99.74%**. The equal split also receives the maximum stated diversification component. This makes the original 60/40 allocation an inferior first-build choice under the newly provided mechanics.

**Policy search, 360 combinations on eight training windows and four later holdouts:** the strongest low-cost policy with at least 99% training shock fill used equal shares, ordered 1.2× supplied weekly demand for weeks 1–8 of each 13-week window, and ordered zero for weeks 9–13. Training proxy cost **37.258M**, shock fill **99.23%**; holdout cost **19.291M**, shock fill **99.999%**. This is compared with equal-thirds ordering every week: training cost **40.776M**, holdout **21.243M**. The result is robust to an alternative initial-stock convention of five times the first week's demand: candidate total cost **56.564M**, shock fill **99.46%**, versus **56.548M** and **99.48%** under the first convention. The search will be added to reproducible code and rerun before the final candidate is named.

## Friday, November 6, 2026 — 7:36 PM

**Final plan rewritten:** `PLAN.md` now consolidates the evidence-driven choices after the user supplied the official evaluation rules. Primary is equal-third shares with 1.2× known demand ordered in the first eight weeks of each independent 13-week window, then zero orders for five weeks. Backup multipliers are 1.1× and 1.3×. Forecasts use 0.8 weekly disruption-state decay and a 25% adjustment from last price toward its recent four-week mean. The old 60/40 plan is retained only as a comparison in `reports/backtest_w208.json`.

**Reproducible search and validation:** `PYTHONDONTWRITEBYTECODE=1 python3 pipeline.py all` completed. The saved `reports/policy_search_w208.json` evaluates **360** policies and identifies the primary as the lowest training cost among policies with at least **99%** training shock-week fill. `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v` passed **4/4** contract tests. `git diff --check` passed. Each candidate CSV has **260 data rows plus one header** and passed the pipeline's ID, numeric-range, share-sum, and eligibility checks.

| Local candidate | SHA-256 | All-window cost proxy | All-window shock fill | Use |
| --- | --- | ---: | ---: | --- |
| `submissions/primary_w208.csv` | `0a4b2b0061a79018f08a8a33092a7b56c67c7fea49670a258bf9a2adc933e46f` | 56.548M | 99.48% | First Kaggle feedback candidate. |
| `submissions/cost_lean_w208.csv` | `26ecd5c53f9352dff5b4371e1b054df0bcae567020f58cc4e8a76551aa1277e3` | 56.010M | 98.61% | Lower-cost backup. |
| `submissions/shock_hedge_w208.csv` | `8348f715725e9b0e9bf57967ed297e3befbaa86d8374294f51cba795a7583737` | 58.079M | 99.75% | Higher-buffer backup. |

**External feedback requested:** The user was asked to upload only `primary_w208.csv` and share its public score/component breakdown. The Kaggle CLI is installed but no Kaggle token file is present, and the private competition page is inaccessible here. No external submission has been made. The archived first-pass CSVs under `submissions/archive/` are superseded and should not be uploaded.

## Friday, November 6, 2026 — 7:44 PM

**News warning experiment:** A two-week lookback over competition headlines was joined to each material through material names, eligible supplier countries, and their shipping lanes. Warning phrases were weighted by recency, negated stories were ignored, the warning score was capped at 1, and the score was added to the state-decay forecast with a 0.6 multiplier and 0.95 weekly decay. Every historical fold used only news available by its origin. On the twelve 13-week folds, Brier improved from **0.150651** (state-only) to **0.134981** (news-aware), and from **0.13851** to **0.13307** on the four later folds. Relative to the historical-rate baseline **0.198628**, Brier skill is **32.04%**. Nine of ten materials improved or tied in material-level Brier; natural gas worsened slightly (**0.0413 → 0.0431**). Rumors may fail in the unseen weeks, so the state-only CSV remains a control.

**New local candidate:** `submissions/news_primary_w208.csv` has the same order quantities, supplier shares, and price forecasts as `primary_w208.csv`; only 104 disruption probabilities changed. It has **260 rows**, passed validation and all four existing contract tests, and its SHA-256 is `9683650ddd2477dc38fff05cbd7b70540884087e2bf9e239c892db710d003d5b`. For example, week-209 lithium risk changes from **0.041346** to **0.587779** because of recent relevant warnings. This is a forecast, not a hidden-label claim. If no file has been uploaded yet, this is now the preferred first Kaggle feedback candidate; the earlier request to upload state-only `primary_w208.csv` remains useful only as a control if already submitted.

**Plan amended:** `PLAN.md` names the news-aware forecast as the preferred first experiment and retains the state-only file as a separate control. The order policy and simulated economics remain unchanged. No Kaggle submission has been made from this workspace.

## Friday, November 6, 2026 — 7:45 PM

**First real Kaggle feedback:** The user submitted the state-only `submissions/primary_w208.csv` and reported a **public score of 44.95**. This is the first official score; the local simulator's cost/fill figures are still proxies. At least one of today's five team submission slots has been used; the exact team-wide count should be checked on Kaggle before further uploads.

**Next isolated test requested:** Upload `submissions/news_primary_w208.csv` (SHA-256 `9683650ddd2477dc38fff05cbd7b70540884087e2bf9e239c892db710d003d5b`) and report its public score. Orders, shares, and price forecasts are byte-for-byte equal at the row-field level; only 104 disruption probabilities differ. If visible, request the P/F/R breakdown to learn which component limits the score. No Kaggle upload was performed by the local pipeline.

## Friday, November 6, 2026 — 7:48 PM

**Submission clarification and quota:** The user showed two successful Kaggle entries named `primary_w208.csv`, both scoring **44.95**. The second was a duplicate after an unclear prompt from the assistant; no news-aware candidate has been uploaded. The screenshot establishes **at least two** of the five daily slots used, assuming no other team submissions. The user was asked to pause further uploads until the remaining experiment sequence is clear. The intended comparison file is **`news_primary_w208.csv`**, not another copy of `primary_w208.csv`.

**Databricks opt-in steering:** The user says they selected Delta tables, Spark/SQL feature engineering, MLflow tracking, AI functions/LLMs on news, and dashboards on the opt-in form. None is implemented in the current local Python pipeline, so those selections are **not yet demonstrated use**. The official track weighs leaderboard rank 70% and Databricks platform use 30%; the plan's earlier skip decision is superseded by the user's opt-in intent. Form submission status, workspace access, and the Code-tab starter notebook were requested. The next implementation should create a verifiable Databricks run for features actually used and avoid claiming unsupported ones.

## Friday, November 6, 2026 — 7:53 PM

**Duplicate score resolved:** The screenshot confirms that both 44.95 entries are the same `primary_w208.csv`, so the second upload supplied no new model comparison. The user was told not to upload that file again. `news_primary_w208.csv` remains unsubmitted. At most three Friday slots remain if the team has made no other submissions; this has not been checked in Kaggle.

**Additional policy experiment:** A 64-variant risk-scaled multiplier search (material multiplier `1 + slope × historical disruption rate`) found an eligible best with equal shares, no start skip, four-week end stop, and slope 0.6. Its training proxy cost was **37.065M** versus primary **37.258M**, but later holdout cost was **19.530M** versus primary **19.291M**. It did not improve the full historical result, so the primary policy stays unchanged. These are local simulator costs, not official F.

**Databricks implementation prepared:** Added `databricks/track_notebook.py`, `databricks/dashboard_queries.sql`, and `databricks/README.md`. The notebook ingests all ten CSVs as Delta tables, builds released-week rolling price/disruption/news features with Spark SQL, runs the same `pipeline.py` backtests and CSV generator, logs metrics and candidates to MLflow, and creates dashboard source tables. Its optional `ai_query` cell labels up to 20 recent released news headlines in a review table; it does not silently alter a submission. The AI endpoint and dashboard require a real workspace run. `python3 -m py_compile databricks/track_notebook.py` and `git diff --check` passed; **no Databricks cell has run** here, so none of the selected features is yet verified as used. The plan now records the user's track choice and the run evidence required.

## Friday, November 6, 2026 — 7:56 PM

**Opt-in status confirmed:** The user says the Databricks form has been submitted and is editable. They expect access to Databricks Free Edition, but no starter notebook was provided. Notebook creation on Kaggle is available; that is a separate environment from the Databricks workspace. The selected form answers are appropriate targets; the remaining task is to run and document them in Databricks.

**Substantial Databricks migration added:** The notebook now rebuilds the preferred 260-row submission with Spark SQL using historical disruption rates, recent benchmark prices, 26-week supplier cost ranking, known demand, and the same released-news warning scores as `pipeline.py`. It writes `databricks_news_primary_w208.csv`, validates submission contracts, and requires exact field parity with the independent local generator before logging the native CSV in MLflow or exposing it in the dashboard table. This is a prepared execution path, not a cloud run. Phase 2 updates both private-window origin and output suffix automatically when the released files extend to week 221.

**Static and local verification:** The Databricks source notebook parses as Python. The five local pipeline contract tests passed in **0.586 s**. `git diff --check` passed. The original uploaded candidate still hashes to `0a4b2b0061a79018f08a8a33092a7b56c67c7fea49670a258bf9a2adc933e46f`; the unsubmitted news candidate still hashes to `9683650ddd2477dc38fff05cbd7b70540884087e2bf9e239c892db710d003d5b`. Spark/Delta/MLflow/AI/dashboard behavior cannot be validated until the notebook runs in the team's workspace.

**AI cell revised:** The default path now uses Databricks `ai_classify` on up to 20 recent released headlines and logs the actual count to MLflow. An available `ai_query` endpoint remains optional. If neither can run in the workspace, the cell prints its error and leaves the submission unchanged. AI labels are review evidence only until a proper backtest shows they help.
