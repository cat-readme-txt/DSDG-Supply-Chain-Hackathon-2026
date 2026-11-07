# DSDG Supply Chain Disruptor — final implementation plan

**Goal:** maximize final placement. The private leaderboard supplies 80% of final placement; judges score the methodology video for the remaining 20% if the team reaches the top eight. This document consolidates the chosen policy and remaining gates. [PROGRESS.md](PROGRESS.md) records actual work, commands, and results.

**Sources:** the participant cheat sheet, the competition overview/evaluation text supplied by the user on November 6, and the ten CSVs in `data/`. The competition page is private from this workspace. The local simulator follows the supplied mechanics with explicit assumptions below; it is **not** the official scorer. No hidden future labels or outside world data are used.

## Confirmed decisions

| Decision | Chosen implementation | Why it is the current first-place attempt |
| --- | --- | --- |
| Environment | Dependency-free local Python pipeline in `pipeline.py`. | The competition CSVs are local; the pipeline can be rerun after Saturday's Phase 2 release and can produce exact upload files. |
| Disruption probability | **News-aware state decay:** start with `p₀(m,h) = r_m + (y_{m,t} - r_m) × 0.8^h`; add `(1-p₀) × 0.6 × warning_score(m,t) × 0.95^h`, then clip to [0,1]. The warning score uses only the last two released weeks of relevant competition headlines and is capped at 1. | On twelve disjoint 13-week historical folds, Brier **0.134981** versus **0.198628** for the historical-rate baseline (**32.04% Brier skill**) and **0.150651** for state decay alone. Later holdout Brier also improved from **0.13851** to **0.13307**. The state-only file remains a control for false rumors. |
| Benchmark price | `last benchmark + 0.25 × (last-four-week mean - last benchmark)`, positive and fixed from each window's forecast origin. | Historical MAPE **5.3407%** versus **5.3681%** for last-price baseline; **0.51% price skill**. The pure four-week mean was worse on the 13-week test. |
| Primary order quantity | In **each independent 13-week window**, order `round(1.2 × supplied demand_units)` in weeks 1–8; order **0** in weeks 9–13. | The windows each start with five weeks of stock, and late orders lose value at 70% salvage. Among 360 searched policies, this had the lowest training cost subject to at least 99% training shock-week fill. |
| Primary supplier allocation | Equal thirds among the material's three eligible countries (`0.333333`, `0.333333`, `0.333334`). | Maximum diversification subscore; strongest historical shock protection and lower local total cost than the earlier 60/40 policy. |
| Backup policies | `primary_w208.csv` uses the same orders/shares with state-only probabilities; `cost_lean` uses **1.1×** demand; `shock_hedge` uses **1.3×** demand. | The first control isolates the value of news; the other two give a financial versus shortage tradeoff while preserving maximum diversification. Generate all four locally, then select up to two final submissions using the most relevant evidence available by Sunday noon. |
| Main track | Focus on the main competition; do not spend tonight's critical path on the optional Databricks track. | Its separate award has the same leaderboard plus platform-use judging. Main first place and the required video are the priorities. |
| Video | Build a three-minute explanation from measured signals, forecast-to-order logic, shock response, and limitations. | Only top-eight private leaderboard teams receive judge review, so clear evidence can affect 20% of final placement. |

**Evidence can change a decision:** Phase 2 data, an official leaderboard result, or a corrected interpretation of the simulator may justify a new candidate. Record the old/new choice, matched comparison, and exact file hash in `PROGRESS.md` before uploading. The currently generated files are local candidates, not submitted entries.

## Official scoring and simulator mechanics

`Score = 100 × (0.30 P + 0.40 F + 0.30 R)`, with each part clipped to [0,1] and scaled so the sample submission scores 0.

- **Prediction P:** 60% Brier skill for `disruption_prob` relative to each material's historical-rate baseline; 40% price MAPE skill relative to carrying the last known benchmark forward. A negative skill component may be clipped by the official implementation; only positive improvements are treated as evidence.
- **Financial F:** `(naive cost - our cost) / (naive cost - hindsight cost)`. Hindsight cost is unavailable, so the local tool reports cost and savings versus a comparable naive reference, **not official F**.
- **Resilience R:** 70% improvement in demand fill during disrupted material-weeks plus four recovery weeks, relative to naive; 30% diversification. Diversification uses `1 - Σ share²`, rescaled so a single source is 0 and equal thirds is 1. The exact fill-skill normalization is not supplied, so local fill rates are **not official R**.
- The public weeks **209–221** and private weeks **222–234** are simulated separately. Each starts with five weeks of demand in stock and no orders in transit.
- Each order is split by submitted shares; a supplier fills at most its capacity in the order week, charges that week's landed cost for filled units, and delivers after that week's lead time. Unfilled units are cancelled without charge. Arrivals replenish stock; then that week's demand is used.
- Holding costs **2%** of a material's average landed cost per unit per week. Unmet demand costs **5×** average landed cost. At the end of each window, stock and in-transit units are sold back at **70%** of average landed cost.

**Local simulator conventions requiring caution:** starting stock is the **sum of the first five weekly demands** per material; an alternative `5 × first-week demand` convention was tested and gave nearly identical policy ranking. Orders arrive at the start of `order_week + observed_lead_time`; holding is charged on post-demand inventory; the average landed cost for holding/shortage is the week's three-supplier mean; salvage uses the window mean. The official implementation could differ in timing or averaging. Recheck with an organizer or leaderboard feedback when possible.

## Historical evidence behind the policy

Forecast comparison uses **twelve nonoverlapping 13-week folds**, 1,560 material-week predictions; each fold is trained only on prior weeks. News warnings use only headlines published by each fold's origin. The state-only forecast scored Brier **0.150651**; the news-aware forecast scored **0.134981**. These keywords and weights are a simple historical-data heuristic, so the state-only control protects against the possibility of false rumors in unseen weeks. The 360-policy search trained on the first eight windows (**weeks 53–156**) and reserved the last four (**157–208**) as a later holdout. The search grid covered five supplier splits, three start skips, four end stops, and six order multipliers. The choice required at least **99% training shock-week fill** before minimizing training cost.

| Policy in local 13-week simulator | Training cost, 8 windows | Later holdout cost, 4 windows | All-window shock fill | Diversification |
| --- | ---: | ---: | ---: | ---: |
| One cheapest, order demand every week (reference) | 55.814M | 22.374M | 76.85% | 0 |
| 60/40, order demand every week (old plan) | 43.283M | 20.903M | 92.33% | 0.72 |
| Equal thirds, order demand every week | 40.776M | 21.243M | 99.74% | 1 |
| **Primary: equal thirds, 1.2×, stop last five weeks** | **37.258M** | **19.291M** | **99.48%** | **1** |
| Cost lean: equal thirds, 1.1×, stop last five weeks | 36.968M | 19.042M | 98.61% | 1 |
| Shock hedge: equal thirds, 1.3×, stop last five weeks | 38.217M | 19.862M | 99.75% | 1 |

Costs are in **millions of Aurelian dollars under local assumptions**. The primary saved **21.639M (27.7%)** versus the comparable reference across all twelve windows, with **99.78% overall demand fill** and **99.48% shock-week fill**. The cost-lean candidate saved another **0.538M** but lost **0.87 percentage points** of shock-week fill; the shock hedge cost **1.531M** more than primary for **0.27 points** more shock fill. These are policy tradeoffs, not official F/R scores. The initial-stock sensitivity check changed primary's total local cost from **56.548M** to **56.564M** and shock fill from **99.48%** to **99.46%**.

The original two 26-week-fold forecast figures in the draft plan are superseded for selection by these 13-week results. Results can change after Phase 2 and on private weeks.

## Submission contracts and data boundaries

- `data/sample_submission.csv` supplies the exact header and **260 IDs**: ten materials for each week 209–234. Every generated file preserves that row order and is checked for a complete, unique ID set.
- `disruption_prob` must be finite in [0,1]; `price_forecast` finite and >0; `order_units` finite and >=0; all six country shares finite in [0,1] and summing to 1. Ineligible country shares must be 0, including rows whose order is 0.
- `data/outcomes.csv` and other historical signals currently end at week **208**. `data/demand.csv` gives demand through **234**. `data/suppliers.csv` has three eligible countries per material. The code does not infer hidden weeks or use future realized price, capacity, or disruptions.
- For Phase 1, both windows forecast from week 208. After Saturday noon's Phase 2 release, the public window stays anchored at week 208; the private window uses newly released history through week 221 if present. The full 260-row CSV remains required.
- The user-supplied rules allow general tools and external data, but this fictional world's provided files are the only data used. No hand-labeling or reverse engineering hidden outcomes is part of this plan.

## Execution and deadlines (Central)

| When | Required action |
| --- | --- |
| **Friday night, Nov 6** | Finish and validate local pipeline, 13-week simulator, policy search, and three candidate CSVs; record exact results and hashes in `PROGRESS.md`. If a Kaggle submission is made, log its actual score and remaining daily quota. |
| **Saturday noon, Nov 7** | Diff Phase 2 files against Phase 1; rerun `pipeline.py all`; focus decisions on private weeks 222–234. The public leaderboard is no longer a useful unbiased private-score estimate. |
| **Saturday afternoon/evening** | Refine only when matched historical tests or a rules-grounded leaderboard result supports the change. Prepare the three-minute video using actual numbers and a clear limitation statement. |
| **Sunday before noon, Nov 8** | Validate exact final CSV files; make and select up to two final Kaggle submissions; upload the video to the team's shared folder. Do not rely on automatic selection of the two best public scores. |

**Reproduce locally:** from this directory, run `PYTHONDONTWRITEBYTECODE=1 python3 pipeline.py all`, then `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`. Validate a specific file with `python3 pipeline.py validate submissions/news_primary_w208.csv`. `pipeline.py all` writes `reports/backtest_w208.json`, `reports/policy_search_w208.json`, and validated CSVs under `submissions/`. The preferred first feedback file is `submissions/news_primary_w208.csv`; `primary_w208.csv` is the state-only control. The `_w208` suffix changes to `_w221` if Phase 2 labels extend to week 221.

**Open gate:** Kaggle's official simulator implementation and hindsight cost are unavailable; local replay is an approximation. Public/private leaderboard scores, team registration/merge state, submission count, and video upload status must be recorded from the team's Kaggle/Drive access. No Kaggle submission or video upload has been performed by this local pipeline.
