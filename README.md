# DSDG Supply Chain Disruptor: submission and code map

This is the **single source-to-submission index** for this project. The 16
files named `submissions/submission<score>.csv` are exact CSVs that received
those **public** scores. A public score covers weeks 209–221 only. Final
placement depends chiefly on the hidden private weeks 222–234, so the highest
public file is not automatically the best final file. The current private
candidate pair is in `submissions/candidates/`; neither file has a known
private score. The complete Python-file inventory is at the end of this README.

## Data phases and information boundary

| Phase | Frozen input | Information available to a newly built forecast | Graded rows |
| --- | --- | --- | --- |
| Phase 1, origin 208 | `data/snapshots/w208/` | Prices, outcomes, news, tariffs, shipping, macro, and supply through week 208; demand through 234 | Public 209–221 |
| Phase 2, origin 221 | `data/Phase 2 data/` | Those same observed tables through week 221; demand through 234 | Private 222–234; released public 209–221 is a calibration set |

The simulator starts each 13-week window with the sum of its first five
weeks' demand in stock and no inbound orders. Orders are constrained by each
supplier's capacity, lead time, and landed cost. Holding costs 2% weekly,
unmet demand costs five times average landed cost, and ending stock and
in-transit orders salvage at 70%. The total score weights Prediction 30%,
Financial 40%, and Resilience 30%; each component is clipped to [0, 1]
relative to baselines. Prediction combines Brier skill (60%) and price MAPE
skill (40%). Resilience combines shock/recovery fill (70%) and supplier
diversification (30%). Local scorers approximate the competition's hidden
implementation and must never be called official scores.

For retrospective checks, each timed table is cut at the historical origin;
only demand remains visible beyond it. The Phase 2 public hindsight files
intentionally use **released** 209–221 truth in their public half. Their
private half uses week-221 information only. This distinction is why a
92.62 public score cannot establish private improvement.

## Final private pair and why there are two files

| File | Purchasing policy | Forecast position | Best local use |
| --- | --- | --- | --- |
| [Conditional lean](submissions/candidates/private_closure_conditional_lean_w221.csv) | SK4 stochastic LP with a price path, 30 scenarios, 96% capacity planning, near-cheapest secondary suppliers, and minimum shares for steel/copper/aluminum | Calm hedge on unwarned risks; for the open MERROW closure, use the no-closure branch down to the independent-risk floor | Usually cheaper when the closure does not occur |
| [Conditional insured](submissions/candidates/private_closure_conditional_insured_w221.csv) | SK4 insured rule policy with a tapering stock buffer, risk-weighted supplier allocation, and more orders | Shock hedge on background risks and prices; for MERROW, use the closure-occurs branch | Protects fill and cost when the closure or another large shock hits |

The two files exist because their **orders differ**, not merely because their
probabilities differ. The lean file buys about 947,590 private units; the
insured file buys about 1,073,023. The insured buffer raises cost in calm
worlds but can prevent severe five-times-cost shortages. In the SK4 48-scenario
test, the members' separate mean scores were:

| Scenario branch | SK4 lean | SK4 insured | Conditional lean | Conditional insured |
| --- | ---: | ---: | ---: | ---: |
| No MERROW closure | 81.36 | 69.97 | **81.97** | 63.19 |
| MERROW closure | 79.92 | **81.55** | 79.38 | **82.58** |

The best-of-two result is computed **within each scenario before averaging**;
it is not the maximum of the columns' means. Under the documented
final-selection rule, retain both
specialists if two files can be selected; choosing solely by their identical
public result discards that private protection. Select the two final files
manually; a default ranking by public score could instead retain 92.62 and
92.31, whose private halves are identical and use the older tariff planner.

The SK4 lean planner uses `n_scenarios=30`, `cap_derate=0.96`,
`hazard_mult=0`, `price_path=True`, and a 30% minimum allocation to a second
supplier when its landed cost is within 2.5% of the cheapest. Its additional
minimum supplier shares are steel 20%, copper 10%, and aluminum 10%. The
insured rule planner uses `safety_base=0.25`, `safety_risk=4`,
`overorder=0.333`, `buffer_mid=2.251`, and `taper=2.167`, with risk-weighted
supplier allocation. Both use a market-checked news model for the submitted
forecast columns while their purchasing plans use the planner's original
supplier probabilities; feeding the sharper market probabilities into orders
lost money in an earlier ban window. The lean forecast halves background and
ongoing probabilities beyond week two; the insured forecast raises background
probabilities beyond week four and ramps prices up to 5% by week 13. Live
news stories are normally exempt from those generic hedges.

SK4 also fixes a material private-planning error: the announced 35% KES
tariff effective week 224 on aluminum and semiconductors is included in disruption forecasts and
landed-cost paths, but excluded from supply-capacity/lead-time shock
scenarios. The older 89.45–92.62 private plan conflated a certain tariff with
the MERROW closure story for semiconductors and overbought them. This fix is
why the SK4 pair is the starting point for the new candidate, not the
public-leading 92.62 file.

The new forecast change uses the week-221 snapshot to run the same
market-checked supplier model three times: normal closure conversion
(`p_base`), closure conversion zero (`p_0`), and closure conversion one
(`p_1`). It adjusts only the stored disruption probability:

```text
lean:    clip(p_old - max(0, p_base - p_0), 0, 1)
insured: clip(p_old + max(0, p_1 - p_base), 0, 1)
```

The no-closure branch retains the gradient-boosted model's independent-risk
floor. This matters: an earlier, stronger mixture inversion lowered forecasts
for unrelated shocks after false closure warnings and lost up to 3.13
Brier-derived points in one historical window. The final change affects
**40 lean** and **67 insured** private probabilities. Every public cell and
all private prices, orders, and six supplier shares are byte-identical to the
corresponding SK4 parent. Source:
`experiments/private_closure_conditional.py`. The copied SK4 parent files and
their generating source are under `methods/sk4_phase2/`.

This is the strongest **tested private-focused pair in this project**, not a
proven global optimum. Its 48-scenario local expected best-of-two score is
**82.59**, versus **81.80** for the original SK4 pair, using a 40% closure
probability. The scenario set assumes the announced KES tariff and the two
warned bans, but omits unwarned shocks. Across all ten historical cutoffs
with a live closure warning, the full best-of-two score improved in four,
fell by 0.03 in one, and was unchanged in five. These windows overlap: only
one prior MERROW closure had an advance warning. See
`reports/private_closure_scenario_summary.md`,
`reports/private_closure_backtest.csv`, and
`reports/private_closure_pair_backtest.csv` for the actual comparisons.
The selected members still have local Prediction scores around 0.62–0.72 and
Financial scores around 0.84–0.87 in the simulated branches; neither
component is saturated. This is evidence of potential headroom, not a claim
that an unreleased private outcome can be predicted perfectly.

### What remains open

1. **Event probabilities and dependence.** The closure conversion estimate is
   based on a handful of prior stories. The 48 future scenarios do not model
   the full distribution of unwarned country and lane shocks. A calibrated
   event-level posterior with onset, duration, shared lanes, and market signals
   could change both forecasts and which policy wins. Validate it at every
   earlier origin before adopting it.
2. **Joint two-file optimization.** The SK4 LP minimizes expected purchasing
   cost for one file, while the actual selection values the larger of two
   total private scores. A distributionally robust optimizer for the two
   policies together could spend more on shock fill only where the insured
   member is likely to win. Any change must beat this pair on chronological
   private-style windows and plausible alternate scoring conventions.
3. **Price forecasts.** Current prices use a shock uplift only above a
   probability threshold and half-strength mean reversion. Because MAPE
   rewards a conditional median, an event-mixture quantile model could improve
   the price component, especially for cobalt and rare earths. Historical
   uplift samples are small, so later-origin validation is essential.
4. **Simulator calibration.** Known public scores constrain but do not fully
   reveal cost and diversification aggregation. Public hindsight plans sit
   near a clipped financial ceiling where local predictions have been biased.
   Tune on multiple chronological origins, not on released weeks 209–221 alone.

GPU training is unlikely to be the bottleneck: the historical dataset and
per-window linear programs are small, and the main uncertainty is event
calibration and simulator fidelity. A faster solver may help run more stress
scenarios, but it does not provide new information about private outcomes.

## Scored CSVs and exact generators

Every scored artifact stays directly under `submissions/`; names encode the
reported public score. `methods/check_provenance.py` rebuilds the first 15
byte for byte from the saved code and phase inputs. For 92.62, the current
public optimizer has small numerical drift in two public orders (at most two
units) and some public shares (at most 0.001); its forecasts and all private
rows still match exactly. Preserve the scored file as the canonical bytes.

| Phase | Public score and CSV | Direct generator and exact setting | Distinguishing change |
| --- | --- | --- | --- |
| 1 | [44.95](submissions/submission44.95.csv) | `pipeline.py generate`, `primary` | Original demand multiplier and equal eligible shares |
| 1 | [45.80](submissions/submission45.80.csv) | `pipeline.py generate`, `lead_aware` | Lead-aware supplier allocation |
| 1 | [46.21](submissions/submission46.21.csv) | `pipeline.py generate`, `long_memory` | Longer-memory price forecast |
| 1 | [46.22](submissions/submission46.22.csv) | `pipeline.py generate`, `independent` | Independent local forecast/policy |
| 1 | [48.55](submissions/submission48.55.csv) | `methods/lean_60_66/src/dsdg_supply_chain/cli.py submit insured_hedged` | Earlier insured specialist |
| 1 | [50.44](submissions/submission50.44.csv) | `methods/policy_v1_54_10/build_forecast_hybrid.py` | 54.10 forecasts with 44.95 orders/shares |
| 1 | [54.10](submissions/submission54.10.csv) | `methods/policy_v1_54_10/src/dsdg_supply_chain/cli.py submit policy_v1` | Earlier full teammate policy |
| 1 | [60.66](submissions/submission60.66.csv) | `methods/lean_60_66/src/dsdg_supply_chain/cli.py submit lean_hedged` | Scenario LP with calm forecast hedge |
| 1 | [66.98](submissions/submission66.98.csv) | `experiments/dual_hazard_three_lane.py` | Third lithium lane; public loss vs 67.74 |
| 1 | [67.26](submissions/submission67.26.csv) | `experiments/lithium_share_only.py` | Nine lithium share changes only; public loss |
| 1 | [67.74](submissions/submission67.74.csv) | `methods/lean_split_67_74/src/dsdg_supply_chain/cli.py submit lean_split` | Near-cost second supplier split |
| 2 | [84.27](submissions/submission84.27.csv) | `experiments/phase2_public_oracle.py --convention 5mean_supweek` | Alternative starting-stock rule; public loss |
| 2 | [89.45](submissions/submission89.45.csv) | `experiments/phase2_public_oracle.py --convention sum5` | Released public truth and hindsight plan; old lean private plan |
| 2 | [91.36](submissions/submission91.36.csv) | `experiments/public_diversification_qp.py --diversity-scale 0.5` | Released-public share QP; private unchanged |
| 2 | [92.31](submissions/submission92.31.csv) | `experiments/public_diversification_qp.py --diversity-scale 1.5` | Stronger released-public diversification; private unchanged |
| 2 | [92.62](submissions/submission92.62.csv) | `experiments/public_joint_order_search.py --base submissions/submission92.31.csv --scale 1.5` | Released-public orders/shares; private unchanged; numerical reproduction limit above |

The two SK4 `_w221_pub.csv` parent files under `methods/sk4_phase2/submissions/`
were each reported as public **89.45**, but they are **not** the canonical
`submissions/submission89.45.csv`. They were regenerated byte for byte from
their copied source (`dsdg submit` followed by
`methods/sk4_phase2/eda/33_fill_public.py`). Their private rows contain the
tariff fix and newer policy; the canonical scored 89.45 private rows do not.
Do not rename either SK4 parent to `submission89.45.csv` or treat matching
public scores as identical files. The conditional candidates preserve these
parents' public rows, so a new public upload would repeat 89.45 without
revealing whether the private change works.

The archived source snapshots have known origins: `policy_v1_54_10` is from
SK3 commit `eb85f21`, `lean_60_66` from commit `1158114`,
`lean_split_67_74` from commit `527bfba`, and the clean SK4 Phase 2 copy from
commit `6c97ff9`. These are copied sources, not live imports from the sibling
project directories. The scored files remain in this project's `submissions/`.

## Reproduce and validate

Use Python with NumPy, pandas, SciPy, and scikit-learn installed. From the
project root:

```sh
python methods/check_provenance.py
DSDG_DATA_DIR='data/Phase 2 data' python \
  experiments/private_closure_conditional.py --alpha 1 \
  --output submissions/candidates
python -m experiments.submission_audit
DSDG_DATA_DIR='data/Phase 2 data' PYTHONPATH=methods/sk4_phase2/src python \
  methods/sk4_phase2/eda/30_scenario_pair.py \
  methods/sk4_phase2/submissions/lean_pxlp_mkt_div_w221_pub.csv \
  methods/sk4_phase2/submissions/insured_hedged_mktcol_w221_pub.csv \
  submissions/candidates/private_closure_conditional_lean_w221.csv \
  submissions/candidates/private_closure_conditional_insured_w221.csv
```

The submission audit checks all 260 IDs, valid ranges, eligible country
shares, and share sums, and reports full/public duplicate groups. The two
private candidates intentionally share their public rows; no stored
submission is a full-data duplicate. Do not spend public trials on archived
or public-identical files to infer a private difference.

## Retained history and other entry points

- `submissions/archive/` contains ungraded week-208 controls. They have
  reference value for earlier policy comparisons but are not final candidates.
- `scratch/` contains three rejected independent modeling approaches and
  chronological evaluations. `methods/legacy_unmapped/` contains an old
  notebook with no proven scored-CSV mapping. Neither is a generator of a
  recommended submission.
- `PLAN.md` and `PROGRESS.md` are chronological decision history; this README
  gives the current mapping. The nested READMEs contain execution details for
  their own components, not a second submission recommendation.
- `kaggle/build_notebook.py` and `databricks/track_notebook.py` are platform
  packaging paths. Their execution in an external platform is not evidence
  of a private leaderboard score.
- The old `data/Phase 1 data/` copy was byte-identical to
  `data/snapshots/w208/` and was removed; the frozen snapshot is canonical.
  An ungraded, public-only material-portfolio CSV under
  `reports/generated_candidates/` was removed because its generator and
  research report remain. No graded or final-candidate CSV was removed.

## Complete Python-file index

Each Python file below has one row. “Score mapping” identifies its direct
scored output, a shared source package, or **none** for research/validation
code. A package file's association does not mean that running that file alone
writes a CSV. “Phase” identifies the saved source snapshot or experiment,
not permission to read future outcomes. The generated inventory is checked
against the filesystem so new Python files cannot silently go unmapped.

<!-- BEGIN PYTHON FILE INVENTORY -->
Python files indexed: **218**. Score shorthand `44.95` means `submissions/submission44.95.csv`; `unscored` means no graded score.

### Phase 1 (107 files)

| Python file | Score mapping / artifact | Role |
| --- | --- | --- |
| [`experiments/allocation_sweep.py`](experiments/allocation_sweep.py) | research; no direct scored CSV | Chronological supplier-share sweep for the risk-responsive order policy. |
| [`experiments/brier_shrink_oof.py`](experiments/brier_shrink_oof.py) | research; no direct scored CSV | Fit Brier-optimal shrinkage on archived out-of-fold forecasts. |
| [`experiments/correlated_event_scenarios.py`](experiments/correlated_event_scenarios.py) | research; no direct scored CSV | Score candidates under joint news-story onset and duration scenarios. |
| [`experiments/correlated_lane_buffer.py`](experiments/correlated_lane_buffer.py) | research; no direct scored CSV | Chronological test of a portfolio buffer triggered by shared lane risk. |
| [`experiments/correlated_lane_buffer_candidate.py`](experiments/correlated_lane_buffer_candidate.py) | research; no direct scored CSV | Build the first score-conditioned public candidate from scored 67.74. |
| [`experiments/cvar_backtest.py`](experiments/cvar_backtest.py) | research; no direct scored CSV | Chronological direct-score check of explicit CVaR scenario LP variants. |
| [`experiments/cvar_lp.py`](experiments/cvar_lp.py) | research; no direct scored CSV | Scenario LP with an explicit conditional-value-at-risk cost term. |
| [`experiments/diversification_frontier.py`](experiments/diversification_frontier.py) | research; no direct scored CSV | Cost-service-diversification frontier around the scored lean-split policy. |
| [`experiments/dual_hazard_three_lane.py`](experiments/dual_hazard_three_lane.py) | 66.98 | Build a week-208 public candidate for a material exposed to two independent hazards. |
| [`experiments/graded_comparison.py`](experiments/graded_comparison.py) | research; no direct scored CSV | Compare graded CSV fields and replay their policies on released supply windows. |
| [`experiments/insured_pair_check.py`](experiments/insured_pair_check.py) | research; no direct scored CSV | Compare the reproduced lean policy with the source package's insured policy. |
| [`experiments/leaderboard_inverse.py`](experiments/leaderboard_inverse.py) | research; no direct scored CSV | Infer plausible public-window supply states from three score differences. |
| [`experiments/lean_share_sweep.py`](experiments/lean_share_sweep.py) | research; no direct scored CSV | Chronological test of diversification changes to the reproduced lean policy. |
| [`experiments/lithium_share_only.py`](experiments/lithium_share_only.py) | 67.26 | Public lithium share-only test against the scored 67.74 order schedule. |
| [`experiments/lithium_share_scenarios.py`](experiments/lithium_share_scenarios.py) | research; no direct scored CSV | Evaluate fixed lithium orders under week-208 supplier scenarios. |
| [`experiments/mape_price_model.py`](experiments/mape_price_model.py) | research; no direct scored CSV | Weighted-absolute-error price model with chronological validation. |
| [`experiments/mape_threshold_sweep.py`](experiments/mape_threshold_sweep.py) | research; no direct scored CSV | Chronological check of the MAPE-optimal shock uplift threshold. |
| [`experiments/material_policy_selector.py`](experiments/material_policy_selector.py) | research; no direct scored CSV | Select purchasing policy independently by material on early released folds. |
| [`experiments/private_risk_hedge.py`](experiments/private_risk_hedge.py) | research; no direct scored CSV | Build a private-window hedge from a scored submission without changing public rows. |
| [`experiments/ridge_forecast.py`](experiments/ridge_forecast.py) | research; no direct scored CSV | Chronological, regularized forecasts for the supply-chain submission. |
| [`experiments/robust_lp_sweep.py`](experiments/robust_lp_sweep.py) | research; no direct scored CSV | Chronological comparison of lean LP risk assumptions; no future labels in plans. |
| [`experiments/scenario_planner.py`](experiments/scenario_planner.py) | research; no direct scored CSV | Scenario-based order planning using only supplier history released at origin. |
| [`kaggle/build_notebook.py`](kaggle/build_notebook.py) | platform packaging; no direct score | Build an offline, self-contained Kaggle notebook from the local pipeline. |
| [`methods/lean_60_66/src/dsdg_supply_chain/__init__.py`](methods/lean_60_66/src/dsdg_supply_chain/__init__.py) | shared source: 48.55, 60.66 | DSDG Hackathon 2026: Supply Chain Disruptor pipeline. |
| [`methods/lean_60_66/src/dsdg_supply_chain/backtest.py`](methods/lean_60_66/src/dsdg_supply_chain/backtest.py) | shared source: 48.55, 60.66 | Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline. |
| [`methods/lean_60_66/src/dsdg_supply_chain/cli.py`](methods/lean_60_66/src/dsdg_supply_chain/cli.py) | shared source: 48.55, 60.66 | Command line: `uv run dsdg {backtest,validate,submit} ...`. |
| [`methods/lean_60_66/src/dsdg_supply_chain/data.py`](methods/lean_60_66/src/dsdg_supply_chain/data.py) | shared source: 48.55, 60.66 | Loading the competition files and producing leakage-free snapshots of them. |
| [`methods/lean_60_66/src/dsdg_supply_chain/features/__init__.py`](methods/lean_60_66/src/dsdg_supply_chain/features/__init__.py) | shared source: 48.55, 60.66 | Feature extraction, one module per data source. |
| [`methods/lean_60_66/src/dsdg_supply_chain/features/build.py`](methods/lean_60_66/src/dsdg_supply_chain/features/build.py) | shared source: 48.55, 60.66 | Combine news, supplier-state and market features into one supplier-week table, with checks. |
| [`methods/lean_60_66/src/dsdg_supply_chain/features/market.py`](methods/lean_60_66/src/dsdg_supply_chain/features/market.py) | shared source: 48.55, 60.66 | Market features per supplier: country tension and currency, lane congestion, tariff announcements, |
| [`methods/lean_60_66/src/dsdg_supply_chain/features/news.py`](methods/lean_60_66/src/dsdg_supply_chain/features/news.py) | shared source: 48.55, 60.66 | News → structured shock signals → per-supplier weekly features. |
| [`methods/lean_60_66/src/dsdg_supply_chain/features/supply.py`](methods/lean_60_66/src/dsdg_supply_chain/features/supply.py) | shared source: 48.55, 60.66 | Supplier state features: is the supplier hit right now, how, for how long, and how widespread. |
| [`methods/lean_60_66/src/dsdg_supply_chain/labels.py`](methods/lean_60_66/src/dsdg_supply_chain/labels.py) | shared source: 48.55, 60.66 | Supplier-level ground truth: which (material, country) is hit by an active shock each week. |
| [`methods/lean_60_66/src/dsdg_supply_chain/models/__init__.py`](methods/lean_60_66/src/dsdg_supply_chain/models/__init__.py) | shared source: 48.55, 60.66 | Predictive models: disruption probability and price. |
| [`methods/lean_60_66/src/dsdg_supply_chain/models/disruption.py`](methods/lean_60_66/src/dsdg_supply_chain/models/disruption.py) | shared source: 48.55, 60.66 | Disruption model: P(supplier is hit h weeks after the information week). |
| [`methods/lean_60_66/src/dsdg_supply_chain/models/duration.py`](methods/lean_60_66/src/dsdg_supply_chain/models/duration.py) | shared source: 48.55, 60.66 | How long does an ongoing shock last? A censoring-aware duration model per shock type. |
| [`methods/lean_60_66/src/dsdg_supply_chain/models/price.py`](methods/lean_60_66/src/dsdg_supply_chain/models/price.py) | shared source: 48.55, 60.66 | Price model: benchmark price = normal level x shock uplift. |
| [`methods/lean_60_66/src/dsdg_supply_chain/models/scripts.py`](methods/lean_60_66/src/dsdg_supply_chain/models/scripts.py) | shared source: 48.55, 60.66 | Story-script model: the news follows fixed scripts per shock type, so model the scripts directly. |
| [`methods/lean_60_66/src/dsdg_supply_chain/policy.py`](methods/lean_60_66/src/dsdg_supply_chain/policy.py) | shared source: 48.55, 60.66 | Ordering policy: a lean, risk-aware plan per material and per 13-week window. |
| [`methods/lean_60_66/src/dsdg_supply_chain/policy_lp.py`](methods/lean_60_66/src/dsdg_supply_chain/policy_lp.py) | shared source: 48.55, 60.66 | Stochastic-programming ordering policy (sample average approximation). |
| [`methods/lean_60_66/src/dsdg_supply_chain/scoring.py`](methods/lean_60_66/src/dsdg_supply_chain/scoring.py) | shared source: 48.55, 60.66 | Local replica of the competition metric for one test window. |
| [`methods/lean_60_66/src/dsdg_supply_chain/simulator.py`](methods/lean_60_66/src/dsdg_supply_chain/simulator.py) | shared source: 48.55, 60.66 | Supply chain simulator reproducing the rules on the competition's Evaluation tab. |
| [`methods/lean_60_66/src/dsdg_supply_chain/strategies.py`](methods/lean_60_66/src/dsdg_supply_chain/strategies.py) | shared source: 48.55, 60.66 | Strategies: functions mapping an information snapshot to a submission. |
| [`methods/lean_60_66/src/dsdg_supply_chain/submission.py`](methods/lean_60_66/src/dsdg_supply_chain/submission.py) | shared source: 48.55, 60.66 | Submission format: building, validating, reading and writing. |
| [`methods/lean_split_67_74/eda/01_episodes.py`](methods/lean_split_67_74/eda/01_episodes.py) | research/test for 67.74; no direct CSV | Disruption episodes: when, how long, and which supplier/lane shows the damage. |
| [`methods/lean_split_67_74/eda/02_country_shocks.py`](methods/lean_split_67_74/eda/02_country_shocks.py) | research/test for 67.74; no direct CSV | Per material-country anomaly signals and whether their union reproduces `disrupted`. |
| [`methods/lean_split_67_74/eda/03_news_timeline.py`](methods/lean_split_67_74/eda/03_news_timeline.py) | research/test for 67.74; no direct CSV | Non-routine headlines in chronological order, annotated with shocks starting soon / active. |
| [`methods/lean_split_67_74/eda/04_market.py`](methods/lean_split_67_74/eda/04_market.py) | research/test for 67.74; no direct CSV | Tariff announcements, landed-cost formula, futures, macro tension, shipping vs shocks. |
| [`methods/lean_split_67_74/eda/05_validate_news.py`](methods/lean_split_67_74/eda/05_validate_news.py) | research/test for 67.74; no direct CSV | Validate the news parser and features against supplier-level ground truth. |
| [`methods/lean_split_67_74/eda/06_validate_features.py`](methods/lean_split_67_74/eda/06_validate_features.py) | research/test for 67.74; no direct CSV | Are the supplier-state and market features predictive, and do they add to the news? |
| [`methods/lean_split_67_74/eda/07_disruption_model.py`](methods/lean_split_67_74/eda/07_disruption_model.py) | research/test for 67.74; no direct CSV | Walk-forward evaluation of the disruption model (material-level Brier skill vs the naive rate). |
| [`methods/lean_split_67_74/eda/08_disruption_diagnose.py`](methods/lean_split_67_74/eda/08_disruption_diagnose.py) | research/test for 67.74; no direct CSV | Where does the disruption model lose Brier score? Supplier-level calibration split by |
| [`methods/lean_split_67_74/eda/09_price_model.py`](methods/lean_split_67_74/eda/09_price_model.py) | research/test for 67.74; no direct CSV | Walk-forward evaluation of the price model: MAPE skill vs carrying the last benchmark forward. |
| [`methods/lean_split_67_74/eda/10_price_drivers.py`](methods/lean_split_67_74/eda/10_price_drivers.py) | research/test for 67.74; no direct CSV | Is there predictable structure in prices outside shocks? |
| [`methods/lean_split_67_74/eda/11_tune_policy.py`](methods/lean_split_67_74/eda/11_tune_policy.py) | research/test for 67.74; no direct CSV | Tune the ordering policy's knobs on the backtest windows. |
| [`methods/lean_split_67_74/eda/12_value_of_information.py`](methods/lean_split_67_74/eda/12_value_of_information.py) | research/test for 67.74; no direct CSV | Value of information: how much would the score rise if each forecast were perfect? |
| [`methods/lean_split_67_74/eda/13_brier_breakdown.py`](methods/lean_split_67_74/eda/13_brier_breakdown.py) | research/test for 67.74; no direct CSV | Where does the Brier score go? Squared error by situation, ours vs naive, on public windows. |
| [`methods/lean_split_67_74/eda/14_calibrate_disruption.py`](methods/lean_split_67_74/eda/14_calibrate_disruption.py) | research/test for 67.74; no direct CSV | Post-hoc calibration of material disruption probabilities, using only history up to each origin. |
| [`methods/lean_split_67_74/eda/15_lp_policy.py`](methods/lean_split_67_74/eda/15_lp_policy.py) | research/test for 67.74; no direct CSV | Evaluate the stochastic-programming (LP) policy against policy_v1 on the cached backtest jobs. |
| [`methods/lean_split_67_74/eda/16_pair_tune.py`](methods/lean_split_67_74/eda/16_pair_tune.py) | research/test for 67.74; no direct CSV | Two final submissions: Kaggle scores the better of the selected ones on the private board. |
| [`methods/lean_split_67_74/eda/17_metric_variants.py`](methods/lean_split_67_74/eda/17_metric_variants.py) | research/test for 67.74; no direct CSV | Which reading of the Evaluation tab does Kaggle actually use? |
| [`methods/lean_split_67_74/eda/18_pair_hedge.py`](methods/lean_split_67_74/eda/18_pair_hedge.py) | research/test for 67.74; no direct CSV | Hedge the forecast columns across the two final submissions. |
| [`methods/lean_split_67_74/eda/19_pair_retune.py`](methods/lean_split_67_74/eda/19_pair_retune.py) | research/test for 67.74; no direct CSV | Retune the insured specialist's ordering for the HEDGED pair (X6 in DECISIONS.md). |
| [`methods/lean_split_67_74/eda/20_story_scripts.py`](methods/lean_split_67_74/eda/20_story_scripts.py) | research/test for 67.74; no direct CSV | Story-level statistics of the news scripts: how often each story type becomes a real shock, |
| [`methods/lean_split_67_74/eda/21_script_backtest.py`](methods/lean_split_67_74/eda/21_script_backtest.py) | research/test for 67.74; no direct CSV | Does the story-script model (models/scripts.py) improve the Brier skill out of sample? |
| [`methods/lean_split_67_74/eda/22_script_dense.py`](methods/lean_split_67_74/eda/22_script_dense.py) | research/test for 67.74; no direct CSV | Dense check of the story-script model: an origin every 2 weeks (52..195), 13-week horizon, |
| [`methods/lean_split_67_74/eda/23_independent_like.py`](methods/lean_split_67_74/eda/23_independent_like.py) | research/test for 67.74; no direct CSV | Backtest the ordering rule of the teammate's independent_w208.csv. |
| [`methods/lean_split_67_74/eda/24_score_files.py`](methods/lean_split_67_74/eda/24_score_files.py) | research/test for 67.74; no direct CSV | Score submission CSVs on every released test week with the local scorer (run after Phase 2). |
| [`methods/lean_split_67_74/eda/25_price_fresh.py`](methods/lean_split_67_74/eda/25_price_fresh.py) | research/test for 67.74; no direct CSV | Fresh look at the price process: what predicts the benchmark's change over the next h weeks? |
| [`methods/lean_split_67_74/eda/26_tariff_announcements.py`](methods/lean_split_67_74/eda/26_tariff_announcements.py) | research/test for 67.74; no direct CSV | Do tariff announcements in tariffs.csv come true? (they feed models/disruption.py) |
| [`methods/lean_split_67_74/eda/common.py`](methods/lean_split_67_74/eda/common.py) | research/test for 67.74; no direct CSV | Shared loaders for EDA. |
| [`methods/lean_split_67_74/notebooks/make_notebooks.py`](methods/lean_split_67_74/notebooks/make_notebooks.py) | research/test for 67.74; no direct CSV | Generates the submission notebooks (run: uv run python notebooks/make_notebooks.py). |
| [`methods/lean_split_67_74/tests/test_disruption.py`](methods/lean_split_67_74/tests/test_disruption.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/lean_split_67_74/tests/test_features.py`](methods/lean_split_67_74/tests/test_features.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/lean_split_67_74/tests/test_news.py`](methods/lean_split_67_74/tests/test_news.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/lean_split_67_74/tests/test_policy.py`](methods/lean_split_67_74/tests/test_policy.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/lean_split_67_74/tests/test_price.py`](methods/lean_split_67_74/tests/test_price.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/lean_split_67_74/tests/test_scoring.py`](methods/lean_split_67_74/tests/test_scoring.py) | research/test for 67.74; no direct CSV | Support module |
| [`methods/policy_v1_54_10/build_forecast_hybrid.py`](methods/policy_v1_54_10/build_forecast_hybrid.py) | 50.44 | Recreate the scored 50.44 control from its two scored parent CSVs. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/__init__.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/__init__.py) | shared source: 54.10; 50.44 forecasts | DSDG Hackathon 2026: Supply Chain Disruptor pipeline. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/backtest.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/backtest.py) | shared source: 54.10; 50.44 forecasts | Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/cli.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/cli.py) | shared source: 54.10; 50.44 forecasts | Command line: `uv run dsdg {backtest,validate,submit} ...`. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/data.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/data.py) | shared source: 54.10; 50.44 forecasts | Loading the competition files and producing leakage-free snapshots of them. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/features/__init__.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/features/__init__.py) | shared source: 54.10; 50.44 forecasts | Feature extraction, one module per data source. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/features/build.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/features/build.py) | shared source: 54.10; 50.44 forecasts | Combine news, supplier-state and market features into one supplier-week table, with checks. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/features/market.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/features/market.py) | shared source: 54.10; 50.44 forecasts | Market features per supplier: country tension and currency, lane congestion, tariff announcements, |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/features/news.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/features/news.py) | shared source: 54.10; 50.44 forecasts | News → structured shock signals → per-supplier weekly features. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/features/supply.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/features/supply.py) | shared source: 54.10; 50.44 forecasts | Supplier state features: is the supplier hit right now, how, for how long, and how widespread. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/labels.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/labels.py) | shared source: 54.10; 50.44 forecasts | Supplier-level ground truth: which (material, country) is hit by an active shock each week. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/models/__init__.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/models/__init__.py) | shared source: 54.10; 50.44 forecasts | Predictive models: disruption probability and price. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/models/disruption.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/models/disruption.py) | shared source: 54.10; 50.44 forecasts | Disruption model: P(supplier is hit h weeks after the information week). |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/models/duration.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/models/duration.py) | shared source: 54.10; 50.44 forecasts | How long does an ongoing shock last? A censoring-aware duration model per shock type. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/models/price.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/models/price.py) | shared source: 54.10; 50.44 forecasts | Price model: benchmark price = normal level x shock uplift. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/policy.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/policy.py) | shared source: 54.10; 50.44 forecasts | Ordering policy: a lean, risk-aware plan per material and per 13-week window. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/policy_lp.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/policy_lp.py) | shared source: 54.10; 50.44 forecasts | Stochastic-programming ordering policy (sample average approximation). |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/scoring.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/scoring.py) | shared source: 54.10; 50.44 forecasts | Local replica of the competition metric for one test window. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/simulator.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/simulator.py) | shared source: 54.10; 50.44 forecasts | Supply chain simulator reproducing the rules on the competition's Evaluation tab. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/strategies.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/strategies.py) | shared source: 54.10; 50.44 forecasts | Strategies: functions mapping an information snapshot to a submission. |
| [`methods/policy_v1_54_10/src/dsdg_supply_chain/submission.py`](methods/policy_v1_54_10/src/dsdg_supply_chain/submission.py) | shared source: 54.10; 50.44 forecasts | Submission format: building, validating, reading and writing. |
| [`pipeline.py`](pipeline.py) | 44.95, 45.80, 46.21, 46.22 | Reproducible, dependency-free Supply Chain Disruptor candidate generator. |
| [`scratch/direct_forecast.py`](scratch/direct_forecast.py) | rejected research; no scored CSV | Direct material forecasts trained only on released target weeks. |
| [`scratch/empirical_allocator.py`](scratch/empirical_allocator.py) | rejected research; no scored CSV | Choose supplier shares by replaying released supply blocks, without a shock model. |
| [`scratch/evaluate_allocator.py`](scratch/evaluate_allocator.py) | rejected research; no scored CSV | Chronological comparison of empirical supplier allocation against 67.74. |
| [`scratch/evaluate_direct.py`](scratch/evaluate_direct.py) | rejected research; no scored CSV | Chronological score test of the raw-table direct forecast against 67.74. |
| [`scratch/evaluate_imitation.py`](scratch/evaluate_imitation.py) | rejected research; no scored CSV | Chronological score test for supervised hindsight imitation. |
| [`scratch/hindsight_imitation.py`](scratch/hindsight_imitation.py) | rejected research; no scored CSV | Learn order plans from historical hindsight optima, with chronological training. |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | tests for 44.95–46.22 source | Contract checks for generated submissions and the Phase 2 forecast boundary. |

### Phase 1; reused in 2 (21 files)

| Python file | Score mapping / artifact | Role |
| --- | --- | --- |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/__init__.py`](methods/lean_split_67_74/src/dsdg_supply_chain/__init__.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | DSDG Hackathon 2026: Supply Chain Disruptor pipeline. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/backtest.py`](methods/lean_split_67_74/src/dsdg_supply_chain/backtest.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/cli.py`](methods/lean_split_67_74/src/dsdg_supply_chain/cli.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Command line: `uv run dsdg {backtest,validate,submit} ...`. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/data.py`](methods/lean_split_67_74/src/dsdg_supply_chain/data.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Loading the competition files and producing leakage-free snapshots of them. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/features/__init__.py`](methods/lean_split_67_74/src/dsdg_supply_chain/features/__init__.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Feature extraction, one module per data source. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/features/build.py`](methods/lean_split_67_74/src/dsdg_supply_chain/features/build.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Combine news, supplier-state and market features into one supplier-week table, with checks. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/features/market.py`](methods/lean_split_67_74/src/dsdg_supply_chain/features/market.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Market features per supplier: country tension and currency, lane congestion, tariff announcements, |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/features/news.py`](methods/lean_split_67_74/src/dsdg_supply_chain/features/news.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | News → structured shock signals → per-supplier weekly features. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/features/supply.py`](methods/lean_split_67_74/src/dsdg_supply_chain/features/supply.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Supplier state features: is the supplier hit right now, how, for how long, and how widespread. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/labels.py`](methods/lean_split_67_74/src/dsdg_supply_chain/labels.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Supplier-level ground truth: which (material, country) is hit by an active shock each week. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/models/__init__.py`](methods/lean_split_67_74/src/dsdg_supply_chain/models/__init__.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Predictive models: disruption probability and price. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/models/disruption.py`](methods/lean_split_67_74/src/dsdg_supply_chain/models/disruption.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Disruption model: P(supplier is hit h weeks after the information week). |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/models/duration.py`](methods/lean_split_67_74/src/dsdg_supply_chain/models/duration.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | How long does an ongoing shock last? A censoring-aware duration model per shock type. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/models/price.py`](methods/lean_split_67_74/src/dsdg_supply_chain/models/price.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Price model: benchmark price = normal level x shock uplift. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/models/scripts.py`](methods/lean_split_67_74/src/dsdg_supply_chain/models/scripts.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Story-script model: the news follows fixed scripts per shock type, so model the scripts directly. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/policy.py`](methods/lean_split_67_74/src/dsdg_supply_chain/policy.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Ordering policy: a lean, risk-aware plan per material and per 13-week window. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/policy_lp.py`](methods/lean_split_67_74/src/dsdg_supply_chain/policy_lp.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Stochastic-programming ordering policy (sample average approximation). |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/scoring.py`](methods/lean_split_67_74/src/dsdg_supply_chain/scoring.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Local replica of the competition metric for one test window. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/simulator.py`](methods/lean_split_67_74/src/dsdg_supply_chain/simulator.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Supply chain simulator reproducing the rules on the competition's Evaluation tab. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/strategies.py`](methods/lean_split_67_74/src/dsdg_supply_chain/strategies.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Strategies: functions mapping an information snapshot to a submission. |
| [`methods/lean_split_67_74/src/dsdg_supply_chain/submission.py`](methods/lean_split_67_74/src/dsdg_supply_chain/submission.py) | shared source: 67.74; private of 84.27, 89.45, 91.36, 92.31, 92.62 | Submission format: building, validating, reading and writing. |

### Phase 2 (86 files)

| Python file | Score mapping / artifact | Role |
| --- | --- | --- |
| [`experiments/correlated_brier.py`](experiments/correlated_brier.py) | research; no direct scored CSV | Chronologically test shared-lane corrections to material disruption risk. |
| [`experiments/diversification_metric_inference.py`](experiments/diversification_metric_inference.py) | research; no direct scored CSV | Compare possible diversification formulas using controlled public scores. |
| [`experiments/event_calendar.py`](experiments/event_calendar.py) | research; no direct scored CSV | Export the latest released news state as an auditable event calendar. |
| [`experiments/event_saa_backtest.py`](experiments/event_saa_backtest.py) | research; no direct scored CSV | Chronological check of event-level scenario ordering against lean_split. |
| [`experiments/event_saa_policy.py`](experiments/event_saa_policy.py) | research; no direct scored CSV | Private policy scenarios drawn from whole news events, not weekly marginals. |
| [`experiments/joint_surface_sweep.py`](experiments/joint_surface_sweep.py) | research; no direct scored CSV | Select a joint public order/share plan using controlled Kaggle responses. |
| [`experiments/phase2_metric_check.py`](experiments/phase2_metric_check.py) | research; no direct scored CSV | Fit unresolved scoring conventions to all known public leaderboard scores. |
| [`experiments/phase2_public_oracle.py`](experiments/phase2_public_oracle.py) | 84.27, 89.45 | Public-window perfect-information plan and week-221 private baseline. |
| [`experiments/private_closure_backtest.py`](experiments/private_closure_backtest.py) | unscored pair evidence | Chronological Brier check for closure-conditional forecasts. |
| [`experiments/private_closure_conditional.py`](experiments/private_closure_conditional.py) | unscored candidate pair | Build a pair whose private forecasts condition on the open MERROW closure. |
| [`experiments/private_closure_pair_backtest.py`](experiments/private_closure_pair_backtest.py) | unscored pair evidence | Evaluate the closure forecast hedge on past Phase-2-style windows. |
| [`experiments/public_diversification_qp.py`](experiments/public_diversification_qp.py) | 91.36, 92.31 | Improve released-window supplier shares with a convex quadratic program. |
| [`experiments/public_joint_order_search.py`](experiments/public_joint_order_search.py) | 92.62 (near reproduction) | Optimize released public order timing, quantities, and shares jointly. |
| [`experiments/public_material_portfolio.py`](experiments/public_material_portfolio.py) | research; no direct scored CSV | Choose a public order plan per material by robust score-response optimization. |
| [`experiments/public_multistart.py`](experiments/public_multistart.py) | research; no direct scored CSV | Test whether the joint public optimizer is trapped in a local solution. |
| [`experiments/public_qp_scale_sweep.py`](experiments/public_qp_scale_sweep.py) | research; no direct scored CSV | Compare public diversification weights against graded-score-fitted metrics. |
| [`experiments/score_surface_fit.py`](experiments/score_surface_fit.py) | research; no direct scored CSV | Fit a local public-score response from controlled graded policy pairs. |
| [`methods/sk4_phase2/eda/01_episodes.py`](methods/sk4_phase2/eda/01_episodes.py) | research/test for unscored SK4 pair | Disruption episodes: when, how long, and which supplier/lane shows the damage. |
| [`methods/sk4_phase2/eda/02_country_shocks.py`](methods/sk4_phase2/eda/02_country_shocks.py) | research/test for unscored SK4 pair | Per material-country anomaly signals and whether their union reproduces `disrupted`. |
| [`methods/sk4_phase2/eda/03_news_timeline.py`](methods/sk4_phase2/eda/03_news_timeline.py) | research/test for unscored SK4 pair | Non-routine headlines in chronological order, annotated with shocks starting soon / active. |
| [`methods/sk4_phase2/eda/04_market.py`](methods/sk4_phase2/eda/04_market.py) | research/test for unscored SK4 pair | Tariff announcements, landed-cost formula, futures, macro tension, shipping vs shocks. |
| [`methods/sk4_phase2/eda/05_validate_news.py`](methods/sk4_phase2/eda/05_validate_news.py) | research/test for unscored SK4 pair | Validate the news parser and features against supplier-level ground truth. |
| [`methods/sk4_phase2/eda/06_validate_features.py`](methods/sk4_phase2/eda/06_validate_features.py) | research/test for unscored SK4 pair | Are the supplier-state and market features predictive, and do they add to the news? |
| [`methods/sk4_phase2/eda/07_disruption_model.py`](methods/sk4_phase2/eda/07_disruption_model.py) | research/test for unscored SK4 pair | Walk-forward evaluation of the disruption model (material-level Brier skill vs the naive rate). |
| [`methods/sk4_phase2/eda/08_disruption_diagnose.py`](methods/sk4_phase2/eda/08_disruption_diagnose.py) | research/test for unscored SK4 pair | Where does the disruption model lose Brier score? Supplier-level calibration split by |
| [`methods/sk4_phase2/eda/09_price_model.py`](methods/sk4_phase2/eda/09_price_model.py) | research/test for unscored SK4 pair | Walk-forward evaluation of the price model: MAPE skill vs carrying the last benchmark forward. |
| [`methods/sk4_phase2/eda/10_price_drivers.py`](methods/sk4_phase2/eda/10_price_drivers.py) | research/test for unscored SK4 pair | Is there predictable structure in prices outside shocks? |
| [`methods/sk4_phase2/eda/11_tune_policy.py`](methods/sk4_phase2/eda/11_tune_policy.py) | research/test for unscored SK4 pair | Tune the ordering policy's knobs on the backtest windows. |
| [`methods/sk4_phase2/eda/12_value_of_information.py`](methods/sk4_phase2/eda/12_value_of_information.py) | research/test for unscored SK4 pair | Value of information: how much would the score rise if each forecast were perfect? |
| [`methods/sk4_phase2/eda/13_brier_breakdown.py`](methods/sk4_phase2/eda/13_brier_breakdown.py) | research/test for unscored SK4 pair | Where does the Brier score go? Squared error by situation, ours vs naive, on public windows. |
| [`methods/sk4_phase2/eda/14_calibrate_disruption.py`](methods/sk4_phase2/eda/14_calibrate_disruption.py) | research/test for unscored SK4 pair | Post-hoc calibration of material disruption probabilities, using only history up to each origin. |
| [`methods/sk4_phase2/eda/15_lp_policy.py`](methods/sk4_phase2/eda/15_lp_policy.py) | research/test for unscored SK4 pair | Evaluate the stochastic-programming (LP) policy against policy_v1 on the cached backtest jobs. |
| [`methods/sk4_phase2/eda/16_pair_tune.py`](methods/sk4_phase2/eda/16_pair_tune.py) | research/test for unscored SK4 pair | Two final submissions: Kaggle scores the better of the selected ones on the private board. |
| [`methods/sk4_phase2/eda/17_metric_variants.py`](methods/sk4_phase2/eda/17_metric_variants.py) | research/test for unscored SK4 pair | Which reading of the Evaluation tab does Kaggle actually use? |
| [`methods/sk4_phase2/eda/18_pair_hedge.py`](methods/sk4_phase2/eda/18_pair_hedge.py) | research/test for unscored SK4 pair | Hedge the forecast columns across the two final submissions. |
| [`methods/sk4_phase2/eda/19_pair_retune.py`](methods/sk4_phase2/eda/19_pair_retune.py) | research/test for unscored SK4 pair | Retune the insured specialist's ordering for the HEDGED pair (X6 in DECISIONS.md). |
| [`methods/sk4_phase2/eda/20_story_scripts.py`](methods/sk4_phase2/eda/20_story_scripts.py) | research/test for unscored SK4 pair | Story-level statistics of the news scripts: how often each story type becomes a real shock, |
| [`methods/sk4_phase2/eda/21_script_backtest.py`](methods/sk4_phase2/eda/21_script_backtest.py) | research/test for unscored SK4 pair | Does the story-script model (models/scripts.py) improve the Brier skill out of sample? |
| [`methods/sk4_phase2/eda/22_script_dense.py`](methods/sk4_phase2/eda/22_script_dense.py) | research/test for unscored SK4 pair | Dense check of the story-script model: an origin every 2 weeks (52..195), 13-week horizon, |
| [`methods/sk4_phase2/eda/23_independent_like.py`](methods/sk4_phase2/eda/23_independent_like.py) | research/test for unscored SK4 pair | Backtest the ordering rule of the teammate's independent_w208.csv. |
| [`methods/sk4_phase2/eda/24_score_files.py`](methods/sk4_phase2/eda/24_score_files.py) | research/test for unscored SK4 pair | Score submission CSVs on every released test week with the local scorer (run after Phase 2). |
| [`methods/sk4_phase2/eda/25_price_fresh.py`](methods/sk4_phase2/eda/25_price_fresh.py) | research/test for unscored SK4 pair | Fresh look at the price process: what predicts the benchmark's change over the next h weeks? |
| [`methods/sk4_phase2/eda/26_tariff_announcements.py`](methods/sk4_phase2/eda/26_tariff_announcements.py) | research/test for unscored SK4 pair | Do tariff announcements in tariffs.csv come true? (they feed models/disruption.py) |
| [`methods/sk4_phase2/eda/27_pick_pair.py`](methods/sk4_phase2/eda/27_pick_pair.py) | research/test for unscored SK4 pair | Rank every strategy (alone and as a best-of-two pair) under a chosen reading of the metric, from |
| [`methods/sk4_phase2/eda/28_cost_path.py`](methods/sk4_phase2/eda/28_cost_path.py) | research/test for unscored SK4 pair | Evidence for models/cost_path.py: landed = spot x (1 + tariff) + freight; freight multipliers on |
| [`methods/sk4_phase2/eda/28_market_dense.py`](methods/sk4_phase2/eda/28_market_dense.py) | research/test for unscored SK4 pair | Round 6: does the market check of open stories (models/scripts.py MarketCheck) sharpen the |
| [`methods/sk4_phase2/eda/28_market_signals.py`](methods/sk4_phase2/eda/28_market_signals.py) | research/test for unscored SK4 pair | Fresh-eyes re-analysis (Round 6): which market data move ahead of shocks, and which don't. |
| [`methods/sk4_phase2/eda/29_pxlp_compare.py`](methods/sk4_phase2/eda/29_pxlp_compare.py) | research/test for unscored SK4 pair | lean_split_pxlp vs lean_split on the 66 step-4 windows: alone, public/private halves, the pair with |
| [`methods/sk4_phase2/eda/29_state_report.py`](methods/sk4_phase2/eda/29_state_report.py) | research/test for unscored SK4 pair | State of the world at an information week: what the final submission has to plan for. |
| [`methods/sk4_phase2/eda/30_scenario_pair.py`](methods/sk4_phase2/eda/30_scenario_pair.py) | research/test for unscored SK4 pair | Score candidate files on simulated futures of the private window (weeks 222-234), built from the |
| [`methods/sk4_phase2/eda/31_stage_gate.py`](methods/sk4_phase2/eda/31_stage_gate.py) | research/test for unscored SK4 pair | Phase 2: do the stage gate (models/scripts.py STAGE_GATE) and the announced-tariff fixes |
| [`methods/sk4_phase2/eda/32_holdout_208.py`](methods/sk4_phase2/eda/32_holdout_208.py) | research/test for unscored SK4 pair | Primary objective (Phase 2): run a strategy with the information available at week 208 (the |
| [`methods/sk4_phase2/eda/33_fill_public.py`](methods/sk4_phase2/eda/33_fill_public.py) | unscored SK4 parents | Fill the released public weeks (209-221) of a week-221 submission with the hindsight plan. |
| [`methods/sk4_phase2/eda/34_metric_fit.py`](methods/sk4_phase2/eda/34_metric_fit.py) | research/test for unscored SK4 pair | Phase 2: fit the simulator details the rules leave open to every Kaggle public score we have. |
| [`methods/sk4_phase2/eda/35_div_fit.py`](methods/sk4_phase2/eda/35_div_fit.py) | research/test for unscored SK4 pair | Phase 2: which diversification formula does Kaggle use? P, F and fill come from the local scorer |
| [`methods/sk4_phase2/eda/common.py`](methods/sk4_phase2/eda/common.py) | research/test for unscored SK4 pair | Shared loaders for EDA. |
| [`methods/sk4_phase2/notebooks/make_notebooks.py`](methods/sk4_phase2/notebooks/make_notebooks.py) | research/test for unscored SK4 pair | Generates the submission notebooks (run: uv run python notebooks/make_notebooks.py). |
| [`methods/sk4_phase2/src/dsdg_supply_chain/__init__.py`](methods/sk4_phase2/src/dsdg_supply_chain/__init__.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | DSDG Hackathon 2026: Supply Chain Disruptor pipeline. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/backtest.py`](methods/sk4_phase2/src/dsdg_supply_chain/backtest.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Walk-forward backtests on released weeks, mirroring the competition's two-phase timeline. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/cli.py`](methods/sk4_phase2/src/dsdg_supply_chain/cli.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Command line: `uv run dsdg {backtest,validate,submit} ...`. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/data.py`](methods/sk4_phase2/src/dsdg_supply_chain/data.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Loading the competition files and producing leakage-free snapshots of them. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/features/__init__.py`](methods/sk4_phase2/src/dsdg_supply_chain/features/__init__.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Feature extraction, one module per data source. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/features/build.py`](methods/sk4_phase2/src/dsdg_supply_chain/features/build.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Combine news, supplier-state and market features into one supplier-week table, with checks. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/features/market.py`](methods/sk4_phase2/src/dsdg_supply_chain/features/market.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Market features per supplier: country tension and currency, lane congestion, tariff announcements, |
| [`methods/sk4_phase2/src/dsdg_supply_chain/features/news.py`](methods/sk4_phase2/src/dsdg_supply_chain/features/news.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | News → structured shock signals → per-supplier weekly features. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/features/supply.py`](methods/sk4_phase2/src/dsdg_supply_chain/features/supply.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Supplier state features: is the supplier hit right now, how, for how long, and how widespread. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/labels.py`](methods/sk4_phase2/src/dsdg_supply_chain/labels.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Supplier-level ground truth: which (material, country) is hit by an active shock each week. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/__init__.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/__init__.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Predictive models: disruption probability and price. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/cost_path.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/cost_path.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Landed-cost path model: how a shock moves every supplier's landed cost, week by week. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/disruption.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/disruption.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Disruption model: P(supplier is hit h weeks after the information week). |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/duration.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/duration.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | How long does an ongoing shock last? A censoring-aware duration model per shock type. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/price.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/price.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Price model: benchmark price = normal level x shock uplift. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/models/scripts.py`](methods/sk4_phase2/src/dsdg_supply_chain/models/scripts.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Story-script model: the news follows fixed scripts per shock type, so model the scripts directly. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/policy.py`](methods/sk4_phase2/src/dsdg_supply_chain/policy.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Ordering policy: a lean, risk-aware plan per material and per 13-week window. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/policy_lp.py`](methods/sk4_phase2/src/dsdg_supply_chain/policy_lp.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Stochastic-programming ordering policy (sample average approximation). |
| [`methods/sk4_phase2/src/dsdg_supply_chain/scoring.py`](methods/sk4_phase2/src/dsdg_supply_chain/scoring.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Local replica of the competition metric for one test window. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/simulator.py`](methods/sk4_phase2/src/dsdg_supply_chain/simulator.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Supply chain simulator reproducing the rules on the competition's Evaluation tab. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/strategies.py`](methods/sk4_phase2/src/dsdg_supply_chain/strategies.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Strategies: functions mapping an information snapshot to a submission. |
| [`methods/sk4_phase2/src/dsdg_supply_chain/submission.py`](methods/sk4_phase2/src/dsdg_supply_chain/submission.py) | SK4 parents (public 89.45, noncanonical); unscored candidate pair | Submission format: building, validating, reading and writing. |
| [`methods/sk4_phase2/tests/test_cost_path.py`](methods/sk4_phase2/tests/test_cost_path.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_disruption.py`](methods/sk4_phase2/tests/test_disruption.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_features.py`](methods/sk4_phase2/tests/test_features.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_news.py`](methods/sk4_phase2/tests/test_news.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_policy.py`](methods/sk4_phase2/tests/test_policy.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_price.py`](methods/sk4_phase2/tests/test_price.py) | research/test for unscored SK4 pair | Support module |
| [`methods/sk4_phase2/tests/test_scoring.py`](methods/sk4_phase2/tests/test_scoring.py) | research/test for unscored SK4 pair | Support module |

### Both (4 files)

| Python file | Score mapping / artifact | Role |
| --- | --- | --- |
| [`databricks/track_notebook.py`](databricks/track_notebook.py) | platform packaging; no direct score | DSDG supply-chain track: native data, features, tracking, and submission run. |
| [`experiments/submission_audit.py`](experiments/submission_audit.py) | audit: all stored CSVs | Validate every stored submission and find redundant full or public CSVs. |
| [`methods/check_provenance.py`](methods/check_provenance.py) | audit: all 16 graded files | Check graded CSV provenance against archived code and input snapshots. |
| [`tools/code_inventory.py`](tools/code_inventory.py) | documentation inventory | Write or verify the complete Python-file index in the root README. |
<!-- END PYTHON FILE INVENTORY -->
