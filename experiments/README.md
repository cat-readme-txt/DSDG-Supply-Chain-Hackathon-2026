# Evaluated alternatives

The [root README](../README.md) is the canonical inventory of every Python
file and scored submission. This page records experiment verdicts.

These scripts reproduce alternative forecasts, policy searches, and stress
tests. `pipeline.py` generates the earlier local submissions. The current
67.74 source and all graded-file provenance are mapped in `methods/README.md`.

- `ridge_forecast.py` fits pooled, regularized risk and price regressions using
  only labels released by each historical origin. The tested models underperformed
  the simpler released-news risk forecast and horizon-dependent price forecast.
- `scenario_planner.py` chooses weekly orders by coordinate search over past
  supplier blocks and shock stress cases. It under-ordered before new lead-time
  shocks in historical replay and had worse later-window cost and fill than the
  selected risk-responsive policy.
- `allocation_sweep.py` evaluates 75 cost- and lead-time-weighted supplier
  allocations with chronology-preserving historical replay. Its selected late
  lead-time shift is implemented in `pipeline.py` as the `lead_aware` profile.
- `private_risk_hedge.py` generates an archived experiment that keeps the scored 60.66 file's
  public rows exactly and diversifies private shares in proportion to forecast
  risk. It changes no price forecasts or order quantities. Chronological
  private-window replay did not support submitting it.
- `graded_comparison.py` validates scored files and replays their policies on
  released historical supply windows. Its report is a stress test, not an
  official score estimate.
- `submission_audit.py` validates every CSV under `submissions/` and reports
  both full-data duplicates and public-week duplicates. Run it before using a
  submission trial; see `submissions/README.md` for the file inventory.
- `dual_hazard_three_lane.py` reproduces the scored 66.98 file from the 67.74
  base. The third lithium lane reduced the actual public score; see
  `reports/dual_hazard_three_lane_w208.md` for the prediction error.
- `lithium_share_only.py` reproduces the scored 67.26 controlled test from the
  scored 67.74 file, altering nine public lithium shares while keeping orders
  and forecasts fixed. `lithium_share_scenarios.py` probes its effect under
  week-208 scenario assumptions; those estimates missed the official score.

The modules expose `backtest(data, ...)` for interactive use with
`pipeline.CompetitionData(Path("data"))`. See the latest `PROGRESS.md` entry for
measured results and `reports/independent_w208.json` for the selected candidate's
matched historical metrics.

## Week-221 private forecast experiment

`private_closure_conditional.py` builds a two-file private candidate from the
local SK4 source snapshot. It changes only disruption probabilities for the
open MERROW closure story. `private_closure_backtest.py` measures its Brier
effect at earlier closure warnings with information cut at each origin;
`private_closure_pair_backtest.py` scores the full pair on all ten past private-style
windows. The scenario proxy is in `methods/sk4_phase2/eda/30_scenario_pair.py`.
The generated files and evidence are described in
`submissions/candidates/README.md`. All scores from these scripts are local
proxies; private leaderboard performance remains unknown.
