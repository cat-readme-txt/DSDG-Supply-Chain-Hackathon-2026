# Evaluated alternatives

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

The modules expose `backtest(data, ...)` for interactive use with
`pipeline.CompetitionData(Path("data"))`. See the latest `PROGRESS.md` entry for
measured results and `reports/independent_w208.json` for the selected candidate's
matched historical metrics.
