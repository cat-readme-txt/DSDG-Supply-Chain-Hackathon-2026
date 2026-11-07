# Evaluated alternatives

These scripts reproduce alternative forecasts, policy searches, and stress
tests. The main submission generator is `pipeline.py`.

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
- `private_risk_hedge.py` generates a backup that keeps the scored 60.66 file's
  public rows exactly and diversifies private shares in proportion to forecast
  risk. It changes no price forecasts or order quantities.
- `graded_comparison.py` validates scored files and replays their policies on
  released historical supply windows. Its report is a stress test, not an
  official score estimate.

The modules expose `backtest(data, ...)` for interactive use with
`pipeline.CompetitionData(Path("data"))`. See the latest `PROGRESS.md` entry for
measured results and `reports/independent_w208.json` for the selected candidate's
matched historical metrics.
