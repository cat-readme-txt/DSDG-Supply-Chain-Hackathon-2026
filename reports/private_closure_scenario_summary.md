# Private closure forecast pair, week-221 information

The user selected this exact conditional lean/insured pair as the final two
files and confirmed **89.45 public** for each. Private scores are unknown;
the results below remain local scenario estimates. Exact file hashes and
source lineage are in [`submissions/candidates/README.md`](../submissions/candidates/README.md).

The scorer here is the copied SK4 local simulator, **not** the hidden private
leaderboard. It evaluates 48 synthetic weeks 222–234 windows with announced
KES tariffs, warned VAL cobalt and KES rare-earth bans, and an open MERROW
closure. Its reported expectation weights closure occurrence at 40% and uses
order-weighted diversification. It does not include unwarned shocks.

| Private pair | Expected best-of-two score in these scenarios |
| --- | ---: |
| Original SK4 lean + insured | 81.80 |
| Conditional lean + original insured | 82.11 |
| Original lean + conditional insured | 82.27 |
| Conditional lean + conditional insured | **82.59** |

The conditional pair changes only private disruption probabilities. The lean
file removes the open closure's script contribution down to the GBM and
other-risk floor; the insured file forecasts the closure-occurs branch. Public
rows and all prices, orders, and shares match their SK4 parents. A more
aggressive mixture inversion scored 83.22 in these scenarios but was rejected:
on a false historical NORTH closure warning it lowered a forecast covering
other risks and lost up to 3.13 estimated Brier-derived score points.

Chronological checks at all ten earlier information cutoffs with live closure
warnings found four local best-of-two gains, one loss of 0.03, and five ties.
The two real MERROW cutoffs gained 9.14 and 0.43 points; the largest false
warning gain was 0.14. The windows overlap, so ten cutoffs are not ten
independent events, and these cases do not establish a private result.
See `private_closure_pair_backtest.csv` and `private_closure_backtest.csv`.

Reproduce the scenario comparison with:

```sh
DSDG_DATA_DIR='data/Phase 2 data' PYTHONPATH=methods/sk4_phase2/src python3 \
  methods/sk4_phase2/eda/30_scenario_pair.py \
  methods/sk4_phase2/submissions/lean_pxlp_mkt_div_w221_pub.csv \
  methods/sk4_phase2/submissions/insured_hedged_mktcol_w221_pub.csv \
  submissions/candidates/private_closure_conditional_lean_w221.csv \
  submissions/candidates/private_closure_conditional_insured_w221.csv
```
