# Submission inventory

The [root README](../README.md) is the canonical Python-to-submission map and
selected final-pair record. This page retains artifact history.

The CSVs directly in this folder are **exact files already graded on the public leaderboard**, named by their reported score. Preserve them as evidence. `candidates/` contains the two selected final files; `archive/` contains old experiments. Every stored CSV must have the required 260 rows and pass `CompetitionData.validate_file`. [The methods report](../methods/README.md) links each graded and final file to its source and records one small public-only numerical reproduction difference for 92.62.

| File | Purpose and observed result | Submit or test again? |
| --- | --- | --- |
| `submission44.95.csv` | Original equal-share, 1.2× demand policy; public **44.95**. | No. Historical baseline. |
| `submission45.80.csv` | Lead-aware share variant of the independent model; public **45.80**. | No. |
| `submission46.21.csv` | Longer-memory price variant of `submission46.22.csv`; public **46.21**. | No. |
| `submission46.22.csv` | Independent forecast and ordering approach; public **46.22**. | No. |
| `submission48.55.csv` | Insured specialist (`insured_hedged`); public **48.55**. Exact source is archived under `methods/lean_60_66/`. | Do not repeat a public upload. A possible complementary final selection if the competition takes the better private score of two, which remains unconfirmed here. |
| `submission50.44.csv` | Forecast-only hybrid: same orders and shares as `submission44.95.csv`, different probability and price forecasts; public **50.44**. | No. Forecast control. |
| `submission54.10.csv` | Full teammate policy: same forecasts as `submission50.44.csv`, different orders and shares; public **54.10**. | No. Policy control. |
| `submission60.66.csv` | `lean_hedged` full policy; public **60.66**. | Historical comparator. Do not spend another public trial on the same file. |
| `submission67.74.csv` | `lean_split` full policy from week 208; public **67.74**. Exact source is archived under `methods/lean_split_67_74/`. | Historical comparator. |
| `submission66.98.csv` | Dual-hazard lithium test: replans 10 public lithium rows using Valdmar alongside Kestria and Orinoque. Forecasts and all private rows match 67.74. Public **66.98**. | No. Its public score is **0.76 lower** than 67.74. |
| `submission67.26.csv` | Controlled lithium supplier test: preserves 67.74 orders and forecasts; changes only nine public shares. No private changes. Public **67.26**. | No. Its public score is **0.48 lower** than 67.74. |
| `submission89.45.csv` | Phase 2 public truth and hindsight inventory plan; private `lean_split` forecast and policy fitted only through week 221. Public **89.45**. | Scored parent of 91.36. Do not repeat upload. |
| `submission84.27.csv` | Alternative public inventory plan using five times average demand as starting stock; forecasts and all private rows match 89.45. Public **84.27**, a **5.18-point loss**. | No. Its starting-stock assumption was weaker on the official public scorer. |
| `submission91.36.csv` | Same forecasts and private rows as 89.45. Public supplier shares are optimized for known cost and diversification, with five units of stockout repair. Public **91.36**, a **1.91-point gain**. | Historical comparator. Do not repeat upload. |
| `submission92.31.csv` | Same forecasts and private rows as 91.36; released public supplier shares use a higher diversification weight. Public **92.31**, a **0.95-point gain**. | Public-only comparator. Do not repeat upload. |
| `submission92.62.csv` | Joint public order/share search from 92.31, officially **92.62**. The private rows still match 89.45. | Highest observed public score, but **not** one of the selected final files. |
| `candidates/private_closure_conditional_lean_w221.csv` | SK4 price-aware lean private policy with the MERROW closure forecast component removed down to the independent-risk floor. Public **89.45**; rows match its SK4 parent. | **Selected final file 1 of 2.** No private score observed. |
| `candidates/private_closure_conditional_insured_w221.csv` | SK4 insured private policy with the MERROW closure forecast conditioned on the event occurring. Public **89.45**; rows match its SK4 parent. | **Selected final file 2 of 2.** No private score observed. |
| `archive/experiments/private_risk_hedge_w208.csv` | Unscored private share hedge derived from `submission60.66.csv`. All 130 public rows are identical; 91 of 130 private rows have changed supplier shares. Forecasts and order quantities are identical throughout. | **Do not upload**: it must reproduce 60.66 publicly, and its average chronological private-window score fell in the local replay. Private benefit remains unverified. |

The controlled public-score comparisons are `44.95 → 50.44` for forecast changes only (**+5.49**), `50.44 → 54.10` for order/share changes only (**+3.66**), `60.66 → 67.74` for order/share changes only (**+7.08**), `67.74 → 66.98` for lithium orders and shares (**−0.76**), `67.74 → 67.26` for nine lithium share changes with identical orders and forecasts (**−0.48**), `89.45 → 91.36` for public allocation and rounding repairs (**+1.91**), `91.36 → 92.31` for stronger public diversification (**+0.95**), and `92.31 → 92.62` for jointly changed public orders and shares (**+0.31**). The last four files share exact forecasts and private rows. A public score cannot validate their private policies.

## Archived experiments

`archive/balanced_w208.csv`, `archive/cost_w208.csv`, and `archive/resilience_w208.csv` are early policy variants. `archive/experiments/cost_lean_w208.csv`, `news_primary_w208.csv`, `shock_hedge_w208.csv`, `private_risk_hedge_w208.csv`, and `correlated_lane_buffer_w208.csv` are later unscored experiments. The correlated buffer was moved here after joint-event scenarios predicted a loss. They are retained for analysis, not recommended as uploads given the scored alternatives.

The exact duplicate `archive/policy_v1_w208.csv` was removed because its bytes matched `submission54.10.csv`. There are currently **no full-data duplicate CSVs**. The two active private candidates intentionally share their public rows; the 60.66/archived-hedge pair also shares its public rows. Check again after any new CSV is created:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.submission_audit
```

The audit checks each CSV's header, IDs, probabilities, prices, orders, eligible country shares, and share sums, then compares normalized numeric content by ID. It reports full-data and public-week duplicate groups.

The source package and notebook for 60.66 are in `methods/lean_60_66/`; the 67.74 package and notebook are in `methods/lean_split_67_74/`. The newer week-221 SK4 source is in `methods/sk4_phase2/`. `experiments/private_closure_conditional.py` builds the selected pair from the two archived SK4 parents and Phase 2 data. `methods/check_provenance.py` rebuilds eleven week-208 graded CSVs, five Phase 2 scored files, and the selected final pair; fifteen graded files match exactly, and 92.62 matches private rows and forecasts with small public optimization drift. A public score cannot validate weeks 222–234.
