# Submission inventory (week-208 information)

The CSVs directly in this folder are **exact files already graded on the public leaderboard**, named by their reported score. Preserve them as evidence. `candidates/` contains unscored alternatives; `archive/` contains old experiments. All 15 stored CSVs have the required 260 rows and pass `CompetitionData.validate_file`.

| File | Purpose and observed result | Submit or test again? |
| --- | --- | --- |
| `submission44.95.csv` | Original equal-share, 1.2× demand policy; public **44.95**. | No. Historical baseline. |
| `submission45.80.csv` | Lead-aware share variant of the independent model; public **45.80**. | No. |
| `submission46.21.csv` | Longer-memory price variant of `submission46.22.csv`; public **46.21**. | No. |
| `submission46.22.csv` | Independent forecast and ordering approach; public **46.22**. | No. |
| `submission48.55.csv` | Distinct graded candidate; public **48.55**. Its exact generating code/provenance is not established here. | No. |
| `submission50.44.csv` | Forecast-only hybrid: same orders and shares as `submission44.95.csv`, different probability and price forecasts; public **50.44**. | No. Forecast control. |
| `submission54.10.csv` | Full teammate policy: same forecasts as `submission50.44.csv`, different orders and shares; public **54.10**. | No. Policy control. |
| `submission60.66.csv` | `lean_hedged` full policy; highest confirmed public score **60.66**. | Retain as current leading final-file option. Do not spend another public trial on the same file. |
| `candidates/private_risk_hedge_w208.csv` | Unscored private share hedge derived from `submission60.66.csv`. All 130 public rows are identical; 91 of 130 private rows have changed supplier shares. Forecasts and order quantities are identical throughout. | **Do not upload to test the public leaderboard**: it must reproduce 60.66 there. Consider only as a second final-file option if private shock resilience is valued after the week-221 update. Private benefit is unverified. |

The controlled public-score comparisons are `44.95 → 50.44` for forecast changes only (**+5.49**) and `50.44 → 54.10` for order/share changes only (**+3.66**). The `60.66` improvement mixes several changes, so it does not isolate one component.

## Archived experiments

`archive/balanced_w208.csv`, `archive/cost_w208.csv`, and `archive/resilience_w208.csv` are early policy variants. `archive/experiments/cost_lean_w208.csv`, `news_primary_w208.csv`, and `shock_hedge_w208.csv` are later unscored variants of the older pipeline. They are retained for analysis, not recommended as uploads given the scored alternatives.

The exact duplicate `archive/policy_v1_w208.csv` was removed because its bytes matched `submission54.10.csv`. There are currently **no full-data duplicate CSVs**. The 60.66/hedge pair is the sole public-week duplicate and differs in the private half. Check again after any new CSV is created:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.submission_audit
```

The audit checks each CSV's header, IDs, probabilities, prices, orders, eligible country shares, and share sums, then compares normalized numeric content by ID. It reports full-data and public-week duplicate groups.

`lean_hedged.ipynb` in the project root describes the 60.66 method, but imports `src/dsdg_supply_chain`, which is not present in this workspace. Its saved output is evidence of the earlier run; the notebook cannot currently regenerate the file here. When week-221 data arrives, refresh forecasts and the private policy using the complete source package before choosing final files. A public score cannot validate weeks 222–234.
