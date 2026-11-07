# Submission inventory (week-208 information)

The CSVs directly in this folder are **exact files already graded on the public leaderboard**, named by their reported score. Preserve them as evidence. `candidates/` holds distinct unscored tests; `archive/` contains old experiments. All 18 stored CSVs have the required 260 rows and pass `CompetitionData.validate_file`. [The method inventory](../methods/README.md) links each graded file to code that reproduces it byte for byte.

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
| `submission67.74.csv` | `lean_split` full policy; highest confirmed public score **67.74**. Exact source is archived under `methods/lean_split_67_74/`. | Current leading final-file option. Do not spend another public trial on the same file. |
| `submission66.98.csv` | Dual-hazard lithium test: replans 10 public lithium rows using Valdmar alongside Kestria and Orinoque. Forecasts and all private rows match 67.74. Public **66.98**. | No. Its public score is **0.76 lower** than 67.74. |
| `submission67.26.csv` | Controlled lithium supplier test: preserves 67.74 orders and forecasts; changes only nine public shares. No private changes. Public **67.26**. | No. Its public score is **0.48 lower** than 67.74. |
| `archive/experiments/private_risk_hedge_w208.csv` | Unscored private share hedge derived from `submission60.66.csv`. All 130 public rows are identical; 91 of 130 private rows have changed supplier shares. Forecasts and order quantities are identical throughout. | **Do not upload**: it must reproduce 60.66 publicly, and its average chronological private-window score fell in the local replay. Private benefit remains unverified. |

The controlled public-score comparisons are `44.95 → 50.44` for forecast changes only (**+5.49**), `50.44 → 54.10` for order/share changes only (**+3.66**), `60.66 → 67.74` for order/share changes only (**+7.08**), `67.74 → 66.98` for lithium orders and shares (**−0.76**), and `67.74 → 67.26` for nine lithium share changes with identical orders and forecasts (**−0.48**). The 66.98 file changes both orders and shares in ten public rows, so its score drop cannot be assigned to one field. The `54.10 → 60.66` improvement mixes several changes, so it does not isolate one component.

## Archived experiments

`archive/balanced_w208.csv`, `archive/cost_w208.csv`, and `archive/resilience_w208.csv` are early policy variants. `archive/experiments/cost_lean_w208.csv`, `news_primary_w208.csv`, `shock_hedge_w208.csv`, and `private_risk_hedge_w208.csv` are later unscored experiments. They are retained for analysis, not recommended as uploads given the scored alternatives.

The exact duplicate `archive/policy_v1_w208.csv` was removed because its bytes matched `submission54.10.csv`. There are currently **no full-data duplicate CSVs**. The 60.66/hedge pair is the sole public-week duplicate and differs in the private half. Check again after any new CSV is created:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.submission_audit
```

The audit checks each CSV's header, IDs, probabilities, prices, orders, eligible country shares, and share sums, then compares normalized numeric content by ID. It reports full-data and public-week duplicate groups.

The source package and notebook for 60.66 are now in `methods/lean_60_66/`; the 67.74 package and notebook are in `methods/lean_split_67_74/`. `methods/check_provenance.py` rebuilds all eleven graded CSVs from the frozen week-208 inputs and checks exact bytes. When week-221 data arrives, refresh forecasts and the private policy before choosing final files. A public score cannot validate weeks 222–234.
