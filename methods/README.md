# Source provenance for scored submissions

Every graded CSV lives once, in `submissions/submission<score>.csv`. These method directories hold the code and notebooks that reproduce them. `data/snapshots/w208/` freezes the ten input CSVs used for all week-208 files, so later data releases cannot silently change a reproduction.

| Public score | Exact generating code in this project | Method |
| ---: | --- | --- |
| 44.95 | `pipeline.py`, profile `primary` | Original local policy. |
| 45.80 | `pipeline.py`, profile `lead_aware` | Lead-aware supplier allocation. |
| 46.21 | `pipeline.py`, profile `long_memory` | Longer-memory price forecast. |
| 46.22 | `pipeline.py`, profile `independent` | Independent local model. |
| 48.55 | `methods/lean_60_66/src`, strategy `insured_hedged` | Insured specialist. |
| 50.44 | `methods/policy_v1_54_10/build_forecast_hybrid.py` | 54.10 forecasts with 44.95 orders and shares. |
| 54.10 | `methods/policy_v1_54_10/src`, strategy `policy_v1` | Earlier full policy. |
| 60.66 | `methods/lean_60_66/src`, strategy `lean_hedged` | Scenario LP with calm forecast hedge. |
| 66.98 | `experiments/dual_hazard_three_lane.py` | Adds a third lithium supplier during the public window; actual public score fell from 67.74. |
| 67.26 | `experiments/lithium_share_only.py` | Changes only nine public lithium supplier shares; actual public score fell from 67.74. |
| 67.74 | `methods/lean_split_67_74/src`, strategy `lean_split` | 60.66 approach plus near-cost supplier share constraint. |

The 54.10 source is the `SK3-67.74` repository at Git commit `eb85f21`. The 60.66 source and notebook match commit `1158114` exactly. The 67.74 directory is a copy of the `SK3-67.74` repository at commit `527bfba`, excluding its `.git`, duplicate data, and duplicate submission CSVs. It includes the research scripts, tests, reports, decision log, and notebooks. The unmatched Kaggle baseline notebook is kept separately in `legacy_unmapped/` because no scored CSV has been proven to come from it.

Run the byte-for-byte reproduction check with a Python environment containing NumPy, pandas, SciPy, and scikit-learn:

```sh
python methods/check_provenance.py --python /path/to/python-with-dependencies
```

The checker uses only the frozen week-208 inputs and a temporary output directory. On November 7, it reproduced all eleven scored CSVs byte for byte. The scored files are not rewritten.
