# Independent methods tested at week 208

These experiments use the frozen inputs in `data/snapshots/w208/`. For each historical forecast origin, training uses only target outcomes and prices released by that origin. Future demand is allowed because the competition supplies it in advance. Windows with origins 78, 104, and 130 form the development set; origins 156 and 182 form the later chronological check. The local scorer is an approximation of the official simulator, so these scores rank hypotheses rather than predict a Kaggle score exactly.

| Method | What was learned independently | Development mean | Later mean | Decision |
| --- | --- | ---: | ---: | --- |
| Scored 67.74 policy, replayed historically | Benchmark for the same folds | 51.84 | 41.82 | Keep as incumbent. |
| `direct_forecast.py`: raw-table gradient boosting for both disruption and price | Material-level forecast from past outcomes, prices, supplier capacity, shipping, macro, tariffs, news keywords, and known demand | 50.59 | 36.10 | Reject. |
| Direct model, probability only | Same probability learner, incumbent price | 51.15 | 36.58 | Reject. |
| Half blend of both direct forecasts | Equal mix of independent and incumbent forecasts | 52.53 | 40.23 | Reject: later fold is worse. |
| `hindsight_imitation.py`: learned purchase policy at 1.2× scale | Supplier-week order quantities learned from historical perfect-information plans; each training window must end by the forecast cutoff | 37.61 | 22.43 | Reject. |
| `empirical_allocator.py`: historical supply-block optimizer | Supplier shares chosen by simulating the known future demand and the same orders across released 13-week supply blocks; diverse objective shown | 49.42 | 32.49 | Reject. |
| Empirical allocator, half blend with 67.74 shares | Same supply-block objective with partial incumbent allocation | 49.99 | 33.54 | Reject. |

The direct forecast test is in `reports/direct_forecast_w208.json`; the ordering test is in `reports/hindsight_imitation_w208.json`; the supply-block test is in `reports/empirical_allocator_w208.json`. None produced an upload CSV. To rerun them, use a Python environment with NumPy, pandas, SciPy, and scikit-learn:

```sh
python -m scratch.evaluate_direct
python -m scratch.evaluate_imitation
python -m scratch.evaluate_allocator
```

Both evaluation scripts import the archived 67.74 source only to produce the benchmark policy and calculate matched local scores. The new feature builder and learned purchase policy are in this folder. This dataset is small enough for CPU runs; a GPU would shorten fitting time but would not create additional training labels.
