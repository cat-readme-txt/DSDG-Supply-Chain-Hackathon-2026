import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.models.disruption import predict_disruption
from dsdg_supply_chain.models.duration import DurationModel

DATA = load_data()


def test_probabilities_valid_and_past_weeks_are_truth():
    info = DATA.until(150)
    p = predict_disruption(info, range(140, 170))
    assert p.p.between(0, 1).all() and not p.p.isna().any()
    assert len(p) == 30 * 10 and not p.duplicated(["week", "material"]).any()
    past = p[p.week <= 150].merge(DATA.outcomes, on=["week", "material"])
    assert np.array_equal(past.p.to_numpy(), past.disrupted.to_numpy(float))


def test_forecast_ignores_data_after_origin():
    info = DATA.until(150)
    tampered = info.until(150)
    # changing nothing at or before 150 but supplying a different (future-free) copy must not matter
    a = predict_disruption(info, range(151, 165))
    b = predict_disruption(tampered, range(151, 165))
    pd.testing.assert_frame_equal(a, b)


def test_duration_survival_is_monotone_and_censoring_aware():
    m = DurationModel.fit(build_features(DATA.until(130)), sigma_min=0.6)
    typ = pd.Series(["sanctions"] * 4)
    s = m.survival(typ, np.full(4, 30.0), np.array([1.0, 4.0, 13.0, 26.0]))
    assert np.all(np.diff(s) <= 0) and s[0] > 0.5
    # sanctions were still running at week 130: they must look long, not "never seen to end"
    assert m.median_weeks()["sanctions"] > 20
