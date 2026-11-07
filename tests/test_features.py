import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features

DATA = load_data()


def test_feature_table_is_complete_and_clean():
    f = build_features(DATA)  # build_features runs check_features: rows, duplicates, NaN, finiteness
    assert len(f) == DATA.last_week * len(DATA.suppliers)


def test_features_use_only_past_data():
    full = build_features(DATA)
    snap = build_features(DATA.until(150))
    cols = snap.columns.tolist()
    pd.testing.assert_frame_equal(snap.reset_index(drop=True),
                                  full[full.week <= 150][cols].reset_index(drop=True), check_like=True)
