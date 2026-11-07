import pandas as pd
import pytest

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.news import classify, expand_to_suppliers, news_features, parse_news
from dsdg_supply_chain.labels import supplier_state

DATA = load_data()
KEY = ["week", "material", "country"]


@pytest.fixture(scope="module")
def events():
    return expand_to_suppliers(parse_news(DATA.news), DATA.suppliers)


@pytest.fixture(scope="module")
def feats():
    return news_features(DATA)


def test_classify_examples():
    assert classify("Kestria lithium producers flag maintenance delays") == ("shortage", "early")
    assert classify("Deal reached, Atlantic route disruption averted") == (None, "called_off")
    assert classify("Weather stays mild across Halvard") is None


def test_every_signal_article_maps_to_a_supplier(events):
    assert set(parse_news(DATA.news).article_id) == set(events.article_id)


def test_onset_articles_match_real_hit_starts(events):
    st = supplier_state(DATA)
    hit = set(map(tuple, st[st.hit][KEY].to_numpy()))
    ons = events[events.stage == "onset"]
    # precision: every onset article points at a supplier that is hit that week or the next
    assert all((w, m, c) in hit or (w + 1, m, c) in hit for w, m, c in ons[KEY].to_numpy())
    # recall: every real hit start has an onset article within one week
    st["start"] = st.hit & ~st.groupby(["material", "country"]).hit.shift(1, fill_value=False)
    near = {(w + d, m, c) for w, m, c in ons[KEY].to_numpy() for d in (-1, 0, 1)}
    assert all(tuple(x) in near for x in st[st.start][KEY].to_numpy())


def test_severity_extracted_from_body():
    ev = parse_news(DATA.news)
    row = ev[(ev.week == 36) & ev.headline.str.contains("forecast cut")].iloc[0]
    assert row.severity == pytest.approx(0.67)


def test_called_off_story_is_cleared(feats):
    # Northern Sea strike threat (weeks 74-76) was called off at week 77
    f = feats.set_index(KEY)
    assert f.loc[(76, "rare_earths", "VAL"), "stage_strike"] == 3
    assert f.loc[(77, "rare_earths", "VAL"), "stage_strike"] == 0
    assert f.loc[(77, "rare_earths", "VAL"), "called_off_4w"] == 1


def test_features_use_only_past_articles(feats):
    snap = news_features(DATA.until(120))
    full = feats[feats.week <= 120].reset_index(drop=True)
    pd.testing.assert_frame_equal(snap, full)
