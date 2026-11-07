import numpy as np
import pandas as pd

from dsdg_supply_chain.data import load_data
from dsdg_supply_chain.features.build import build_features
from dsdg_supply_chain.models.disruption import supplier_forecast
from dsdg_supply_chain.models.price import PriceModel, predict_price

DATA = load_data()


def test_prices_positive_complete_and_past_is_truth():
    info = DATA.until(150)
    weeks = range(140, 170)
    sup = supplier_forecast(info, weeks)
    px = predict_price(info, weeks, sup)
    assert len(px) == 300 and not px.duplicated(["week", "material"]).any()
    assert (px.price > 0).all() and np.isfinite(px.price).all()
    past = px[px.week <= 150].merge(DATA.outcomes, on=["week", "material"])
    assert np.allclose(past.price, past.benchmark_price)


def test_no_shock_means_carry_forward():
    info = DATA.until(150)
    weeks = range(151, 160)
    sup = supplier_forecast(info, weeks).assign(p=0.0, hit=0)
    px = PriceModel(uplift={"ban": 0.3}).predict(info, sup, weeks)
    last = info.outcomes[info.outcomes.week == 150].set_index("material").benchmark_price
    assert np.allclose(px.price, px.material.map(last))


def test_likely_new_shock_lifts_price():
    info = DATA.until(150)
    weeks = range(151, 155)
    sup = supplier_forecast(info, weeks).assign(p=0.0, hit=0, shock_type=None)
    sel = (sup.material == "copper") & (sup.country == "ORI")
    sup.loc[sel, ["p", "shock_type"]] = [0.9, "ban"]
    px = PriceModel(uplift={"ban": 0.3}).predict(info, sup, weeks).set_index(["week", "material"]).price
    last = info.outcomes[info.outcomes.week == 150].set_index("material").benchmark_price
    assert np.isclose(px.loc[(151, "copper")], last["copper"] * np.exp(0.3))
    assert np.isclose(px.loc[(151, "steel")], last["steel"])


def test_uplifts_learned_without_future():
    a = PriceModel.fit(DATA.until(170), build_features(DATA.until(170))).uplift
    assert all(v >= 0 for v in a.values()) and "ban" in a
