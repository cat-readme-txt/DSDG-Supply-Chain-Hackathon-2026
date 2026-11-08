import numpy as np

from dsdg_supply_chain.models.cost_path import scenario_costs


def _costs(hit, hit0, typ, announced=(False, False), hike=None):
    landed0, spot0 = np.array([110.0, 120.0]), np.array([100.0, 110.0])
    tariff0, freight0 = np.array([0.05, 0.0]), np.array([5.0, 10.0])
    hike = np.repeat(tariff0[:, None], hit.shape[2], axis=1) if hike is None else hike
    return scenario_costs(landed0, spot0, tariff0, freight0, freight0.copy(), tariff0.copy(), hike,
                          np.array(announced), np.array(hit0), typ, hit,
                          {"shortage": np.log(1.2)}, {"closure": 3.0})


def test_untouched_supplier_weeks_keep_todays_cost_exactly():
    hit = np.zeros((2, 2, 4), bool)
    c = _costs(hit, [False, False], ["shortage", None])
    assert (c == np.array([110.0, 120.0])[None, :, None]).all()


def test_source_shock_lifts_every_supplier_spot_while_it_lasts():
    hit = np.zeros((1, 2, 4), bool)
    hit[0, 0, 1:3] = True                            # supplier 0 short in weeks 1-2
    c = _costs(hit, [False, False], ["shortage", None])
    assert np.isclose(c[0, 1, 1], 110 * 1.2 + 10) and np.isclose(c[0, 1, 3], 120.0)
    assert np.isclose(c[0, 0, 2], 100 * 1.2 * 1.05 + 5)


def test_ongoing_shock_loses_its_uplift_after_the_scenario_end():
    hit = np.zeros((1, 2, 4), bool)
    hit[0, 0, :2] = True                             # ongoing today, ends after week 1
    c = _costs(hit, [True, False], ["shortage", None])
    assert np.isclose(c[0, 1, 0], 120.0) and np.isclose(c[0, 1, 2], 110 / 1.2 + 10)


def test_new_lane_hit_multiplies_freight_only():
    hit = np.zeros((1, 2, 3), bool)
    hit[0, 1, 1] = True
    c = _costs(hit, [False, False], [None, "closure"])
    assert np.isclose(c[0, 1, 1], 110 + 30) and np.isclose(c[0, 0, 1], 110.0)
