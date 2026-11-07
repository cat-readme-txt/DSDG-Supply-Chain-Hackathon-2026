import numpy as np

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.strategies import policy_v1
from dsdg_supply_chain.submission import validate

DATA = load_data()


def test_policy_submission_is_valid_for_test_weeks():
    weeks = range(209, 235)
    sub = policy_v1(DATA, weeks)
    assert validate(sub, DATA, weeks) == []


def test_final_specialists_are_valid_and_differ():
    from dsdg_supply_chain.strategies import insured, lean_lp
    weeks = range(209, 235)
    a, b = lean_lp(DATA, weeks), insured(DATA, weeks)
    assert validate(a, DATA, weeks) == [] and validate(b, DATA, weeks) == []
    # same forecasts, different ordering: the insured plan carries more stock mid-window
    assert np.allclose(a.disruption_prob, b.disruption_prob)
    assert b.order_units.sum() > a.order_units.sum()


def test_lp_plan_matches_hindsight_when_given_the_truth():
    from dsdg_supply_chain.labels import supplier_state as labels
    from dsdg_supply_chain.policy_lp import LPParams, plan_orders_lp
    from dsdg_supply_chain.scoring import score_window
    from dsdg_supply_chain.strategies import naive
    info, win = DATA.until(143), range(144, 157)
    t = labels(DATA)
    # oracle state: the true capacity, lead time and cost of every order week, no uncertainty
    t = t[t.week.isin(win)].assign(p=0.0, severity=1.0, cost=lambda d: d.landed_cost, hit=0, lead_delta=0,
                                   shock_type=None)
    t["base_lead_weeks"] = t.lead_time_weeks
    t["normal_capacity_units"] = t.capacity_units
    plan = plan_orders_lp(info, win, None, None, LPParams(n_scenarios=1, min_share=0.0, hazard_mult=0.0),
                          state=t).set_index(["week", "material"])
    sub = naive(info, range(144, 170)).set_index(["week", "material"])
    sub.loc[plan.index, ["order_units", *SHARE_COLS]] = plan[["order_units", *SHARE_COLS]]
    ws = score_window(sub.reset_index(), naive(info, range(144, 170)), DATA, win)
    assert ws.F > 0.99   # with perfect information the stochastic program is the hindsight optimum


def test_policy_respects_suppliers_and_capacity():
    sub = policy_v1(DATA.until(130), range(131, 157))
    assert (sub.order_units >= 0).all()
    assert np.allclose(sub[SHARE_COLS].sum(axis=1), 1)
    sup = DATA.suppliers.set_index(["material", "country"]).normal_capacity_units
    for c in [c[6:] for c in SHARE_COLS]:
        rows = sub[sub[f"share_{c}"] > 0]
        slice_units = rows.order_units * rows[f"share_{c}"]
        cap = [sup.get((m, c), 0) for m in rows.material]
        assert (slice_units <= np.array(cap) * 1.0001).all(), c   # never ask a supplier beyond capacity


def test_policy_buys_lean_in_calm_window():
    # total bought should be close to what the window needs (demand minus 5 weeks starting stock)
    sub = policy_v1(DATA.until(143), range(144, 170))
    win = sub[sub.week.between(144, 156)]
    dem = DATA.demand[DATA.demand.week.between(144, 156)].groupby("material").demand_units.sum()
    first5 = DATA.demand[DATA.demand.week.between(144, 148)].groupby("material").demand_units.sum()
    ratio = win.groupby("material").order_units.sum() / (dem - first5)
    assert ratio.between(0.9, 1.4).all(), ratio
