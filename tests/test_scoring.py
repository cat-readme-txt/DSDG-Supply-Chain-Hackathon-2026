import numpy as np
import pandas as pd
import pytest

from dsdg_supply_chain.data import SHARE_COLS, load_data
from dsdg_supply_chain.scoring import score_window
from dsdg_supply_chain.simulator import hindsight_plan, simulate, truth_for
from dsdg_supply_chain.strategies import naive
from dsdg_supply_chain.submission import read_submission, validate

DATA = load_data()
WEEKS = range(99, 112)          # KES sanctions + Merrow closure: a window full of shocks
FULL = range(99, 125)


@pytest.fixture(scope="module")
def base():
    return naive(DATA.until(98), FULL)


def test_naive_replicates_sample_submission():
    ours = naive(DATA.until(208), range(209, 235)).set_index("id")   # the sample is built from week 208
    sample = read_submission("data/comp/sample_submission.csv").set_index("id").loc[ours.index]
    for col in ["order_units", *SHARE_COLS]:
        assert np.allclose(ours[col], sample[col]), col
    assert np.allclose(ours.disruption_prob, sample.disruption_prob, atol=1e-4)
    assert np.allclose(ours.price_forecast, sample.price_forecast, atol=1e-4)


def test_sample_submission_is_valid():
    assert validate(read_submission("data/comp/sample_submission.csv"), DATA) == []


def test_validate_catches_bad_shares():
    bad = read_submission("data/comp/sample_submission.csv")
    bad.loc[0, "share_SUN"] = 0.5  # lithium is not sold by SUN, and shares no longer sum to 1
    errs = validate(bad, DATA)
    assert any("sum to 1" in e for e in errs) and any("share_SUN" in e for e in errs)


def test_until_has_no_leakage():
    snap = DATA.until(100)
    for name in ["macro", "news", "outcomes", "prices", "shipping", "supply_history", "tariffs"]:
        assert getattr(snap, name).week.max() == 100
    assert snap.demand.week.max() == DATA.demand.week.max()


def test_naive_scores_zero(base):
    ws = score_window(base, base, DATA, WEEKS)
    assert ws.score == 0 and ws.P == 0 and ws.F == 0 and ws.R == 0


def test_hindsight_plan_scores_full_financial_marks(base):
    sub = base.copy()
    for m in sub.material.unique():
        truth = truth_for(DATA, m, WEEKS)
        cost, units = hindsight_plan(truth)
        orders = units.sum(axis=0)
        shares = np.divide(units, orders, out=np.zeros_like(units), where=orders > 0)
        shares[0, orders == 0] = 1.0
        assert simulate(truth, orders, shares).total_cost == pytest.approx(cost, rel=1e-6)
        idx = sub.index[(sub.material == m) & sub.week.isin(WEEKS)]
        sub.loc[idx, "order_units"] = orders
        sub.loc[idx, SHARE_COLS] = 0.0
        for s, c in enumerate(truth.countries):
            sub.loc[idx, f"share_{c}"] = shares[s]
    ws = score_window(sub, base, DATA, WEEKS)
    assert ws.F == pytest.approx(1.0, abs=1e-6)
    assert ws.fill_rate >= ws.naive_fill_rate


def test_simulator_capacity_and_cancellation():
    truth = truth_for(DATA, "lithium", WEEKS)
    huge = np.full(len(WEEKS), 1e9)
    shares = np.zeros((3, len(WEEKS)))
    shares[0] = 1.0
    res = simulate(truth, huge, shares)
    # only capacity is filled and charged; the rest is cancelled
    assert np.allclose(res.filled[0], truth.capacity[0])
    assert res.purchase == pytest.approx((truth.capacity[0] * truth.cost[0]).sum())
