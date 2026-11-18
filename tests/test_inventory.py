"""Tests for the inventory simulation (forecast.inventory)."""

import numpy as np
import pandas as pd
import pytest

from forecast.inventory import inventory_at_fill_rate, simulate, summarize, tradeoff_curve


def test_hand_computed_example():
    """S = 6 at every review, demand 3, 5, 0, 4. Worked out week by week:

    start: on hand 6 (= S). Review week 0: order 6 - 6 = 0.
    week 1: sell 3 -> 3 left. Review: order 6 - 3 - 0 = 3, arrives week 3.
    week 2: demand 5, only 3 on hand -> sell 3, lose 2, 0 left.
            Review: on order 3 -> order 6 - 0 - 3 = 3, arrives week 4.
    week 3: +3 arrives, demand 0 -> 3 left. (No review: S only given for weeks 0-2.)
    week 4: +3 arrives -> 6, sell 4 -> 2 left.
    """
    demand = np.array([[3.0, 5.0, 0.0, 4.0]])
    s = np.full((1, 3), 6.0)
    sim = simulate(demand, s)
    np.testing.assert_array_equal(sim["sold"], [[3, 3, 0, 4]])
    np.testing.assert_array_equal(sim["lost"], [[0, 2, 0, 0]])
    np.testing.assert_array_equal(sim["end_on_hand"], [[3, 0, 3, 2]])

    summary = summarize(sim, demand)
    assert summary["fill_rate"] == pytest.approx(10 / 12)
    assert summary["avg_on_hand"] == pytest.approx(2.0)
    assert summary["stockout_week_share"] == pytest.approx(1 / 4)


def test_lost_sales_are_not_backordered():
    """A stockout loses the demand; it does not come back as extra demand later."""
    demand = np.array([[10.0, 0.0, 0.0, 0.0]])
    sim = simulate(demand, np.full((1, 3), 4.0))
    assert sim["sold"].sum() == 4
    assert sim["lost"].sum() == 6


def test_steady_demand_with_two_weeks_cover_never_stocks_out():
    d = 7.0
    demand = np.full((3, 12), d)
    sim = simulate(demand, np.full((3, 11), 2 * d))
    assert not sim["stockout"].any()
    assert summarize(sim, demand)["fill_rate"] == pytest.approx(1.0)


def test_order_arrives_after_one_week_lead_time():
    """Order placed at the end of week 1 is on the shelf at the start of week 3, not 2."""
    demand = np.array([[5.0, 5.0, 5.0]])
    sim = simulate(demand, np.array([[5.0, 10.0, 10.0]]))
    # week 1 sells the initial 5; week 2 has nothing (order still in transit); week 3 gets it.
    np.testing.assert_array_equal(sim["sold"], [[5, 0, 5]])


def test_more_stock_never_lowers_fill_rate():
    rng = np.random.default_rng(42)
    demand = rng.poisson(3, size=(50, 16)).astype(float)
    s = rng.uniform(2, 10, size=(50, 15))
    curve = tradeoff_curve(demand, s)
    assert curve["fill_rate"].is_monotonic_increasing
    assert curve["avg_on_hand"].is_monotonic_increasing


def test_inventory_at_fill_rate_interpolates():
    curve = pd.DataFrame({"fill_rate": [0.8, 0.9, 1.0], "avg_on_hand": [2.0, 4.0, 8.0]})
    assert inventory_at_fill_rate(curve, 0.85) == pytest.approx(3.0)
    assert np.isnan(inventory_at_fill_rate(curve, 0.5))


def test_arrivals_record_the_week_an_order_lands():
    """Hand example above: orders of 3 placed at the end of weeks 1 and 2 land in weeks 3 and 4."""
    sim = simulate(np.array([[3.0, 5.0, 0.0, 4.0]]), np.full((1, 3), 6.0))
    assert sim["arrivals"][0, 3] == 3
    assert sim["arrivals"][0, 4] == 3
    assert sim["arrivals"][0].sum() == 6
