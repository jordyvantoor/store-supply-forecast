"""Tests for forecast.data on a tiny synthetic M5-shaped dataset.

Layout of the fixture (days start on a Saturday, like M5):
- Week 1 (d_1..d_7):   2011-12-17..23, complete
- Week 2 (d_8..d_14):  2011-12-24..30, complete, contains Dec 25
- Week 3 (d_15..d_16): 2011-12-31..2012-01-01, partial (data ends on a Sunday)
The calendar runs a few days past the sales data, as the real one does.
"""

import numpy as np
import pandas as pd
import pytest

from forecast.data import build_weekly

N_DAYS = 16
DAY_COLS = [f"d_{i}" for i in range(1, N_DAYS + 1)]
WEEKS = (1, 2, 3)  # wm_yr_wk codes for the fixture


@pytest.fixture
def calendar() -> pd.DataFrame:
    n = N_DAYS + 5
    dates = pd.date_range("2011-12-17", periods=n, freq="D")
    week = [WEEKS[min(i // 7, 2)] for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "wm_yr_wk": week,
        "d": [f"d_{i}" for i in range(1, n + 1)],
        "event_name_1": [None] * n,
        "snap_TX": [1 if i % 3 == 0 else 0 for i in range(n)],
    })


@pytest.fixture
def sales() -> pd.DataFrame:
    rows = [
        ("A", "TX_1"),  # listed from week 1
        ("B", "TX_1"),  # first price in week 2 -> week 1 must be dropped
        ("A", "TX_2"),
        ("A", "CA_1"),  # not Texas -> must not appear
    ]
    data = {
        "item_id": [r[0] for r in rows],
        "dept_id": ["FOODS_1"] * len(rows),
        "cat_id": ["FOODS"] * len(rows),
        "store_id": [r[1] for r in rows],
    }
    for j, col in enumerate(DAY_COLS):
        data[col] = np.arange(len(rows), dtype="int16") + j  # distinct, known values
    return pd.DataFrame(data)


@pytest.fixture
def prices() -> pd.DataFrame:
    rows = [("A", "TX_1", w) for w in WEEKS] + [("B", "TX_1", w) for w in (2, 3)] \
        + [("A", "TX_2", w) for w in WEEKS] + [("A", "CA_1", w) for w in WEEKS]
    return pd.DataFrame(rows, columns=["item_id", "store_id", "wm_yr_wk"]).assign(sell_price=1.5)


@pytest.fixture
def weekly(sales, calendar, prices) -> pd.DataFrame:
    return build_weekly(sales, calendar, prices, stores=("TX_1", "TX_2"), parallel=False)


def get(df: pd.DataFrame, item: str, store: str, week: int) -> pd.Series:
    row = df[(df.item_id == item) & (df.store_id == store) & (df.wm_yr_wk == week)]
    assert len(row) == 1
    return row.iloc[0]


def test_weekly_sales_equal_sum_of_daily(weekly, sales):
    a_tx1 = sales[(sales.item_id == "A") & (sales.store_id == "TX_1")].iloc[0]
    assert get(weekly, "A", "TX_1", 1)["sales"] == a_tx1[DAY_COLS[0:7]].sum()
    assert get(weekly, "A", "TX_1", 2)["sales"] == a_tx1[DAY_COLS[7:14]].sum()


def test_partial_final_week_is_dropped(weekly):
    assert set(weekly["wm_yr_wk"]) == {1, 2}


def test_weeks_before_first_price_are_dropped(weekly):
    b = weekly[weekly.item_id == "B"]
    assert b["wm_yr_wk"].tolist() == [2]


def test_christmas_week_has_six_open_days(weekly):
    week2 = weekly[weekly.wm_yr_wk == 2]
    week1 = weekly[weekly.wm_yr_wk == 1]
    assert (week2["days_open"] == 6).all() and week2["is_christmas_week"].all()
    assert (week1["days_open"] == 7).all() and not week1["is_christmas_week"].any()


def test_only_requested_stores(weekly):
    assert set(weekly["store_id"]) == {"TX_1", "TX_2"}


def test_one_row_per_item_store_week(weekly):
    assert not weekly.duplicated(["item_id", "store_id", "wm_yr_wk"]).any()


def test_parallel_matches_serial(sales, calendar, prices, weekly):
    par = build_weekly(sales, calendar, prices, stores=("TX_1", "TX_2"), parallel=True)
    pd.testing.assert_frame_equal(par, weekly)
