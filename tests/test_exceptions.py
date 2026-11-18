"""Tests for the exception flags and the order plan (forecast.exceptions).

Toy data: `make_inputs` turns a dict of weekly sales per item into the same
two tables the pipeline uses: a weekly sales table and a forecast table with
one row per series x origin x horizon. Every toy forecast made at origin o is
the mean of the last 4 weeks of sales up to o, so it never sees later sales.
"""

import numpy as np
import pandas as pd
import pytest

from forecast.exceptions import (build_exceptions, drift_flags, order_plan, stockout_flags,
                                 zero_run_flags)

STORE = "TX_1"


def make_inputs(sales: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly, fc = [], []
    for item, s in sales.items():
        s = np.asarray(s, float)
        n = len(s)
        weekly.append(pd.DataFrame({"item_id": item, "store_id": STORE,
                                    "week_idx": np.arange(n), "sales": s}))
        for o in range(n):
            level = s[max(0, o - 3):o + 1].mean()
            for h in range(1, 5):
                if o + h < n:
                    fc.append({"item_id": item, "store_id": STORE, "dept_id": "D",
                               "demand_type": "smooth", "origin_idx": o, "h": h,
                               "target_idx": o + h, "actual": s[o + h], "ma4": level,
                               "hgbr": level, "p50": level, "p90": 1.5 * level + 1})
    return pd.concat(weekly, ignore_index=True), pd.DataFrame(fc)


def h1_rows(item: str, actual: float, forecast: float, as_of: int = 10) -> pd.DataFrame:
    """Four h=1 rows for weeks as_of-3..as_of, splitting the totals evenly."""
    weeks = np.arange(as_of - 3, as_of + 1)
    return pd.DataFrame({"item_id": item, "store_id": STORE, "h": 1, "target_idx": weeks,
                         "origin_idx": weeks - 1, "actual": actual / 4, "hgbr": forecast / 4})


# ------------------------------------------------------------------ drift

def test_drift_fires_beyond_threshold_with_volume():
    fc = pd.concat([h1_rows("big_miss", 40, 20),      # +100%, enough volume -> flag
                    h1_rows("small_miss", 29, 20),    # +45%: under the 50% threshold
                    h1_rows("low_volume", 10, 4)])    # +150% but under 20 units
    flags = drift_flags(fc, as_of=10)
    assert flags["item_id"].tolist() == ["big_miss"]
    assert flags["units_at_stake"].iloc[0] == pytest.approx(20)
    assert flags["reason"].iloc[0] == "Sold 40 units in the last 4 weeks vs 20 forecast (+100%)."


def test_drift_ignores_weeks_after_as_of():
    fc = pd.concat([h1_rows("a", 40, 40), h1_rows("a", 400, 0, as_of=14)])
    assert drift_flags(fc, as_of=10).empty


# ------------------------------------------------------------------ zero run

@pytest.mark.parametrize("sales, flagged", [
    ([3.0] * 13 + [0, 0, 0], True),                 # steady, then 3 zeros
    ([3.0] * 14 + [0, 0], False),                   # only 2 zeros
    ([1.5] * 13 + [0, 0, 0], False),                # steady but avg below 2/week
    ([5.0] * 11 + [0, 0] + [0, 0, 0], False),       # sold in 11 of 13 weeks (< 90%)
    ([3.0] * 13 + [0, 1, 0], False),                # one sale breaks the run
])
def test_zero_run_rule(sales, flagged):
    weekly, _ = make_inputs({"x": sales})
    flags = zero_run_flags(weekly, as_of=len(sales) - 1)
    assert (len(flags) == 1) is flagged


def test_zero_run_reason_and_units():
    weekly, _ = make_inputs({"x": [4.0] * 13 + [0, 0, 0]})
    row = zero_run_flags(weekly, as_of=15).iloc[0]
    assert row["units_at_stake"] == pytest.approx(12)
    assert row["reason"].startswith("No sales for 3 weeks after selling in 13 of the previous "
                                    "13 weeks (avg 4.0/week)")


def test_zero_run_ignores_items_not_yet_listed():
    """A series listed 5 weeks ago can't have a 13-week steady stretch."""
    weekly, _ = make_inputs({"x": [3.0] * 13 + [0, 0, 0]})
    weekly = weekly[weekly["week_idx"] >= 11]
    assert zero_run_flags(weekly, as_of=15).empty


# ------------------------------------------------------------------ stockout

def test_stockout_flags_shortfall_of_at_least_one_unit():
    plan = pd.DataFrame({"item_id": ["short", "tiny_gap", "ok"], "store_id": STORE,
                         "on_hand": [2.0, 4.6, 9.0], "arriving": [1.0, 0.0, 0.0],
                         "p50_w1": [6.0, 5.0, 5.0]})
    flags = stockout_flags(plan)
    assert flags["item_id"].tolist() == ["short"]
    assert flags["units_at_stake"].iloc[0] == pytest.approx(3)
    assert flags["reason"].iloc[0] == ("Simulated stock 2 + arriving 1 is below next week's "
                                       "median forecast of 6.")


def test_order_plan_matches_hand_simulation():
    """Same case as the hand example in test_inventory: S = 6 at every review,
    demand 3, 5, 0 in weeks 1-3. End of week 3: 3 on hand, 3 arriving in week 4."""
    _, fc = make_inputs({"x": [0.0, 3, 5, 0, 1, 1, 1, 1]})
    fc["p90"] = 3.0                                  # S = 3 + 3 = 6
    plan = order_plan(fc, as_of=3, first_review=0)
    assert plan.loc[0, "on_hand"] == pytest.approx(3)
    assert plan.loc[0, "arriving"] == pytest.approx(3)
    assert {"p50_w1", "p50_w4", "p90_w4", "ma4", "dept_id"} <= set(plan.columns)


# ------------------------------------------------------------------ leakage

def scramble_after(weekly: pd.DataFrame, week: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(0)
    out = {}
    for item, g in weekly.groupby("item_id"):
        s = g.sort_values("week_idx")["sales"].to_numpy().copy()
        s[week + 1:] = rng.integers(0, 500, len(s) - week - 1)
        out[item] = s.tolist()
    return out


def test_flags_and_plan_ignore_everything_after_as_of():
    """Scrambling every sale after the as-of week (and so every actual and
    forecast that depends on them) must not change a flag or the order plan."""
    rng = np.random.default_rng(42)
    sales = {
        "steady_then_zero": [5.0] * 27 + [0, 0, 0] + [5.0] * 6,      # zero run
        "spike": [2.0] * 29 + [40.0] + [2.0] * 6,                    # drift + stockout
        "noise": rng.poisson(4, 36).astype(float).tolist(),
    }
    as_of, first_review = 29, 20
    weekly, fc = make_inputs(sales)
    flags, plan = build_exceptions(fc, weekly, as_of, first_review)
    assert set(flags["flag"]) == {"drift", "projected_stockout", "zero_run"}

    weekly2, fc2 = make_inputs(scramble_after(weekly, as_of))
    flags2, plan2 = build_exceptions(fc2, weekly2, as_of, first_review)
    pd.testing.assert_frame_equal(flags, flags2)
    pd.testing.assert_frame_equal(plan, plan2)
