"""Tests for the backtest harness and the baselines (forecast.backtest, forecast.models)."""

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import (add_week_index, bias, demand_type, make_origins, split,
                               to_dept_level, wape)
from forecast.models import baseline_forecasts

N_WEEKS = 70


def wm_yr_wk(idx: int) -> int:
    """Walmart-style week code: jumps at year ends (11152 -> 11201), like M5."""
    return 11101 + (idx // 52) * 100 + idx % 52


def make_weekly(series: dict[str, tuple[int, list[float]]]) -> pd.DataFrame:
    """Build a weekly frame. `series` maps item_id -> (first week, sales list)."""
    rows = []
    for item, (start, sales) in series.items():
        for k, s in enumerate(sales):
            idx = start + k
            rows.append({"item_id": item, "store_id": "TX_1", "dept_id": "D1",
                         "wm_yr_wk": wm_yr_wk(idx), "sales": s})
    return pd.DataFrame(rows)


@pytest.fixture
def weekly() -> pd.DataFrame:
    """'old': sales = week number, all 70 weeks. 'young': listed from week 30."""
    return add_week_index(make_weekly({
        "old": (0, list(range(N_WEEKS))),
        "young": (30, [5.0] * (N_WEEKS - 30)),
    }))


# ----------------------------------------------------------------- week index

def test_week_index_is_consecutive_across_year_end(weekly):
    old = weekly[weekly.item_id == "old"]
    assert old["week_idx"].tolist() == list(range(N_WEEKS))
    assert old["wm_yr_wk"].iloc[52] - old["wm_yr_wk"].iloc[51] > 1  # the jump exists


def test_week_index_rejects_series_with_a_gap():
    df = make_weekly({"a": (0, [1.0] * 10), "b": (0, [1.0] * 10)})
    df = df.drop(index=df[(df.item_id == "a")].index[4])  # remove one week of 'a'
    with pytest.raises(ValueError):
        add_week_index(df)


# -------------------------------------------------------- origins and splits

def test_origins_cover_last_16_weeks_without_overlap():
    origins = make_origins(277)
    assert origins == [260, 264, 268, 272]
    targets = [o + h for o in origins for h in (1, 2, 3, 4)]
    assert sorted(targets) == list(range(261, 277))


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_split_never_trains_on_targets_after_the_origin(h):
    frame = pd.DataFrame({"origin_idx": np.repeat(np.arange(100), 2)})
    for origin in (60, 70, 80):
        train, test = split(frame, origin, h)
        assert (train["origin_idx"] + h).max() == origin  # latest target = origin, not later
        assert (test["origin_idx"] == origin).all()


# -------------------------------------------------------------------- metrics

def test_wape_and_bias_on_known_numbers():
    actual = pd.Series([10, 0, 5])
    assert wape(actual, pd.Series([8, 2, 5])) == pytest.approx(4 / 15)
    assert bias(actual, pd.Series([8, 2, 5])) == pytest.approx(0)
    assert bias(actual, pd.Series([12, 0, 5])) > 0  # over-forecast -> positive bias


def test_dept_level_lets_item_errors_cancel():
    fc = pd.DataFrame({"origin_idx": [0, 0], "h": [1, 1], "dept_id": ["D1", "D1"],
                       "store_id": ["TX_1", "TX_1"], "actual": [5, 5], "m": [7.0, 3.0]})
    dept = to_dept_level(fc, ["m"])
    assert wape(fc["actual"], fc["m"]) == pytest.approx(0.4)
    assert wape(dept["actual"], dept["m"]) == pytest.approx(0.0)


# ------------------------------------------------------------------ baselines

def get(fc: pd.DataFrame, item: str, origin: int, h: int) -> pd.Series:
    row = fc[(fc.item_id == item) & (fc.origin_idx == origin) & (fc.h == h)]
    assert len(row) == 1
    return row.iloc[0]


def test_baselines_give_expected_values(weekly):
    fc = baseline_forecasts(weekly, origins=[60])
    row = get(fc, "old", 60, 2)               # 'old' sells exactly its week number
    assert row["actual"] == 62
    assert row["naive"] == 60
    assert row["ma4"] == pytest.approx((57 + 58 + 59 + 60) / 4)
    assert row["snaive"] == 60 + 2 - 52      # same week last year
    assert not row["snaive_fallback"]


def test_seasonal_naive_falls_back_for_young_series(weekly):
    fc = baseline_forecasts(weekly, origins=[60])
    row = get(fc, "young", 60, 1)             # week 9 is before 'young' was listed
    assert row["snaive_fallback"]
    assert row["snaive"] == row["naive"]


def test_ma4_uses_available_weeks_for_very_young_series():
    df = add_week_index(make_weekly({"old": (0, [0.0] * 70), "new": (59, [2.0, 4.0] + [0.0] * 9)}))
    fc = baseline_forecasts(df, origins=[60])
    assert get(fc, "new", 60, 1)["ma4"] == pytest.approx(3.0)


def test_baselines_do_not_use_sales_after_the_origin(weekly):
    """Leakage test: scrambling every sale after the origin must not move a forecast."""
    origin = 60
    before = baseline_forecasts(weekly, origins=[origin])

    rng = np.random.default_rng(42)
    future = weekly["week_idx"] > origin
    scrambled = weekly.copy()
    scrambled.loc[future, "sales"] = rng.integers(1000, 9999, future.sum())
    after = baseline_forecasts(scrambled, origins=[origin])

    cols = ["naive", "ma4", "snaive", "snaive_fallback"]
    pd.testing.assert_frame_equal(before[cols], after[cols])
    assert not before["actual"].equals(after["actual"])  # the scramble really happened


# ---------------------------------------------------------------- demand type

def test_demand_type_classes_and_cutoff():
    pattern = {
        "smooth": [10.0] * 8,                 # sells every week, steady
        "erratic": [1.0, 20.0] * 4,           # sells every week, sizes vary a lot
        "intermittent": [10.0, 0.0] * 4,      # every other week, steady size
        "lumpy": [1.0, 0.0, 20.0, 0.0] * 2,   # gaps and varying sizes
    }
    # After the cutoff every series turns smooth; that must not affect the label.
    df = add_week_index(make_weekly({k: (0, v + [10.0] * 8) for k, v in pattern.items()}))
    types = demand_type(df, cutoff_idx=7)
    for item in pattern:
        assert types.loc[(item, "TX_1")] == item
