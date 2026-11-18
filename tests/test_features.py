"""Tests for forecast.features, including one leakage test per feature (CLAUDE.md rule 6).

The leakage tests are parametrized over the feature lists in forecast.features,
so a newly added feature is covered automatically.
"""

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import add_week_index
from forecast.features import (FEATURES, HORIZON_SALES_FEATURES, ID_FEATURES, KNOWN_FEATURES,
                               SALES_FEATURES, build_features, build_horizon_frame)

N_WEEKS = 70
ORIGIN = 55
PRICE_FEATURES = ["sell_price", "price_ratio_13", "price_rel_dept"]


def make_raw(seed: int = 42) -> pd.DataFrame:
    """Two stores x three items; item C is listed from week 20. Random sales and prices."""
    rng = np.random.default_rng(seed)
    rows = []
    for store in ("TX_1", "TX_2"):
        for item, start in (("A", 0), ("B", 0), ("C", 20)):
            for idx in range(start, N_WEEKS):
                rows.append({
                    "item_id": item, "dept_id": "D1", "cat_id": "C1", "store_id": store,
                    "wm_yr_wk": 11101 + (idx // 52) * 100 + idx % 52,
                    "sales": int(rng.poisson(3)),
                    "sell_price": float(rng.choice([1.0, 1.5, 2.0])),
                    "week_start": pd.Timestamp("2011-01-29") + pd.Timedelta(weeks=idx),
                    "n_snap_days": int(rng.integers(0, 6)), "n_event_days": 0,
                    "days_open": 7, "is_christmas_week": False,
                })
    df = pd.DataFrame(rows)
    df[ID_FEATURES] = df[ID_FEATURES].astype("category")
    return add_week_index(df)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="module")
def frames(raw) -> dict[int, pd.DataFrame]:
    feats = build_features(raw, parallel=False)
    return {h: build_horizon_frame(feats, h) for h in (1, 2, 3, 4)}


def scrambled_frame(raw: pd.DataFrame, column: str, after_week: int, h: int) -> pd.DataFrame:
    """Overwrite `column` for all weeks after `after_week` and rebuild the horizon frame."""
    rng = np.random.default_rng(0)
    df = raw.copy()
    future = df["week_idx"] > after_week
    df.loc[future, column] = rng.integers(500, 999, future.sum()).astype(df[column].dtype)
    return build_horizon_frame(build_features(df, parallel=False), h)


# ------------------------------------------------------------ leakage tests

@pytest.mark.parametrize("h", [1, 2, 3, 4])
@pytest.mark.parametrize("feature", SALES_FEATURES + HORIZON_SALES_FEATURES + KNOWN_FEATURES)
def test_feature_does_not_use_sales_after_origin(raw, frames, feature, h):
    """Scrambling every sale after the origin must not change any feature at that origin."""
    before = frames[h]
    after = scrambled_frame(raw, "sales", ORIGIN, h)
    rows = before["origin_idx"] <= ORIGIN
    pd.testing.assert_series_equal(before.loc[rows, feature].reset_index(drop=True),
                                   after.loc[rows, feature].reset_index(drop=True))


def test_scramble_really_changes_future_targets(raw, frames):
    """Guard for the test above: the scramble must actually change something."""
    after = scrambled_frame(raw, "sales", ORIGIN, 1)
    at_origin = frames[1]["origin_idx"] == ORIGIN
    assert not frames[1].loc[at_origin, "target"].reset_index(drop=True).equals(
        after.loc[at_origin, "target"].reset_index(drop=True))


@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("feature", PRICE_FEATURES)
def test_price_features_do_not_look_past_the_target_week(raw, frames, feature, h):
    """Prices are known in advance, but only up to the target week itself."""
    target_week = ORIGIN + h
    before = frames[h]
    after = scrambled_frame(raw, "sell_price", target_week, h)
    rows = before["target_idx"] <= target_week
    pd.testing.assert_series_equal(before.loc[rows, feature].reset_index(drop=True),
                                   after.loc[rows, feature].reset_index(drop=True))


def test_every_model_feature_has_a_leakage_test():
    tested = set(SALES_FEATURES + HORIZON_SALES_FEATURES + KNOWN_FEATURES)
    assert set(FEATURES) - set(ID_FEATURES) == tested


# ------------------------------------------------------------- exact values

def sales_at(raw: pd.DataFrame, item: str, store: str, week: int) -> int:
    row = raw[(raw.item_id == item) & (raw.store_id == store) & (raw.week_idx == week)]
    return int(row["sales"].iloc[0])


def row_at(frame: pd.DataFrame, item: str, store: str, origin: int) -> pd.Series:
    row = frame[(frame.item_id == item) & (frame.store_id == store) & (frame.origin_idx == origin)]
    assert len(row) == 1
    return row.iloc[0]


def test_lags_rolling_and_target_values(raw, frames):
    h = 3
    r = row_at(frames[h], "A", "TX_2", ORIGIN)
    s = lambda w: sales_at(raw, "A", "TX_2", w)
    assert r["lag_1"] == s(ORIGIN)
    assert r["lag_4"] == s(ORIGIN - 3)
    assert r["roll_mean_4"] == pytest.approx(np.mean([s(w) for w in range(ORIGIN - 3, ORIGIN + 1)]))
    assert r["sales_same_week_last_year"] == s(ORIGIN + h - 52)
    assert r["target"] == s(ORIGIN + h)
    assert r["target_idx"] == ORIGIN + h


def test_young_series_features(frames):
    r = row_at(frames[1], "C", "TX_1", 25)             # 'C' listed at week 20
    assert r["weeks_since_listed"] == 5
    assert np.isnan(r["lag_8"])                       # 7 weeks back is before listing
    assert np.isnan(r["sales_same_week_last_year"])


def test_known_features_come_from_the_target_week(raw, frames):
    h = 2
    r = row_at(frames[h], "B", "TX_1", ORIGIN)
    target = raw[(raw.item_id == "B") & (raw.store_id == "TX_1") & (raw.week_idx == ORIGIN + h)].iloc[0]
    assert r["sell_price"] == target["sell_price"]
    assert r["n_snap_days"] == target["n_snap_days"]
    assert r["month"] == target["week_start"].month


def test_last_h_weeks_have_no_row(frames):
    for h, frame in frames.items():
        assert frame["origin_idx"].max() == N_WEEKS - 1 - h


# ---------------------------------------------------------------- pipeline

def test_parallel_matches_serial(raw):
    pd.testing.assert_frame_equal(build_features(raw, parallel=True),
                                  build_features(raw, parallel=False))


def test_rejects_series_with_a_gap(raw):
    gap = raw.drop(index=raw[(raw.item_id == "A") & (raw.week_idx == 30)].index)
    with pytest.raises(ValueError):
        build_features(gap, parallel=False)
