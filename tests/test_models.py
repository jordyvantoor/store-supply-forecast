"""Tests for the ML models and the baseline P90 (forecast.models).

Uses the synthetic generator from test_features.py: 2 stores x 3 items, 70 weeks.
"""

import numpy as np
import pandas as pd
import pytest

from forecast.features import build_features, build_horizon_frame
from forecast.models import (ML_MODELS, SCALED_FEATURES, LevelScaledRegressor, baseline_forecasts,
                             baseline_p90, fit_fold, fold_of, predict)
from test_features import make_raw

FOLDS = [56, 60]
ORIGIN = 56


def scramble_after(raw: pd.DataFrame, week: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = raw.copy()
    future = df["week_idx"] > week
    df.loc[future, "sales"] = rng.integers(500, 999, future.sum())
    return df


def forecast_at(raw: pd.DataFrame, origin: int, h: int) -> pd.DataFrame:
    frame = build_horizon_frame(build_features(raw, parallel=False), h)
    fitted = fit_fold(frame, origin, h)
    preds, _ = predict(fitted, frame[frame["origin_idx"] == origin])
    return preds.reset_index(drop=True)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.mark.parametrize("h", [1, 4])
def test_model_forecasts_ignore_sales_after_origin(raw, h):
    """End-to-end leakage test: features, training rows and model together.

    Scrambling every sale after the origin must not change a single forecast
    made at that origin. (Training is deterministic, so any difference means
    future sales reached the model.)
    """
    before = forecast_at(raw, ORIGIN, h)
    after = forecast_at(scramble_after(raw, ORIGIN), ORIGIN, h)
    pd.testing.assert_frame_equal(before, after)


def test_training_is_deterministic(raw):
    pd.testing.assert_frame_equal(forecast_at(raw, ORIGIN, 2), forecast_at(raw, ORIGIN, 2))


def test_predictions_are_non_negative_and_p90_not_below_p50(raw):
    preds = forecast_at(raw, ORIGIN, 1)
    assert (preds[list(ML_MODELS)] >= 0).all().all()
    assert (preds["p90"] >= preds["p50"]).all()


def test_fold_of_maps_each_week_to_latest_fold_origin():
    weeks = np.array([259, 260, 263, 264, 271, 272, 275])
    expected = np.array([-1, 260, 260, 264, 268, 272, 272])
    np.testing.assert_array_equal(fold_of(weeks, [260, 264, 268, 272]), expected)


def test_baseline_p90_ignores_sales_after_fold_origin(raw):
    weeks = [ORIGIN]
    before = baseline_p90(raw, FOLDS, weeks)
    after = baseline_p90(scramble_after(raw, ORIGIN), FOLDS, weeks)
    pd.testing.assert_frame_equal(before, after)


def test_baseline_p90_is_ma4_plus_error_quantile(raw):
    h = 1
    res = baseline_p90(raw, FOLDS, [ORIGIN])
    row = res[(res.item_id == "A") & (res.store_id == "TX_1") & (res.h == h)].iloc[0]

    hist = baseline_forecasts(raw, list(range(0, ORIGIN + 1)), (h,))
    hist = hist[(hist.item_id == "A") & (hist.store_id == "TX_1")]
    window = hist[(hist.target_idx <= ORIGIN) & (hist.target_idx > ORIGIN - 52)]
    q90 = (window["actual"] - window["ma4"]).quantile(0.9)
    ma4_now = hist.loc[hist.origin_idx == ORIGIN, "ma4"].iloc[0]
    assert row["baseline_p90"] == pytest.approx(max(ma4_now + q90, 0))


# ------------------------------------------------------------ level scaling

def test_level_scaling_divides_sales_features_by_floored_level():
    X = pd.DataFrame({"roll_mean_13": [200.0, 0.4], "lag_1": [240.0, 1.0],
                      "sales_same_week_last_year": [100.0, 0.0], "sell_price": [3.0, 2.0]})
    for c in SCALED_FEATURES:
        if c not in X:
            X[c] = X["lag_1"]
    lvl = LevelScaledRegressor.level(X)
    scaled = LevelScaledRegressor.scale(X, lvl)
    assert lvl.tolist() == [200.0, 1.0]                     # slow seller floored at 1
    assert scaled["lag_1"].tolist() == [1.2, 1.0]
    assert scaled["sales_same_week_last_year"].tolist() == [0.5, 0.0]
    assert scaled["sell_price"].tolist() == [3.0, 2.0]       # not a sales feature: unchanged
    assert scaled["level"].tolist() == [200.0, 1.0]


def poisson_deviance(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - y + mu)


def pinball(y, q_pred, q):
    diff = np.asarray(y, float) - np.asarray(q_pred, float)
    return np.maximum(q * diff, (q - 1) * diff)


def test_weighted_scaled_loss_equals_unscaled_loss():
    """Why rows are weighted by their level: loss(y/L, f/L) * L == loss(y, f).

    So the scaled model minimises exactly the same loss as an unscaled one;
    only how the features are presented changes.
    """
    y = np.array([0.0, 3.0, 120.0, 480.0])
    f = np.array([0.5, 2.0, 150.0, 400.0])
    L = np.array([1.0, 2.5, 140.0, 450.0])
    assert (poisson_deviance(y / L, f / L) * L).sum() == pytest.approx(poisson_deviance(y, f).sum())
    for q in (0.5, 0.9):
        assert (pinball(y / L, f / L, q) * L).sum() == pytest.approx(pinball(y, f, q).sum())


def test_scaled_model_predicts_in_units(raw):
    """Predictions come back on the original scale (units), not as ratios."""
    preds = forecast_at(raw, ORIGIN, 1)
    frame = build_horizon_frame(build_features(raw, parallel=False), 1)
    level = frame.loc[frame.origin_idx == ORIGIN, "roll_mean_13"]
    assert preds["hgbr"].mean() == pytest.approx(level.mean(), rel=0.5)
