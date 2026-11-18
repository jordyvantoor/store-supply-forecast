"""Forecasting methods: the three baselines and the gradient-boosted models.

Baselines (use only sales up to and including the origin week `o`):
- naive:          sales in week o ("next week looks like last week")
- ma4:            mean of weeks o-3..o (fewer weeks if the series is younger)
- seasonal naive: sales in week o + h - 52 ("same week last year"). For h <= 4
                  that week is always at or before the origin, so it is known.
                  Series listed less than 52 weeks before the target have no
                  such week; they fall back to naive (flagged in
                  `snaive_fallback`) so every method is scored on the same rows.

ML models (direct strategy: one model per horizon, retrained at each fold origin):
- hgbr: HistGradientBoostingRegressor, Poisson loss -> the main point forecast (a mean)
- xgb:  XGBoost, Poisson objective -> comparison model
- p50:  HistGradientBoostingRegressor, quantile loss 0.5 -> the median
- p90:  HistGradientBoostingRegressor, quantile loss 0.9 -> drives stocking

Run with:  python -m forecast.models   (trains every fold; about 10 minutes)
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from forecast import PROCESSED_DIR
from forecast.backtest import (HORIZONS, MODELS_DIR, REPORTS_DIR, SERIES, add_week_index,
                               demand_type, make_origins, p90_coverage, scoreboard, split,
                               update_metrics, volume_band_scores)
from forecast.features import (FEATURES, HORIZON_SALES_FEATURES, SALES_FEATURES,
                               build_features, build_horizon_frame)

RANDOM_STATE = 42

BASELINES = {"naive": "Last week", "ma4": "4-week moving average",
             "snaive": "Seasonal naive (same week last year)"}
ML_MODELS = ("hgbr", "xgb", "p50", "p90")
POINT_METHODS = ["naive", "ma4", "snaive", "hgbr", "xgb", "p50"]  # scored on WAPE and bias

# HGBR accepts at most 255 levels per categorical feature; item_id has 3,049.
# The item's own sales level reaches HGBR through the lag and rolling features.
HGBR_FEATURES = [f for f in FEATURES if f != "item_id"]
ERROR_WINDOW = 52  # weeks of past forecast errors used for the baseline P90

# Level scaling for HGBR (see LevelScaledRegressor): the series' level is its
# 13-week average up to the origin, floored at 1 unit/week.
LEVEL_FEATURE = "roll_mean_13"
LEVEL_FLOOR = 1.0
SCALED_FEATURES = ([f for f in SALES_FEATURES if f.startswith(("lag_", "roll_mean_", "roll_std_"))]
                   + HORIZON_SALES_FEATURES)


class LevelScaledRegressor(RegressorMixin, BaseEstimator):
    """Fit a model on sales relative to each series' own level, predict in units.

    Why: HGBR sorts every feature into at most 255 bins. Sales are very skewed,
    so all top sellers (above ~160 units/week) share one bin and look alike.
    Dividing the sales features and the target by the series' level puts every
    series on the same scale, so the bins stop clipping the top.

    How:
    - level = max(roll_mean_13, 1): the 13-week average up to the origin (only
      sales <= origin, so no leakage), floored so slow sellers aren't blown up.
    - Sales features (lags, rolling stats, same week last year) and the target
      are divided by the level; `level` itself is added as a feature so the
      model still knows how big the series is.
    - Each row is weighted by its level. Poisson deviance and pinball (quantile)
      loss both scale linearly, so loss(y/L, f/L) * L == loss(y, f): the model
      minimises exactly the same loss as without scaling.
    - Predictions are multiplied back by the level.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    @staticmethod
    def level(X: pd.DataFrame) -> pd.Series:
        return X[LEVEL_FEATURE].clip(lower=LEVEL_FLOOR)

    @staticmethod
    def scale(X: pd.DataFrame, level: pd.Series) -> pd.DataFrame:
        out = X.copy()
        for c in SCALED_FEATURES:
            out[c] = X[c] / level
        out["level"] = level
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series):
        lvl = self.level(X)
        self.estimator_ = clone(self.estimator).fit(self.scale(X, lvl), y / lvl,
                                                     sample_weight=lvl)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lvl = self.level(X)
        return self.estimator_.predict(self.scale(X, lvl)) * lvl.to_numpy()


# ----------------------------------------------------------------- baselines

def baseline_forecasts(df: pd.DataFrame, origins: list[int],
                       horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Baseline forecasts for every series listed at each origin, for each horizon.

    `df` must come from `backtest.add_week_index` (sorted, gap-free series),
    so `shift(k)` within a series means "k weeks earlier".
    Returns one row per series x origin x horizon with the actual and each forecast.
    """
    g = df.groupby(SERIES, observed=True)["sales"]
    base = df[SERIES + ["dept_id", "week_idx", "sales"]].copy()
    base["ma4"] = (g.rolling(4, min_periods=1).mean()
                   .reset_index(level=list(range(len(SERIES))), drop=True))

    last_week = df["week_idx"].max()
    parts = []
    for h in horizons:
        frame = base.assign(
            h=h,
            actual=g.shift(-h),           # sales in the target week o + h
            snaive=g.shift(52 - h),       # sales in week o + h - 52
        )
        # Origins whose target lies past the end of the data have no row for this h.
        frame = frame[frame["week_idx"].isin(origins) & (frame["week_idx"] + h <= last_week)]
        parts.append(frame)

    fc = pd.concat(parts, ignore_index=True).rename(columns={"week_idx": "origin_idx",
                                                             "sales": "naive"})
    if fc["actual"].isna().any():
        raise ValueError("Missing actuals: a series ends before its target week")

    fc["target_idx"] = fc["origin_idx"] + fc["h"]
    fc["snaive_fallback"] = fc["snaive"].isna()
    fc["snaive"] = fc["snaive"].fillna(fc["naive"])
    fc = fc.astype({"actual": "int32", "naive": "float64", "snaive": "float64"})

    cols = SERIES + ["dept_id", "origin_idx", "h", "target_idx", "actual",
                     "naive", "ma4", "snaive", "snaive_fallback"]
    return fc[cols].sort_values(SERIES + ["origin_idx", "h"], ignore_index=True)


def baseline_p90(df: pd.DataFrame, fold_origins: list[int], weeks: list[int],
                 horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Baseline P90 = 4-week moving average + 90th percentile of its past errors.

    For a forecast made in week w, the errors come from the fold origin
    o <= w: that series' own ma4 errors for the same horizon, over the
    `ERROR_WINDOW` weeks whose targets are at or before o. This is the classic
    "safety stock from historical forecast error" a planner would use.
    """
    first = fold_origins[0] - ERROR_WINDOW - max(horizons)
    hist = baseline_forecasts(df, list(range(first, max(weeks) + 1)), horizons)
    hist["error"] = hist["actual"] - hist["ma4"]

    out = hist[hist["origin_idx"].isin(weeks)][SERIES + ["origin_idx", "h", "ma4"]].copy()
    out["fold_origin"] = fold_of(out["origin_idx"].to_numpy(), fold_origins)

    parts = []
    for o in fold_origins:
        window = hist[(hist["target_idx"] <= o) & (hist["target_idx"] > o - ERROR_WINDOW)]
        q90 = (window.groupby(SERIES + ["h"], observed=True)["error"].quantile(0.9)
               .rename("error_q90").reset_index())
        part = out[out["fold_origin"] == o].merge(q90, on=SERIES + ["h"], how="left")
        parts.append(part)

    res = pd.concat(parts, ignore_index=True)
    res["baseline_p90"] = (res["ma4"] + res["error_q90"].fillna(0)).clip(lower=0)
    return res[SERIES + ["origin_idx", "h", "baseline_p90"]]


# ----------------------------------------------------------------- ML models

def make_model(name: str):
    """A fresh, unfitted model with settings close to the library defaults.

    HGBR: `early_stopping=False` because the default early stopping holds out a
    random 10% of training rows, which is a random split inside time-series
    data. Fixed 100 iterations (the default) keep training deterministic.
    The HGBR models are wrapped in LevelScaledRegressor (top-seller fix).
    """
    hgbr = dict(categorical_features="from_dtype", early_stopping=False,
                random_state=RANDOM_STATE)
    if name == "hgbr":
        return LevelScaledRegressor(HistGradientBoostingRegressor(loss="poisson", **hgbr))
    if name == "p50":
        return LevelScaledRegressor(HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **hgbr))
    if name == "p90":
        return LevelScaledRegressor(HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **hgbr))
    if name == "xgb":
        return XGBRegressor(objective="count:poisson", tree_method="hist",
                            enable_categorical=True, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model {name!r}")


def feature_columns(name: str) -> list[str]:
    """XGBoost can use item_id as a categorical; HGBR cannot (see HGBR_FEATURES)."""
    return FEATURES if name == "xgb" else HGBR_FEATURES


def fold_of(weeks: np.ndarray, fold_origins: list[int]) -> np.ndarray:
    """For each week, the latest fold origin at or before it (-1 if none)."""
    idx = np.searchsorted(np.asarray(fold_origins), weeks, side="right") - 1
    return np.where(idx >= 0, np.asarray(fold_origins)[np.clip(idx, 0, None)], -1)


def fit_fold(frame: pd.DataFrame, fold_origin: int, h: int,
             models: tuple[str, ...] = ML_MODELS) -> dict:
    """Fit each model on rows whose target is at or before `fold_origin`."""
    train, _ = split(frame, fold_origin, h)
    fitted = {}
    for name in models:
        cols = feature_columns(name)
        fitted[name] = make_model(name).fit(train[cols], train["target"])
    return fitted


def predict(fitted: dict, rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Predictions per model (clipped at 0). Returns the frame and the number of
    rows where P90 fell below P50 and was raised to P50 (quantile crossing)."""
    out = rows[SERIES + ["origin_idx", "h"]].copy()
    for name, model in fitted.items():
        out[name] = np.clip(model.predict(rows[feature_columns(name)]), 0, None)
    crossed = 0
    if {"p50", "p90"} <= set(out.columns):
        crossed = int((out["p90"] < out["p50"]).sum())
        out["p90"] = out[["p50", "p90"]].max(axis=1)
    return out, crossed


def importance(model, rows: pd.DataFrame, name: str) -> pd.Series:
    """Permutation importance: how much MAE gets worse when a feature is shuffled."""
    cols = feature_columns(name)
    res = permutation_importance(model, rows[cols], rows["target"], n_repeats=5,
                                 scoring="neg_mean_absolute_error", random_state=RANDOM_STATE)
    return pd.Series(res.importances_mean, index=cols)


def model_forecasts(feats: pd.DataFrame, fold_origins: list[int],
                    models: tuple[str, ...] = ML_MODELS,
                    horizons: tuple[int, ...] = HORIZONS,
                    importance_horizons: tuple[int, ...] = (1, 4)):
    """Train every fold x horizon and forecast every week from the first fold origin.

    Week w is forecast by the model of the latest fold origin at or before w,
    using w's own features. That model only saw targets up to its fold origin,
    which is earlier than w, so there is no leakage. Returns the forecasts, the
    permutation importance table, the quantile-crossing count and the fitted
    models of the last fold.
    """
    preds, imps, crossed, last_fold = [], [], 0, {}
    for h in horizons:
        frame = build_horizon_frame(feats, h)
        fold = fold_of(frame["origin_idx"].to_numpy(), fold_origins)
        for o in fold_origins:
            start = time.perf_counter()
            fitted = fit_fold(frame, o, h, models)
            rows = frame[fold == o]
            p, c = predict(fitted, rows)
            preds.append(p)
            crossed += c
            if h in importance_horizons and "hgbr" in fitted:
                at_origin = rows[rows["origin_idx"] == o]
                imps.append(importance(fitted["hgbr"], at_origin, "hgbr")
                            .rename("importance").reset_index()
                            .rename(columns={"index": "feature"}).assign(h=h, fold_origin=o))
            if o == fold_origins[-1]:
                last_fold[h] = fitted
            print(f"  h={h} fold origin {o}: {len(rows):,} forecast rows, "
                  f"{time.perf_counter() - start:.0f}s")
    imp = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
    return pd.concat(preds, ignore_index=True), imp, crossed, last_fold


# ---------------------------------------------------------------------- main

def compact(fc: pd.DataFrame) -> pd.DataFrame:
    """Shrink the forecast table for storage: float32 forecasts, small ints, categories."""
    floats = fc.select_dtypes("float64").columns
    out = fc.astype({c: "float32" for c in floats})
    out = out.astype({"origin_idx": "int16", "h": "int8", "target_idx": "int16",
                      "fold_origin": "int16", "demand_type": "category"})
    return out


def coverage_table(fc: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """P90 coverage per column: overall, by horizon and by demand type."""
    rows = []
    for by in [None, "h", "demand_type"]:
        groups = [("all", fc)] if by is None else fc.groupby(by, observed=True)
        for key, part in groups:
            for c in cols:
                rows.append({"group_by": by or "overall", "group": str(key), "method": c,
                             "coverage": p90_coverage(part["actual"], part[c]),
                             "n_rows": len(part)})
    return pd.DataFrame(rows)


def main() -> None:
    """Train all folds, write forecasts, scoreboard, coverage and importance."""
    start = time.perf_counter()
    df = add_week_index(pd.read_parquet(PROCESSED_DIR / "tx_weekly.parquet"))
    last_week = int(df["week_idx"].max())
    folds = make_origins(last_week + 1)
    weeks = list(range(folds[0], last_week))
    feats = build_features(df)

    print("Training models (4 folds x 4 horizons) ...")
    ml, imp, crossed, last_fold = model_forecasts(feats, folds)

    keys = SERIES + ["origin_idx", "h"]
    fc = (baseline_forecasts(df, weeks)
          .merge(ml, on=keys, how="inner")
          .merge(baseline_p90(df, folds, weeks), on=keys, how="left")
          .merge(demand_type(df, cutoff_idx=folds[0]).reset_index(), on=SERIES, how="left"))
    fc["fold_origin"] = fold_of(fc["origin_idx"].to_numpy(), folds)
    fc = compact(fc)

    MODELS_DIR.mkdir(exist_ok=True)
    fc.to_parquet(MODELS_DIR / "backtest_forecasts.parquet", index=False)
    for h, fitted in last_fold.items():
        for name in ("hgbr", "p50", "p90"):
            joblib.dump(fitted[name], MODELS_DIR / f"{name}_h{h}.joblib")

    # Scores use the 4 fold origins only (CLAUDE.md); other weeks feed the simulation.
    scored = fc[fc["origin_idx"].isin(folds)]
    board = scoreboard(scored, POINT_METHODS)
    cov = coverage_table(scored, ["p90", "baseline_p90"])
    imp_mean = (imp.groupby(["h", "feature"], as_index=False)["importance"].mean()
                .sort_values(["h", "importance"], ascending=[True, False]))
    board.to_csv(REPORTS_DIR / "model_scoreboard.csv", index=False)
    volume_band_scores(scored, ["ma4", "hgbr", "xgb", "p50"]).to_csv(
        REPORTS_DIR / "volume_bands.csv", index=False)
    cov.to_csv(REPORTS_DIR / "coverage.csv", index=False)
    imp_mean.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    overall = board[board["group_by"] == "overall"]
    update_metrics({"models": {
        "generated_by": "python -m forecast.models",
        "methods": POINT_METHODS,
        "overall": overall.to_dict(orient="records"),
        "p90_coverage": cov[cov["group_by"] == "overall"].to_dict(orient="records"),
        "quantile_crossing_rows": crossed,
        "n_forecast_rows_all_weeks": len(fc),
        "n_scored_rows": len(scored),
        "runtime_minutes": round((time.perf_counter() - start) / 60, 1),
    }})

    print(overall.pivot(index="method", columns="level", values=["wape", "bias"])
          .loc[POINT_METHODS].round(3))
    print(cov[cov["group_by"] == "overall"][["method", "coverage"]].to_string(index=False))
    print(f"Quantile crossing fixed on {crossed:,} rows. "
          f"Done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
