"""Forecasting methods. Today: the three baselines every model must beat.

All baselines use only sales up to and including the origin week `o`:
- naive:          sales in week o ("next week looks like last week")
- ma4:            mean of weeks o-3..o (fewer weeks if the series is younger)
- seasonal naive: sales in week o + h - 52 ("same week last year"). For h <= 4
                  that week is always at or before the origin, so it is known.
                  Series listed less than 52 weeks before the target have no
                  such week; they fall back to naive (flagged in
                  `snaive_fallback`) so every method is scored on the same rows.
"""

from __future__ import annotations

import pandas as pd

from forecast.backtest import HORIZONS, SERIES

BASELINES = {"naive": "Last week", "ma4": "4-week moving average",
             "snaive": "Seasonal naive (same week last year)"}


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

    parts = []
    for h in horizons:
        frame = base.assign(
            h=h,
            actual=g.shift(-h),           # sales in the target week o + h
            snaive=g.shift(52 - h),       # sales in week o + h - 52
        )
        frame = frame[frame["week_idx"].isin(origins)]
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
