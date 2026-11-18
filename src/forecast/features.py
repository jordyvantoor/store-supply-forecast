"""Leakage-safe features for the direct (one model per horizon) strategy.

Two steps:
1. `build_features`: one row per series x week. Read each row as "standing at
   origin week o". Sales features on that row use sales up to and including
   week o only (lags are shifts, rolling windows end at o). The row also holds
   that week's known-in-advance values (price, calendar).
2. `build_horizon_frame(features, h)`: for horizon h, pair each origin row with
   the target week o + h: the target (sales at o + h), the target week's
   known-in-advance features, and same-week-last-year sales (week o + h - 52,
   which is at or before o for h <= 4).

Prices, events and SNAP days are known in advance (CLAUDE.md rule 4), so the
target week's values may be used. Sales never are.

Run with:  python -m forecast.features   (builds the table and prints a summary)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from forecast import PROCESSED_DIR
from forecast.backtest import SERIES, add_week_index, check_no_gaps

LAGS = (1, 2, 4, 8, 13)
WINDOWS = (4, 8, 13)
ZERO_WINDOW = 52
PRICE_WINDOW = 13

ID_FEATURES = ["item_id", "dept_id", "cat_id", "store_id"]

# Computed from sales up to and including the origin week.
SALES_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + [f"roll_std_{w}" for w in WINDOWS]
    + [f"zero_share_{ZERO_WINDOW}", "weeks_since_listed"]
)
# Depends on the horizon: sales in week o + h - 52.
HORIZON_SALES_FEATURES = ["sales_same_week_last_year"]
# Known in advance; taken from the target week.
KNOWN_FEATURES = [
    "sell_price", f"price_ratio_{PRICE_WINDOW}", "price_rel_dept",
    "n_snap_days", "n_event_days", "days_open", "is_christmas_week",
    "week_of_year", "month",
]
FEATURES = ID_FEATURES + SALES_FEATURES + HORIZON_SALES_FEATURES + KNOWN_FEATURES


def _rolling(g: pd.core.groupby.SeriesGroupBy, window: int, stat: str,
             min_periods: int = 1) -> pd.Series:
    """Rolling statistic per series, over the window ending at the current row."""
    r = getattr(g.rolling(window, min_periods=min_periods), stat)()
    return r.reset_index(level=list(range(len(SERIES))), drop=True)


def store_features(df: pd.DataFrame) -> pd.DataFrame:
    """Origin-level features for one store (sorted, gap-free; runs in a worker)."""
    out = df[ID_FEATURES + ["week_idx", "wm_yr_wk", "sales"]].copy()
    g = df.groupby(SERIES, observed=True)["sales"]

    # Sales features: lag_1 is the origin week itself, lag_k is k-1 weeks earlier.
    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k - 1)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = _rolling(g, w, "mean")
        out[f"roll_std_{w}"] = _rolling(g, w, "std", min_periods=2)
    is_zero = df["sales"].eq(0).groupby([df[c] for c in SERIES], observed=True)
    out[f"zero_share_{ZERO_WINDOW}"] = _rolling(is_zero, ZERO_WINDOW, "mean")
    out["weeks_since_listed"] = g.cumcount()

    # Known-in-advance values for this row's own week (shifted to the target later).
    price = df.groupby(SERIES, observed=True)["sell_price"]
    out["sell_price"] = df["sell_price"]
    out[f"price_ratio_{PRICE_WINDOW}"] = df["sell_price"] / _rolling(price, PRICE_WINDOW, "mean")
    dept_mean = df.groupby(["dept_id", "store_id", "week_idx"], observed=True)["sell_price"].transform("mean")
    out["price_rel_dept"] = df["sell_price"] / dept_mean
    for c in ["n_snap_days", "n_event_days", "days_open", "is_christmas_week"]:
        out[c] = df[c]
    out["week_of_year"] = df["week_start"].dt.isocalendar().week.astype("int8")
    out["month"] = df["week_start"].dt.month.astype("int8")
    return out


def build_features(df: pd.DataFrame, parallel: bool = True) -> pd.DataFrame:
    """Origin-level feature table for all stores. `df` needs `week_idx` (see backtest).

    Set `parallel=False` to process stores one by one (easier to debug).
    """
    check_no_gaps(df)
    stores = [df[df["store_id"] == s] for s in df["store_id"].cat.categories
              if (df["store_id"] == s).any()]
    if parallel:
        with ProcessPoolExecutor(max_workers=len(stores)) as pool:
            parts = list(pool.map(store_features, stores))
    else:
        parts = [store_features(s) for s in stores]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(SERIES + ["week_idx"], ignore_index=True)


def build_horizon_frame(features: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rows for horizon h: origin features + target-week features + target.

    Rows whose target week is past the end of the data are dropped.
    """
    g = features.groupby(SERIES, observed=True)
    out = features[ID_FEATURES + ["week_idx"] + SALES_FEATURES].rename(
        columns={"week_idx": "origin_idx"})
    out.insert(len(ID_FEATURES) + 1, "h", h)
    out.insert(len(ID_FEATURES) + 2, "target_idx", out["origin_idx"] + h)

    out["sales_same_week_last_year"] = g["sales"].shift(52 - h)
    for c in KNOWN_FEATURES:
        out[c] = g[c].shift(-h)
    out["target"] = g["sales"].shift(-h)

    out = out[out["target"].notna()].copy()
    return out.astype({"target": "int32", "is_christmas_week": "bool",
                       "n_snap_days": "int8", "n_event_days": "int8", "days_open": "int8",
                       "week_of_year": "int8", "month": "int8"})


def main(parallel: bool = True) -> None:
    """Build the feature table from the processed parquet and print a summary."""
    start = time.perf_counter()
    df = add_week_index(pd.read_parquet(PROCESSED_DIR / "tx_weekly.parquet"))
    feats = build_features(df, parallel=parallel)
    print(f"Origin-level table: {len(feats):,} rows, {feats.shape[1]} columns, "
          f"{feats.memory_usage(deep=True).sum() / 1e6:.0f} MB, "
          f"built in {time.perf_counter() - start:.1f}s")
    for h in (1, 2, 3, 4):
        frame = build_horizon_frame(feats, h)
        print(f"h={h}: {len(frame):,} rows, {len(FEATURES)} features, "
              f"missing share per feature max {frame[FEATURES].isna().mean().max():.1%}")


if __name__ == "__main__":
    main()
