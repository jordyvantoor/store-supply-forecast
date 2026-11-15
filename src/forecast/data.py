"""Build the Texas weekly sales table from the M5 raw files.

Output: data/processed/tx_weekly.parquet, one row per item x store x Walmart week,
with units sold, the weekly sell price, and weekly calendar counts.

Steps:
1. Keep only the Texas stores (TX_1, TX_2, TX_3).
2. Per store (in parallel): reshape daily columns to rows, map each day to its
   Walmart week, sum to weekly units, and drop weeks before the item's first
   price week ("not yet listed", not zero demand).
3. Attach weekly calendar info: SNAP days, event days, Christmas closure.

Only complete weeks (7 days in the data) are kept. The M5 evaluation file ends
on a Sunday, so its last 2 days form a partial week that would look like a
sales drop. See docs/decisions.md.

Run with:  python -m forecast.data
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from forecast import PROCESSED_DIR, RAW_DIR

TX_STORES = ("TX_1", "TX_2", "TX_3")
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id"]
KEY_COLS = ["item_id", "store_id", "wm_yr_wk"]
OUTPUT_PATH = PROCESSED_DIR / "tx_weekly.parquet"


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three M5 files, keeping only Texas rows for sales and prices."""
    day_cols = [c for c in pd.read_csv(raw_dir / "sales_train_evaluation.csv", nrows=0).columns
                if c.startswith("d_")]
    # Daily unit sales fit easily in int16 (max is well below 32,767); this
    # cuts the memory of the wide table by 4x compared with int64.
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv",
                        dtype={c: "int16" for c in day_cols})
    sales = sales[sales["store_id"].isin(TX_STORES)].reset_index(drop=True)

    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["store_id"].isin(TX_STORES)].reset_index(drop=True)
    return sales, calendar, prices


def weekly_calendar(calendar: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """Summarise the daily calendar to one row per Walmart week.

    Only days present in the sales data (`day_cols`) are counted, so a week
    cut off at the end of the data gets `n_days` < 7.
    """
    cal = calendar[calendar["d"].isin(day_cols)].copy()
    cal["is_christmas"] = (cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)
    cal["has_event"] = cal["event_name_1"].notna()

    weeks = cal.groupby("wm_yr_wk", as_index=False).agg(
        week_start=("date", "min"),
        n_days=("d", "size"),
        n_snap_days=("snap_TX", "sum"),
        n_event_days=("has_event", "sum"),
        n_christmas_days=("is_christmas", "sum"),
    )
    # Stores are closed on Dec 25, so that week has one fewer selling day.
    weeks["days_open"] = weeks["n_days"] - weeks["n_christmas_days"]
    weeks["is_christmas_week"] = weeks["n_christmas_days"] > 0
    return weeks.drop(columns="n_christmas_days")


def store_to_weekly(
    store_sales: pd.DataFrame,
    store_prices: pd.DataFrame,
    day_to_week: dict[str, int],
    complete_weeks: list[int],
) -> pd.DataFrame:
    """Turn one store's wide daily sales into weekly rows from first listing on.

    Runs in a worker process, so it only uses its arguments (no globals).
    """
    long = store_sales.melt(id_vars=ID_COLS, var_name="d", value_name="sales")
    long["wm_yr_wk"] = long["d"].map(day_to_week)
    long = long[long["wm_yr_wk"].isin(complete_weeks)]

    weekly = (long.groupby(ID_COLS + ["wm_yr_wk"], as_index=False)["sales"]
              .sum()
              .astype({"sales": "int32"}))

    # Left join keeps listed weeks even if a price were ever missing mid-life.
    weekly = weekly.merge(store_prices[KEY_COLS + ["sell_price"]], on=KEY_COLS, how="left")

    # "Not yet listed": drop weeks before the item-store's first price week.
    # Item-stores that never have a price are dropped entirely.
    first_price_week = (store_prices.groupby(["item_id", "store_id"])["wm_yr_wk"]
                        .min().rename("first_price_week").reset_index())
    weekly = weekly.merge(first_price_week, on=["item_id", "store_id"], how="inner")
    weekly = weekly[weekly["wm_yr_wk"] >= weekly["first_price_week"]]
    return weekly.drop(columns="first_price_week")


def build_weekly(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    stores: tuple[str, ...] = TX_STORES,
    parallel: bool = True,
) -> pd.DataFrame:
    """Build the item x store x week table for `stores` from in-memory frames.

    Set `parallel=False` to process stores one by one (easier to debug).
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    weeks = weekly_calendar(calendar, day_cols)
    complete_weeks = weeks.loc[weeks["n_days"] == 7, "wm_yr_wk"].tolist()
    day_to_week = dict(zip(calendar["d"], calendar["wm_yr_wk"]))

    jobs = [
        (sales.loc[sales["store_id"] == s, ID_COLS + day_cols],
         prices[prices["store_id"] == s],
         day_to_week,
         complete_weeks)
        for s in stores
    ]
    if parallel:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            parts = list(pool.map(store_to_weekly, *zip(*jobs)))
    else:
        parts = [store_to_weekly(*job) for job in jobs]

    df = pd.concat(parts, ignore_index=True)
    df = df.merge(weeks[weeks["n_days"] == 7].drop(columns="n_days"),
                  on="wm_yr_wk", how="left")

    df = df.astype({
        "item_id": "category", "dept_id": "category", "cat_id": "category",
        "store_id": "category", "wm_yr_wk": "int32", "sell_price": "float32",
        "n_snap_days": "int8", "n_event_days": "int8", "days_open": "int8",
    })
    df = df.sort_values(["store_id", "item_id", "wm_yr_wk"], ignore_index=True)

    if df.duplicated(KEY_COLS).any():
        raise ValueError("Duplicate item x store x week rows in the output")
    return df


def main(raw_dir: Path = RAW_DIR, out_path: Path = OUTPUT_PATH, parallel: bool = True) -> None:
    """Load raw files, build the weekly table, and write it to parquet."""
    sales, calendar, prices = load_raw(raw_dir)
    df = build_weekly(sales, calendar, prices, parallel=parallel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({len(df):,} rows, {size_mb:.1f} MB)")
    print(f"Weeks: {df['wm_yr_wk'].nunique()}, item-stores: "
          f"{df.groupby(['item_id', 'store_id'], observed=True).ngroups:,}")
    print(f"Weeks with missing price after first listing: {df['sell_price'].isna().sum():,}")


if __name__ == "__main__":
    main()
