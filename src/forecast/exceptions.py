"""Exception flags and the order plan at the as-of week.

The app plans from one week, the "as-of" week: the last backtest fold origin
(week 272, 2016-04-16). Everything here uses only sales up to and including
that week and forecasts made at or before it, so a planner standing in that
week could have produced exactly this list.

Three flags (CLAUDE.md), each with a plain-language reason:
- drift:              the last 4 weeks' sales are far from what was forecast
- projected_stockout: simulated stock won't cover next week's median forecast
- zero_run:           a run of zero-sales weeks after steady sales

Stock on hand is *simulated* (the model-driven policy from forecast.inventory,
run up to the as-of week). M5 has no inventory data.

Run with:  python -m forecast.exceptions   (needs models/backtest_forecasts.parquet)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import PROCESSED_DIR
from forecast.backtest import MODELS_DIR, SERIES, add_week_index, update_metrics
from forecast.inventory import demand_matrix, order_up_to_matrix, simulate

# Thresholds, calibrated on week 272 so the list stays workable (decisions.md).
DRIFT_WEEKS = 4
DRIFT_THRESHOLD = 0.5      # |actual - forecast| / forecast
DRIFT_MIN_UNITS = 20       # actual or forecast over the 4 weeks, so noise on slow items doesn't flag
STOCKOUT_MIN_SHORTFALL = 1.0
ZERO_RUN_WEEKS = 3
STEADY_WEEKS = 13
STEADY_MIN_SHARE = 0.9     # sold in at least 90% of the steady weeks
STEADY_MIN_MEAN = 2.0      # units per week

FLAG_COLUMNS = SERIES + ["flag", "units_at_stake", "reason"]


def drift_flags(fc: pd.DataFrame, as_of: int, point: str = "hgbr") -> pd.DataFrame:
    """Flag series whose last 4 weeks of sales drift from the 1-week-ahead forecasts.

    Uses the h=1 forecasts for weeks as_of-3 .. as_of (made at the week before
    each one), so every actual and forecast involved is known at the as-of week.
    """
    weeks = range(as_of - DRIFT_WEEKS + 1, as_of + 1)
    rows = fc[(fc["h"] == 1) & fc["target_idx"].isin(weeks)]
    tot = rows.groupby(SERIES, observed=True)[["actual", point]].sum()
    gap = tot["actual"] - tot[point]
    rel = gap / tot[point].clip(lower=1.0)
    hit = (rel.abs() > DRIFT_THRESHOLD) & (tot[["actual", point]].max(axis=1) >= DRIFT_MIN_UNITS)
    out = tot[hit].assign(units_at_stake=gap[hit].abs(), rel=rel[hit])
    out["reason"] = [
        f"Sold {a:.0f} units in the last {DRIFT_WEEKS} weeks vs {f:.0f} forecast ({r:+.0%})."
        for a, f, r in zip(out["actual"], out[point], out["rel"])]
    return out.reset_index().assign(flag="drift")[FLAG_COLUMNS]


def zero_run_flags(weekly: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """Flag series with no sales in the last 3 weeks after a steady selling stretch.

    weekly: one row per series x week with `week_idx` and `sales`. Only weeks
    up to the as-of week are read.
    """
    run = list(range(as_of - ZERO_RUN_WEEKS + 1, as_of + 1))
    steady = list(range(run[0] - STEADY_WEEKS, run[0]))
    past = weekly[weekly["week_idx"].between(steady[0], as_of)]
    wide = past.pivot_table(index=SERIES, columns="week_idx", values="sales", observed=True)
    wide = wide.reindex(columns=steady + run)        # NaN = not yet listed
    before = wide[steady]
    sold_weeks = (before > 0).sum(axis=1)
    mean = before.mean(axis=1)
    hit = ((wide[run] == 0).all(axis=1)
           & (sold_weeks >= STEADY_MIN_SHARE * STEADY_WEEKS)
           & (mean >= STEADY_MIN_MEAN))
    out = pd.DataFrame({"units_at_stake": mean[hit] * ZERO_RUN_WEEKS})
    out["reason"] = [
        f"No sales for {ZERO_RUN_WEEKS} weeks after selling in {n} of the previous "
        f"{STEADY_WEEKS} weeks (avg {m:.1f}/week). Possible stockout or phantom inventory."
        for n, m in zip(sold_weeks[hit], mean[hit])]
    return out.reset_index().assign(flag="zero_run")[FLAG_COLUMNS]


def order_plan(fc: pd.DataFrame, as_of: int, first_review: int) -> pd.DataFrame:
    """Per series: simulated stock at the as-of week and the forecasts made there.

    Runs the model-driven policy (S = P90 h1 + P90 h2) from `first_review`
    up to the as-of week: demand weeks first_review+1 .. as_of, reviews
    first_review .. as_of-1. `arriving` is the order placed at the review
    before the as-of week, due at the start of the next week.
    """
    demand = demand_matrix(fc, list(range(first_review + 1, as_of + 1)))
    s = order_up_to_matrix(fc, list(range(first_review, as_of)), "p90").reindex(demand.index)
    if demand.isna().any().any() or s.isna().any().any():
        raise ValueError("Missing forecasts or actuals before the as-of week")
    d = demand.to_numpy(float)
    sim = simulate(d, s.to_numpy(float))
    plan = pd.DataFrame({"on_hand": sim["end_on_hand"][:, -1],
                         "arriving": sim["arrivals"][:, d.shape[1] + 1]}, index=demand.index)

    now = fc[fc["origin_idx"] == as_of]
    wide = now.pivot_table(index=SERIES, columns="h", values=["p50", "p90"], observed=True)
    wide.columns = [f"{q}_w{h}" for q, h in wide.columns]
    info = now[now["h"] == 1].set_index(SERIES)[["dept_id", "demand_type", "ma4"]]
    plan = plan.join(info).join(wide).reset_index()
    first = ["item_id", "store_id", "dept_id", "demand_type"]
    return plan[first + [c for c in plan.columns if c not in first]]


def stockout_flags(plan: pd.DataFrame) -> pd.DataFrame:
    """Flag series whose stock plus next week's arrival is below next week's P50."""
    supply = plan["on_hand"] + plan["arriving"]
    short = plan["p50_w1"] - supply
    hit = short >= STOCKOUT_MIN_SHORTFALL
    out = plan.loc[hit, SERIES].assign(units_at_stake=short[hit])
    out["reason"] = [
        f"Simulated stock {o:.0f} + arriving {a:.0f} is below next week's median "
        f"forecast of {p:.0f}."
        for o, a, p in zip(plan.loc[hit, "on_hand"], plan.loc[hit, "arriving"],
                           plan.loc[hit, "p50_w1"])]
    return out.assign(flag="projected_stockout")[FLAG_COLUMNS]


def build_exceptions(fc: pd.DataFrame, weekly: pd.DataFrame, as_of: int,
                     first_review: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All flags (one row per series x flag) and the order plan at the as-of week."""
    plan = order_plan(fc, as_of, first_review)
    flags = pd.concat([drift_flags(fc, as_of), stockout_flags(plan),
                       zero_run_flags(weekly, as_of)], ignore_index=True)
    flags = flags.merge(plan[SERIES + ["dept_id"]], on=SERIES, how="left")
    flags = flags.sort_values("units_at_stake", ascending=False, ignore_index=True)
    return flags[SERIES + ["dept_id"] + FLAG_COLUMNS[2:]], plan


def main() -> None:
    """Write the flags and the order plan for the app and the Excel export."""
    fc = pd.read_parquet(MODELS_DIR / "backtest_forecasts.parquet")
    weekly = add_week_index(pd.read_parquet(PROCESSED_DIR / "tx_weekly.parquet"))
    first_review = int(fc["fold_origin"].min())
    as_of = int(fc["fold_origin"].max())
    flags, plan = build_exceptions(fc, weekly, as_of, first_review)

    for col in ["item_id", "store_id", "dept_id", "flag"]:
        flags[col] = flags[col].astype("category")
    flags.to_parquet(PROCESSED_DIR / "exceptions.parquet", index=False)
    plan.astype({c: "float32" for c in plan.select_dtypes("float").columns}).to_parquet(
        PROCESSED_DIR / "order_plan.parquet", index=False)

    week_start = weekly.drop_duplicates("week_idx").set_index("week_idx")["week_start"]
    counts = flags["flag"].value_counts().to_dict()
    update_metrics({"exceptions": {
        "generated_by": "python -m forecast.exceptions",
        "as_of_idx": as_of,
        "as_of_week_start": str(pd.Timestamp(week_start[as_of]).date()),
        "flag_counts": {k: int(v) for k, v in counts.items()},
        "n_flagged_series": int(flags[SERIES].drop_duplicates().shape[0]),
        "n_series": len(plan),
    }})
    print(f"As of week {as_of}: {len(plan):,} series in the order plan")
    print(flags["flag"].value_counts().to_string())
    print(f"{flags[SERIES].drop_duplicates().shape[0]:,} series with at least one flag")


if __name__ == "__main__":
    main()
