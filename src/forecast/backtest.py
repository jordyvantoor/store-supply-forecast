"""Rolling-origin backtest: week index, origins, splits, metrics and scoreboard.

Vocabulary used throughout the project:
- origin `o`: the last week whose sales are known when the forecast is made
- horizon `h`: how many weeks ahead we forecast (1-4)
- target week: `o + h`

All arithmetic on weeks uses `week_idx` (0, 1, 2, ...), never `wm_yr_wk`,
because `wm_yr_wk` jumps at year ends (e.g. 11152 -> 11201).

Run with:  python -m forecast.backtest   (writes the baseline scoreboard to reports/)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import PROCESSED_DIR, REPO_ROOT

SERIES = ["item_id", "store_id"]
HORIZONS = (1, 2, 3, 4)
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

# Syntetos-Boylan cutoffs (CLAUDE.md)
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


# ---------------------------------------------------------------- week index

def add_week_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add `week_idx` (0 = first week in the data) and check series have no gaps.

    The rest of the pipeline uses `groupby(series).shift(k)` to mean
    "k weeks earlier". That is only true if every series has one row per week
    from its first listed week to the end, so this is checked here.
    """
    weeks = np.sort(df["wm_yr_wk"].unique())
    out = df.sort_values(SERIES + ["wm_yr_wk"], ignore_index=True)
    out["week_idx"] = np.searchsorted(weeks, out["wm_yr_wk"]).astype("int16")
    check_no_gaps(out)
    return out


def check_no_gaps(df: pd.DataFrame) -> None:
    """Raise if any series skips a week or is out of order (needs `week_idx`)."""
    step = df.groupby(SERIES, observed=True)["week_idx"].diff()
    if (step.dropna() != 1).any():
        raise ValueError("Some series have missing weeks; shift-based lags would be wrong")


def make_origins(n_weeks: int, n_origins: int = 4, step: int = 4,
                 max_h: int = max(HORIZONS)) -> list[int]:
    """Forecast origins at the end of the data, `step` weeks apart.

    The last origin is placed so its longest horizon lands on the final week.
    With 277 weeks this gives [260, 264, 268, 272]: the targets of the four
    folds cover the last 16 weeks exactly, without overlap.
    """
    last = n_weeks - 1 - max_h
    return [last - step * i for i in reversed(range(n_origins))]


def split(frame: pd.DataFrame, origin: int, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split for one origin and horizon (frame has `origin_idx`).

    Test: rows made at this origin. Train: rows whose *target* week is at or
    before the origin, i.e. `origin_idx <= origin - h`. Using
    `origin_idx < origin` instead would leak: a training row made at
    `origin - 1` for horizon 4 has its target 3 weeks after the origin.
    """
    train = frame[frame["origin_idx"] <= origin - h]
    test = frame[frame["origin_idx"] == origin]
    return train, test


# ------------------------------------------------------------------- metrics

def wape(actual: pd.Series, forecast: pd.Series) -> float:
    """Weighted absolute percentage error: sum|actual - forecast| / sum(actual)."""
    return float(np.abs(actual - forecast).sum() / actual.sum())


def bias(actual: pd.Series, forecast: pd.Series) -> float:
    """sum(forecast - actual) / sum(actual). Positive = over-forecast."""
    return float((forecast - actual).sum() / actual.sum())


def to_dept_level(fc: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Sum actuals and forecasts to department x store x target week.

    Errors of items in the same department partly cancel when summed, which is
    why department-level WAPE is lower than item-level WAPE. Bias does not
    change with aggregation (it is a ratio of sums either way).
    """
    keys = ["origin_idx", "h", "dept_id", "store_id"]
    return fc.groupby(keys, observed=True, as_index=False)[["actual"] + methods].sum()


def score_table(fc: pd.DataFrame, methods: list[str], by: str | None) -> pd.DataFrame:
    """WAPE and bias per method, overall (`by=None`) or per value of column `by`."""
    groups = [("all", fc)] if by is None else fc.groupby(by, observed=True)
    rows = []
    for key, part in groups:
        for m in methods:
            rows.append({"group_by": by or "overall", "group": str(key), "method": m,
                         "wape": wape(part["actual"], part[m]),
                         "bias": bias(part["actual"], part[m]),
                         "n_rows": len(part), "actual_units": int(part["actual"].sum())})
    return pd.DataFrame(rows)


def scoreboard(fc: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Full scoreboard: both levels, overall and by horizon and origin.

    The demand-type split is item-store level only, because demand type is a
    property of a single series.
    """
    dept = to_dept_level(fc, methods)
    parts = []
    for level, frame, splits in [
        ("item-store-week", fc, [None, "h", "origin_idx", "demand_type"]),
        ("dept-store-week", dept, [None, "h", "origin_idx"]),
    ]:
        for by in splits:
            parts.append(score_table(frame, methods, by).assign(level=level))
    cols = ["level", "group_by", "group", "method", "wape", "bias", "n_rows", "actual_units"]
    return pd.concat(parts, ignore_index=True)[cols]


# --------------------------------------------------------------- demand type

def demand_type(df: pd.DataFrame, cutoff_idx: int) -> pd.Series:
    """Syntetos-Boylan demand class per series, from weeks <= `cutoff_idx` only.

    ADI = listed weeks / weeks with sales (average gap between sales).
    CV2 = (std / mean)^2 of the non-zero weekly sales (population std).
    smooth: ADI < 1.32 and CV2 < 0.49 | erratic: ADI < 1.32, CV2 >= 0.49
    intermittent: ADI >= 1.32, CV2 < 0.49 | lumpy: ADI >= 1.32, CV2 >= 0.49

    Using only history up to the first origin keeps the label itself from
    peeking at the test weeks. Series listed after the cutoff are "new";
    series with no sales before it are "no sales".
    """
    hist = df[df["week_idx"] <= cutoff_idx]
    nonzero = hist[hist["sales"] > 0]
    g_all = hist.groupby(SERIES, observed=True)["sales"]
    g_nz = nonzero.groupby(SERIES, observed=True)["sales"]

    stats = pd.DataFrame({"n_weeks": g_all.size(), "n_nonzero": g_nz.size(),
                          "nz_mean": g_nz.mean(), "nz_std": g_nz.std(ddof=0)})
    stats["adi"] = stats["n_weeks"] / stats["n_nonzero"]
    stats["cv2"] = (stats["nz_std"] / stats["nz_mean"]) ** 2

    label = np.select(
        [stats["n_nonzero"].isna(),
         (stats["adi"] < ADI_CUTOFF) & (stats["cv2"] < CV2_CUTOFF),
         (stats["adi"] < ADI_CUTOFF),
         (stats["cv2"] < CV2_CUTOFF)],
        ["no sales", "smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    out = pd.Series(label, index=stats.index, name="demand_type")
    all_series = df[SERIES].drop_duplicates().set_index(SERIES).index
    return out.reindex(all_series, fill_value="new")


# ---------------------------------------------------------------------- main

def main(data_path: Path = PROCESSED_DIR / "tx_weekly.parquet") -> None:
    """Run the baselines over all origins and write the scoreboard to reports/."""
    from forecast.models import BASELINES, baseline_forecasts

    df = add_week_index(pd.read_parquet(data_path))
    origins = make_origins(df["week_idx"].max() + 1)
    fc = baseline_forecasts(df, origins)

    types = demand_type(df, cutoff_idx=origins[0])
    fc = fc.merge(types.reset_index(), on=SERIES, how="left")

    board = scoreboard(fc, list(BASELINES))
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    board.to_csv(REPORTS_DIR / "baseline_scoreboard.csv", index=False)
    fc.to_parquet(MODELS_DIR / "baseline_forecasts.parquet", index=False)

    week_of = df.drop_duplicates("week_idx").set_index("week_idx")
    metrics = {
        "generated_by": "python -m forecast.backtest",
        "origins": [{"origin_idx": int(o), "wm_yr_wk": int(week_of.loc[o, "wm_yr_wk"]),
                     "week_start": str(week_of.loc[o, "week_start"].date())} for o in origins],
        "horizons": list(HORIZONS),
        "n_series": int(fc[SERIES].drop_duplicates().shape[0]),
        "n_forecast_rows": len(fc),
        "snaive_fallback_share": float(fc["snaive_fallback"].mean()),
        "demand_type_counts": types.value_counts().to_dict(),
        "baselines": board.to_dict(orient="records"),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    overall = board[board["group_by"] == "overall"]
    print(overall.pivot(index="method", columns="level", values=["wape", "bias"]).round(3))
    print(f"Seasonal naive fell back to naive on {metrics['snaive_fallback_share']:.1%} of rows")
    print(f"Wrote {REPORTS_DIR / 'metrics.json'} and {REPORTS_DIR / 'baseline_scoreboard.csv'}")


if __name__ == "__main__":
    main()
