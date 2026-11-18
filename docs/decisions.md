# Decision log

Important modeling and engineering choices, with the reasoning behind them.

## 2026-11-15: Python environment via a conda env at `.venv`
- **Decision:** Create a Python 3.11 conda environment at `./.venv` (`conda create -p`). Conda supplies only Python and, on macOS, the OpenMP runtime (`llvm-openmp`) that XGBoost needs. Every Python package is installed with pip from `requirements.txt`.
- **Reason:** The system and conda-base Python is 3.9. A separate environment keeps the project isolated, and `requirements.txt` remains the single source of truth, which Streamlit Community Cloud also reads.
- **Alternatives:** Upgrading Python in conda `base` (risky, can break conda itself and other projects); Homebrew Python plus `venv` (Homebrew isn't installed).

## 2026-11-15: Source file is `sales_train_evaluation.csv`
- **Decision:** Use the M5 *evaluation* sales file (days 1–1941).
- **Reason:** It contains the full history, including the final 28 days that were held out during the competition.
- **Alternatives:** `sales_train_validation.csv` (days 1–1913), which drops the most recent 4 weeks for no benefit here.

## 2026-11-15: "Not yet listed" = weeks before the first price
- **Decision:** For each item-store, drop the weeks before its first week with a sell price.
- **Reason:** A price is direct evidence that the item was on the shelf. Zeros before that are "not listed", not "zero demand", and would bias models toward low forecasts.
- **Alternatives:** Start at the first week with a sale. Rejected because it would also drop real zero-sales weeks of slow movers that were already listed.

## 2026-11-15: Christmas week is kept and flagged
- **Decision:** Keep the week containing Dec 25 and add `days_open` (6 instead of 7) and `is_christmas_week`. Lost sales are not filled in.
- **Reason:** Dropping the week would leave a gap in the weekly series and break lag features such as "same week last year". Flagging it lets models and baselines account for the shorter week.
- **Alternatives:** Drop the week; scale its sales up by 7/6 (invents numbers).

## 2026-11-15: Keep only complete weeks (drop the partial final week)
- **Decision:** `data.py` keeps only Walmart weeks (Saturday to Friday) that have all 7 days in the data, and logs how many days were dropped. Days 1–1941 start on Saturday 2011-01-29, so they give 277 full weeks plus 2 days (Sat–Sun 2016-05-21/22) of a 278th week. That partial week is dropped.
- **Reason:** A 2-day week would look like a ~70% sales drop in the most recent week, right where the backtest origins sit, and would distort the lags and baselines.
- **Alternatives:** Scale the partial week up to 7 days (invents numbers); keep it with a `days_in_data` column (every downstream step would have to handle it). The original plan assumed exactly 277 full weeks; that was an arithmetic error caught while writing this entry.

## 2026-11-15: matplotlib for notebooks, Plotly for the app
- **Decision:** Notebooks use matplotlib (static PNG charts) with a shared style in `src/forecast/plotting.py`. The Streamlit app keeps Plotly. Added `matplotlib`, `ipykernel` and `nbconvert` to `requirements.txt`.
- **Reason:** Readers will view the notebooks on GitHub, which does not render Plotly's interactive output (charts show up blank). `ipykernel` and `nbconvert` let the notebooks be run top to bottom from VS Code and from the command line (`jupyter nbconvert --to notebook --execute --inplace`).
- **Alternatives:** Plotly with static export via kaleido (one library, but kaleido needs Chrome and is fiddly); interactive Plotly only (blank charts on GitHub).

## 2026-11-15: Commit the processed parquet only if it is small
- **Decision:** Commit `data/processed/tx_weekly.parquet`. The first build is 3.5 MB (2,046,387 rows), far below the 50 MB limit.
- **Reason:** The deployed app may only load precomputed files, never raw data.
- **Alternatives:** Rebuild from raw data in the app (not allowed by the project rules).

## 2026-11-18: Weeks are counted with `week_idx`, and lags are shifts within a series
- **Decision:** All week arithmetic uses `week_idx` (0–276), the position of each `wm_yr_wk`. "k weeks earlier" is `groupby(series).shift(k)`. `add_week_index` checks that every series has one row per week from its first listing to the end, and raises an error if not.
- **Reason:** `wm_yr_wk` jumps at year ends (11152 → 11201), so subtracting week codes gives wrong distances. A shift counts rows, so it only means "weeks" when there are no gaps; that is why the check exists.
- **Alternatives:** Join on computed week codes (more code, and easy to get wrong at year ends).

## 2026-11-18: Backtest origins and the training-window rule
- **Decision:** Four origins at `week_idx` 260, 264, 268 and 272 (week starts 2016-01-23 to 2016-04-16), with horizons 1–4. For a forecast at origin `o` and horizon `h`, training rows must have their *target* at or before `o`, so `origin_idx ≤ o − h`.
- **Reason:** The last origin's 4-week-ahead target is the final week, so the four folds' targets cover the last 16 weeks with no overlap. The training rule is the subtle part: a row made at `o − 1` for horizon 4 has its target at `o + 3`, after the origin, so `origin_idx < o` would leak future sales. Tests check both points.
- **Alternatives:** Overlapping origins one week apart (more folds, but correlated); a single holdout (one origin can be unrepresentative).

## 2026-11-18: Seasonal-naive fallback and short histories
- **Decision:** Seasonal naive uses the sales in week `o + h − 52`. If the series wasn't listed then, it falls back to naive (flagged as `snaive_fallback`; 0.7% of backtest rows). The 4-week moving average uses the weeks available if the series is younger than 4 weeks.
- **Reason:** Every method has to be scored on exactly the same rows, otherwise WAPE comparisons are unfair. Naive is the simplest honest stand-in.
- **Alternatives:** Drop rows without last-year data (each method would be scored on a different set); fill with zero (unfair to seasonal naive).

## 2026-11-18: Demand type from history up to the first origin, item level only
- **Decision:** Syntetos–Boylan class per item-store, from weeks up to and including the first origin (`week_idx` 260). ADI = listed weeks ÷ weeks with sales; CV² uses the population standard deviation of non-zero weeks. Results are split by demand type at item-store level only.
- **Reason:** A label computed from the test weeks would quietly use future data when choosing how to slice the results. Demand type describes a single series, so it has no clean meaning for a department total that mixes types.
- **Alternatives:** Classify on the full history (leaks); reclassify at each origin (more precise, but the groups would change between folds).
