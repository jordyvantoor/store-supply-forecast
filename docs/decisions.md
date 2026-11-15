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

## 2026-11-15: Commit the processed parquet only if it is small
- **Decision:** Decide after the first build. Commit `data/processed/tx_weekly.parquet` only if it is well under 50 MB.
- **Reason:** The deployed app may only load precomputed files, never raw data.
- **Alternatives:** Rebuild from raw data in the app (not allowed by the project rules).
