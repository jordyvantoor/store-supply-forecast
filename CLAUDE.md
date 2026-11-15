# CLAUDE.md — Store Supply Forecast

## What this project is
A portfolio project that forecasts weekly unit sales per item and store for the three Texas stores (TX_1, TX_2, TX_3) in the Walmart M5 dataset, turns those forecasts into stocking decisions, and serves them in a Streamlit planning app.

Readers will go through the README, open the live app, and expect every modeling choice to be explained. **The owner must understand and be able to defend every line.** Favor clear, readable code over clever code.

Full scope and day-by-day plan: `docs/project-plan.md` (a local file, intentionally not committed). Check it before starting a new step.

## Scope
- MVP: pipeline, baselines, models, evaluation, inventory simulation, Streamlit app, deployment.
- Out of scope: deep learning, extensive hyperparameter tuning, other states or daily granularity, leaderboard chasing.
- If a request would add work beyond the current plan step, say so and suggest whether it belongs in the MVP or after it.

## Stack and commands
- Python 3.11+, virtual environment in `.venv`, pinned versions in `requirements.txt`
- Libraries: pandas, numpy, pyarrow, scikit-learn, xgboost, plotly, streamlit, openpyxl, joblib, pytest
- Install: `pip install -r requirements.txt`
- Tests: `pytest -q`
- App: `streamlit run app/streamlit_app.py`
- Pipeline entry points live in `src/forecast/` and can be run as modules, e.g. `python -m forecast.data`

## Repo layout
```
data/raw/          # M5 raw files, gitignored, filled by a download script
data/processed/    # TX weekly parquet and app-ready outputs (keep small)
notebooks/         # 01_eda, 02_baselines_features, 03_models_evaluation
src/forecast/      # data, features, backtest, models, inventory, exceptions, export
app/               # streamlit_app.py
models/            # saved models and precomputed forecasts
reports/           # metrics.json and result tables written by code
tests/
docs/              # project-plan.md, decisions.md, ai-workflow.md
```

## Data rules
- Source: M5 Forecasting (Walmart). Credit Walmart, the M5 organizers and Kaggle in the README. Never commit raw data.
- Keep only stores TX_1, TX_2, TX_3. Aggregate daily sales to Walmart weeks (`wm_yr_wk` from `calendar.csv`).
- Target: units sold per item × store × week.
- Prices are weekly and missing before an item's first price; treat weeks before first sale/price as "not yet listed", not as zero demand.
- Stores are closed on Dec 25; handle that week explicitly.
- Sales are not demand: M5 has no inventory data, so zero sales may mean no stock. Note this wherever it affects conclusions.

## Forecasting rules (non-negotiable)
1. **No random train/test splits.** Only time-based splits.
2. **Rolling-origin backtest:** 4 forecast origins, 4 weeks apart, at the end of the data; horizons h = 1–4 weeks.
3. **No leakage.** A feature for a forecast made at origin *t* may only use sales up to and including week *t*. Lags and rolling windows must be shifted accordingly for each horizon.
4. Prices, events and SNAP days count as known in advance and may be used for the target week.
5. **Direct strategy:** one model per horizon.
6. **Every new feature gets a leakage test** in `tests/`.
7. **Always compare to baselines:** naive (last week), 4-week moving average, seasonal naive (same week last year).
8. If a result looks surprisingly good, suspect leakage first and check before reporting it.
9. Use `random_state=42` everywhere randomness exists.

## Metric definitions
- **WAPE** = Σ|actual − forecast| / Σ actual
- **Bias** = Σ(forecast − actual) / Σ actual (positive = over-forecast)
- **P90 coverage** = share of actuals ≤ P90 forecast (target ≈ 0.90)
- Report at two levels: item-store-week and department-store-week.
- Also split results by demand type (Syntetos–Boylan: smooth, erratic, intermittent, lumpy; ADI cutoff 1.32, CV² cutoff 0.49).
- All reported numbers are written by code to `reports/` (e.g. `reports/metrics.json`). **Never type result numbers into the README or notebooks by hand,** and never invent or estimate them.

## Models
- Baselines as above.
- scikit-learn `HistGradientBoostingRegressor` with Poisson loss, and XGBoost with Poisson or Tweedie objective. Global models across all series; item, dept, category and store as categorical features.
- P50 and P90 via quantile loss (`HistGradientBoostingRegressor(loss="quantile")`).
- Keep hyperparameters close to defaults; at most a small, documented search.
- Feature importance: permutation importance on the backtest folds.

## Inventory simulation
- Weekly review, 1-week lead time, lost sales, starting inventory = order-up-to level.
- Order-up-to level S = P90 demand over lead time + review period (2 weeks).
- Compare baseline-driven vs. model-driven policies on: unit fill rate, average on-hand units, stockout weeks.
- Summing weekly P90s overstates the 2-week P90. Accept that simplification for the MVP and state it in the README.

## Exception flags
Flag item-stores where (a) trailing 4-week actuals drift from forecast beyond a threshold with a minimum volume, (b) the simulation projects a stockout, or (c) a sudden run of zero-sales weeks follows steady sales. Each flag carries a plain-language reason.

## Code conventions
- Logic lives in `src/forecast/`; notebooks call into it and tell the story. Notebooks must run top to bottom without errors.
- Small functions with type hints and short docstrings. Vectorized pandas; no row-wise loops over large frames.
- Paths via `pathlib`, relative to the repo root. Store tables as parquet.
- Multiprocessing: per-store work through `concurrent.futures.ProcessPoolExecutor`, with a `parallel=False` option for debugging.
- No secrets in the repo. Any API key goes in `st.secrets` / environment variables.

## Streamlit app
- Loads only precomputed files from `data/processed/`, `models/` and `reports/`; never raw data. Keep app data under ~50 MB.
- Cache with `st.cache_data` and `st.cache_resource`.
- Build order: Overview KPIs → Forecast explorer (Plotly, history + P50/P90 + baseline) → Exceptions → Excel export. After the MVP: planner overrides with reason codes (logged, session state) and an upload page with schema validation.
- Streamlit Community Cloud storage is ephemeral; the override log is per session. Say so in the app.

## Excel export
openpyxl workbook with sheets Summary, Forecast, Exceptions (and Overrides later). Header formatting, frozen panes, conditional formatting on exceptions, and live formulas where a planner would edit (e.g. order quantity = order-up-to − on hand).

## Git workflow
- One feature branch per plan step (e.g. `feat/backtest-harness`); never commit directly to `main`.
- Small commits with clear messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`). Keep the Co-Authored-By trailer.
- Run `pytest -q` before every commit.
- PR description: what, why, how it was tested, and key results (from `reports/`).
- Never commit raw data, secrets, or files over 50 MB.

## How to work with me
- For anything touching more than one file, **propose a short plan first and wait for my OK.**
- When you make a modeling choice, explain it in 2–3 plain sentences: what, why, and the alternative.
- Log important decisions in `docs/decisions.md` (date, decision, reason, alternatives considered). I use this to explain choices later.
- If the model loses to a baseline, say so plainly and help me understand why. An honest result beats a flattering one.
- Point out anything I should be able to explain but might not yet understand.
