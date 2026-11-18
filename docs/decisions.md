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

## 2026-11-18: Feature table built in origin frame, then shifted per horizon
- **Decision:** `build_features` makes one row per series × week, read as "standing at origin week `o`". Sales features on that row use only sales ≤ `o`: lags are shifts and rolling windows end at `o`. `build_horizon_frame(features, h)` then attaches the target (sales at `o + h`) and the target week's known-in-advance features with `shift(-h)`.
- **Reason:** Leakage becomes hard by construction. No sales feature is ever computed from a row after the origin, and the only forward shift touches the target and the known-in-advance columns. One origin-level table serves all four horizon models.
- **Alternatives:** Compute features directly in the target week with horizon-specific shifts (the same result, but every feature needs its own `h` offset, which is easy to get wrong).

## 2026-11-18: "Lag 52" means same week last year relative to the target
- **Decision:** The plan's lag 52 is `sales_same_week_last_year` = sales in week `o + h − 52`. It differs per horizon. The other lags (1, 2, 4, 8, 13) are counted back from the origin.
- **Reason:** Its job is seasonality, so it has to line up with the week being forecast. For h ≤ 4 that week is always at or before the origin, so it is allowed.
- **Alternatives:** Sales 52 weeks before the origin (misaligned with the target by `h` weeks).

## 2026-11-18: Zero-sales share over 52 weeks, and weeks since listed
- **Decision:** `zero_share_52` = share of zero-sales weeks in the trailing 52 weeks (or since listing, if shorter). Added `weeks_since_listed` as a feature; it was not in the original plan.
- **Reason:** A year-long window makes the zero share a stable description of how intermittent a series is, matching the demand-type idea. The EDA showed new listings sell less than established items, so a series' age is useful information. It uses only data up to the origin and has its own leakage test.
- **Alternatives:** A 13-week zero share (reacts faster but is noisier); leave age out.

## 2026-11-18: Price features use prices up to the target week only
- **Decision:** `price_ratio_13` = price in the target week ÷ mean price over the 13 weeks ending at the target week. `price_rel_dept` = price ÷ mean price of the same department × store in that week.
- **Reason:** Prices are known in advance (CLAUDE.md rule 4), so the target week's price is allowed. Prices after the target week are not; a leakage test checks this.
- **Alternatives:** Ratio against the price at the origin (simpler, but ignores the price history before it).

## 2026-11-18: `item_id` is not a feature in HistGradientBoosting (deviation from CLAUDE.md)
- **Decision:** HGBR models (`hgbr`, `p50`, `p90`) use department, category and store as categoricals but leave out `item_id`. XGBoost (`xgb`) does use `item_id` as a categorical.
- **Reason:** HGBR's native categorical support allows at most 255 levels, and there are 3,049 items. Each item's own sales level still reaches the model through the lag and rolling features, which matter more than a fixed item identity anyway. XGBoost shows whether the item identity adds anything.
- **Alternatives:** Integer-encode `item_id` as a number (the tree would split on arbitrary ID order); target-encode it (needs its own leakage-safe scheme per fold); one-hot (3,049 sparse columns).

## 2026-11-18: HGBR without early stopping
- **Decision:** `early_stopping=False`, keeping the default 100 iterations.
- **Reason:** With more than 10,000 rows, HGBR's default early stopping holds out a *random* 10% of the training rows. That's a random split inside time-series data, and the result depends on the draw. Fixed iterations are deterministic, which the end-to-end leakage test relies on, and stay close to the defaults.
- **Alternatives:** Early stopping on a time-based validation slice (more code; a later tuning step).

## 2026-11-18: Point forecast = Poisson mean; P50 reported next to it
- **Decision:** `hgbr` (Poisson loss) is the main point forecast. The median model `p50` is scored alongside it on the same rows.
- **Reason:** Poisson suits non-negative counts with many zeros, and it predicts the *mean*, which keeps total bias near zero. WAPE, however, is minimized by the *median*. For skewed or intermittent series the median is lower than the mean (often 0), so `p50` may win on WAPE while under-forecasting totals. Both are shown so the trade-off is visible instead of hidden.
- **Alternatives:** Use only the model that wins on WAPE (flatters the metric, under-stocks); Tweedie loss (an extra parameter to tune).

## 2026-11-18: P90 from quantile loss; crossing fixed; baseline P90 from past errors
- **Decision:** `p90` = HGBR with quantile loss 0.9. If it comes out below `p50` on a row, it is raised to `p50`, and the count is reported. The baseline's P90 = 4-week moving average + the 90th percentile of that series' own moving-average errors for the same horizon, over the 52 weeks whose targets are at or before the fold origin.
- **Reason:** Two separately trained quantile models can cross, and a P90 below the median makes no sense. The baseline needs a P90 too, so the stocking comparison is fair. "Safety stock from historical forecast error" is what a planner would do without a model.
- **Alternatives:** Assume Poisson-distributed demand for the baseline P90 (understates the spread of retail sales, which would handicap the baseline).

## 2026-11-18: Forecasts for every week of the test window; scores on the fold origins only
- **Decision:** Every week from the first fold origin onwards is forecast by the model of the latest fold origin at or before it, using that week's features. WAPE, bias and coverage use only the 4 fold origins. The inventory simulation uses all weeks.
- **Reason:** A weekly review policy needs an order-up-to level every week. The model only saw targets up to its fold origin, which is earlier than the week it forecasts, so this doesn't leak. The scoreboard follows CLAUDE.md's 4-origin design.
- **Alternatives:** Retrain every week (16 times the training cost for little gain); simulate only at fold origins (not a weekly review).

## 2026-11-18: Inventory comparison at equal fill rate
- **Decision:** Besides simulating both policies as they are, scale every order-up-to level by a factor `k` (0.5–2.0) and trace fill rate vs. average on-hand. The headline is the baseline's inventory, interpolated at the model policy's fill rate, compared with the model's inventory.
- **Reason:** As-is results usually differ on *both* fill rate and inventory, which makes "better" ambiguous. Comparing inventory at the same fill rate turns it into one number. `k` is chosen on the test window for both sides; it is a way to compare, not a policy someone would run.
- **Alternatives:** Report only the as-is numbers (ambiguous); tune the target service level per policy (more moving parts).

## 2026-11-18: The app shows the last backtest origin, not a future forecast
- **Decision:** The app's forecast is the one made at the last fold origin (2016-04-16) for the 4 weeks after it, shown next to what actually sold.
- **Reason:** Forecasting past the end of the data would need prices and calendar for weeks outside the processed table, which is a pipeline extension beyond the MVP. Showing forecast vs. actual is also more convincing for a portfolio.
- **Alternatives:** Extend the pipeline with M5's future-week prices and calendar (after the MVP).

## 2026-11-18: Known weakness: HGBR on the top-selling item-stores (see the level-scaling entry below)
- **Finding:** HGBR's item-level WAPE is far worse than the 4-week moving average on the top 1% of item-stores by volume (18% of units). It beats the moving average in the 50–99% volume bands. Cause: HGBR sorts each feature into at most 255 quantile bins (95 for `roll_mean_4`), and everything above about 160 units/week shares the top bin, so the model can't tell a 200-unit seller from a 500-unit one. XGBoost avoids this through `item_id`.
- **Decision for now:** keep HGBR as the main model, report the weakness openly (notebook 03, section 1b), and propose a fix as the next step.
- **Proposed fix:** train on sales relative to each series' own level (e.g. ÷ the 13-week average, with the lag and rolling features scaled the same way), then multiply back. All series then share one scale, so the bins no longer clip the top. This needs leakage tests for the scaling.
- **Alternatives:** use XGBoost as the main point forecast (it is better at item level, but there is no XGBoost P90 yet); accept the weakness.

## 2026-11-18: Level-scaled HGBR tried, not adopted (follow-up to the top-seller weakness)
- **What was tried** (commit `experiment: level-scaled HGBR…`, reverted in the next commit): divide the sales features and the target by each series' level (`max(roll_mean_13, 1)`), weight rows by that level, add `level` as a feature, and multiply predictions back. Weighting by the level makes the scaled Poisson and pinball losses equal to the unscaled ones, so only the presentation of the features changes.
- **Decision rule, set before the results:** adopt only if overall item-level WAPE improves and department WAPE and P90 coverage don't get meaningfully worse. Time box 1.5 hours.
- **What happened:**
  - **Poisson + scaling was numerically unstable.** On the h=4, origin-260 fold, some leaf values reached 73 on the log scale, and raw scores on real test rows reached 10.5, meaning forecasts tens of thousands of times a series' level. HGBR sizes each leaf as gradient ÷ hessian; for rows predicted near zero the hessian is tiny, and the level weights inflate the gradients. XGBoost has `max_delta_step` for exactly this; HGBR doesn't. The full run failed in permutation importance with infinite predictions.
  - **Squared-error loss on the scaled target** (stable: its leaves are weighted means) was tested on one fold. One week ahead it improved overall WAPE (0.359 vs 0.367) and the top 1% (0.150 vs 0.194). Four weeks ahead it was worse overall (0.484 vs 0.453) and in every band except the top 1%.
- **Where the numbers come from:** the leaf values, raw scores and single-fold WAPEs above were printed by one-off diagnostic scripts during the experiment. They are not pipeline outputs in `reports/`, so treat them as diagnostic evidence for this decision, not as project results.
- **Decision:** keep the current models. Neither variant met the decision rule within the time box, and the squared-error variant would also give up the Poisson loss CLAUDE.md asks for. The weakness stays documented (notebook 03, section 1b).
- **Kept from the attempt:** `backtest.volume_band_scores` and `reports/volume_bands.csv`, so the weakness is tracked by code from now on.
- **Next options (after the MVP):** use XGBoost as the point forecast (it already handles top sellers) and add an XGBoost P90 (`reg:quantileerror`); or finish the squared-error variant and validate it on all folds and horizons.
