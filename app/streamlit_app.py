"""Store Supply Forecast: a planning app for the three Texas stores in Walmart M5.

Pages: Overview -> Forecast explorer -> Exceptions -> Excel export.

The app only reads precomputed files (CLAUDE.md): forecasts from
models/backtest_forecasts.parquet, history from data/processed/tx_weekly.parquet,
the order plan and flags from `python -m forecast.exceptions`, and every
headline number from reports/. It never loads raw data or trains a model.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast import PROCESSED_DIR
from forecast.backtest import MODELS_DIR, REPORTS_DIR, SERIES, add_week_index, volume_band
from forecast.export import CAVEATS, FLAG_LABELS, build_workbook, headline_kpis
from forecast.plotting import METHOD_COLORS, MUTED
from forecast.readme import compare

REPO_URL = "https://github.com/jordyvantoor/store-supply-forecast"
HISTORY_WEEKS = 26
ACTUAL_COLOR = MUTED          # mid grey: readable in light and dark themes
MODEL_COLOR = METHOD_COLORS["hgbr"]
BASELINE_COLOR = METHOD_COLORS["ma4"]
PAGES: dict[str, st.Page] = {}   # filled in main(), used to jump between pages


# ------------------------------------------------------------------ data

@st.cache_data
def load_metrics() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


@st.cache_data
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / name)


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """Backtest forecasts: one row per item-store x origin x horizon."""
    cols = SERIES + ["dept_id", "origin_idx", "h", "target_idx", "actual",
                     "ma4", "hgbr", "p50", "p90", "fold_origin"]
    return pd.read_parquet(MODELS_DIR / "backtest_forecasts.parquet", columns=cols)


@st.cache_data
def load_weekly() -> pd.DataFrame:
    """Weekly sales history with `week_idx`."""
    cols = SERIES + ["dept_id", "wm_yr_wk", "week_start", "sales"]
    return add_week_index(pd.read_parquet(PROCESSED_DIR / "tx_weekly.parquet", columns=cols))


@st.cache_data
def week_starts() -> pd.Series:
    """Week start date per `week_idx`."""
    w = load_weekly()
    return w.drop_duplicates("week_idx").set_index("week_idx")["week_start"].sort_index()


@st.cache_data
def load_plan() -> pd.DataFrame:
    """Order plan at the as-of week, with each item-store's volume band."""
    plan = pd.read_parquet(PROCESSED_DIR / "order_plan.parquet")
    fc = load_forecasts()
    scored = fc[fc["origin_idx"] == fc["fold_origin"]]          # the 4 fold origins
    bands = (scored.assign(band=volume_band(scored))
             .drop_duplicates(SERIES).set_index(SERIES)["band"].astype(str))
    return plan.join(bands, on=SERIES)


@st.cache_data
def load_flags() -> pd.DataFrame:
    flags = pd.read_parquet(PROCESSED_DIR / "exceptions.parquet")
    return flags.assign(flag_label=flags["flag"].astype(str).map(FLAG_LABELS))


def as_of() -> tuple[int, str]:
    e = load_metrics()["exceptions"]
    return e["as_of_idx"], e["as_of_week_start"]


# ------------------------------------------------------------------ helpers

def pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def flag_options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(df[col].astype(str).unique())


# ------------------------------------------------------------------ pages

def overview() -> None:
    """Headline KPIs, total units over the backtest, and where the model loses."""
    m = load_metrics()
    k = headline_kpis(m)
    _, as_of_date = as_of()

    st.title("Store Supply Forecast")
    st.markdown(
        "How much of each item will each Texas Walmart store sell over the next 4 weeks, "
        "how sure are we, and how much should we stock? Weekly forecasts for "
        f"{m['n_series']:,} item-stores (M5 data, stores TX_1–TX_3), made with gradient-boosted "
        "models and turned into order quantities.")
    st.caption(f"Planning week: the week starting **{as_of_date}**. The weeks after it are in "
               "the data, so every forecast here can be checked against what actually sold.")

    st.subheader("Forecast accuracy (4 backtest origins × 4 weeks ahead)")
    item_m = k["WAPE item-store-week, model (HGBR)"]
    item_b = k["WAPE item-store-week, 4-week moving average"]
    dept_m = k["WAPE dept-store-week, model (HGBR)"]
    dept_b = k["WAPE dept-store-week, 4-week moving average"]
    c = st.columns(4)
    c[0].metric("WAPE, item-store-week", f"{item_m:.1%}", pts(item_m - item_b),
                delta_color="inverse", border=True,
                help="Sum of absolute errors ÷ units sold. Delta vs the 4-week moving average.")
    c[1].metric("WAPE, dept-store-week", f"{dept_m:.1%}", pts(dept_m - dept_b),
                delta_color="inverse", border=True,
                help="Same metric after adding items up to department level.")
    c[2].metric("Bias", f"{k['Bias, model (HGBR)']:+.1%}", border=True,
                help="Σ(forecast − actual) ÷ Σ actual. Negative = under-forecast.")
    c[3].metric("P90 coverage", f"{k['P90 coverage, model']:.1%}",
                pts(k["P90 coverage, model"] - k["P90 coverage, baseline"]), border=True,
                help="Share of actuals at or below the P90 forecast (target 90%). "
                     "Delta vs the baseline P90 (moving average + historical error).")
    st.markdown(
        f"Per item and week the model is **{compare(item_m, item_b)}** the 4-week moving "
        f"average. Added up to departments it is **{compare(dept_m, dept_b)}** it. Summing "
        "items cancels most item-level noise; what is left is largely shared movement "
        "(events, SNAP days, prices, season), which the model sees and the moving average "
        "doesn't.")

    st.subheader("Stocking (simulated, weekly review, 1-week lead time)")
    inv = m["inventory"]
    c = st.columns(3)
    c[0].metric("Fill rate, model policy", f"{k['Fill rate, model policy']:.1%}",
                pts(k["Fill rate, model policy"] - k["Fill rate, baseline policy"]), border=True,
                help="Share of units sold out of units demanded. Delta vs the baseline policy.")
    c[1].metric("Less inventory at equal fill rate", f"{k['Inventory saving at equal fill rate']:.1%}",
                border=True,
                help="Average stock the baseline policy needs to match the model's fill rate, "
                     "compared with the model's average stock.")
    flagged = m["exceptions"]["n_flagged_series"]
    c[2].metric("Item-stores flagged this week", f"{flagged:,}", border=True,
                help="Drift, projected stockout or zero-sales run. See Exceptions.")
    by_target = pd.DataFrame(inv["saving_by_target_fill_rate"])
    grows = by_target["inventory_saving"].is_monotonic_increasing
    st.caption("Saving by target fill rate: " + ", ".join(
        f"{r.fill_rate:.0%} → {r.inventory_saving:.1%}" for r in by_target.itertuples())
        + (". The higher the service level, the more a sharper P90 is worth." if grows else "."))

    st.subheader("Total weekly units, 1 week ahead")
    fc = load_forecasts()
    store = st.segmented_control("Store", ["All stores", "TX_1", "TX_2", "TX_3"],
                                 default="All stores", key="ov_store")
    rows = fc[fc["h"] == 1]
    if store and store != "All stores":
        rows = rows[rows["store_id"] == store]
    tot = rows.groupby("target_idx")[["actual", "hgbr", "ma4"]].sum()
    tot.index = week_starts().reindex(tot.index).to_numpy()
    fig = go.Figure()
    fig.add_scatter(x=tot.index, y=tot["actual"], name="Actual", mode="lines+markers",
                    line=dict(color=ACTUAL_COLOR, width=2))
    fig.add_scatter(x=tot.index, y=tot["hgbr"], name="Model (HGBR)", mode="lines",
                    line=dict(color=MODEL_COLOR, width=2))
    fig.add_scatter(x=tot.index, y=tot["ma4"], name="4-week moving average", mode="lines",
                    line=dict(color=BASELINE_COLOR, width=2, dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="Units", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each week's forecast was made one week earlier, using only sales up to then.")

    st.subheader("Where the model wins and loses")
    board = load_report("model_scoreboard.csv")
    dt = board[(board["level"] == "item-store-week") & (board["group_by"] == "demand_type")]
    table = dt.pivot(index="group", columns="method", values="wape")[["ma4", "hgbr", "xgb"]]
    table.columns = ["Moving avg", "HGBR", "XGBoost"]
    counts = pd.Series(m["demand_type_counts"], name="Item-stores")
    st.markdown("**WAPE by demand type** (Syntetos–Boylan), item-store-week:")
    st.dataframe(table.join(counts).rename_axis("Demand type").style.format(
        {c: "{:.1%}" for c in table.columns} | {"Item-stores": "{:,}"}), width="content")

    bands = load_report("volume_bands.csv").set_index("band")
    top = bands.loc["top 1%"]
    st.markdown(
        f"**Known weakness: the top sellers.** The top 1% of item-stores by volume carry "
        f"{top['share_of_units']:.0%} of units. There HGBR's WAPE is {top['wape_hgbr']:.1%}, "
        f"against {top['wape_ma4']:.1%} for the moving average and {top['wape_xgb']:.1%} for "
        "XGBoost. HGBR sorts each feature into at most 255 bins, so very high sales levels "
        "share a bin. For those items, read the moving average as the point forecast; "
        "the explorer marks them.")
    st.dataframe(bands[["share_of_units", "wape_ma4", "wape_hgbr", "wape_xgb"]]
                 .rename(columns={"share_of_units": "Share of units", "wape_ma4": "Moving avg",
                                  "wape_hgbr": "HGBR", "wape_xgb": "XGBoost"})
                 .rename_axis("Volume band").style.format("{:.1%}"), width="content")


def explorer() -> None:
    """History, P50/P90 and the baseline for one item-store, plus its order."""
    plan, flags, weekly = load_plan(), load_flags(), load_weekly()
    as_of_idx, as_of_date = as_of()
    st.title("Forecast explorer")
    st.caption(f"Forecasts made at the end of the week starting {as_of_date}, for the 4 weeks "
               "after it, shown next to what actually sold.")

    c = st.columns(3)
    store = pick(c[0], "Store", flag_options(plan, "store_id"), "ex_store")
    dept = pick(c[1], "Department", flag_options(plan, "dept_id"), "ex_dept")
    items = plan[(plan["store_id"] == store) & (plan["dept_id"] == dept)]
    items = items.sort_values("ma4", ascending=False)
    avg = dict(zip(items["item_id"].astype(str), items["ma4"]))
    item = pick(c[2], "Item (by recent volume)", list(avg), "ex_item",
                fmt=lambda i: f"{i} · {avg[i]:,.1f}/wk")
    row = items[items["item_id"] == item].iloc[0]

    badges = [f":blue-badge[{row['demand_type']}]", f":gray-badge[volume: {row['band']}]"]
    st.markdown(" ".join(badges))
    if row["band"] == "top 1%":
        st.warning("Top-1% seller: HGBR is weak on these (see Overview). Read the moving "
                   "average as the point forecast here.", icon=":material/warning:")
    for f in flags[(flags["item_id"] == item) & (flags["store_id"] == store)].itertuples():
        st.error(f"**{f.flag_label}.** {f.reason}", icon=":material/flag:")

    hist = weekly[(weekly["item_id"] == item) & (weekly["store_id"] == store)]
    past = hist[hist["week_idx"].between(as_of_idx - HISTORY_WEEKS + 1, as_of_idx)]
    after = hist[hist["week_idx"] > as_of_idx]
    weeks = week_starts().reindex(range(as_of_idx + 1, as_of_idx + 5)).to_numpy()
    p50 = [row[f"p50_w{h}"] for h in range(1, 5)]
    p90 = [row[f"p90_w{h}"] for h in range(1, 5)]

    fig = go.Figure()
    fig.add_scatter(x=past["week_start"], y=past["sales"], name="Sales (history)",
                    mode="lines", line=dict(color=ACTUAL_COLOR, width=2))
    fig.add_scatter(x=weeks, y=p90, mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=weeks, y=p50, name="P50–P90 range", mode="lines", fill="tonexty",
                    fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), hoverinfo="skip")
    fig.add_scatter(x=weeks, y=p90, name="P90", mode="lines+markers",
                    line=dict(color=MODEL_COLOR, width=1, dash="dash"))
    fig.add_scatter(x=weeks, y=p50, name="P50", mode="lines+markers",
                    line=dict(color=MODEL_COLOR, width=2.5))
    fig.add_scatter(x=weeks, y=[row["ma4"]] * 4, name="4-week moving average",
                    mode="lines", line=dict(color=BASELINE_COLOR, width=2, dash="dot"))
    fig.add_scatter(x=after["week_start"], y=after["sales"], name="Actual (after planning week)",
                    mode="markers", marker=dict(color=ACTUAL_COLOR, size=9, symbol="diamond"))
    fig.add_vline(x=week_starts()[as_of_idx], line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="Units per week", yaxis_rangemode="tozero",
                      legend=dict(orientation="h", y=1.12, traceorder="normal"))
    st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame({
        "Week starting": pd.to_datetime(weeks).strftime("%Y-%m-%d"),
        "P50": p50, "P90": p90, "Moving avg": [row["ma4"]] * 4,
        "Actual": after.set_index("week_idx")["sales"].reindex(
            range(as_of_idx + 1, as_of_idx + 5)).to_numpy()})
    st.dataframe(table.style.format({"P50": "{:.1f}", "P90": "{:.1f}", "Moving avg": "{:.1f}",
                                     "Actual": "{:.0f}"}), hide_index=True, width="content")

    st.subheader("This week's order")
    s = row["p90_w1"] + row["p90_w2"]
    qty = max(0.0, s - row["on_hand"] - row["arriving"])
    c = st.columns(4)
    c[0].metric("On hand (simulated)", f"{row['on_hand']:.1f}", border=True)
    c[1].metric("Arriving next week", f"{row['arriving']:.1f}", border=True)
    c[2].metric("Order-up-to level", f"{s:.1f}", border=True,
                help="P90 for next week + P90 for the week after (1-week lead time + "
                     "1-week review period).")
    c[3].metric("Order quantity", f"{qty:.1f}", border=True,
                help="max(0, order-up-to − on hand − arriving). Arrives in 2 weeks.")
    st.caption(CAVEATS[0])


def pick(container, label: str, options: list[str], key: str, fmt=str) -> str:
    """A select box whose value survives option changes (and jumps from Exceptions)."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return container.selectbox(label, options, key=key, format_func=fmt)


def exceptions() -> None:
    """Flagged item-stores with reasons; pick a row to open it in the explorer."""
    flags = load_flags()
    _, as_of_date = as_of()
    st.title("Exceptions")
    st.caption(f"Item-stores that need a planner's attention in the week starting {as_of_date}.")

    c = st.columns(3)
    stores = c[0].multiselect("Store", flag_options(flags, "store_id"), placeholder="All")
    depts = c[1].multiselect("Department", flag_options(flags, "dept_id"), placeholder="All")
    kinds = c[2].multiselect("Flag", list(FLAG_LABELS.values()), placeholder="All")
    view = flags
    if stores:
        view = view[view["store_id"].astype(str).isin(stores)]
    if depts:
        view = view[view["dept_id"].astype(str).isin(depts)]
    if kinds:
        view = view[view["flag_label"].isin(kinds)]

    c = st.columns(3)
    for col, (flag, label) in zip(c, FLAG_LABELS.items()):
        col.metric(label, f"{(view['flag'] == flag).sum():,}", border=True)

    shown = view[["item_id", "store_id", "dept_id", "flag_label", "units_at_stake", "reason"]]
    event = st.dataframe(
        shown, hide_index=True, width="stretch", height=420,
        on_select="rerun", selection_mode="single-row", key="exc_table",
        column_config={
            "item_id": "Item", "store_id": "Store", "dept_id": "Dept", "flag_label": "Flag",
            "units_at_stake": st.column_config.NumberColumn(
                "Units at stake", format="%.0f",
                help="Drift: gap between sales and forecast. Stockout: shortfall vs next "
                     "week's P50. Zero run: typical sales over the silent weeks."),
            "reason": st.column_config.TextColumn("Reason", width="large")})
    rows = event.selection.rows if event else []
    if rows and st.button("Open in forecast explorer", type="primary"):
        r = shown.iloc[rows[0]]
        st.session_state.update(ex_store=str(r["store_id"]), ex_dept=str(r["dept_id"]),
                                ex_item=str(r["item_id"]))
        st.switch_page(PAGES["explorer"])
    st.caption("Select a row to open it in the forecast explorer. "
               "Stock is simulated (M5 has no inventory data), and zero sales may mean "
               "no stock rather than no demand.")


@st.cache_data
def workbook_bytes(stores: tuple[str, ...], depts: tuple[str, ...]) -> bytes:
    plan, flags = load_plan(), load_flags()
    keep_p = plan["store_id"].astype(str).isin(stores) & plan["dept_id"].astype(str).isin(depts)
    keep_f = flags["store_id"].astype(str).isin(stores) & flags["dept_id"].astype(str).isin(depts)
    return build_workbook(load_metrics(), plan[keep_p].drop(columns="band"),
                          flags[keep_f].drop(columns="flag_label"))


def export() -> None:
    """Download the planner workbook for a selection of stores and departments."""
    plan = load_plan()
    as_of_idx, as_of_date = as_of()
    st.title("Excel export")
    st.markdown(
        "A workbook for the planning week starting "
        f"{as_of_date} with three sheets: **Summary** (headline numbers and caveats), "
        "**Forecast** (one row per item-store: P50/P90 for 4 weeks, simulated stock and the "
        "order) and **Exceptions** (flags with reasons). Order-up-to and order quantity are "
        "live formulas, so editing a forecast or the on-hand count updates the order.")
    c = st.columns(2)
    all_stores, all_depts = flag_options(plan, "store_id"), flag_options(plan, "dept_id")
    stores = c[0].multiselect("Stores", all_stores, default=all_stores)
    depts = c[1].multiselect("Departments", all_depts, default=all_depts)
    if not stores or not depts:
        st.info("Pick at least one store and one department.")
        return
    n = int((plan["store_id"].astype(str).isin(stores)
             & plan["dept_id"].astype(str).isin(depts)).sum())
    st.caption(f"{n:,} item-stores selected.")
    data = workbook_bytes(tuple(stores), tuple(depts))
    st.download_button("Download Excel workbook", data, type="primary",
                       file_name=f"store_supply_plan_{as_of_date}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:")
    st.markdown("**Notes**\n" + "\n".join(f"- {note}" for note in CAVEATS))


# ------------------------------------------------------------------ app

def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"[Code and method on GitHub]({REPO_URL})")
        st.caption(
            "Data: M5 Forecasting (Walmart), via the M5 organizers and Kaggle. "
            "Sales are not demand: M5 has no inventory data.")
        st.caption(
            "This app reads precomputed files only and saves nothing. Storage on Streamlit "
            "Community Cloud is temporary, so anything entered here lasts for this session only.")


def main() -> None:
    st.set_page_config(page_title="Store Supply Forecast", page_icon=":material/inventory_2:",
                       layout="wide")
    PAGES.update(
        overview=st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
        explorer=st.Page(explorer, title="Forecast explorer", icon=":material/query_stats:",
                         url_path="forecast-explorer"),
        exceptions=st.Page(exceptions, title="Exceptions", icon=":material/flag:",
                           url_path="exceptions"),
        export=st.Page(export, title="Excel export", icon=":material/download:",
                       url_path="excel-export"))
    sidebar()
    st.navigation(list(PAGES.values())).run()


if __name__ == "__main__":
    main()
