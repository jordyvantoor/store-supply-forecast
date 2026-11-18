"""Excel export for planners: Summary, Forecast and Exceptions sheets (openpyxl).

The workbook is built in memory and returned as bytes, so the Streamlit app
can hand it straight to `st.download_button`.

The Forecast sheet keeps live formulas where a planner would edit:
    Order-up-to = P90 week 1 + P90 week 2          (lead time + review period)
    Order qty   = MAX(0, order-up-to - on hand - arriving)
Changing a P90 or the on-hand cell updates the order. This is the same policy
as the simulation in forecast.inventory.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from forecast.backtest import SERIES

HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")
FLAG_FILLS = {"drift": "FDE2C8", "projected_stockout": "F8C9C4", "zero_run": "D9D2F5"}
FLAG_LABELS = {"drift": "Forecast drift", "projected_stockout": "Projected stockout",
               "zero_run": "Zero-sales run"}

CAVEATS = [
    "Stock on hand is simulated (model-driven policy); M5 has no inventory data.",
    "Sales are not demand: weeks with zero sales may have had no stock.",
    "Order-up-to sums two weekly P90s, which slightly overstates the 2-week P90.",
    "Forecasts are made at the as-of week; the weeks after it are a backtest.",
]


def headline_kpis(metrics: dict) -> dict[str, float]:
    """The headline numbers, read from reports/metrics.json (never typed by hand)."""
    overall = {(r["level"], r["method"]): r for r in metrics["models"]["overall"]}
    cov = {r["method"]: r["coverage"] for r in metrics["models"]["p90_coverage"]}
    pol = {r["policy"]: r for r in metrics["inventory"]["policies"]}
    item, dept = "item-store-week", "dept-store-week"
    return {
        "WAPE item-store-week, model (HGBR)": overall[(item, "hgbr")]["wape"],
        "WAPE item-store-week, 4-week moving average": overall[(item, "ma4")]["wape"],
        "WAPE dept-store-week, model (HGBR)": overall[(dept, "hgbr")]["wape"],
        "WAPE dept-store-week, 4-week moving average": overall[(dept, "ma4")]["wape"],
        "Bias, model (HGBR)": overall[(item, "hgbr")]["bias"],
        "P90 coverage, model": cov["p90"],
        "P90 coverage, baseline": cov["baseline_p90"],
        "Fill rate, model policy": pol["model"]["fill_rate"],
        "Fill rate, baseline policy": pol["baseline"]["fill_rate"],
        "Inventory saving at equal fill rate": metrics["inventory"]["inventory_saving_at_equal_fill_rate"],
    }


def _header(ws: Worksheet, columns: list[str], widths: dict[str, int] | None = None) -> None:
    """Write a styled header row and set column widths."""
    ws.append(columns)
    for i, name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=i)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(name, 12)
    ws.row_dimensions[1].height = 30


def _summary_sheet(ws: Worksheet, metrics: dict, n_series: int, n_flags: int) -> None:
    """KPIs from metrics.json plus caveats."""
    _header(ws, ["Item", "Value"], {"Item": 48, "Value": 16})
    ws.append(["As-of week (Walmart week starting)", metrics["exceptions"]["as_of_week_start"]])
    ws.append(["Item-stores in this export", n_series])
    ws.append(["Exception flags in this export", n_flags])
    for label, value in headline_kpis(metrics).items():
        ws.append([label, value])
        ws.cell(row=ws.max_row, column=2).number_format = "0.0%"
    ws.append([])
    ws.append(["Notes"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    for note in CAVEATS:
        ws.append([note])
    ws.freeze_panes = "A2"


def _forecast_sheet(ws: Worksheet, plan: pd.DataFrame, flags: pd.DataFrame) -> None:
    """One row per item-store with forecasts, stock and live order formulas."""
    labels = (flags.assign(label=flags["flag"].astype(str).map(FLAG_LABELS))
              .groupby(SERIES, observed=True)["label"].agg(", ".join))
    plan = plan.join(labels, on=SERIES)

    weeks = [1, 2, 3, 4]
    cols = (["Item", "Store", "Dept", "Demand type", "MA4"]
            + [f"P50 wk{w}" for w in weeks] + [f"P90 wk{w}" for w in weeks]
            + ["On hand (sim.)", "Arriving wk1", "Order-up-to", "Order qty", "Flags"])
    _header(ws, cols, {"Item": 16, "Dept": 11, "Demand type": 12, "Flags": 36})
    letter = {name: get_column_letter(i) for i, name in enumerate(cols, start=1)}

    ids = plan[["item_id", "store_id", "dept_id", "demand_type"]].astype(str).to_numpy()
    numbers = plan[["ma4"] + [f"p50_w{w}" for w in weeks] + [f"p90_w{w}" for w in weeks]
                   + ["on_hand", "arriving"]].astype(float).round(2).to_numpy()
    flag_text = plan["label"].fillna("").to_numpy()
    for r, (id_row, num_row, text) in enumerate(zip(ids, numbers, flag_text), start=2):
        s = f"={letter['P90 wk1']}{r}+{letter['P90 wk2']}{r}"
        q = (f"=MAX(0,{letter['Order-up-to']}{r}-{letter['On hand (sim.)']}{r}"
             f"-{letter['Arriving wk1']}{r})")
        ws.append([*id_row, *num_row.tolist(), s, q, text])

    last = ws.max_row
    for name in cols[4:-1]:
        for (cell,) in ws[f"{letter[name]}2:{letter[name]}{last}"]:
            cell.number_format = "0.0"
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{letter['Flags']}{last}"
    if last >= 2:   # shade rows that carry a flag
        ws.conditional_formatting.add(
            f"A2:{letter['Flags']}{last}",
            FormulaRule(formula=[f'${letter["Flags"]}2<>""'],
                        fill=PatternFill("solid", fgColor="FFF4D6")))


def _exceptions_sheet(ws: Worksheet, flags: pd.DataFrame) -> None:
    """One row per flag, colored by flag type."""
    _header(ws, ["Item", "Store", "Dept", "Flag", "Units at stake", "Reason"],
            {"Item": 16, "Flag": 20, "Reason": 100})
    for row in flags.itertuples(index=False):
        ws.append([str(row.item_id), str(row.store_id), str(row.dept_id),
                   FLAG_LABELS[str(row.flag)], round(float(row.units_at_stake), 1), row.reason])
    last = ws.max_row
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:F{last}"
    if last >= 2:
        for flag, color in FLAG_FILLS.items():
            ws.conditional_formatting.add(
                f"D2:D{last}",
                FormulaRule(formula=[f'$D2="{FLAG_LABELS[flag]}"'],
                            fill=PatternFill("solid", fgColor=color)))


def build_workbook(metrics: dict, plan: pd.DataFrame, flags: pd.DataFrame) -> bytes:
    """The planner workbook as .xlsx bytes. `plan` and `flags` may be filtered."""
    wb = Workbook()
    _summary_sheet(wb.active, metrics, len(plan), len(flags))
    wb.active.title = "Summary"
    _forecast_sheet(wb.create_sheet("Forecast"), plan, flags)
    _exceptions_sheet(wb.create_sheet("Exceptions"), flags)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
