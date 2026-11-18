"""Tests for the planner workbook (forecast.export)."""

import json
from io import BytesIO

import pandas as pd
import pytest
from openpyxl import load_workbook

from forecast.backtest import REPORTS_DIR
from forecast.export import build_workbook, headline_kpis


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


@pytest.fixture(scope="module")
def workbook(metrics):
    plan = pd.DataFrame({
        "item_id": ["A", "B"], "store_id": "TX_1", "dept_id": "FOODS_1",
        "demand_type": ["smooth", "lumpy"], "ma4": [5.0, 1.0],
        **{f"p50_w{w}": [5.0, 1.0] for w in range(1, 5)},
        **{f"p90_w{w}": [8.0, 3.0] for w in range(1, 5)},
        "on_hand": [4.0, 0.0], "arriving": [2.0, 0.0]})
    flags = pd.DataFrame({"item_id": ["B"], "store_id": ["TX_1"], "dept_id": ["FOODS_1"],
                          "flag": ["zero_run"], "units_at_stake": [6.0],
                          "reason": ["No sales for 3 weeks ..."]})
    return load_workbook(BytesIO(build_workbook(metrics, plan, flags)))


def test_sheets_and_frozen_panes(workbook):
    assert workbook.sheetnames == ["Summary", "Forecast", "Exceptions"]
    assert workbook["Summary"].freeze_panes == "A2"
    assert workbook["Forecast"].freeze_panes == "C2"
    assert workbook["Exceptions"].freeze_panes == "A2"


def test_order_columns_are_live_formulas(workbook):
    ws = workbook["Forecast"]
    header = [c.value for c in ws[1]]
    row = {name: cell.value for name, cell in zip(header, ws[2])}
    assert row["Order-up-to"] == "=J2+K2"                  # P90 wk1 + P90 wk2
    assert header[9:11] == ["P90 wk1", "P90 wk2"]
    assert row["Order qty"] == "=MAX(0,P2-N2-O2)"         # S - on hand - arriving
    assert header[13:16] == ["On hand (sim.)", "Arriving wk1", "Order-up-to"]


def test_flags_appear_on_the_forecast_row_and_are_highlighted(workbook):
    ws = workbook["Forecast"]
    assert ws["R2"].value in (None, "")
    assert ws["R3"].value == "Zero-sales run"
    assert len(ws.conditional_formatting) == 1
    assert len(workbook["Exceptions"].conditional_formatting) >= 1
    assert workbook["Exceptions"]["D2"].value == "Zero-sales run"


def test_summary_kpis_come_from_metrics(workbook, metrics):
    values = {r[0]: r[1] for r in workbook["Summary"].iter_rows(min_row=2, values_only=True)}
    for label, value in headline_kpis(metrics).items():
        assert values[label] == pytest.approx(value)
    assert values["Item-stores in this export"] == 2
