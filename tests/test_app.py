"""Smoke test for the Streamlit app: every page renders without an exception.

Uses Streamlit's AppTest on the real precomputed files, so it also catches a
missing or renamed column in data/processed/, models/ or reports/.
"""


import pytest
from streamlit.testing.v1 import AppTest

from forecast import REPO_ROOT

APP_DIR = str(REPO_ROOT / "app")
PAGES = ["overview", "explorer", "exceptions", "export"]


def render(page: str, app_dir: str) -> None:
    """Runs inside AppTest: import the app module and draw one page."""
    import sys
    sys.path.insert(0, app_dir)
    import streamlit_app
    getattr(streamlit_app, page)()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = AppTest.from_function(render, args=(page, APP_DIR), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_explorer_shows_order_metrics():
    at = AppTest.from_function(render, args=("explorer", APP_DIR), default_timeout=120).run()
    labels = [m.label for m in at.metric]
    assert {"On hand (simulated)", "Order-up-to level", "Order quantity"} <= set(labels)


def test_main_builds_navigation():
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Store Supply Forecast"
