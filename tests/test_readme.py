"""Tests for the generated README results section (forecast.readme)."""

import json

import pandas as pd
import pytest

from forecast import REPO_ROOT
from forecast.backtest import REPORTS_DIR
from forecast.readme import END, START, compare, results_markdown, update_readme


@pytest.fixture(scope="module")
def block() -> str:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    return results_markdown(metrics, pd.read_csv(REPORTS_DIR / "volume_bands.csv"))


def test_update_replaces_only_the_marked_block():
    text = f"intro\n{START}\nold numbers\n{END}\noutro"
    out = update_readme(text, "new")
    assert out == f"intro\n{START}\nnew\n{END}\noutro"
    assert update_readme(out, "new") == out            # running twice changes nothing


def test_update_requires_markers():
    with pytest.raises(ValueError):
        update_readme("no markers here", "new")


def test_committed_readme_matches_metrics(block):
    """The README on disk must equal what the generator writes from reports/ now."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert update_readme(readme, block) == readme, "run: python -m forecast.readme"


@pytest.mark.parametrize("model, base, expected", [
    (0.400, 0.402, "about level with"), (0.30, 0.40, "better than"), (0.50, 0.40, "worse than")])
def test_compare(model, base, expected):
    assert compare(model, base) == expected
