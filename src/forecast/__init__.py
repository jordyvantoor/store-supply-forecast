"""Store Supply Forecast: weekly demand forecasts for the M5 Texas stores."""

from pathlib import Path

# Repo root, resolved from this file: src/forecast/__init__.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
