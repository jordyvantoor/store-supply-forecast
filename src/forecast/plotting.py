"""Shared matplotlib style for the notebooks (static charts that render on GitHub).

The Streamlit app uses Plotly instead; this module is for notebooks only.
Colors: the first three slots of a colorblind-validated categorical palette.
Those three stay distinguishable in every pairing, which matters because the
three stores often appear together on one chart.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

STORE_COLORS = {"TX_1": "#2a78d6", "TX_2": "#eb6834", "TX_3": "#1baf7a"}
SINGLE_COLOR = "#2a78d6"  # for charts with one series

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Set matplotlib defaults: thin lines, recessive grid and axes, no chart junk."""
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
    })


def thousands(ax: plt.Axes, axis: str = "y") -> None:
    """Format an axis with thousands separators (e.g. 60,000)."""
    fmt = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
