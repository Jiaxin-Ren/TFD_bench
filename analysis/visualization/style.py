"""Shared publication style for benchmark visualizations.

Keep visual choices in one place so figures generated independently and by
``plot_all.py`` use the same typography, axes, grids, lines, and method colors.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager

from analysis.methods import METHOD_ABBREVIATIONS, PLOT_METHOD_ORDER


MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 90
ONE_HALF_COLUMN_MM = 140
DOUBLE_COLUMN_MM = 190
SUBMISSION_DPI = 500


def figure_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Convert a final printed figure size from millimetres to inches."""
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def _publication_font() -> str:
    for family in ("Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Serif"


FONT_FAMILY = _publication_font()
FONT_SIZE = 7
AXIS_LABEL_SIZE = 8
PANEL_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
LEGEND_SIZE = 7
ANNOTATION_SIZE = 7

LINE_WIDTH = 1.0
SEED_LINE_WIDTH = 0.45
AXIS_LINE_WIDTH = 0.6
GRID_LINE_WIDTH = 0.35
MARKER_SIZE = 3.5


def apply_publication_style() -> None:
    """Apply the common, reproducible style used by every benchmark plot."""

    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": PANEL_LABEL_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": AXIS_LINE_WIDTH,
            "ytick.major.width": AXIS_LINE_WIDTH,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_SIZE,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "grid.color": "#b8b8b8",
            "grid.linestyle": "--",
            "grid.linewidth": GRID_LINE_WIDTH,
            "grid.alpha": 0.35,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


_TAB20 = mpl.colormaps["tab20"].colors
METHOD_COLORS = {
    method: _TAB20[index % len(_TAB20)]
    for index, method in enumerate(PLOT_METHOD_ORDER)
}


def method_color(method: str):
    """Return the fixed color for a raw method name or paper abbreviation."""

    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    for raw_method, abbreviation in METHOD_ABBREVIATIONS.items():
        if method == abbreviation:
            return METHOD_COLORS.get(raw_method, "#4c4c4c")
    return "#4c4c4c"


apply_publication_style()
