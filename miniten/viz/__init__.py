"""
MiniTen Visualization Engine

An optimized, Cython-based visualization library designed for
high-performance plotting and real-time training visualization.

Features:
- Fast line plots, bar charts, scatter plots, histograms
- Image display and heatmaps
- Real-time plot updates
- Export to PNG, SVG formats
- Minimal dependencies
- Optimized for edge devices

Example:
    >>> from miniten import viz
    >>> fig = viz.Figure()
    >>> ax = fig.add_subplot()
    >>> ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
    >>> fig.save("output.png")
"""

from .figure import Figure
from .axes import Axes
from .plotting import (
    plot,
    scatter,
    bar,
    hist,
    imshow,
    heatmap,
)
from .colors import colormap, get_color
from .export import save_figure

__all__ = [
    "Figure",
    "Axes",
    "plot",
    "scatter",
    "bar",
    "hist",
    "imshow",
    "heatmap",
    "colormap",
    "get_color",
    "save_figure",
]
