"""
Convenience plotting functions for MiniTen visualization.

Similar to matplotlib.pyplot interface for easy use.
"""

from .figure import Figure, figure as create_figure, subplots

# Global state for pyplot-like interface
_current_figure = None
_current_axes = None


def figure(figsize=(8, 6), dpi=100, **kwargs):
    """
    Create a new figure.
    
    Args:
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        **kwargs: Additional figure arguments
        
    Returns:
        Figure object
    """
    global _current_figure, _current_axes
    _current_figure = create_figure(figsize=figsize, dpi=dpi, **kwargs)
    _current_axes = None
    return _current_figure


def gcf():
    """Get current figure."""
    global _current_figure
    if _current_figure is None:
        figure()
    return _current_figure


def gca():
    """Get current axes."""
    global _current_axes
    if _current_axes is None:
        fig = gcf()
        _current_axes = fig.add_subplot(1, 1, 1)
    return _current_axes


def plot(x, y=None, **kwargs):
    """
    Plot y vs x.
    
    Args:
        x: X data or Y data if y is None
        y: Y data
        **kwargs: Plot options
        
    Returns:
        Line object
    """
    ax = gca()
    return ax.plot(x, y, **kwargs)


def scatter(x, y, **kwargs):
    """
    Create a scatter plot.
    
    Args:
        x: X coordinates
        y: Y coordinates
        **kwargs: Scatter options
        
    Returns:
        PathCollection object
    """
    ax = gca()
    return ax.scatter(x, y, **kwargs)


def bar(x, height, **kwargs):
    """
    Create a bar chart.
    
    Args:
        x: X positions
        height: Bar heights
        **kwargs: Bar options
        
    Returns:
        BarContainer object
    """
    ax = gca()
    return ax.bar(x, height, **kwargs)


def hist(x, bins=10, **kwargs):
    """
    Create a histogram.
    
    Args:
        x: Input data
        bins: Number of bins
        **kwargs: Histogram options
        
    Returns:
        (counts, bin_edges, patches)
    """
    ax = gca()
    return ax.hist(x, bins=bins, **kwargs)


def imshow(data, **kwargs):
    """
    Display an image.
    
    Args:
        data: Image data
        **kwargs: Image options
        
    Returns:
        AxesImage object
    """
    ax = gca()
    return ax.imshow(data, **kwargs)


def heatmap(data, **kwargs):
    """
    Create a heatmap.
    
    Args:
        data: 2D array
        **kwargs: Heatmap options
        
    Returns:
        AxesImage object
    """
    ax = gca()
    return ax.heatmap(data, **kwargs)


def xlabel(label, **kwargs):
    """Set x-axis label."""
    return gca().set_xlabel(label, **kwargs)


def ylabel(label, **kwargs):
    """Set y-axis label."""
    return gca().set_ylabel(label, **kwargs)


def title(label, **kwargs):
    """Set axes title."""
    return gca().set_title(label, **kwargs)


def xlim(left=None, right=None):
    """Set x-axis limits."""
    return gca().set_xlim(left, right)


def ylim(bottom=None, top=None):
    """Set y-axis limits."""
    return gca().set_ylim(bottom, top)


def legend(**kwargs):
    """Show legend."""
    return gca().legend(**kwargs)


def grid(visible=True, **kwargs):
    """Configure grid."""
    return gca().grid(visible, **kwargs)


def subplot(nrows, ncols, index):
    """
    Add a subplot.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        index: Subplot index (1-based)
        
    Returns:
        Axes object
    """
    global _current_axes
    fig = gcf()
    _current_axes = fig.add_subplot(nrows, ncols, index)
    return _current_axes


def savefig(filename, **kwargs):
    """Save the current figure."""
    gcf().save(filename, **kwargs)


def show():
    """Display the current figure."""
    return gcf().show()


def close(fig=None):
    """Close a figure."""
    global _current_figure, _current_axes
    if fig is None:
        if _current_figure is not None:
            _current_figure.close()
        _current_figure = None
        _current_axes = None
    else:
        fig.close()


def tight_layout():
    """Adjust layout."""
    return gcf().tight_layout()


def suptitle(title, **kwargs):
    """Set figure title."""
    return gcf().suptitle(title, **kwargs)
