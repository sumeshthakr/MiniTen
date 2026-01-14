"""
Figure class for MiniTen visualization.

Provides a container for plots and subplots with efficient rendering.
"""

import numpy as np
from .axes import Axes


class Figure:
    """
    A figure container for plots.
    
    Similar to matplotlib.figure.Figure but optimized for performance.
    
    Args:
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering
        facecolor: Background color
        
    Example:
        >>> fig = Figure(figsize=(8, 6))
        >>> ax = fig.add_subplot(1, 1, 1)
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> fig.save("plot.png")
    """
    
    def __init__(self, figsize=(8, 6), dpi=100, facecolor='white'):
        """Initialize a new Figure."""
        self.figsize = figsize
        self.dpi = dpi
        self.facecolor = facecolor
        self.axes = []
        self.title = None
        
        # Compute pixel dimensions
        self.width = int(figsize[0] * dpi)
        self.height = int(figsize[1] * dpi)
        
        # Create canvas (RGB array)
        self._canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
    
    def add_subplot(self, nrows=1, ncols=1, index=1):
        """
        Add a subplot to the figure.
        
        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            index: Index of this subplot (1-based)
            
        Returns:
            Axes object for the subplot
        """
        # Calculate position for this subplot
        row = (index - 1) // ncols
        col = (index - 1) % ncols
        
        # Calculate bounds (with margins)
        margin = 0.1
        subplot_width = (1.0 - 2 * margin) / ncols
        subplot_height = (1.0 - 2 * margin) / nrows
        
        left = margin + col * subplot_width
        bottom = margin + (nrows - 1 - row) * subplot_height
        
        ax = Axes(self, [left, bottom, subplot_width * 0.9, subplot_height * 0.9])
        self.axes.append(ax)
        return ax
    
    def suptitle(self, title, fontsize=14):
        """Set the figure title."""
        self.title = title
        return self
    
    def tight_layout(self):
        """Adjust layout to prevent overlapping."""
        # Simplified version - adjusts margins
        for ax in self.axes:
            ax._update_bounds()
        return self
    
    def _render(self):
        """Render the figure to the internal canvas."""
        # Clear canvas
        self._canvas[:] = 255
        
        # Render each axes
        for ax in self.axes:
            ax._render(self._canvas)
        
        # Render title if present
        if self.title:
            self._render_title()
        
        return self._canvas
    
    def _render_title(self):
        """Render the figure title."""
        # Simple text rendering placeholder
        # In full implementation, would render text
        pass
    
    def save(self, filename, format=None, dpi=None):
        """
        Save the figure to a file.
        
        Args:
            filename: Output filename
            format: File format (png, svg, etc.)
            dpi: Resolution
        """
        from .export import save_figure
        save_figure(self, filename, format=format, dpi=dpi or self.dpi)
    
    def show(self):
        """Display the figure (for interactive use)."""
        # Render and display
        canvas = self._render()
        print(f"Figure displayed: {self.width}x{self.height} pixels")
        return canvas
    
    def close(self):
        """Close the figure and free resources."""
        self._canvas = None
        self.axes = []
    
    def savefig(self, filename, **kwargs):
        """Alias for save() for matplotlib compatibility."""
        return self.save(filename, **kwargs)


def figure(figsize=(8, 6), dpi=100, facecolor='white'):
    """
    Create a new figure.
    
    Convenience function similar to matplotlib.pyplot.figure()
    
    Args:
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        facecolor: Background color
        
    Returns:
        Figure object
    """
    return Figure(figsize=figsize, dpi=dpi, facecolor=facecolor)


def subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a figure with subplots.
    
    Similar to matplotlib.pyplot.subplots()
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size
        **kwargs: Additional figure arguments
        
    Returns:
        (Figure, Axes or array of Axes)
    """
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    
    fig = Figure(figsize=figsize, **kwargs)
    
    if nrows == 1 and ncols == 1:
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax
    
    axes = []
    for i in range(1, nrows * ncols + 1):
        axes.append(fig.add_subplot(nrows, ncols, i))
    
    if nrows == 1 or ncols == 1:
        return fig, axes
    
    # Reshape to 2D array
    axes_array = np.array(axes).reshape(nrows, ncols)
    return fig, axes_array
