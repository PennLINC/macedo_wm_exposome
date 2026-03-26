# ------------------------------------------------------------------------------
# --- Figure formatting utilities ---
# ------------------------------------------------------------------------------

# This script contains utilities for formatting figures at an exact physical
# size (mm) with consistent styling (font size, font type, line width, etc.).

# ------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, Union

MM_PER_INCH = 25.4


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH

# "rc" stands for "runtime configuration", which in matplotlib refers to
# settings controling the appearance of plots (fonts, colors, linewidths, etc)


def _apply_rc(
    *,
    base_pt=7, label_pt=7, title_pt=7,
    font_family="sans-serif",
    sans_list: Union[list[str], str] = "Helvetica",
    axes_linewidth=0.8, line_width=1,
):
    # Convert sans_list to list if it's a string
    font_list = [sans_list] if isinstance(sans_list, str) else list(sans_list)

    mpl.rcParams.update({
        # Fonts / sizes
        "font.family": font_family,
        "font.sans-serif": font_list,
        "axes.titlesize": title_pt,
        "axes.labelsize": label_pt,
        "xtick.labelsize": base_pt,
        "ytick.labelsize": base_pt,
        "legend.fontsize": base_pt,

        # Strokes
        "axes.linewidth": axes_linewidth,
        "lines.linewidth": line_width,

        # Keep live text in PDF/SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # NO automatic layout that can push/resize things
        "figure.constrained_layout.use": False,
        "savefig.bbox": "standard",   # never 'tight'
        "savefig.pad_inches": 0.0,
    })


def setup_figure(
    width_mm: float,
    height_mm: float,
    *,
    base_pt: int = 7,
    label_pt: int = 7,
    title_pt: int = 7,
    font_family: str = "sans-serif",
    sans_list: Union[list[str], str] = "Helvetica",
    axes_linewidth: float = 0.8,
    line_width: float = 1,
    margins_mm: Tuple[float, float, float, float] = (0, 0, 0, 0),
    nrows: int = 1,
    ncols: int = 1,
    sharex=False,
    sharey=False,
    axes_aspect: str | float = "auto",
):
    """
    Create a figure with exact physical size (no tight cropping), fixed
    margins, and optional strict check that content fits inside the page.

    This function creates a matplotlib figure with precise physical dimensions
    in millimeters. Unlike figures saved with `bbox_inches='tight'`, this
    ensures the saved figure matches the specified dimensions exactly. The
    function also applies consistent styling (fonts, line widths) and can
    create subplot grids.

    Parameters
    ----------
    width_mm : float
        Figure width in millimeters.
    height_mm : float
        Figure height in millimeters.
    base_pt : int, default=7
        Font size in points for tick labels, legend text, and other base
        elements.
    label_pt : int, default=7
        Font size in points for axis labels (xlabel, ylabel).
    title_pt : int, default=7
        Font size in points for axis titles and figure titles.
    font_family : str, default="sans-serif"
        Main font family. Common values: "sans-serif", "serif", "monospace".
    sans_list : list[str] or str, default="Helvetica"
        Preferred sans-serif font(s). Can be a single font name (str) or a list
        of fonts.
        If a list, matplotlib uses the first available font from the list.
        Used when font_family="sans-serif".
    axes_linewidth : float, default=0.8
        Line width in points for axes (spines, ticks, grid lines).
    line_width : float, default=1
        Default line width in points for plot lines.
    margins_mm : tuple[float, float, float, float],
                 default=(18.0, 2.0, 18.0, 8.0)
        Margins in millimeters as (left, right, bottom, top). These define the
        space between the figure edges and the plot area.
    nrows : int, default=1
        Number of rows in the subplot grid.
    ncols : int, default=1
        Number of columns in the subplot grid.
    sharex : bool, default=False
        If True, subplots share x-axis. See matplotlib's subplots() for
        details.
    sharey : bool, default=False
        If True, subplots share y-axis. See matplotlib's subplots() for
        details.
    axes_aspect : str or float, default="auto"
        Aspect ratio control for axes, can be "auto", "equal", or
        a numeric aspect:
        - "auto" (recommended): Prevents aspect from forcing layout expansion.
          Use this unless you specifically need a fixed aspect ratio.
        - "equal": Forces equal aspect ratio (x and y scales match).
        - float: Numeric aspect ratio (e.g., 1.0 for square, 2.0 for 2:1).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : matplotlib.axes.Axes or array of Axes
        Single axes object if nrows=ncols=1, or array of axes for subplots.

    Examples
    --------
    >>> # Single plot with default margins
    >>> fig, ax = setup_figure(width_mm=100, height_mm=80)
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure(fig, 'plot.svg')

    >>> # Subplot grid with custom margins
    >>> fig, axes = setup_figure(
    ...     width_mm=150, height_mm=100,
    ...     margins_mm=(10, 10, 10, 10),  # left, right, bottom, top
    ...     nrows=2, ncols=2
    ... )

    >>> # Figure with equal aspect ratio
    >>> fig, ax = setup_figure(
    ...     width_mm=80, height_mm=80,
    ...     axes_aspect="equal"
    ... )

    >>> # Figure with single font (string)
    >>> fig, ax = setup_figure(width_mm=100, height_mm=80, sans_list="Arial")

    >>> # Figure with font list
    >>> fig, ax = setup_figure(
    ...     width_mm=100, height_mm=80,
    ...     sans_list=["Helvetica", "Arial"]
    ... )

    Notes
    -----
    - The figure size is fixed and will not be cropped or resized during
    saving.
    - Use `strict_overflow_check=True` to detect when content might exceed
    margins.
    - For best results, use `axes_aspect="auto"` unless you specifically need
      a fixed aspect ratio, as fixed aspects can cause layout issues with
      margins.
    """

    _apply_rc(
        base_pt=base_pt, label_pt=label_pt, title_pt=title_pt,
        font_family=font_family, sans_list=sans_list,
        axes_linewidth=axes_linewidth, line_width=line_width,
    )

    fig = plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))
    # Convert mm margins to fractions
    # The left and bottom margins are specified directly as fractions
    # from 0 (edge) to left/bottom.
    # For the right and top margins, matplotlib expects the *end* location,
    # not fraction width.
    # So, to get the correct fraction for right/top, we subtract (margin/size)
    # from 1.0.
    L, R, B, T = margins_mm
    left = L / width_mm
    right = 1.0 - (R / width_mm)
    bottom = B / height_mm
    top = 1.0 - (T / height_mm)

    # Store intended content rectangle so save_figure respects margins
    fig._figformat_content_rect = (left, right, bottom, top)

    # Build the grid inside the fixed page
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols,
                          left=left, right=right,
                          bottom=bottom, top=top)
    axes = gs.subplots(sharex=sharex, sharey=sharey)

    # This block sets the aspect ratio for each subplot axis so that it
    # matches the user's request.
    # The function _set_aspect sets the aspect ratio of a given axis to
    # either "auto" or a user-specified value.
    # It then applies this aspect logic to every axis in the figure,
    # regardless of whether there is
    # a single axis, a list/tuple of axes, or a numpy array of axes:
    def _set_aspect(ax):
        if axes_aspect == "auto":
            ax.set_aspect("auto", adjustable="box")
        else:
            ax.set_aspect(axes_aspect, adjustable="box")

    # Apply the aspect ratio to each axis in axes (an axis is a single plot)
    # Handles the potential cases for axes being a single object, tuple/list,
    # or numpy array
    if isinstance(axes, (list, tuple)):
        for ax in axes:
            _set_aspect(ax)
    else:
        try:
            # If axes is a numpy.ndarray, iterate through all axes flat
            for ax in axes.ravel():
                _set_aspect(ax)
        except Exception:
            # If axes is a single object, just set its aspect
            _set_aspect(axes)

    return fig, axes


def _fit_content_within_fixed_canvas(fig: mpl.figure.Figure, *,
                                     max_iter: int = 8,
                                     eps: float = 1e-4) -> None:
    """
    Shrink/shift axes so that the tight bounding box fits inside the
    *intended content rectangle* (set by setup_figure margins), while keeping
    the overall figure size fixed.
    """
    # Default target is full canvas if not provided
    target = getattr(fig, "_figformat_content_rect", (0.0, 1.0, 0.0, 1.0))
    target_left, target_right, target_bottom, target_top = target

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        tight = fig.get_tightbbox(renderer)  # inches
        fig_w, fig_h = fig.get_size_inches()

        # tight bbox edges in figure-fraction coords
        x0 = tight.x0 / fig_w
        y0 = tight.y0 / fig_h
        x1 = tight.x1 / fig_w
        y1 = tight.y1 / fig_h

        # Overflow relative to the TARGET rectangle (not 0..1)
        overflow_left = max(0.0, target_left - x0)
        overflow_bottom = max(0.0, target_bottom - y0)
        overflow_right = max(0.0, x1 - target_right)
        overflow_top = max(0.0, y1 - target_top)

        total = overflow_left + overflow_right + overflow_bottom + overflow_top
        if total < eps:
            return

        # Apply correction to all axes
        for ax in fig.axes:
            pos = ax.get_position()

            new_x0 = pos.x0 + overflow_left
            new_y0 = pos.y0 + overflow_bottom
            new_w = pos.width - (overflow_left + overflow_right)
            new_h = pos.height - (overflow_bottom + overflow_top)

            new_w = max(new_w, 0.01)
            new_h = max(new_h, 0.01)

            ax.set_position([new_x0, new_y0, new_w, new_h])


def save_figure(fig: mpl.figure.Figure, path: str, *,
                autofit: bool = True) -> None:
    """
    Save with the page size exactly as requested in setup_figure
    (no tight bbox), while optionally shrinking axes so that
    labels/titles/ticks fit within the page.

    This preserves the styling from _apply_rc() unchanged.
    """
    if autofit:
        _fit_content_within_fixed_canvas(fig)

    # Keep the page size fixed (rcParams already set savefig.bbox='standard')
    fig.savefig(path)
