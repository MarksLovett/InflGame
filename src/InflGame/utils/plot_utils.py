"""
.. module:: plot_utils
   :synopsis: Provides plotting utilities for creating and manipulating figures in influencer games visualizations.

Plot Utils Module
=================

This module provides utility functions for creating and manipulating matplotlib figures, particularly
for side-by-side plot comparisons with support for 2D plots, 3D plots, heat maps, and shared colorbars.

The module is designed to work with the `InflGame` package and supports creating publication-quality
figures with consistent styling and formatting.

Dependencies:
-------------
- matplotlib
- numpy

Usage:
------
The `side_by_side_plots` function can be used to combine two existing plots into a single figure
with optional shared colorbars and axis labels.

Example:
--------

.. code-block:: python
    
    from InflGame.utils.plot_utils import side_by_side_plots
    import matplotlib.pyplot as plt
    import numpy as np

    # Create two example plots
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 4, 9])
    ax1.set_title("Plot 1")
    
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [1, 2, 3])
    ax2.set_title("Plot 2")
    
    # Combine them side by side
    combined_fig = side_by_side_plots(
        ax1, ax2, 
        title_main="Combined Plots",
        cbar_params={'common_cbar': False},
        axis_params={'common_axis': True, 'axis_xlabel': 'X-axis', 'axis_ylabel': 'Y-axis'}
    )
    combined_fig.show()
"""

from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def side_by_side_plots(ax1: plt.Axes,
                       ax2: plt.Axes,
                       title_main: str,
                       title_ads: list = [],
                       cbar_params: dict = {'common_cbar': False, 'cbar_title': ''},
                       axis_params: dict = {'common_axis': False, 'axis_ylabel': '', 'axis_xlabel': ''},
                       font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                       legend_params: dict = {'external_legend': False, 'legend_title': ''},
                       limits_params: dict = {'xlim_left': None, 'xlim_right': None, 'ylim': None}):
    """
    Create a side-by-side comparison figure from two existing plot axes.
    
    This function copies plot elements from two source axes and combines them into a single figure
    with two subplots placed side by side. It handles various plot types including line plots,
    scatter plots (2D and 3D), heat maps, contour plots, and preserves colorbars.

    The function supports:
    
    - **Line plots**: Standard 2D line plots with all styling preserved
    - **Scatter plots**: Both 2D and 3D scatter plots with colors and sizes
    - **Heat maps**: Image plots (imshow, contourf, pcolormesh) with discrete colorbars
    - **Contour plots**: Filled contour plots with proper styling
    - **3D plots**: Full 3D plot support with axis labels and limits
    - **Colorbars**: Individual or shared colorbars with discrete levels

    Parameters
    ----------
    ax1 : plt.Axes
        The first source axes to copy from (left subplot in output).
    ax2 : plt.Axes
        The second source axes to copy from (right subplot in output).
    title_main : str
        Main title for the combined figure.
    title_ads : list, optional
        Additional title components to append to the main title, by default [].
    cbar_params : dict, optional
        Colorbar configuration dictionary, by default ``{'common_cbar': False, 'cbar_title': ''}``.
        
        - ``'common_cbar'`` (bool): If True, create a single shared colorbar; if False, create individual colorbars
        - ``'cbar_title'`` (str): Title/label for the colorbar
    axis_params : dict, optional
        Axis configuration dictionary, by default ``{'common_axis': False, 'axis_ylabel': '', 'axis_xlabel': ''}``.
        
        - ``'common_axis'`` (bool): If True, use shared axis labels
        - ``'axis_ylabel'`` (str): Common y-axis label
        - ``'axis_xlabel'`` (str): Common x-axis label
    font : dict, optional
        Font configuration dictionary, by default ``{'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'}``.
        
        - ``'default_size'`` (int): Default font size for general text and tick labels
        - ``'cbar_size'`` (int): Font size for colorbar tick labels
        - ``'title_size'`` (int): Font size for figure suptitle
        - ``'subtitle_size'`` (int): Font size for subplot titles (falls back to ``title_size``)
        - ``'legend_size'`` (int): Font size for legend text
        - ``'axis_size'`` (int): Font size for axis labels (also accepts ``'label_size'``)
        - ``'font_family'`` (str): Font family (e.g., 'sans-serif', 'serif')
    limits_params : dict, optional
        Axis limits and tick configuration, by default ``{'xlim_left': None, 'xlim_right': None, 'ylim': None}``.
        
        - ``'xlim_left'`` (list): [min, max] x-axis limits for left subplot
        - ``'xlim_right'`` (list): [min, max] x-axis limits for right subplot
        - ``'ylim'`` (list): [min, max] y-axis limits for both subplots
        - ``'tick_count_left'`` (int): Maximum number of ticks on each axis of the left subplot (>= 2)
        - ``'tick_count_right'`` (int): Maximum number of ticks on each axis of the right subplot (>= 2)

    Returns
    -------
    matplotlib.figure.Figure
        The new figure containing both plots arranged side by side.
        
    Notes
    -----
    - The function creates discrete colorbars with centered labels for heat maps
    - 3D plots are properly detected and handled with appropriate projection settings
    - All axis limits, tick positions, and labels are preserved from the source plots
    - The function closes the original figures to prevent memory leaks
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> # Create two sample plots
    >>> fig1, ax1 = plt.subplots()
    >>> x = np.linspace(0, 10, 100)
    >>> ax1.plot(x, np.sin(x))
    >>> ax1.set_title("Sine Wave")
    >>> 
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.plot(x, np.cos(x))
    >>> ax2.set_title("Cosine Wave")
    >>> 
    >>> # Combine with shared x-axis label
    >>> combined = side_by_side_plots(
    ...     ax1, ax2,
    ...     title_main="Trigonometric Functions",
    ...     axis_params={'common_axis': True, 'axis_xlabel': 'x', 'axis_ylabel': 'f(x)'}
    ... )
    """
    # Set font properties
    font['font.family'] = font.get('font_family', 'sans-serif')
    axis_size = font.get('axis_size', font.get('label_size', 15))
    cbar_font_size = font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 25)
    subtitle_font_size = font.get('subtitle_size', title_font_size)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    # Extract parameters
    common_cbar = cbar_params.get('common_cbar', False)
    cbar_title = cbar_params.get('cbar_title', '')
    common_axis = axis_params.get('common_axis', False)
    axis_ylabel = axis_params.get('axis_ylabel', '')
    axis_xlabel = axis_params.get('axis_xlabel', '')
    show_legend = legend_params.get('show_legend', True)
    external_legend = legend_params.get('external_legend', False)
    legend_title = legend_params.get('legend_title', '')
    xlim_left = limits_params.get('xlim_left', [0.03, 0.3])
    xlim_right = limits_params.get('xlim_right',[0.03, 0.3])
    ylim = limits_params.get('ylim', None)
    tick_count_left = limits_params.get('tick_count_left', None)
    tick_count_right = limits_params.get('tick_count_right', None)
    if tick_count_left is not None:
        tick_count_left = max(2, int(tick_count_left))
    if tick_count_right is not None:
        tick_count_right = max(2, int(tick_count_right))

    # Use GridSpec for better control when external legend is needed
    from matplotlib.gridspec import GridSpec
    if show_legend and external_legend:
        # Create figure with GridSpec: 2 plot columns + 1 legend column (wider legend space)
        fig = plt.figure(figsize=(40, 16))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.25], wspace=0.5)
    else:
        fig = plt.figure(figsize=(25, 13))
        gs = None
    
    # Variables to store image mappables for common colorbar
    left_images = []
    right_images = []
    
    # Check if ax1 is 3D
    if hasattr(ax1, 'zaxis'):
        if gs is not None:
            ax_left = fig.add_subplot(gs[0], projection='3d')
        else:
            ax_left = fig.add_subplot(121, projection='3d')
        # Copy 3D data from ax1
        for line in ax1.get_lines():
            if hasattr(line, '_verts3d'):
                x, y, z = line._verts3d
                ax_left.plot(x, y, z, color=line.get_color(), label=line.get_label(),
                           linewidth=line.get_linewidth(), linestyle=line.get_linestyle())
        # Copy 3D scatter plots if any
        for collection in ax1.collections:
            if hasattr(collection, '_offsets3d'):
                x, y, z = collection._offsets3d
                ax_left.scatter(x, y, z, c=collection.get_facecolors(), 
                              s=collection.get_sizes(), alpha=collection.get_alpha(),
                              label=collection.get_label())
        # Set 3D labels and title
        ax_left.set_xlabel(ax1.get_xlabel(), fontsize=axis_size)
        ax_left.set_ylabel(ax1.get_ylabel(), fontsize=axis_size)
        ax_left.set_zlabel(ax1.get_zlabel(), fontsize=axis_size)
        # Copy 3D axis limits
        ax_left.set_xlim(xlim_left if xlim_left is not None else ax1.get_xlim())
        ax_left.set_ylim(ylim if ylim is not None else ax1.get_ylim())
        ax_left.set_zlim(ax1.get_zlim())
        # Only copy tick labels if no custom limits provided (otherwise let matplotlib auto-generate)
        if xlim_left is None:
            ax_left.set_xticks(ax1.get_xticks())
            ax_left.set_xticklabels(ax1.get_xticklabels())
        if ylim is None:
            ax_left.set_yticks(ax1.get_yticks())
            ax_left.set_yticklabels(ax1.get_yticklabels())
        ax_left.set_zticks(ax1.get_zticks())
        ax_left.set_zticklabels(ax1.get_zticklabels())
    else:
        if gs is not None:
            ax_left = fig.add_subplot(gs[0])
        else:
            ax_left = fig.add_subplot(121)
        # Copy 2D data from ax1
        for line in ax1.get_lines():
            ax_left.plot(line.get_xdata(), line.get_ydata(), 
                        color=line.get_color(), label=line.get_label(),
                        linewidth=line.get_linewidth(), linestyle=line.get_linestyle())
        
        # Copy images (heat maps, contour plots, etc.)
        for image in ax1.get_images():
            # Get image data and extent
            array = image.get_array()
            extent = image.get_extent()
            cmap = image.get_cmap()
            vmin, vmax = image.get_clim()
            alpha = image.get_alpha()
            
            # Create discrete colormap and normalization
            n_levels = int(vmax)  # Number of discrete levels
            boundaries = np.linspace(0, vmax, n_levels + 1)
            norm = BoundaryNorm(boundaries, ncolors=n_levels)
            
            # Create discrete colormap by sampling the original colormap
            colors = cmap(np.linspace(0, 1, n_levels))
            discrete_cmap = ListedColormap(colors)
            
            # Use pcolormesh instead of imshow so heatmap data is vector (not raster) in SVG
            _arr = np.array(array)
            if _arr.ndim > 2:
                _arr = _arr[:, :, 0]  # take first channel if RGB/A
            # Downsample to cap SVG cell count (nearest-neighbour, preserves discrete values)
            _max_cells = 60
            rows, cols = _arr.shape
            row_step = max(1, rows // _max_cells)
            col_step = max(1, cols // _max_cells)
            _arr = _arr[::row_step, ::col_step]
            rows, cols = _arr.shape
            x_mesh = np.linspace(extent[0], extent[1], cols + 1)
            y_mesh = np.linspace(extent[2], extent[3], rows + 1)
            X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)
            im_left = ax_left.pcolormesh(X_mesh, Y_mesh, _arr, cmap=discrete_cmap, norm=norm,
                                         alpha=alpha)
            left_images.append(im_left)
            
            # Add individual colorbar only if not using common colorbar
            if not common_cbar:
                try:
                    cbar = plt.colorbar(im_left, ax=ax_left)
                    cbar.ax.tick_params(labelsize=cbar_font_size)
                    
                    # Make individual colorbar discrete with centered labels
                    vmin, vmax = im_left.get_clim()
                    n_levels = int(vmax)  # Number of discrete levels
                    
                    # Create boundaries for discrete levels
                    boundaries = np.linspace(0, vmax, n_levels + 1)
                    
                    # Create centered tick positions and labels
                    center_positions = (boundaries[:-1] + boundaries[1:]) / 2
                    center_labels = [f'{int(val)}' for val in center_positions]
                    
                    # Set ticks at center positions with center labels
                    cbar.set_ticks(center_positions)
                    cbar.set_ticklabels(center_labels)
                except Exception as e:
                    # If error occurs, just continue
                    print(f"Warning: Could not create colorbar for ax1: {e}")
                    pass
        
        # Copy collections (scatter plots, contour fills, etc.)
        from matplotlib.collections import PathCollection as _PathCollection
        for collection in ax1.collections:
            if isinstance(collection, _PathCollection):
                # Scatter plot — reconstruct with ax.scatter so SVG stays compact
                offsets = collection.get_offsets()
                c_array = collection.get_array()
                sizes = collection.get_sizes()
                _s = sizes[0] if len(sizes) == 1 else (sizes if len(sizes) > 1 else 10)
                alpha = collection.get_alpha()
                if c_array is not None and len(c_array) > 0:
                    sc = ax_left.scatter(offsets[:, 0], offsets[:, 1], c=np.asarray(c_array),
                                         cmap=collection.get_cmap(), norm=collection.norm,
                                         s=_s, alpha=alpha, linewidths=0, rasterized=False)
                    left_images.append(sc)
                elif len(offsets) > 0:
                    fc = collection.get_facecolors()
                    ax_left.scatter(offsets[:, 0], offsets[:, 1],
                                    color=fc[:len(offsets)] if len(fc) >= len(offsets) else fc,
                                    s=_s, alpha=alpha, linewidths=0, rasterized=False)
            elif hasattr(collection, 'get_array') and collection.get_array() is not None:
                try:
                    ax_left.add_collection(type(collection)(collection._paths, **collection._get_patch_kwargs()))
                except Exception:
                    ax_left.add_collection(collection)

        ax_left.set_xlabel(ax1.get_xlabel(), fontsize=axis_size)
        ax_left.set_ylabel(ax1.get_ylabel(), fontsize=axis_size)
        # Copy 2D axis limits (use provided limits if available)
        ax_left.set_xlim(xlim_left if xlim_left is not None else ax1.get_xlim())
        ax_left.set_ylim(ylim if ylim is not None else ax1.get_ylim())
        # Only copy tick labels if no custom limits provided (otherwise let matplotlib auto-generate)
        if xlim_left is None:
            ax_left.set_xticks(ax1.get_xticks())
            ax_left.set_xticklabels(ax1.get_xticklabels())
        if ylim is None:
            ax_left.set_yticks(ax1.get_yticks())
            ax_left.set_yticklabels(ax1.get_yticklabels())
        if ax1.get_aspect() != 'auto':
            ax_left.set_aspect(ax1.get_aspect())
        
        # Apply common axis labels if specified
        if common_axis:
            if axis_xlabel:
                ax_left.set_xlabel('')  # Remove individual x-label for left plot
            if axis_ylabel:
                ax_left.set_ylabel(axis_ylabel, size=axis_size)  # Set common y-label on left plot

    # Check if ax2 is 3D
    if hasattr(ax2, 'zaxis'):
        if gs is not None:
            ax_right = fig.add_subplot(gs[1], projection='3d')
        else:
            ax_right = fig.add_subplot(122, projection='3d')
        # Copy 3D data from ax2
        for line in ax2.get_lines():
            if hasattr(line, '_verts3d'):
                x, y, z = line._verts3d
                ax_right.plot(x, y, z, color=line.get_color(), label=line.get_label(),
                            linewidth=line.get_linewidth(), linestyle=line.get_linestyle())
        # Copy 3D scatter plots if any
        for collection in ax2.collections:
            if hasattr(collection, '_offsets3d'):
                x, y, z = collection._offsets3d
                ax_right.scatter(x, y, z, c=collection.get_facecolors(), 
                               s=collection.get_sizes(), alpha=collection.get_alpha(),
                               label=collection.get_label())
        # Set 3D labels and title
        ax_right.set_xlabel(ax2.get_xlabel(), fontsize=axis_size)
        ax_right.set_ylabel(ax2.get_ylabel(), fontsize=axis_size)
        ax_right.set_zlabel(ax2.get_zlabel(), fontsize=axis_size)
        # Copy 3D axis limits (use provided limits if available)
        ax_right.set_xlim(xlim_right if xlim_right is not None else ax2.get_xlim())
        ax_right.set_ylim(ylim if ylim is not None else ax2.get_ylim())
        ax_right.set_zlim(ax2.get_zlim())
        # Only copy tick labels if no custom limits provided (otherwise let matplotlib auto-generate)
        if xlim_right is None:
            ax_right.set_xticks(ax2.get_xticks())
            ax_right.set_xticklabels(ax2.get_xticklabels())
        if ylim is None:
            ax_right.set_yticks(ax2.get_yticks())
            ax_right.set_yticklabels(ax2.get_yticklabels())
        ax_right.set_zticks(ax2.get_zticks())
        ax_right.set_zticklabels(ax2.get_zticklabels())
    else:
        if gs is not None:
            ax_right = fig.add_subplot(gs[1])
        else:
            ax_right = fig.add_subplot(122)
        # Copy 2D data from ax2
        for line in ax2.get_lines():
            ax_right.plot(line.get_xdata(), line.get_ydata(), 
                         color=line.get_color(), label=line.get_label(),
                         linewidth=line.get_linewidth(), linestyle=line.get_linestyle())
        
        # Copy images (heat maps, contour plots, etc.)
        for image in ax2.get_images():
            # Get image data and extent
            array = image.get_array()
            extent = image.get_extent()
            cmap = image.get_cmap()
            vmin, vmax = image.get_clim()
            alpha = image.get_alpha()
            
            # Create discrete colormap and normalization
            n_levels = int(vmax)  # Number of discrete levels
            boundaries = np.linspace(0, vmax, n_levels + 1)
            norm = BoundaryNorm(boundaries, ncolors=n_levels)
            
            # Create discrete colormap by sampling the original colormap
            colors = cmap(np.linspace(0, 1, n_levels))
            discrete_cmap = ListedColormap(colors)
            
            # Use pcolormesh instead of imshow so heatmap data is vector (not raster) in SVG
            _arr = np.array(array)
            if _arr.ndim > 2:
                _arr = _arr[:, :, 0]  # take first channel if RGB/A
            # Downsample to cap SVG cell count (nearest-neighbour, preserves discrete values)
            _max_cells = 60
            rows, cols = _arr.shape
            row_step = max(1, rows // _max_cells)
            col_step = max(1, cols // _max_cells)
            _arr = _arr[::row_step, ::col_step]
            rows, cols = _arr.shape
            x_mesh = np.linspace(extent[0], extent[1], cols + 1)
            y_mesh = np.linspace(extent[2], extent[3], rows + 1)
            X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)
            im_right = ax_right.pcolormesh(X_mesh, Y_mesh, _arr, cmap=discrete_cmap, norm=norm,
                                           alpha=alpha)
            right_images.append(im_right)
            
            # Add individual colorbar only if not using common colorbar
            if not common_cbar:
                try:
                    cbar = plt.colorbar(im_right, ax=ax_right)
                    cbar.ax.tick_params(labelsize=cbar_font_size)
                    
                    # Make individual colorbar discrete with centered labels
                    vmin, vmax = im_right.get_clim()
                    n_levels = int(vmax)  # Number of discrete levels
                    
                    # Create boundaries for discrete levels
                    boundaries = np.linspace(0, vmax, n_levels + 1)
                    
                    # Create centered tick positions and labels
                    center_positions = (boundaries[:-1] + boundaries[1:]) / 2
                    center_labels = [f'{int(val)}' for val in center_positions]
                    
                    # Set ticks at center positions with center labels
                    cbar.set_ticks(center_positions)
                    cbar.set_ticklabels(center_labels)
                except Exception as e:
                    # If error occurs, just continue
                    print(f"Warning: Could not create colorbar for ax2: {e}")
                    pass
        
        # Copy collections (scatter plots, contour fills, etc.)
        from matplotlib.collections import PathCollection as _PathCollection
        for collection in ax2.collections:
            if isinstance(collection, _PathCollection):
                # Scatter plot — reconstruct with ax.scatter so SVG stays compact
                offsets = collection.get_offsets()
                c_array = collection.get_array()
                sizes = collection.get_sizes()
                _s = sizes[0] if len(sizes) == 1 else (sizes if len(sizes) > 1 else 10)
                alpha = collection.get_alpha()
                if c_array is not None and len(c_array) > 0:
                    sc = ax_right.scatter(offsets[:, 0], offsets[:, 1], c=np.asarray(c_array),
                                          cmap=collection.get_cmap(), norm=collection.norm,
                                          s=_s, alpha=alpha, linewidths=0, rasterized=False)
                    right_images.append(sc)
                elif len(offsets) > 0:
                    fc = collection.get_facecolors()
                    ax_right.scatter(offsets[:, 0], offsets[:, 1],
                                     color=fc[:len(offsets)] if len(fc) >= len(offsets) else fc,
                                     s=_s, alpha=alpha, linewidths=0, rasterized=False)
            elif hasattr(collection, 'get_array') and collection.get_array() is not None:
                try:
                    ax_right.add_collection(type(collection)(collection._paths, **collection._get_patch_kwargs()))
                except Exception:
                    ax_right.add_collection(collection)

        ax_right.set_xlabel(ax2.get_xlabel(), fontsize=axis_size)
        ax_right.set_ylabel(ax2.get_ylabel(), fontsize=axis_size)
        # Copy 2D axis limits (use provided limits if available)
        ax_right.set_xlim(xlim_right if xlim_right is not None else ax2.get_xlim())
        ax_right.set_ylim(ylim if ylim is not None else ax2.get_ylim())
        # Only copy tick labels if no custom limits provided (otherwise let matplotlib auto-generate)
        if xlim_right is None:
            ax_right.set_xticks(ax2.get_xticks())
            ax_right.set_xticklabels(ax2.get_xticklabels())
        if ylim is None:
            ax_right.set_yticks(ax2.get_yticks())
            ax_right.set_yticklabels(ax2.get_yticklabels())
        if ax2.get_aspect() != 'auto':
            ax_right.set_aspect(ax2.get_aspect())
        
        # Apply common axis labels if specified
        if common_axis:
            if axis_xlabel:
                ax_right.set_xlabel('')  # Remove individual x-label for right plot
            if axis_ylabel:
                ax_right.set_ylabel('')  # Remove y-label from right plot when using common labels
    
    # Copy titles
    ax_left.set_title(ax1.get_title(), fontsize=subtitle_font_size)
    ax_right.set_title(ax2.get_title(), fontsize=subtitle_font_size)
    
    # Handle legends - collect all labeled lines for potential external legend
    all_lines = []
    all_labels = []
    
    # Collect lines from ax1/ax_left
    for line in ax_left.get_lines():
        label = line.get_label()
        if label and not label.startswith('_'):
            all_lines.append(line)
            all_labels.append(label)
    
    # Add legends based on external_legend setting (only if show_legend is True)
    if show_legend:
        if external_legend and gs is not None:
            # Create external legend axis using GridSpec
            if all_lines:
                # Get unique labels (avoid duplicates)
                unique_labels = []
                unique_lines = []
                for line, label in zip(all_lines, all_labels):
                    if label not in unique_labels:
                        unique_labels.append(label)
                        unique_lines.append(line)
                
                # Create a dedicated axis for the legend in the third GridSpec column
                ax_legend = fig.add_subplot(gs[2])
                ax_legend.axis('off')  # Hide the axis
                
                # Create legend in the dedicated axis
                ax_legend.legend(unique_lines, unique_labels, 
                               loc='center',
                               fontsize=legend_font_size,
                               title=legend_title if legend_title else None,
                               title_fontsize=legend_font_size,
                               frameon=True)
        else:
            # Original behavior: place legends inside each subplot
            if any(line.get_label() and not line.get_label().startswith('_') for line in ax1.get_lines()):
                # Get original legend location if it exists
                original_legend = ax1.get_legend()
                if original_legend:
                    # Get the location from the original legend
                    loc = original_legend._loc
                    ax_left.legend(loc='lower center')
                else:
                    ax_left.legend()
            
            if any(line.get_label() and not line.get_label().startswith('_') for line in ax2.get_lines()):
                # Get original legend location if it exists
                original_legend = ax2.get_legend()
                if original_legend:
                    # Get the location from the original legend
                    loc = original_legend._loc
                    ax_right.legend(loc='lower center')
                else:
                    ax_right.legend()

    # Set title for the entire figure
    if len(title_ads) > 0:
        for item in title_ads:
            title_main += " " + item
    fig.suptitle(title_main, fontsize=title_font_size)
    
    # Add common x-axis label if specified
    if common_axis and axis_xlabel:
        fig.text(0.5, -0.02, axis_xlabel, ha='center', va='bottom', fontsize=axis_size)
    
    # Add common colorbar if requested and there are images
    if common_cbar and (left_images or right_images):
        try:
            # Use the first available image for the colorbar
            if left_images:
                reference_image = left_images[0]
            elif right_images:
                reference_image = right_images[0]
            
            # Create common colorbar as inset to the right of ax2
            # Only create if ax2 is 2D (not 3D)
            if not hasattr(ax_right, 'zaxis'):
                # Create inset axes to the right of ax2
                cax = inset_axes(ax_right, 
                               width="5%",  # width of colorbar
                               height="100%",  # height of colorbar
                               loc='center left',
                               bbox_to_anchor=(1.02, 0., 1, 1),
                               bbox_transform=ax_right.transAxes,
                               borderpad=0)
                
                cbar = fig.colorbar(reference_image, cax=cax)
                cbar.ax.tick_params(labelsize=cbar_font_size)
                
                # Add colorbar title if specified
                if cbar_title:
                    cbar.set_label(cbar_title, fontsize=default_font_size)
                
                # Make colorbar discrete with centered labels
                # Get the data range and create discrete levels
                vmin, vmax = reference_image.get_clim()
                
                # Create discrete levels
                n_levels = int(vmax)  # Number of discrete levels
                boundaries = np.linspace(0, vmax, n_levels + 1)
                
                # Create centered tick positions and labels
                center_positions = (boundaries[:-1] + boundaries[1:]) / 2
                center_labels = [f'{int(val)}' for val in center_positions]
                
                # Set ticks at center positions with center labels
                cbar.set_ticks(center_positions)
                cbar.set_ticklabels(center_labels)
            else:
                print("Warning: Common colorbar not supported for 3D plots")
        except Exception as e:
            print(f"Warning: Could not create common colorbar: {e}")
            pass
    
    # Apply tick count limits if specified
    if tick_count_left is not None:
        ax_left.xaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_left))
        ax_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_left))
        if hasattr(ax_left, 'zaxis'):
            ax_left.zaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_left))
    if tick_count_right is not None:
        ax_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_right))
        ax_right.yaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_right))
        if hasattr(ax_right, 'zaxis'):
            ax_right.zaxis.set_major_locator(ticker.MaxNLocator(nbins=tick_count_right))

    # Apply tick label font sizes explicitly
    ax_left.tick_params(axis='both', labelsize=default_font_size)
    ax_right.tick_params(axis='both', labelsize=default_font_size)
    if hasattr(ax_left, 'zaxis'):
        ax_left.zaxis.set_tick_params(labelsize=default_font_size)
    if hasattr(ax_right, 'zaxis'):
        ax_right.zaxis.set_tick_params(labelsize=default_font_size)

    plt.tight_layout()
    plt.close()
    return fig