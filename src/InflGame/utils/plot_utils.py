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
                       cbar_params: dict = {'common_cbar': False,'cbar_title':''},
                       axis_params: dict = {'common_axis': False,'axis_ylabel':'', 'axis_xlabel': ''},
                       font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},):
    """Put two plots side by side, handling 2D, 3D plots, and heat maps
    
    This function copies plot elements including:
    - Line plots
    - Scatter plots (2D and 3D)
    - Heat maps (imshow, contourf, pcolormesh)
    - Contour plots
    - Colorbars (when present)
    
    Parameters:
    -----------
    ax1, ax2 : plt.Axes
        The source axes to copy from
    title_main : str
        Main title for the combined figure
    title_ads : list, optional
        Additional title components to append
    cbar_params : dict, optional
        Colorbar configuration: {'common_cbar': bool, 'cbar_title': str}
    axis_params : dict, optional
        Axis configuration: {'common_axis': bool, 'axis_ylabel': str, 'axis_xlabel': str}
    font : dict, optional
        Font configuration dictionary
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The new figure with side-by-side plots
    """
    # Set font properties
    font['font.family'] = font.get('font_family', 'sans-serif')
    axis_size= font.get('axis_size', 15)
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 25)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    # Extract parameters
    common_cbar = cbar_params.get('common_cbar', False)
    cbar_title = cbar_params.get('cbar_title', '')
    common_axis = axis_params.get('common_axis', False)
    axis_ylabel = axis_params.get('axis_ylabel', '')
    axis_xlabel = axis_params.get('axis_xlabel', '')

    
    
    fig = plt.figure(figsize=(16, 8))
    
    # Variables to store image mappables for common colorbar
    left_images = []
    right_images = []
    
    # Check if ax1 is 3D
    if hasattr(ax1, 'zaxis'):
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
        ax_left.set_xlim(ax1.get_xlim())
        ax_left.set_ylim(ax1.get_ylim())
        ax_left.set_zlim(ax1.get_zlim())
        # Copy 3D tick labels
        ax_left.set_xticks(ax1.get_xticks())
        ax_left.set_xticklabels(ax1.get_xticklabels())
        ax_left.set_yticks(ax1.get_yticks())
        ax_left.set_yticklabels(ax1.get_yticklabels())
        ax_left.set_zticks(ax1.get_zticks())
        ax_left.set_zticklabels(ax1.get_zticklabels())
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
            
            im_left = ax_left.imshow(array, extent=extent, cmap=discrete_cmap, norm=norm,
                                   alpha=alpha, aspect=ax1.get_aspect(), origin='lower')
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
        for collection in ax1.collections:
            if hasattr(collection, 'get_array') and collection.get_array() is not None:
                # This handles contour fills, scatter plots with color mapping, etc.
                if hasattr(collection, '_paths'):  # ContourSet or similar
                    # For contour plots, we need to recreate them
                    # This is complex, so we'll copy the collection directly for now
                    ax_left.add_collection(type(collection)(collection._paths, **collection._get_patch_kwargs()))
                else:
                    ax_left.add_collection(collection)
        
        ax_left.set_xlabel(ax1.get_xlabel(), fontsize=axis_size)
        ax_left.set_ylabel(ax1.get_ylabel(), fontsize=axis_size)
        # Copy 2D axis limits
        ax_left.set_xlim(ax1.get_xlim())
        ax_left.set_ylim(ax1.get_ylim())
        # Copy tick labels and positions
        ax_left.set_xticks(ax1.get_xticks())
        ax_left.set_xticklabels(ax1.get_xticklabels())
        ax_left.set_yticks(ax1.get_yticks())
        ax_left.set_yticklabels(ax1.get_yticklabels())
        if ax1.get_aspect() != 'auto':
            ax_left.set_aspect(ax1.get_aspect())
        
        # Apply common axis labels if specified
        if common_axis:
            if axis_xlabel:
                ax_left.set_xlabel('')  # Remove individual x-label for left plot
            if axis_ylabel:
                ax_left.set_ylabel(axis_ylabel,size=axis_size)  # Set common y-label on left plot

    # Check if ax2 is 3D
    if hasattr(ax2, 'zaxis'):
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
        # Copy 3D axis limits
        ax_right.set_xlim(ax2.get_xlim())
        ax_right.set_ylim(ax2.get_ylim())
        ax_right.set_zlim(ax2.get_zlim())
        # Copy 3D tick labels
        ax_right.set_xticks(ax2.get_xticks())
        ax_right.set_xticklabels(ax2.get_xticklabels())
        ax_right.set_yticks(ax2.get_yticks())
        ax_right.set_yticklabels(ax2.get_yticklabels())
        ax_right.set_zticks(ax2.get_zticks())
        ax_right.set_zticklabels(ax2.get_zticklabels())
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
            
            im_right = ax_right.imshow(array, extent=extent, cmap=discrete_cmap, norm=norm,
                                     alpha=alpha, aspect=ax2.get_aspect(), origin='lower')
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
        for collection in ax2.collections:
            if hasattr(collection, 'get_array') and collection.get_array() is not None:
                # This handles contour fills, scatter plots with color mapping, etc.
                if hasattr(collection, '_paths'):  # ContourSet or similar
                    # For contour plots, we need to recreate them
                    # This is complex, so we'll copy the collection directly for now
                    ax_right.add_collection(type(collection)(collection._paths, **collection._get_patch_kwargs()))
                else:
                    ax_right.add_collection(collection)
        
        ax_right.set_xlabel(ax2.get_xlabel(), fontsize=axis_size)
        ax_right.set_ylabel(ax2.get_ylabel(), fontsize=axis_size)
        # Copy 2D axis limits
        ax_right.set_xlim(ax2.get_xlim())
        ax_right.set_ylim(ax2.get_ylim())
        # Copy tick labels and positions
        ax_right.set_xticks(ax2.get_xticks())
        ax_right.set_xticklabels(ax2.get_xticklabels())
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
    ax_left.set_title(ax1.get_title())
    ax_right.set_title(ax2.get_title())
    
    # Add legends if there are labeled lines
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
    if len(title_ads)>0:
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
    
    plt.tight_layout()
    plt.close()
    return fig