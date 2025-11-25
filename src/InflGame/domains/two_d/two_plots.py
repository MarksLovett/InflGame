
"""
.. module:: two_plots
   :synopsis: Provides 2D visualization tools for analyzing agent dynamics and resource distributions in influencer games.

2D Visualization Module
=======================

This module provides visualization tools for analyzing and understanding the dynamics of agents and resource distributions in 2D domains for influencer games.
It includes utilities for plotting agent positions, influence distributions, and bifurcation dynamics in 2D rectangular domains.

The module is designed to work with the `InflGame.adaptive` subpackage and supports creating visual representations of agent behaviors and resource distributions in 2D environments.


Usage:
------
The functions in this module can be used to visualize agent dynamics and resource distributions in 2D domains. For example, the `dist_and_pos_plot_2d_simple` function
can be used to plot agent positions over time and their influence distributions.

"""

from typing import Optional
import numpy as np
import torch
import colorsys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pylab
import matplotlib.figure
import matplotlib as mpl
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def dist_and_pos_plot_2d_simple(num_agents: int,
                                bin_points: np.ndarray,
                                cmap1,
                                cmap2,
                                pos_matrix: torch.Tensor,
                                infl_dist: torch.Tensor,
                                resource_type: str,
                                x_min: Optional[float] = None,
                                y_min: Optional[float] = None,
                                domain_bounds: Optional[torch.Tensor] = None,
                                resources: torch.Tensor = 0,
                                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif','sub_title_size':12},
                                 ) -> matplotlib.figure.Figure:
    """
    Plots the positions of agents over time and their influence distributions.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Points representing resource bins.
    :type bin_points: np.ndarray
    :param rect_X: X-coordinates of the rectangular grid.
    :type rect_X: np.ndarray
    :param rect_Y: Y-coordinates of the rectangular grid.
    :type rect_Y: np.ndarray
    :param cmap1: Colormap for agent positions.
    :type cmap1: Any
    :param cmap2: Colormap for influence distributions.
    :type cmap2: Any
    :param pos_matrix: Tensor containing agent positions over time.
    :type pos_matrix: torch.Tensor
    :param infl_dist: Tensor containing influence distributions.
    :type infl_dist: torch.Tensor
    :param resource_type: Type of resource distribution.
    :type resource_type: str
    :param resources: Resource values, defaults to 0.
    :type resources: str, optional
    :returns: The generated plot figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    sub_title_font_size = font.get('sub_title_size', 12)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    x_coords = bin_points[:, 0].numpy() if torch.is_tensor(bin_points) else bin_points[:, 0]
    y_coords = bin_points[:, 1].numpy() if torch.is_tensor(bin_points) else bin_points[:, 1]

    # Generate a distinct, high-contrast color for each agent using HSV spacing.
    # This scales to any number of agents and returns a suggested edge color
    # (black or white) chosen for good contrast against the marker fill.
    def make_agent_colors(n: int, saturation: float = 0.65, value: float = 0.9):
        cols = []
        for i in range(n):
            # Evenly space hues around the circle
            h = float(i) / max(1, n)
            r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
            # Perceived luminance to select an edge color for contrast
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            edge = 'black' if lum > 0.5 else 'white'
            cols.append(((r, g, b), edge))
        return cols

    agent_colors = make_agent_colors(num_agents)
    cm = pylab.get_cmap(cmap1)
    fig = plt.figure(figsize=(24, 18))

    # Arrange the right column into a square grid (k x k) if possible so
    # that multiple agent influence plots form a compact square layout.
    k = int(np.ceil(np.sqrt(max(1, num_agents))))
    outer_gs = GridSpec(nrows=k, ncols=2, width_ratios=[1, 1], wspace=.5,
                        hspace=.1,top=.65)

    # Left column: big positions/heatmap spanning all k rows
    ax0 = fig.add_subplot(outer_gs[:, 0])
    ax0.set_box_aspect(1)
    for a_id in range(num_agents):
        new_coor = pos_matrix[:, a_id]
        y = new_coor.detach().cpu().numpy()
        fill_color, edge_color = agent_colors[a_id]
        # Ensure agents are drawn above the heatmap by increasing zorder
        ax0.scatter(y[-1, 0], y[-1, 1], s=70, color=fill_color, linewidth=0.3,
                    label='Agent ' + str(a_id), zorder=6)
        # Starting position: hollow marker with a contrasting edge
        ax0.scatter(y[0, 0], y[0, 1], s=70, facecolors='none', edgecolors=edge_color,
                    linewidth=1, zorder=7)
        ax0.plot(y[:, 0], y[:, 1], color=fill_color, linewidth=2, zorder=5)
        
    # Ensure domain_bounds are numeric floats (handle torch tensors gracefully)
    if domain_bounds is None:
        domain_bounds = np.array([[np.min(x_coords), np.max(x_coords)], [np.min(y_coords), np.max(y_coords)]], dtype=float)
    else:
        # If domain_bounds are torch tensors, convert to numpy
        if torch.is_tensor(domain_bounds):
            domain_bounds = domain_bounds.detach().cpu().numpy()
        domain_bounds = np.asarray(domain_bounds, dtype=float)

    # Create a regular grid for interpolation using numeric floats
    grid_x, grid_y = np.mgrid[domain_bounds[0, 0]:domain_bounds[0, 1]:100j,
                              domain_bounds[1, 0]:domain_bounds[1, 1]:100j]

    # Interpolate resource values onto the grid
    points = np.column_stack((x_coords, y_coords))
    grid_z = griddata(points, resources, (grid_x, grid_y), method='cubic', fill_value=0)

    # Create interpolated heatmap on the positions axis (ax0) so positions overlay the heatmap
    # Draw heatmap below agent markers
    im = ax0.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap=cmap1, alpha=.9, zorder=1)
    ax0.set_xlabel('Trait 1', fontsize=14)
    ax0.set_ylabel('Trait 2', fontsize=14)
    ax0.set_title('Resource Heatmap', fontsize=16, fontweight='bold')
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlim(domain_bounds[0, 0], domain_bounds[0, 1])
    ax0.set_ylim(domain_bounds[1, 0], domain_bounds[1, 1])
    if resource_type=='custom_2d_rect':
        ax0.set_xticks([0,5])
        ax0.set_xticklabels([f'{x_min}','1'],fontdict={'size':20})
        ax0.set_yticks([0,5])
        ax0.set_yticklabels([f'{y_min}','1'],fontdict={'size':20})
    ax0.legend(loc='upper right', fontsize=10)
    # Place the first colorbar in an explicit axes just to the right of ax0
    # so there is guaranteed empty space between the heatmap and its colorbar.
    try:
        pos0 = ax0.get_position()
        gap0 = 0.01
        cax0_width = 0.03
        cax0_height = pos0.height * 0.9
        cax0_x = pos0.x1 + gap0
        cax0_y = pos0.y0 + 0.05 * pos0.height
        cax0 = fig.add_axes([cax0_x, cax0_y, cax0_width, cax0_height])
        cbar1 = fig.colorbar(im, cax=cax0)
        cbar1.set_label('Resource Values', rotation=270, labelpad=20, fontsize=12)
    except Exception:
        # Fallback to automatic colorbar placement
        cbar1 = fig.colorbar(im, ax=ax0)
        cbar1.set_label('Resource Values', rotation=270, labelpad=20, fontsize=12)


    # Create a container axis for the right column (used for title + colorbar)
    ax_right_container = fig.add_subplot(outer_gs[:, 1])
    # Reduce the visual height of the right-column container so the column
    # of small agent plots appears shorter. We move the bottom up slightly
    # and shrink height to keep the title and colorbar aligned.
    try:
        pos = ax_right_container.get_position()
        new_pos = [pos.x0, pos.y0 + 0.08 * pos.height, pos.width, pos.height * 0.82]
        ax_right_container.set_position(new_pos)
    except Exception:
        # If position manipulation fails in some backends, fall back silently
        pass
    # Subdivide the right column into a k x k grid
    right_spec = GridSpecFromSubplotSpec(k, k, subplot_spec=ax_right_container.get_subplotspec(),
                                         hspace=0.001, wspace=0.15)

    pcm = None
    for a_id in range(num_agents):
        r = a_id // k
        c = a_id % k
        ax1 = fig.add_subplot(right_spec[r, c])
        ax1.set_box_aspect(1)
        pvals = infl_dist[a_id].numpy()
        grid_w = griddata(points, pvals, (grid_x, grid_y), method='cubic', fill_value=0)
        pcm = ax1.pcolormesh(grid_x, grid_y, grid_w, cmap=cmap2)
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"Agent {a_id}", fontsize=max(8, sub_title_font_size - 1))

    # Blank-out any remaining cells in the k x k grid (if k*k > num_agents)
    total_cells = k * k
    for idx in range(num_agents, total_cells):
        r = idx // k
        c = idx % k
        ax_empty = fig.add_subplot(right_spec[r, c])
        ax_empty.axis('off')
    
    # Create a (mostly empty) subplot on the right column and attach a
    # narrow colorbar axis to it so the colorbar appears small regardless
    # of the figure size.
    # Create an explicit colorbar axis placed a fixed gap to the right of
    # the agent column so there is guaranteed empty space between the
    # agent plots and the colorbar regardless of GridSpec scaling.
    try:
        pos = ax_right_container.get_position()
        gap = 0.03  # fraction of figure width to separate the colorbar
        cax_width = 0.04
        cax_height = pos.height * 0.85
        cax_x = pos.x1 + gap
        cax_y = pos.y0 + 0.07 * pos.height
        # Add axes in figure coordinates [x, y, width, height]
        cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])
        if pcm is not None:
            cbar2 = fig.colorbar(pcm, cax=cax, extend='max')
            cbar2.set_label('Influence Values', rotation=270, labelpad=20, fontsize=12)
    except Exception:
        # Fallback: use make_axes_locatable if explicit placement isn't supported
        divider = make_axes_locatable(ax_right_container)
        cax = divider.append_axes("right", size="6%", pad=0.5)
        if pcm is not None:
            cbar2 = fig.colorbar(pcm, cax=cax, extend='max')
            cbar2.set_label('Influence Values', rotation=270, labelpad=20, fontsize=12)

    # Create a dedicated title axes above the right-column container so the
    # title has its own space and won't be clipped by the agent subplots.
    try:
        pos = ax_right_container.get_position()
        title_h = 0.1  # height of title area in figure coords
        # place title just above the container, clamped to fit inside figure
        title_y = min(0.98 - title_h, pos.y1 + 0.01)
        title_ax = fig.add_axes([pos.x0, title_y, pos.width, title_h])
        title_ax.text(0.5, 0.5, "Agents' influence distributions",
                      ha='center', va='center', fontsize=22)
        title_ax.axis('off')
    except Exception:
        # Fallback: set title on container (may be clipped on some backends)
        ax_right_container.set_title("Agents' influence distributions", fontsize=sub_title_font_size, pad=12)
    ax_right_container.axis('off')
    
    plt.close()
    return fig

def dist_plot_2d(agent_id: int,
                 infl_dist: torch.Tensor,
                 rect_Y: np.ndarray,
                 rect_X: np.ndarray) -> matplotlib.figure.Figure:
    """
    Plots the influence distribution of a single agent.

    :param agent_id: ID of the agent.
    :type agent_id: int
    :param infl_dist: Tensor containing influence distributions.
    :type infl_dist: torch.Tensor
    :param rect_Y: Y-coordinates of the rectangular grid.
    :type rect_Y: np.ndarray
    :param rect_X: X-coordinates of the rectangular grid.
    :type rect_X: np.ndarray
    :returns: The generated plot figure.
    :rtype: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    pval=infl_dist[agent_id].numpy()
    pval=pval.reshape(len(rect_Y),len(rect_X))
    im = ax.pcolormesh(rect_X,rect_Y, pval)

    # Make the plot square
    ax.set_box_aspect(1) 

    # Add a colorbar
    fig.colorbar(im)
    plt.close()
    return fig

def equilibrium_bifurcation_plot_2d_simple(num_agents: int,
                                           domain_bounds: np.ndarray,
                                           reach_num_points: int,
                                           final_pos_matrix: torch.Tensor,
                                           title_ads: list,
                                           font: dict =  {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                           ) -> matplotlib.figure.Figure:
    """
    Plots the bifurcation of agents' final positions for different parameter values.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param domain_bounds: Bounds of the domain.
    :type domain_bounds: np.ndarray
    :param reach_num_points: Number of points in the reach.
    :type reach_num_points: int
    :param final_pos_matrix: Tensor containing final positions of agents.
    :type final_pos_matrix: torch.Tensor
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list
    :returns: The generated plot figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    c=range(reach_num_points)
    for agent_id in range(num_agents): 
        ax.scatter(final_pos_matrix[:,agent_id][:,0],final_pos_matrix[:,agent_id][:,1],c=c, cmap='rainbow')
        #lines=colored_line(final_pos_matrix[:,agent_id][:[:,0].numpy(),final_pos_matrix[:,agent_id][:,1].numpy(), sigmas, ax, linewidth=1, cmap="plasma")
    #fig.colorbar(lines)  # add a color legend
    ax.set_xlim(domain_bounds[0][0], domain_bounds[0][1])
    ax.set_ylim(domain_bounds[1][0], domain_bounds[1][1])
    title="Bifurcation of " +str(num_agents)+r" agents for different $\sigma$ values"
    if len(title_ads)>0:
        for title_ads in title_ads:
            title=title+" "+title_ads
    ax.set_title(title,fontsize=title_font_size)
    ax.set_xlabel('Strat Comp 1')
    ax.set_ylabel('Strat Comp 2')

    plt.close()
    return fig

## Incomplete 

# def vector_plot_2d():
#     fig,ax = plt.subplots()
#     ax.set_box_aspect(1)
#     Y, X = self.rect_Y,self.rect_X
#     U,V = self.direction[:,0].reshape((10,10)),self.direction[:,1].reshape((10,10))
#     strm = ax.streamplot(X, Y, U, V, **kwargs)
