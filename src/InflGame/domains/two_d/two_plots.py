
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

from typing import Optional, Tuple
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

    # Convert bin_points to numpy array
    if torch.is_tensor(bin_points):
        bin_points = bin_points.cpu().numpy()
    elif not isinstance(bin_points, np.ndarray):
        bin_points = np.array(bin_points)
    if torch.is_tensor(pos_matrix):
        pos_matrix = pos_matrix.cpu().numpy()
    elif not isinstance(pos_matrix, np.ndarray):
        pos_matrix = np.array(pos_matrix)
    if torch.is_tensor(infl_dist):
        infl_dist = infl_dist.cpu().numpy()
    elif not isinstance(infl_dist, np.ndarray):
        infl_dist = np.array(infl_dist)
    if torch.is_tensor(resources):
        resources = resources.cpu().numpy()
    elif not isinstance(resources, np.ndarray):
        resources = np.array(resources)
    x_coords = bin_points[:, 0]
    y_coords = bin_points[:, 1]

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
        y = new_coor
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
        pvals = infl_dist[a_id].cpu().numpy() if torch.is_tensor(infl_dist[a_id]) else infl_dist[a_id]
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


def pos_plot_2d(num_agents: int,
                pos_matrix: torch.Tensor,
                domain_bounds: np.ndarray,
                title_ads: Optional[list] = [],
                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                axis_return: Optional[bool] = False,
                line_thickness: float = 2,
                marker_size: float = 8,
                black:bool=False,
                fig_size:Tuple=(18, 18) 
                ) -> matplotlib.figure.Figure:
    """
    Plot agent position trajectories over time in a 2D domain.
    
    Creates a plot showing how agent positions change over gradient ascent iterations
    in a 2D space. Each agent's trajectory is plotted as a separate line with a distinct
    color. Start positions are marked with open circles and end positions with filled circles.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param pos_matrix: Matrix of agent positions over time (shape: [time_steps, num_agents, 2]).
    :type pos_matrix: torch.Tensor
    :param domain_bounds: Bounds of the 2D domain as [[x_min, x_max], [y_min, y_max]].
    :type domain_bounds: np.ndarray
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[list]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: Optional[bool]
    :param line_thickness: Thickness of trajectory lines.
    :type line_thickness: float
    :param marker_size: Size of start/end markers.
    :type marker_size: float
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    # Convert to numpy if tensor
    if torch.is_tensor(pos_matrix):
        pos_matrix = pos_matrix.cpu().numpy()
    
    # Convert domain_bounds to numpy if tensor (for GPU compatibility)
    if torch.is_tensor(domain_bounds):
        domain_bounds = domain_bounds.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_box_aspect(1)
    if num_agents>10:
        black=True
    # Generate colors for agents
    if black:
        #grey for all 
        colors = ['black']*num_agents
        alph=.1
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
        alph=1
    
    for a_id in range(num_agents):
        x1 = pos_matrix[:, a_id, 0]  # First coordinate
        x2 = pos_matrix[:, a_id, 1]  # Second coordinate
        
        # Plot trajectory
        ax.plot(x1, x2, color=colors[a_id], label=f'Agent {a_id + 1}', linewidth=line_thickness,alpha=alph)
        # Start position (open circle)
        ax.plot(x1[0], x2[0], 'o', color=colors[a_id],alpha=alph, mfc='none', markersize=marker_size)
        # End position (filled circle)
        ax.plot(x1[-1], x2[-1], 'o', color=colors[a_id],alpha=alph, markersize=marker_size)
    
    ax.set_xlabel(r'$x_1$', fontsize=default_font_size)
    ax.set_ylabel(r'$x_2$', fontsize=default_font_size)
    ax.set_xlim(domain_bounds[0, 0], domain_bounds[0, 1])
    ax.set_ylim(domain_bounds[1, 0], domain_bounds[1, 1])
    
    # Horizontal legend
    ncols = min(num_agents, 8)
    if num_agents<=10:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=ncols, fontsize=legend_font_size)
    
    title = "Agent Position Trajectories"
    if len(title_ads) > 0:
        for item in title_ads:
            title += " " + item
    ax.set_title(title, fontsize=title_font_size)
    
    plt.tight_layout()
    plt.close()
    
    if axis_return:
        return ax
    else:
        return fig


def dist_plot_2d(agent_id: int,
                 infl_dist: torch.Tensor,
                 rect_Y: np.ndarray,
                 rect_X: np.ndarray,
                 font: dict) -> matplotlib.figure.Figure:
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
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    fig, ax = plt.subplots()
    # Convert to numpy for matplotlib (handle GPU tensors)
    pval = infl_dist[agent_id].cpu().numpy() if torch.is_tensor(infl_dist[agent_id]) else infl_dist[agent_id]
    pval = pval.reshape(len(rect_Y), len(rect_X))
    im = ax.pcolormesh(rect_X, rect_Y, pval)

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


def agent_density_3d_2d(
    pos_matrix: np.ndarray,
    num_agents: int,
    domain_bounds: np.ndarray,
    bins: int = 25,
    distance_threshold: float = 0.05,
    cmap: str = 'viridis',
    font: dict = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
    figsize: Tuple = (24, 20),
    xlabel: str = r'$x_1$',
    ylabel: str = r'$x_2$',
    zlabel: str = 'Number of Agents',
    axis_return: bool = False,
    edgecolor: str = 'black',
    linewidth: float = 0.2,
    alpha: float = 0.9,
    title_ads: list = [],
    save: bool = False,
    name_ads: list = [],
    save_types: list = ['.png', '.svg'],
    paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'agent_density_3d'},
    id: int = 0,
    cap_z_axis: bool = True,
    integer_ticks: bool = True
) -> matplotlib.figure.Figure:
    """
    Create a 3D histogram showing agent density at final positions for 2D rectangular domain.
    
    :param pos_matrix: Position matrix of shape (time_steps, num_agents, 2).
    :type pos_matrix: np.ndarray or torch.Tensor
    :param num_agents: Number of agents.
    :type num_agents: int
    :param domain_bounds: Domain bounds of shape (2, 2) as [[x_min, x_max], [y_min, y_max]].
    :type domain_bounds: np.ndarray
    :param bins: Number of bins in each dimension.
    :type bins: int
    :param distance_threshold: Distance threshold for clustering nearby agents.
    :type distance_threshold: float
    :param cmap: Colormap name.
    :type cmap: str
    :param font: Font configuration dictionary.
    :type font: dict
    :param figsize: Figure size as (width, height).
    :type figsize: tuple
    :param xlabel: Label for x-axis.
    :type xlabel: str
    :param ylabel: Label for y-axis.
    :type ylabel: str
    :param zlabel: Label for z-axis.
    :type zlabel: str
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: bool
    :param edgecolor: Color of outlines around bars.
    :type edgecolor: str
    :param linewidth: Width of bar edge lines.
    :type linewidth: float
    :param alpha: Bar transparency.
    :type alpha: float
    :param title_ads: Additional titles for the plot.
    :type title_ads: list
    :param save: Whether to save the plot.
    :type save: bool
    :param name_ads: Additional names for saved files.
    :type name_ads: list
    :param save_types: File types to save the plot.
    :type save_types: list
    :param paper_figure: Dictionary for paper figure naming.
    :type paper_figure: dict
    :param id: Identifier for file naming.
    :type id: int
    :param cap_z_axis: If True, cap the z-axis maximum at num_agents.
    :type cap_z_axis: bool
    :param integer_ticks: If True, only show integer ticks on the z-axis.
    :type integer_ticks: bool
    :return: The generated plot figure.
    :rtype: matplotlib.figure.Figure
    """
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from InflGame.utils import data_management
    
    title = f'Agent Density'
    if title_ads:
        title = title + ' - ' + ' - '.join(title_ads)
    
    if torch.is_tensor(pos_matrix):
        pos_matrix = pos_matrix.cpu().numpy()
    
    if torch.is_tensor(domain_bounds):
        domain_bounds = domain_bounds.cpu().numpy()

    if pos_matrix.ndim == 3:
        final_positions = pos_matrix[-1, :, :]
    else:
        final_positions = pos_matrix

    # Cluster nearby agents
    if len(final_positions) > 1:
        distances = pdist(final_positions)
        if len(distances) > 0:
            Z = linkage(distances, method='complete')
            clusters = fcluster(Z, t=distance_threshold, criterion='distance')
            
            unique_clusters = np.unique(clusters)
            centers = []
            counts = []
            
            for c in unique_clusters:
                mask = clusters == c
                cluster_positions = final_positions[mask]
                centers.append(cluster_positions.mean(axis=0))
                counts.append(mask.sum())
            
            cluster_centers = np.array(centers)
            cluster_counts = np.array(counts)
        else:
            cluster_centers = final_positions
            cluster_counts = np.ones(len(final_positions))
    else:
        cluster_centers = final_positions
        cluster_counts = np.ones(len(final_positions))

    x1_final = cluster_centers[:, 0]
    x2_final = cluster_centers[:, 1]

    h, xedges, yedges = np.histogram2d(
        x1_final, x2_final,
        bins=bins,
        range=[[domain_bounds[0, 0], domain_bounds[0, 1]],
               [domain_bounds[1, 0], domain_bounds[1, 1]]],
        weights=cluster_counts
    )

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xpos, ypos = np.meshgrid(xcenters, ycenters, indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = (xedges[1] - xedges[0]) * np.ones_like(xpos)
    dy = (yedges[1] - yedges[0]) * np.ones_like(ypos)
    dz = h.ravel()

    nonzero = dz > 0
    xpos, ypos, zpos = xpos[nonzero], ypos[nonzero], zpos[nonzero]
    dx, dy, dz = dx[nonzero], dy[nonzero], dz[nonzero]

    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    norm = Normalize(vmin=dz.min() if dz.size else 0, vmax=dz.max() if dz.size else 1)
    colors_arr = plt.get_cmap(cmap)(norm(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_arr, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, shade=True)

    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_xlim(domain_bounds[0, 0], domain_bounds[0, 1])
    ax.set_ylim(domain_bounds[1, 0], domain_bounds[1, 1])
    ax.set_ylabel(ylabel, labelpad=15)
    ax.set_zlabel(zlabel, labelpad=15)
    
    if cap_z_axis:
        ax.set_zlim(0, num_agents)
    if integer_ticks:
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_title(title, fontsize=title_font_size)

    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot', 'plot_type': 'agent_density_3d', 'section': paper_figure['section'], 
             'figure_id': paper_figure.get('figure_id', 'agent_density_3d'), "num_agents": num_agents, 'domain_type': '2d'},
            name_ads=name_ads + [f'id_{id}'] if id else name_ads,
            save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            if file_name.lower().endswith('.svg'):
                plt.rcParams['svg.fonttype'] = 'none'
            fig.savefig(file_name, dpi=600 if file_name.lower().endswith('.svg') else 300, bbox_inches='tight')
    
    return ax if axis_return else fig
