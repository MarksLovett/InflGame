"""
.. module:: one_plots
   :synopsis: Provides 1D visualization tools for analyzing agent dynamics and resource distributions in influencer games.

1D Visualization Module
========================

This module provides visualization tools for analyzing and understanding the dynamics of agents 
and resource distributions in 1D domains for influencer games. It includes utilities for plotting 
agent positions, gradients, influence distributions, and bifurcation dynamics in 1D domains.

The module is designed to work with the :mod:`InflGame.adaptive` package and supports creating 
visual representations of agent behaviors and resource distributions in 1D environments.

Dependencies:
-------------
- matplotlib
- NumPy
- PyTorch
- InflGame.utils
- InflGame.domains

Usage:
------
The functions in this module can be used to visualize agent dynamics and resource distributions 
in 1D domains. For example, the :func:`pos_plot_1d` function can be used to plot agent positions 
over time, while the :func:`dist_and_pos_plot_1d` function can visualize both agent positions 
and influence distributions.

Example:
--------

.. code-block:: python

    from InflGame.domains.one_d.one_plots import pos_plot_1d, equilibrium_bifurcation_plot_1d
    import torch
    import numpy as np
    
    # Plot agent positions over time
    fig = pos_plot_1d(
        num_agents=3,
        pos_matrix=torch.randn(100, 3),
        domain_bounds=(0, 1),
        title_ads=['Example Plot']
    )
    fig.show()
"""

from InflGame.utils import data_management
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict, Optional
import matplotlib.figure
from matplotlib import ticker
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Wedge, Rectangle
from matplotlib.collections import PolyCollection, PathCollection

import InflGame.utils.general as general
import InflGame.domains.one_d.one_utils as one_utils

def pos_plot_1d(num_agents: int,
                pos_matrix: torch.Tensor,
                domain_bounds: Tuple[float, float],
                title_ads: Optional[List[str]] = [],
                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                axis_return: Optional[bool] = False,
                line_thickness: float = 2,
                fig_size:Tuple = (18, 18)
                ) -> matplotlib.figure.Figure:
    """
    Plot agent positions over time in a 1D domain.
    
    Creates a line plot showing how agent positions change over gradient ascent iterations.
    Each agent's trajectory is plotted as a separate line with a distinct color.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param pos_matrix: Matrix of agent positions over time (shape: [time_steps, num_agents]).
    :type pos_matrix: torch.Tensor
    :param domain_bounds: Minimum and maximum bounds of the 1D domain.
    :type domain_bounds: Tuple[float, float]
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: Optional[bool]
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    num_points=len(pos_matrix)
    domain=np.linspace(0,num_points,num_points)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_box_aspect(1)
    for a_id in range(num_agents):
        ax.plot(domain,pos_matrix[:,a_id].cpu().numpy(),label='Agent '+ str(a_id+1),linewidth=line_thickness)
    #ax.axhline(y=self.mean,color='r', linestyle='--',label='Mean')
    ax.set_xlabel('Steps',fontsize=default_font_size)
    ax.set_ylabel('Influencer location',fontsize=default_font_size)
    plt.xlim(0,num_points)
    plt.ylim(domain_bounds[0],domain_bounds[1])
    plt.legend()
    title="Adaptive Agents' Trajectories"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title, fontsize=title_font_size)
    plt.close()
    if axis_return:
        return ax
    else:
        return fig

def gradient_plot_1d(num_agents: int,
                     grad_matrix: torch.Tensor,
                     title_ads: Optional[List[str]] = [],
                     font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}
                     ) -> matplotlib.figure.Figure:
    """
    Plot agent gradients over time in a 1D domain.
    
    Creates a line plot showing how the gradient values for each agent change over gradient
    ascent iterations. Each agent's gradient trajectory is plotted as a separate line.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param grad_matrix: Matrix of agent gradients over time (shape: [time_steps, num_agents]).
    :type grad_matrix: torch.Tensor
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    
    :return: The generated matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    num_points=len(grad_matrix)
    domain=np.linspace(0,num_points,num_points)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    for a_id in range(num_agents):
        ax.plot(domain,grad_matrix[:,a_id],label='Player '+ str(a_id))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Player gradient')
    plt.xlim(0,num_points)
    plt.legend()
    title="Player Gradients"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title,fontsize=title_font_size)
    plt.close()
    return fig


def resource_distribution_plot_1d(bin_points: np.ndarray,
                                   resource_distribution: np.ndarray,
                                   alpha: float = None,
                                   show_alpha_line: bool = True,
                                   title: str = 'Resource distribution',
                                   fig_size: Tuple = (12, 8),
                                   line_width: float = 2,
                                   font: dict = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'alpha_size': 20, 'font_family': 'sans-serif'},
                                   y_padding: float = 1.25,
                                   save: bool = False,
                                   name_ads: List[str] = [],
                                   save_types: List[str] = ['.png', '.svg'],
                                   paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'resource_dist'}
                                   ) -> matplotlib.figure.Figure:
    r"""
    Plot the resource distribution with optional alpha line annotation for bimodal distributions.
    
    For bimodal Gaussian distributions, this function draws a dashed line between the two peaks 
    and labels it with :math:`\alpha`, representing the separation distance between peaks.
    The peak positions are calculated as :math:`0.5 - \alpha/2` and :math:`0.5 + \alpha/2`.
    
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource density values at each bin point.
    :type resource_distribution: np.ndarray
    :param alpha: The separation parameter for bimodal distributions. If provided, a dashed line 
                  will be drawn between the peaks at positions (0.5 - alpha/2) and (0.5 + alpha/2).
    :type alpha: float, optional
    :param show_alpha_line: Whether to show the alpha annotation line between peaks.
    :type show_alpha_line: bool
    :param title: Title for the plot.
    :type title: str
    :param fig_size: Figure size as (width, height).
    :type fig_size: Tuple
    :param line_width: Width of the distribution line.
    :type line_width: float
    :param font: Font configuration dictionary with keys: 'default_size', 'title_size', 'alpha_size', 'font_family'.
    :type font: dict
    :param y_padding: Multiplier for y-axis upper limit to add space for labels.
    :type y_padding: float
    :param save: Whether to save the plot.
    :type save: bool
    :param name_ads: Additional names for saved files.
    :type name_ads: List[str]
    :param save_types: File types to save the plot.
    :type save_types: List[str]
    :param paper_figure: Configuration for paper figure saving.
    :type paper_figure: dict
    
    :return: The generated matplotlib figure.
    :rtype: matplotlib.figure.Figure
    
    Example:
    --------
    .. code-block:: python
    
        import numpy as np
        from InflGame.domains.one_d.one_plots import resource_distribution_plot_1d
        import InflGame.domains.rd as rd
        
        bin_points = np.linspace(0.001, 0.999, 100)
        alpha = 0.5
        resource_params = [[0.1, 0.1], [0.5 - alpha/2, 0.5 + alpha/2], [1, 1]]
        resource_dist = rd.resource_distribution_choice(
            bin_points=bin_points,
            resource_type='multi_modal_gaussian_distribution_1D',
            resource_parameters=resource_params
        )
        
        fig = resource_distribution_plot_1d(bin_points, resource_dist, alpha=alpha, save=True)
        fig.show()
    """
    # Convert to numpy if needed
    bin_points_np = bin_points.cpu().numpy() if torch.is_tensor(bin_points) else np.array(bin_points)
    resource_dist_np = resource_distribution.cpu().numpy() if torch.is_tensor(resource_distribution) else np.array(resource_distribution)
    
    # Set font properties
    font_family = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 15)
    title_font_size = font.get('title_size', 18)
    alpha_fontsize = font.get('alpha_size', 20)
    
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font_family})
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the distribution
    ax.plot(bin_points_np, resource_dist_np, linewidth=line_width)
    
    # Draw alpha line if requested and alpha is provided
    if show_alpha_line and alpha is not None:
        # Calculate peak positions
        peak1_pos = 0.5 - alpha / 2
        peak2_pos = 0.5 + alpha / 2
        
        # Find the indices closest to peak positions
        peak1_idx = np.argmin(np.abs(bin_points_np - peak1_pos))
        peak2_idx = np.argmin(np.abs(bin_points_np - peak2_pos))
        
        # Get peak heights from the resource distribution data
        peak1_height = resource_dist_np[peak1_idx]
        peak2_height = resource_dist_np[peak2_idx]
        
        # Draw dashed line between the two peaks at their actual heights
        ax.plot([bin_points_np[peak1_idx], bin_points_np[peak2_idx]], 
                [peak1_height, peak2_height], 
                linestyle='--', color='black', linewidth=1.5)
        
        # Add alpha label above the dashed line (centered)
        mid_x = (bin_points_np[peak1_idx] + bin_points_np[peak2_idx]) / 2
        mid_y = (peak1_height + peak2_height) / 2
        ax.text(mid_x, mid_y * 1.05, r'$\alpha$', 
                fontsize=alpha_fontsize, ha='center', va='bottom')
        
        # Adjust y-axis limits to add space for the title
        ax.set_ylim(bottom=0, top=max(resource_dist_np) * y_padding)
    
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel('loc')
    ax.set_ylabel('Resource density')
    ax.set_box_aspect(1)
    
    # Save logic
    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot', 'plot_type': 'resource_distribution', 'domain_type': '1d','num_agents': '0',
             'section': paper_figure['section'], 
             'figure_id': paper_figure.get('figure_id', 'resource_dist')},
            name_ads=name_ads, save_types=save_types, paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')
    
    return fig


def prob_plot_1d(num_agents: int,
                 agents_pos: List[float],
                 bin_points: np.ndarray,
                 domain_bounds: List[float],
                 prob: List[np.ndarray],
                 voting_configs: Dict[str, bool],
                 title_ads: Optional[List[str]],
                 font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}
                 ) -> matplotlib.figure.Figure:
    r"""
    Plot probability distribution of agent influence in a 1D domain.
    
    Visualizes the probability that each agent influences each bin/resource point via their
    relative influence. The probability is computed as:
    
    .. math::
        G_{i,k}(\mathbf{x},b_k) = \frac{f_i(x_i, b_k)}{\sum_{j=1}^N f_j(x_j, b_k)}
    
    where :math:`f_i(x_i, b_k)` is the influence of agent :math:`i` at bin point :math:`b_k`.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param agents_pos: Current positions of all agents.
    :type agents_pos: List[float]
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param domain_bounds: Minimum and maximum bounds of the 1D domain.
    :type domain_bounds: List[float]
    :param prob: Probability distributions for each agent (one array per agent).
    :type prob: List[np.ndarray]
    :param voting_configs: Configuration dictionary with keys 'fixed_party' and 'abstain' for voting behavior.
    :type voting_configs: Dict[str, bool]
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    
    :return: The generated matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig, ax  = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    for agent_id in range(num_agents):
        ax.plot(bin_points,prob[agent_id],label=f'Agent {agent_id+1}')
        ax.scatter(agents_pos[agent_id],0)


    if voting_configs['fixed_party']==True:
        ax.plot(bin_points,prob[num_agents],label=f'Fixed Party')
        if voting_configs['abstain']==True:
            ax.plot(bin_points,prob[num_agents+1],label=f'Abstaining')
    elif voting_configs['abstain']==True:
        ax.plot(bin_points,prob[num_agents],label=f'Abstaining')
    
        
    
    plt.legend()
    plt.xlim(domain_bounds[0],domain_bounds[1])
    plt.ylabel('Probability')
    plt.xlabel('Resource position')
    title="Agent probability of influence"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title, fontsize=title_font_size)
    plt.close()
    return fig

def three_agent_dynamics(pos_matrix: np.ndarray,
                          x_star: float,
                          title_ads: List[str],
                          font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                          axis_return: Optional[bool] = False
                          ) -> matplotlib.figure.Figure:
    """
    Visualize three-agent dynamics in 3D trajectory space.
    
    Creates a 3D plot where each axis represents one agent's position over time, demonstrating
    the instability and complex dynamics of three-player influencer games in 1D domains.
    Only applicable for exactly 3 agents with 1D strategy spaces.

    :param pos_matrix: Matrix of agent positions over time (shape: [time_steps, 3]).
    :type pos_matrix: np.ndarray
    :param x_star: Equilibrium or reference position (e.g., symmetric Nash equilibrium).
    :type x_star: float
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: List[str]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: Optional[bool]
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    new_pos=pos_matrix.T
    
    x=new_pos[0,:]
    y=new_pos[1,:]
    z=new_pos[2,:]

    mpl.rcParams['legend.fontsize'] = 10

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})

    ax.plot(x, y, z, label='Agents\' path')
    ax.scatter(x[0],y[0],z[0],label='initial position')
    ax.scatter(x_star,x_star,x_star,label='mean')
    ax.set_zlim(0,1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.legend()
    ax.set_box_aspect((1,1,1))
    plt.title("Agent 3d Positions", fontsize=title_font_size)
    plt.close()
    if axis_return== False:
        return fig
    else:
        return ax

def vector_plot_1d(ids: List[int],
                   gradient: torch.Tensor,
                   title_ads: Optional[List[str]],
                   font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                   **kwargs
                   ) -> matplotlib.figure.Figure:
    """
    Plot vector field of gradients for two agents in a 1D domain.
    
    Creates a streamplot showing the gradient vector field for a two-agent system, where
    each axis represents one agent's position. The vectors indicate the direction and
    magnitude of gradient ascent at each point in the position space.

    :param ids: Agent IDs to include in the vector field (must be exactly 2 agents).
    :type ids: List[int]
    :param gradient: Gradient matrix for the vector field (shape: [grid_points, 2]).
    :type gradient: torch.Tensor
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    :param **kwargs: Additional keyword arguments passed to matplotlib streamplot function.
    
    :return: The generated matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    
    # Convert to torch tensor if needed
    if torch.is_tensor(gradient):
        gradient_torch = gradient
    else:
        gradient_torch = torch.tensor(gradient, dtype=torch.float32)
    
    # Create coordinate grid using torch to match np.mgrid behavior
    y_coords = torch.linspace(0, 1, 100)
    x_coords = torch.linspace(0, 1, 100)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')  # This matches np.mgrid order
    
    # Extract U and V components using torch operations
    U = gradient_torch[:, 0].reshape(100, 100)
    V = gradient_torch[:, 1].reshape(100, 100)
    
    # Convert to numpy only for matplotlib (matplotlib requires numpy arrays)
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    U_np = U.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    # Filter out any 'kwargs' key if it exists (this can happen with nested kwargs)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'kwargs'}
    ax.streamplot(X_np, Y_np, U_np, V_np, **filtered_kwargs)
    #plot a x=y line 
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='sym pos')
    plt.xlabel(f"Agent {ids[0]+1}'s position")
    plt.ylabel(f"Agent {ids[1]+1}'s position")
    plt.legend()
    title=f"Agent {ids[0]+1} and {ids[1]+1}'s vector field"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title, fontsize=title_font_size)
    plt.close()
    return fig

def dist_and_pos_plot_1d(num_agents: int,
                         bin_points: np.ndarray,
                         resource_distribution: np.ndarray,
                         pos_matrix: torch.Tensor,
                         len_grad_matrix: int,
                         infl_dist: List[torch.Tensor],
                         cm: mpl.colors.Colormap,
                         NUM_COLORS: int,
                         title_ads: Optional[List[str]],
                         font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}
                         ) -> matplotlib.figure.Figure:
    """
    Plot agent influence distributions and positions over time in a 1D domain.
    
    Creates a side-by-side visualization with:
    
    - Left panel: Influence distributions for each agent overlaid with resource distribution
    - Right panel: Agent positions changing over gradient ascent iterations
    
    This provides comprehensive insight into both the spatial influence patterns and
    the temporal evolution of agent positions.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource values at each bin point.
    :type resource_distribution: np.ndarray
    :param pos_matrix: Matrix of agent positions over time (shape: [time_steps, num_agents]).
    :type pos_matrix: torch.Tensor
    :param len_grad_matrix: Number of time steps for x-axis scaling.
    :type len_grad_matrix: int
    :param infl_dist: Influence distribution arrays for each agent.
    :type infl_dist: List[torch.Tensor]
    :param cm: Matplotlib colormap for agent colors.
    :type cm: mpl.colors.Colormap
    :param NUM_COLORS: Total number of colors in the colormap (typically num_agents + 1).
    :type NUM_COLORS: int
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    
    :return: The generated matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    
    
    fig = plt.figure(figsize=(19, 7))
    num_points=len(pos_matrix)
    gs = GridSpec(nrows=num_agents, ncols=2,width_ratios=[1, 1],wspace=0.2, hspace=0.2, top=1, bottom=0.05, left=0.17, right=0.845)
    domain=np.linspace(0,num_points,num_points)
    ax0 = fig.add_subplot(gs[:, 1])
    for a_id in range(num_agents):
        ax0.scatter(0,pos_matrix[:,a_id][0],s=70,color=cm(1.*a_id/NUM_COLORS),linewidth=0.3,label='Player '+str(a_id+1))
        ax0.scatter(len(pos_matrix),pos_matrix[:,a_id][-1],s=70,facecolors='none',edgecolors=cm(1.*a_id/NUM_COLORS),linewidth=1)
        ax0.plot(domain,pos_matrix[:,a_id].cpu().numpy(),color=cm(1.*a_id/NUM_COLORS))
        ax0.set_xlim(xmin=0,xmax=len_grad_matrix)
    #ax0.axhline(y=self.mean,color='r', linestyle='--',label='Mean')
    ax0.set_xlabel('Steps')
    ax0.set_ylabel('Agent location')
    if num_agents<=10:
        plt.legend()
    plt.xlim(0,len_grad_matrix)
    plt.ylim(0,1)
    plt.title('Agents positions in time',y=1)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(bin_points,resource_distribution,color=cm(1.*(a_id+1)/NUM_COLORS),label='Resource distribution')
    for agent_id in range(num_agents):
        ax1.plot(bin_points,infl_dist[agent_id].cpu().numpy(),color=cm(1.*agent_id/NUM_COLORS),label='Player '+str(agent_id))
    if num_agents<=10:
        plt.legend()
    plt.xlabel('pos')
    plt.ylabel('influence')
    title='Players\' influence distribution'
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title, fontsize=title_font_size)
    plt.close()
    return fig

def equilibrium_bifurcation_plot_1d(num_agents: int,
                                    bin_points: np.ndarray,
                                    resource_distribution: np.ndarray,
                                    infl_type: str,
                                    reach_parameters: List[float],
                                    final_pos_matrix: np.ndarray,
                                    reach_start: float,
                                    reach_end: float,
                                    refinements: int,
                                    plot_type: str,
                                    title_ads: Optional[List[str]],
                                    short_title: bool = False,
                                    norm:bool = True,
                                    infl_cshift: bool = False,
                                    cmaps: dict = {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys'},
                                    font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                    cbar_config: dict = {'center_labels': True, 'label_alignment': 'center', 'shrink': 0.8},
                                    axis_return: bool = False,
                                    show_pred: bool = False,
                                    optional_vline: List[float] = None
                                    ) -> matplotlib.figure.Figure:
    r"""
    Plot equilibrium bifurcation diagram for agents in a 1D domain.
    
    Visualizes how equilibrium positions change as a function of the reach parameter
    (e.g., variance :math:`\sigma` for Gaussian influence kernels). As :math:`\sigma`
    decreases, agents bifurcate from symmetric positions to asymmetric equilibria.
    
    Each agent has a vector of final positions :math:`X_i = [x_1, x_2, \dots, x_A]`
    where :math:`A` is the number of test parameters and :math:`x_i` is the equilibrium
    position at parameter value :math:`i`.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource values at each bin point.
    :type resource_distribution: np.ndarray
    :param infl_type: Type of influence kernel ('gaussian', 'beta', 'multi_gaussian', etc.).
    :type infl_type: str
    :param reach_parameters: Array of reach parameter values to test.
    :type reach_parameters: List[float]
    :param final_pos_matrix: Matrix of final equilibrium positions (shape: [num_params, num_agents]).
    :type final_pos_matrix: np.ndarray
    :param reach_start: Starting value of reach parameter range.
    :type reach_start: float
    :param reach_end: Ending value of reach parameter range.
    :type reach_end: float
    :param refinements: Number of refinements for critical value estimation.
    :type refinements: int
    :param plot_type: Type of plot ('line' or 'heat').
    :type plot_type: str
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param short_title: Use abbreviated title format.
    :type short_title: bool
    :param norm: Normalize heatmap values.
    :type norm: bool
    :param infl_cshift: Whether influence uses center shift.
    :type infl_cshift: bool
    :param cmaps: Color map configuration dictionary with keys 'heat', 'trajectory', 'crit'.
    :type cmaps: dict
    :param font: Font configuration dictionary.
    :type font: dict
    :param cbar_config: Colorbar configuration dictionary.
    :type cbar_config: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: bool
    :param show_pred: Show predicted critical values (only for Gaussian kernels).
    :type show_pred: bool
    :param optional_vline: Optional vertical lines to add to plot.
    :type optional_vline: Optional[List[float]]
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    
    crit_cmap = cmaps.get('crit', 'Greys')
    trajectory_cmap = cmaps.get('trajectory', '#851321')
    heat_cmap = cmaps.get('heat', 'Blues')
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    cbar_center_labels = cbar_config.get('center_labels', True)
    cbar_label_alignment = cbar_config.get('label_alignment', 'center')
    cbar_shrink = cbar_config.get('shrink', 1)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    if plot_type == "line":
        for agent_id in range(num_agents):
            ax.plot(reach_parameters, final_pos_matrix[:, agent_id])
    elif plot_type == "heat":
        # For heatmap, we need to create a density matrix
        # Extract position data
        if isinstance(final_pos_matrix, tuple):
            # If it's (figure, positions)
            pos_data = final_pos_matrix[1] if len(final_pos_matrix) > 1 else final_pos_matrix[0]
        else:
            pos_data = final_pos_matrix
        
        # Convert to numpy if needed
        if hasattr(pos_data, 'numpy'):
            positions = pos_data.cpu().numpy()
        elif isinstance(pos_data, (list, tuple)):
            positions = np.array(pos_data)
        else:
            positions = pos_data
        
        # Create parameter range
        reach_parameters = np.linspace(reach_start, reach_end, len(reach_parameters))
        
        
        
        if positions.ndim == 2:
            # 2D array - parameters x agents
            min_pos = max(0, positions.min() - 0.5)
            max_pos = min(1, positions.max() + 0.5)
            position_bins = np.linspace(min_pos, max_pos, 100)
            
            # Create density matrix
            density_matrix = np.zeros((len(position_bins)-1, min(positions.shape[0], len(reach_parameters))))
            
            # Count agents in each position bin for each parameter
            for i in range(min(positions.shape[0], len(reach_parameters))):
                agent_positions = positions[i, :]
                # Remove NaN values
                valid_positions = agent_positions[~np.isnan(agent_positions)]
                if len(valid_positions) > 0:
                    counts, _ = np.histogram(valid_positions, bins=position_bins)
                    density_matrix[:, i] = counts
            
            # Adjust reach_parameters to match data
            reach_parameters = reach_parameters[:density_matrix.shape[1]]
        
        
        # Create discrete colormap for agent count
        max_agents = int(density_matrix.max())
        if max_agents > 0:
            # Define discrete levels based on agent count
            levels = np.arange(0, max_agents + 2, 1)  # 0, 1, 2, ..., max_agents+1
            norm = mpl.colors.BoundaryNorm(levels, ncolors=256)
            
            # Create the heatmap with discrete colormap
            im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, norm=norm, origin='lower',
                        extent=[reach_parameters[0], reach_parameters[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')  # Use 'nearest' for discrete appearance
        else:
            # Fallback for empty data
            im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, origin='lower',
                        extent=[reach_parameters[0], reach_parameters[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')
        
        
                
    
    #Bifurcations critical values (works for gaussian only)
    if infl_type=='gaussian' and show_pred==True:
        _,means,crit_stds=one_utils.critical_values_plot(num_agents=num_agents,bin_points=bin_points,resource_distribution=resource_distribution,axis=ax,reach_start=reach_start,reach_end=reach_end,refinements=refinements,crit_cs=crit_cmap)
        crit_stds=general.flatten_list(xss=crit_stds)
        crit_stds.sort()
        for std in crit_stds:
            if std < reach_start or std > reach_end:
                #remove the std from crit_stds if it is outside the reach range
                crit_stds.remove(std)
            
        std_ticks = [float(np.around(i,decimals=2)) for i in np.linspace(reach_end, reach_start, num=5)]
        crit_means=np.around(means,decimals=3)
        mean_ticks= [0,1]+list(crit_means)
        mean_ticks.sort()

        std_removed=np.setdiff1d(np.array(std_ticks),np.around(crit_stds,2))
        std_tick_vals=np.array(list(std_removed)+crit_stds)
        std_tick_vals.sort()
        crit_std_locs=[]
        for std_id in range(len(crit_stds)):
            crit_std_locs.append(int(np.where(std_tick_vals==crit_stds[std_id])[0][0]))
        std_tick_labels=list(std_tick_vals.copy())
        for std_loc_id in range(len(crit_stds)):
            if std_loc_id ==len(crit_stds)-1:
                std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_*$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]]) 
            else:
                std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_'+str(len(crit_stds)-std_loc_id-1)+r'$' #+r'='+str(std_vals[crit_std_locs[std_loc_id]])
        if infl_cshift==False:
            ax.xaxis.set_ticks(std_tick_vals)
            ax.xaxis.set_ticklabels(std_tick_labels)
            
        ax.yaxis.set_ticks(crit_means)
        ax.yaxis.set_ticklabels(crit_means)
    
    #Plot features
    if plot_type == "heat":
        # Create discrete colorbar for integer agent counts
        max_agents = int(density_matrix.max()) if 'density_matrix' in locals() else num_agents
        tick_levels = np.arange(0, max_agents + 2, 1)  # 0, 1, 2, ..., max_agents
        
        if axis_return == False:
            # Use the same norm as the image if it exists
            if 'norm' in locals():
                cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
            else:
                cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
            
            cbar.set_label('Number of Agents', fontsize=cbar_font_size)
            
            # Apply custom colorbar adjustments if requested
            if cbar_center_labels and len(tick_levels) > 1:
                # Calculate centered positions between tick marks
                centered_positions = []
                for i in range(len(tick_levels) - 1):
                    centered_positions.append((tick_levels[i] + tick_levels[i + 1]) / 2)
                
                # Set centered tick positions and labels
                if len(centered_positions) > 0:
                    cbar.set_ticks(centered_positions)
                    # Create labels for centered positions - each represents the actual agent count for that color band
                    # The first band (0-1) gets label "0", second band (1-2) gets label "1", etc.
                    centered_labels = [str(i) for i in range(len(centered_positions))]
                    cbar.set_ticklabels(centered_labels)
            else:
                # Keep original tick behavior when not centering
                cbar.set_ticks(tick_levels)
                cbar.set_ticklabels([str(int(level)) for level in tick_levels])
            
            # Apply label alignment if specified
            if cbar_label_alignment == 'center':
                cbar.ax.tick_params(axis='y', which='major', pad=5)
                for label in cbar.ax.get_yticklabels():
                    label.set_horizontalalignment('center')


    if optional_vline is not None:
        for vline_id in range(len(optional_vline)):
            ax.vlines(x=optional_vline[vline_id], ymin=0, ymax=1, colors='black', linestyles='dashed', label=r'$\sigma^*_' + str(vline_id + 1) + r'=$'+str(np.around(optional_vline[vline_id],decimals=4)))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='lower center')
    if short_title==True:
        title='Adaptive Agents'
    else:
        title=str(num_agents)+' Adaptive Agents\' Bifurcation of Equilibria'
    if len(title_ads)>0:
        for title_addition in title_ads:
            title=title+" "+title_addition 
    plt.title(title,fontsize=title_font_size)
    if infl_type=='gaussian':
        plt.xlabel(r"$\sigma$ (std)")
    else: 
        plt.xlabel(r"$\sigma$")
    plt.ylim(0,1)
    plt.ylabel("Agent Position")
    plt.tight_layout()
    plt.close()
    if axis_return:
        return ax
    else:
        return fig

def equilibrium_bifurcation_envelope_plot_1d(num_agents: int,
                                    bin_points: np.ndarray,
                                    resource_distribution: np.ndarray,
                                    infl_type: str,
                                    reach_parameters: List[float],
                                    extreme_positions: Dict[str, torch.Tensor],
                                    reach_start: float,
                                    reach_end: float,
                                    refinements: int,
                                    plot_type: str,
                                    title_ads: Optional[List[str]],
                                    short_title: bool = False,
                                    norm: bool = True,
                                    infl_cshift: bool = False,
                                    cmaps: dict = {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys', 'envelope': '#FF6B6B'},
                                    font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                    cbar_config: dict = {'center_labels': True, 'label_alignment': 'center', 'shrink': 0.8},
                                    axis_return: bool = False,
                                    show_pred: bool = False,
                                    optional_vline: List[float] = None,
                                    envelope_alpha: float = 0.3,
                                    show_bif_labels: bool = True,
                                    bifurcation_key_tolerance: int = 3
                                    ) -> matplotlib.figure.Figure:
    r"""
    Plot equilibrium bifurcation envelope showing extreme agent positions in a 1D domain.
    
    Creates an envelope plot showing the maximum and minimum positions achieved by agents
    across different reach parameters, providing insight into the spread and stability of
    equilibria. The envelope reveals the range of positions agents explore as the influence
    parameter varies.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource values at each bin point.
    :type resource_distribution: np.ndarray
    :param infl_type: Type of influence kernel ('gaussian', 'beta', 'multi_gaussian', etc.).
    :type infl_type: str
    :param reach_parameters: Array of reach parameter values to test.
    :type reach_parameters: List[float]
    :param extreme_positions: Dictionary with 'max' and 'min' keys containing extreme position tensors.
    :type extreme_positions: Dict[str, torch.Tensor]
    :param reach_start: Starting value of reach parameter range.
    :type reach_start: float
    :param reach_end: Ending value of reach parameter range.
    :type reach_end: float
    :param refinements: Number of refinements for critical value estimation.
    :type refinements: int
    :param plot_type: Type of plot ('line', 'envelope', or 'heat').
    :type plot_type: str
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param short_title: Use abbreviated title format.
    :type short_title: bool
    :param norm: Normalize heatmap values.
    :type norm: bool
    :param infl_cshift: Whether influence uses center shift.
    :type infl_cshift: bool
    :param cmaps: Color map configuration dictionary with keys 'heat', 'trajectory', 'crit', 'envelope'.
    :type cmaps: dict
    :param font: Font configuration dictionary.
    :type font: dict
    :param cbar_config: Colorbar configuration dictionary.
    :type cbar_config: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: bool
    :param show_pred: Show predicted critical values (only for Gaussian kernels).
    :type show_pred: bool
    :param optional_vline: Optional vertical lines to add to plot.
    :type optional_vline: Optional[List[float]]
    :param envelope_alpha: Transparency level for envelope fill (0-1).
    :type envelope_alpha: float
    :param show_bif_labels: Whether to show bifurcation labels on the plot.
    :type show_bif_labels: bool
    :param bifurcation_key_tolerance: Minimum key distance between bifurcations to include both. Bifurcations with keys closer than this tolerance to the previous one will be ignored.
    :type bifurcation_key_tolerance: int
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    
    crit_cmap = cmaps.get('crit', 'Greys')
    trajectory_cmap = cmaps.get('trajectory', '#851321')
    heat_cmap = cmaps.get('heat', 'Blues')
    envelope_cmap = cmaps.get('envelope', '#FF6B6B')
    
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size = font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    table_font_size = font.get('table_size', 10)
    rect_label_font_size = font.get('rect_label_size', 16)
    rect_sigma_font_size = font.get('rect_sigma_size', 12)
    cbar_center_labels = cbar_config.get('center_labels', True)
    cbar_label_alignment = cbar_config.get('label_alignment', 'center')
    cbar_shrink = cbar_config.get('shrink', 1)
    vline_id=0
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    # Create figure with GridSpec for main plot and table subplot
    # Use different layout based on show_bif_labels
    if show_bif_labels:
        fig = plt.figure(figsize=(28, 16))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
    else:
        # When show_bif_labels is False, use manual positioning for right column elements
        # to minimize whitespace and avoid overlap
        fig = plt.figure(figsize=(28, 16))
        # Create a simple 1x2 grid for the main plot, right column will use manual positioning
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)
    
    # Extract extreme positions following project patterns
    max_positions = extreme_positions['max']
    min_positions = extreme_positions['min']

    
    
    # Convert to numpy if needed following project patterns
    if hasattr(max_positions, 'numpy'):
        max_pos_np = max_positions.cpu().numpy()
    else:
        max_pos_np = np.array(max_positions)
        
    if hasattr(min_positions, 'numpy'):
        min_pos_np = min_positions.cpu().numpy()
    else:
        min_pos_np = np.array(min_positions)
    
    # Ensure reach_parameters is numpy array and properly 1D
    if hasattr(reach_parameters, 'numpy'):
        reach_params_np = reach_parameters.cpu().numpy()
    else:
        reach_params_np = np.array(reach_parameters)
    
    # Critical fix: Ensure reach_parameters is 1D for matplotlib
    if reach_params_np.ndim > 1:
        if reach_params_np.shape[1] == 1:
            # If it's a column vector, flatten it
            reach_params_np = reach_params_np.flatten()
        else:
            # For multi-agent case, use the first parameter (they should be the same for reach)
            reach_params_np = reach_params_np[:, 0]
    
    # Ensure it's truly 1D
    reach_params_np = np.atleast_1d(reach_params_np).flatten()
    
    if plot_type == "line":
        # Plot individual agent trajectories using max/min data
        if max_pos_np.ndim == 2:
            for agent_id in range(min(num_agents, max_pos_np.shape[1])):
                ax.plot(reach_params_np, max_pos_np[:, agent_id], 
                       color=trajectory_cmap, linestyle='-', alpha=0.7,
                       label=f'Agent {agent_id+1} Max' if agent_id == 0 else "")
                ax.plot(reach_params_np, min_pos_np[:, agent_id], 
                       color=trajectory_cmap, linestyle='--', alpha=0.7,
                       label=f'Agent {agent_id+1} Min' if agent_id == 0 else "")
        else:
            # 1D case - single agent or aggregate
            ax.plot(reach_params_np, max_pos_np, color=trajectory_cmap, 
                   linestyle='-', linewidth=2, label='Maximum Position')
            ax.plot(reach_params_np, min_pos_np, color=trajectory_cmap, 
                   linestyle='--', linewidth=2, label='Minimum Position')
            
    elif plot_type == "envelope":
        # Create envelope plot showing the spread of positions
        if max_pos_np.ndim == 2:
            # For multi-agent case, find overall max and min across all agents
            overall_max = np.max(max_pos_np, axis=1)
            overall_min = np.min(min_pos_np, axis=1)
        else:
            # Single agent or already aggregated case
            overall_max = max_pos_np
            overall_min = min_pos_np
        
        # Ensure all arrays are 1D and same length for fill_between
        overall_max = np.atleast_1d(overall_max).flatten()
        overall_min = np.atleast_1d(overall_min).flatten()
        
        # Ensure all arrays have the same length by trimming to the shortest
        min_length = min(len(reach_params_np), len(overall_max), len(overall_min))
        reach_params_np = reach_params_np[:min_length]
        overall_max = overall_max[:min_length]
        overall_min = overall_min[:min_length]
        
        # Fill the envelope area following project patterns
        ax.fill_between(reach_params_np, overall_min, overall_max, 
                       color=envelope_cmap, alpha=envelope_alpha, 
                       label='Position Envelope')
        
        # Add boundary lines
        ax.plot(reach_params_np, overall_max, color=envelope_cmap, 
               linewidth=2, label='Maximum Position')
        ax.plot(reach_params_np, overall_min, color=envelope_cmap, 
               linewidth=2, linestyle='--', label='Minimum Position')
        
    elif plot_type == "heat":

        # Create density matrix using envelope positions
        # Count each agent once per parameter (avoid double-counting max/min)
        
        # For each parameter point, compute one representative position per agent
        combined_positions = []
        for i in range(max_pos_np.shape[0]):  # For each parameter value
            # Get max and min positions for this parameter
            max_row = max_pos_np[i, :]
            min_row = min_pos_np[i, :]
            
            agent_positions = []
            for agent_id in range(num_agents):
                max_val = max_row[agent_id] 
                min_val = min_row[agent_id]
                
                if np.isnan(max_val) and np.isnan(min_val):
                    continue
                
                if not np.isnan(max_val) and not np.isnan(min_val):
                    representative = (max_val + min_val) / 2.0
                else:
                    representative = max_val if not np.isnan(max_val) else min_val
                
                agent_positions.append(round(representative, 6))
            
            # Store representative agent positions for this parameter point
            combined_positions.append(np.array(agent_positions))
        
        # Find the maximum number of extremes across all parameter points
        max_extremes_per_param = max(len(pos) for pos in combined_positions) if combined_positions else 0
        
        # Create consistent array structure with NaN padding
        if max_extremes_per_param > 0:
            combined_positions_array = np.full((len(combined_positions), max_extremes_per_param), np.nan)
            for i, pos_list in enumerate(combined_positions):
                combined_positions_array[i, :len(pos_list)] = pos_list
            combined_positions = combined_positions_array
        else:
            combined_positions = np.array([])
                
        # Create position bins and density matrix
        if combined_positions.size > 0:
            all_positions = combined_positions.flatten()
            valid_positions = all_positions[~np.isnan(all_positions)]
            
            if len(valid_positions) > 0:
                min_pos = max(0, valid_positions.min() - 0.05)
                max_pos = min(1, valid_positions.max() + 0.05)
                position_bins = np.linspace(min_pos, max_pos, 100)
                
                # Create density matrix
                density_matrix = np.zeros((len(position_bins)-1, len(reach_params_np)))
                
                # Count extreme positions per agent in each bin for each parameter
                for i in range(min(len(reach_params_np), combined_positions.shape[0])):
                    # Get extreme positions for this parameter
                    agent_positions = combined_positions[i, :]
                    valid_pos = agent_positions[~np.isnan(agent_positions)]
                    
                    if len(valid_pos) > 0:
                        # Count positions in each bin - this gives the number of extreme positions per bin
                        # Each agent can contribute at most 2 extreme positions (min and max)
                        # But if min == max, they only contribute 1
                        counts, _ = np.histogram(valid_pos, bins=position_bins)
                        density_matrix[:, i] = counts
                
                # Create discrete colormap for per-agent position count
                max_extremes = int(density_matrix.max())
                if max_extremes > 0:
                    # The maximum should be <= num_agents (one position per agent)
                    levels = np.arange(0, max_extremes + 2, 1)
                    norm = mpl.colors.BoundaryNorm(levels, ncolors=256)
                    
                    im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                                norm=norm, origin='lower',
                                extent=[reach_params_np[0], reach_params_np[-1], 
                                    position_bins[0], position_bins[-1]],
                                interpolation='nearest')
                else:
                    im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                                origin='lower',
                                extent=[reach_params_np[0], reach_params_np[-1], 
                                    position_bins[0], position_bins[-1]],
                                interpolation='nearest')
            else:
                # No valid data case
                density_matrix = np.zeros((10, len(reach_params_np)))
                im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                            origin='lower',
                            extent=[reach_params_np[0], reach_params_np[-1], 0, 1],
                            interpolation='nearest')
        else:
            # Empty data case
            density_matrix = np.zeros((10, len(reach_params_np)))
            im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                        origin='lower',
                        extent=[reach_params_np[0], reach_params_np[-1], 0, 1],
                        interpolation='nearest')
        
        
            
        # Add envelope overlay using original extreme positions data
        # Only show envelope where there are actual differences between min and max
        locs = torch.where(torch.round(extreme_positions['min'][:,0], decimals=2) != torch.round(extreme_positions['max'][:,0], decimals=2))
        
        # Filter out isolated elements - keep only sequences of 2 or more consecutive values
        if len(locs[0]) > 0:
            locs_list = locs[0].tolist()
            locs_list.sort()
            
            filtered_locs = []
            for i, val in enumerate(locs_list):
                # Check if value is part of a sequence (has consecutive neighbor)
                has_prev = (i > 0 and locs_list[i-1] == val - 1)
                has_next = (i < len(locs_list) - 1 and locs_list[i+1] == val + 1)
                
                # Keep value if it's part of a sequence (has at least one consecutive neighbor)
                if has_prev or has_next:
                    filtered_locs.append(val)
            
            locs = (torch.tensor(filtered_locs),)
        
        if len(locs[0]) > 0:  # Only plot envelope if there are actual differences after filtering
            envelope_params = reach_parameters[locs]
            max_pos_envelope = extreme_positions['max'][locs]
            min_pos_envelope = extreme_positions['min'][locs]
            
            # Convert to numpy for plotting
            if hasattr(envelope_params, 'numpy'):
                envelope_params_np = envelope_params[:, 0].cpu().numpy()
            else:
                envelope_params_np = envelope_params[:, 0]
                
            if hasattr(max_pos_envelope, 'numpy'):
                max_pos_envelope_np = max_pos_envelope.cpu().numpy()
                min_pos_envelope_np = min_pos_envelope.cpu().numpy()
            else:
                max_pos_envelope_np = max_pos_envelope
                min_pos_envelope_np = min_pos_envelope
            
            # Plot envelope for each agent
            for i in range(num_agents):
                if i == 0:
                    ax.plot(envelope_params_np, max_pos_envelope_np[:, i], 
                        color='orange', linewidth=1, alpha=0.5, linestyle='--', label='Upper envelope')
                    ax.plot(envelope_params_np, min_pos_envelope_np[:, i], 
                        color='red', linewidth=1, linestyle='--', alpha=0.5, label='Lower envelope')
                else:
                    ax.plot(envelope_params_np, max_pos_envelope_np[:, i], 
                        color='orange', linewidth=1, linestyle='--', alpha=0.5)
                    ax.plot(envelope_params_np, min_pos_envelope_np[:, i], 
                        color='red', linewidth=1, linestyle='--', alpha=0.5)
        
    # Bifurcations critical values (works for gaussian only) following project patterns
    if infl_type == 'gaussian' and show_pred == True:
        _, means, crit_stds = one_utils.critical_values_plot(
            num_agents=num_agents, bin_points=bin_points, 
            resource_distribution=resource_distribution, axis=ax, 
            reach_start=reach_start, reach_end=reach_end, 
            refinements=refinements, crit_cs=crit_cmap)
        
        crit_stds = general.flatten_list(xss=crit_stds)
        crit_stds.sort()
        
        # Filter critical values within reach range
        crit_stds = [std for std in crit_stds if reach_start <= std <= reach_end]
        
        std_ticks = [float(np.around(i, decimals=2)) for i in np.linspace(reach_end, reach_start, num=5)]
        crit_means = np.around(means, decimals=3)
        
        std_removed = np.setdiff1d(np.array(std_ticks), np.around(crit_stds, 2))
        std_tick_vals = np.array(list(std_removed) + crit_stds)
        std_tick_vals.sort()
        
        crit_std_locs = []
        for std_id in range(len(crit_stds)):
            crit_std_locs.append(int(np.where(std_tick_vals == crit_stds[std_id])[0][0]))
        
        std_tick_labels = list(std_tick_vals.copy())
        for std_loc_id in range(len(crit_stds)):
            if std_loc_id == len(crit_stds) - 1:
                std_tick_labels[int(crit_std_locs[std_loc_id])] = r'$t_*$'
            else:
                std_tick_labels[int(crit_std_locs[std_loc_id])] = r'$t_' + str(len(crit_stds) - std_loc_id - 1) + r'$'
                
        if infl_cshift == False:
            ax.xaxis.set_ticks(std_tick_vals)
            ax.xaxis.set_ticklabels(std_tick_labels)
            
        ax.yaxis.set_ticks(crit_means)
        ax.yaxis.set_ticklabels(crit_means)

    else:
        bifurcation_types = one_utils.bifurcation_type_helper(matrix=extreme_positions,reach_parameters=reach_parameters)
        locs = torch.where(torch.round(extreme_positions['min'][:,0], decimals=2) != torch.round(extreme_positions['max'][:,0], decimals=2))
        
        # Filter out isolated elements - keep only sequences of 2 or more consecutive values
        if len(locs[0]) > 0:
            locs_list = locs[0].tolist()
            locs_list.sort()
            
            filtered_locs = []
            for i, val in enumerate(locs_list):
                # Check if value is part of a sequence (has consecutive neighbor)
                has_prev = (i > 0 and locs_list[i-1] == val - 1)
                has_next = (i < len(locs_list) - 1 and locs_list[i+1] == val + 1)
                
                # Keep value if it's part of a sequence (has at least one consecutive neighbor)
                if has_prev or has_next:
                    filtered_locs.append(val)
            
            locs = (torch.tensor(filtered_locs),)
        
        # Convert locs to a set of excluded indices
        excluded_indices = set(locs[0].tolist()) if len(locs[0]) > 0 else set()

        # Separate bifurcations by type and get classifications, keeping track of which is which
        type1_bifurcations = []
        type2_bifurcations = []
        bifurcation_info = []  # Store all bifurcations with their info

        # First, collect all valid bifurcations sorted by key
        all_bifurcations_by_key = []
        for key, value in bifurcation_types.items():
            # Skip if this key is in the excluded indices
            if int(key) in excluded_indices:
                continue
            all_bifurcations_by_key.append((int(key), key, value))
        
        # Sort by key to process in order
        all_bifurcations_by_key.sort(key=lambda x: x[0])
        
        # Keys that form the left/right boundary of a cycle region must never be filtered out,
        # otherwise we lose the vertical lines at cycle start/end.
        cycle_boundary_keys = set()
        if excluded_indices:
            exc_sorted = sorted(excluded_indices)
            runs = []
            curr = [exc_sorted[0]]
            for j in range(1, len(exc_sorted)):
                if exc_sorted[j] == curr[-1] + 1:
                    curr.append(exc_sorted[j])
                else:
                    runs.append(curr)
                    curr = [exc_sorted[j]]
            runs.append(curr)
            for r in runs:
                k = min(r) - 1
                if k >= 0:
                    cycle_boundary_keys.add(k)
                k_right = max(r) + 1
                if k_right >= 0 and str(k_right) in bifurcation_types:
                    cycle_boundary_keys.add(k_right)
        
        # Filter out bifurcations that are too close to the previous one
        # Never filter cycle-boundary keys (vertical lines at cycle start/end).
        last_accepted_key = None
        for int_key, key, value in all_bifurcations_by_key:
            is_cycle_boundary = int_key in cycle_boundary_keys
            if not is_cycle_boundary and last_accepted_key is not None and (int_key - last_accepted_key) <= bifurcation_key_tolerance:
                continue
            
            last_accepted_key = int_key
            reach_param = value['reach_parameter']
            bif_type = value['type']
            classification = value['classification_new']
            
            bifurcation_info.append({
                'reach': reach_param,
                'type': bif_type,
                'classification': classification,
                'key': key
            })
            
            if bif_type == '1':
                type1_bifurcations.append(reach_param)
            else:
                type2_bifurcations.append(reach_param)

        # Sort bifurcation_info by reach parameter (left to right)
        bifurcation_info.sort(key=lambda x: x['reach'])

        xlim_start, xlim_end = reach_start, reach_end

        # Define color palette for shaded regions
        region_colors = plt.cm.Pastel1(np.linspace(0, 1, 9))

        # Create shaded regions from right to left
        # First, collect all boundaries including xlim_start and xlim_end
        boundaries = [xlim_start] + [info['reach'] for info in bifurcation_info] + [xlim_end]

        # Get reach parameters for cycle indices
        cycle_reach_params = []
        if len(excluded_indices) > 0:
            for idx in excluded_indices:
                cycle_reach_params.append(reach_parameters[idx][0].item())

        # Shade regions and track legend patches
        from matplotlib.patches import Patch

        # First pass: collect all unique labels and assign colors
        # We need to identify if any regions contain cycles and split them
        label_to_color = {}
        all_labels = []
        final_boundaries = []
        final_labels = []

        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            
            # Check if any cycle reach values fall within this region
            cycles_in_region = []
            cycle_indices = []
            for idx, cycle_reach in enumerate(cycle_reach_params):
                if x_start < cycle_reach < x_end:
                    cycles_in_region.append(cycle_reach)
                    cycle_indices.append(idx)
            
            # Get the original region's classification
            if i == len(boundaries) - 2:
                original_label = f'$({num_agents})$'
            elif i < len(bifurcation_info):
                original_label = bifurcation_info[i]['classification']
            else:
                original_label = ''
            
            if cycles_in_region:
                # Sort cycle points to find the range they span
                sorted_pairs = sorted(zip(cycles_in_region, cycle_indices))
                cycles_in_region = [val for val, idx in sorted_pairs]
                cycle_indices = [idx for val, idx in sorted_pairs]
                
                # Get the reach values at index-1 and index+1
                min_idx = cycle_indices[0]
                max_idx = cycle_indices[-1]
                
                cycle_start = x_start  # Use the boundary start instead of calculated cycle_start
                cycle_end_raw = cycle_reach_params[min(len(cycle_reach_params) - 1, max_idx + 2)]
                cycle_end = min(x_end, cycle_end_raw)
                
                # Create up to 2 sub-regions (no gap before cycles):
                # 1. The cycle region extends from the boundary start
                final_boundaries.append((cycle_start, cycle_end))
                final_labels.append('Cycles')
                
                # 2. After cycles (if there's space)
                if cycle_end < x_end:
                    final_boundaries.append((cycle_end, x_end))
                    final_labels.append(original_label)
                
            else:
                # No cycles in this region, keep it as is
                final_boundaries.append((x_start, x_end))
                final_labels.append(original_label)

        # Assign colors to unique labels with specific colors for certain patterns
        import matplotlib.colors as mcolors

        # Define specific colors for key patterns
        specific_colors = {
            f'$({num_agents})$': '#87CEEB',  # Sky blue for all agents together
            'Cycles': '#FFD700',  # Gold for cycles
            '(2,1,1,2)': '#FF6B6B',  # Coral red for (2,1,1,2)
            '(1,1,1,1,1,1)': '#9370DB'  # Medium purple for (1,1,1,1,1,1)
        }

        # Additional predefined color palette for other labels
        additional_colors = ['#98D8C8', '#F7B7A3', '#EA5F89', '#9D84B7', '#A8E6CF', 
                            '#FFD3B6', '#FFAAA5', '#FF8B94', '#C7CEEA', '#B5EAD7']

        # Assign colors to labels
        color_index = 0
        for label in final_labels:
            if label and label not in label_to_color:
                if label in specific_colors:
                    label_to_color[label] = specific_colors[label]
                else:
                    # Use predefined color palette
                    label_to_color[label] = additional_colors[color_index % len(additional_colors)]
                    color_index += 1

        # Second pass: draw regions with consistent colors and add to legend
        # Track all regions with their positions for ordered legend
        region_legend_items = []  # Store (order_index, label, handle) for ordering
        
        for i, (x_start, x_end) in enumerate(final_boundaries):
            label = final_labels[i]
            color = label_to_color.get(label, region_colors[0])
            
            # Add shaded region (always add, but we'll handle legend separately)
            zorder = 1 if label == 'Cycles' else 0
            span = ax.axvspan(x_start, x_end, alpha=0.1, color=color, zorder=zorder)
            
            # Track this region for legend with its order index
            # Use negative index for right-to-left ordering
            if label:
                region_legend_items.append((i, label, span))
        
        # Create ordered, deduplicated legend entries based on last occurrence
        # Deduplicate by keeping only the LAST occurrence of each label (for right-to-left ordering)
        seen_labels_ordered = {}
        for order_idx, label, handle in region_legend_items:
            # Always update - this keeps the last occurrence
            seen_labels_ordered[label] = (order_idx, handle)
        
        # Sort by order index in reverse (right to left as they appear in final_boundaries)
        sorted_region_items = sorted(seen_labels_ordered.items(), key=lambda x: x[1][0], reverse=True)
        
        # Store the ordered region items for later use in legend
        ordered_region_handles = []
        ordered_region_labels = []
        for label, (order_idx, handle) in sorted_region_items:
            ordered_region_handles.append(handle)
            ordered_region_labels.append(label)

        # Plot type 1 bifurcation lines without text labels
        for i, reach_param in enumerate(type1_bifurcations):
            ax.axvline(x=reach_param, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=10,
                        label=f'$\\sigma_{i+1}^1 = {reach_param:.4f}$')

        # Plot type 2 bifurcation lines without text labels
        for j, reach_param in enumerate(type2_bifurcations):
            ax.axvline(x=reach_param, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=10,
                        label=f'$\\sigma_{j+1}^2 = {reach_param:.4f}$')

        # Plot cycle bifurcation line without text label (if cycles exist)
        if len(cycle_reach_params) > 0:
            cycle_end_param = max(cycle_reach_params)
            ax.axvline(x=cycle_end_param, color='purple', linestyle='-.', linewidth=2, alpha=0.5, zorder=10,
                        label=f'$\\sigma^{{cycle}} = {cycle_end_param:.4f}$')

        # Simplified legend: gather all labeled elements automatically
        handles, labels = ax.get_legend_handles_labels()

        # Separate into categories
        bifurcation_items = []
        envelope_items = []

        for handle, label in zip(handles, labels):
            if 'sigma' in label.lower():
                bifurcation_items.append((handle, label))
            elif label in ['Upper envelope', 'Lower envelope']:
                envelope_items.append((handle, label))

        # Extract handles and labels from envelope items
        envelope_handles = [h for h, l in envelope_items]
        envelope_labels = [l for h, l in envelope_items]

        # Combine regions (already ordered right-to-left) and envelope for legend (no bifurcations)
        combined_handles = ordered_region_handles + envelope_handles
        combined_labels = ordered_region_labels + envelope_labels

        # Create legend in upper right corner of main plot (only if show_bif_labels is True)
        if show_bif_labels==True:
            ax.legend(handles=combined_handles, labels=combined_labels, 
                        loc='upper right', 
                        fontsize=legend_font_size, title='Legend', framealpha=0.9)
                        

        ax.set_xlim(reach_start,reach_end)

        
        # Create rectangle bifurcation plot in separate subplot
        if len(bifurcation_items) > 0:
            import matplotlib.patches as patches
            from matplotlib.lines import Line2D
            
            # Create rectangle plot axes - position depends on show_bif_labels
            if show_bif_labels:
                ax_rect = fig.add_subplot(gs[1])
            else:
                # Use manual positioning for tighter layout when show_bif_labels is False
                # Position rectangle plot at top of right column with minimal height
                ax_rect = fig.add_axes([0.52, 0.65, 0.45, 0.25])  # [left, bottom, width, height]
            ax_rect.set_axis_off()
            
            # Define rectangle parameters
            rect_height = 0.4
            rect_y_start = 1
            sigma_min = reach_start
            sigma_max = reach_end
            rect_x_start = 0
            rect_total_width = 8.0
            
            # Function to convert sigma value to x-coordinate
            def sigma_to_x(sigma):
                return rect_x_start + (sigma - sigma_min) / (sigma_max - sigma_min) * rect_total_width
            
            # Draw colored segments
            for i, (x_start_sigma, x_end_sigma) in enumerate(final_boundaries):
                segment_x_start = sigma_to_x(x_start_sigma)
                segment_x_end = sigma_to_x(x_end_sigma)
                segment_width = segment_x_end - segment_x_start
                
                label = final_labels[i]
                color = label_to_color.get(label, '#CCCCCC')
                
                rectangle = patches.Rectangle(
                    (segment_x_start, rect_y_start),
                    segment_width,
                    rect_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax_rect.add_patch(rectangle)
            
            # Combine all bifurcations
            all_bifurcations = []
            for i, reach_param in enumerate(type1_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    all_bifurcations.append(('type1', i, reach_param))
            
            for j, reach_param in enumerate(type2_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    all_bifurcations.append(('type2', j, reach_param))
            
            if len(cycle_reach_params) > 0:
                cycle_end_param = max(cycle_reach_params)
                if sigma_min <= cycle_end_param <= sigma_max:
                    all_bifurcations.append(('cycle', 0, cycle_end_param))
            
            # Sort by position in descending order (right to left, largest sigma first)
            all_bifurcations.sort(key=lambda x: x[2], reverse=True)
            
            # Draw vertical bifurcation lines with alternating labels
            # Use separate counters for each type to label them 1, 2, 3... from right to left
            label_counter = 0
            type1_label_counter = 1
            type2_label_counter = 1
            for bif_type, idx, reach_param in all_bifurcations:
                x_pos = sigma_to_x(reach_param)
                
                if bif_type == 'type1':
                    color = 'red'
                    linestyle = '--'
                    alpha = 0.9
                    label_text = f'$\\sigma_{type1_label_counter}^1$'
                    type1_label_counter += 1
                elif bif_type == 'type2':
                    color = 'blue'
                    linestyle = ':'
                    alpha = 0.9
                    label_text = f'$\\sigma_{type2_label_counter}^2$'
                    type2_label_counter += 1
                else:  # cycle
                    color = 'purple'
                    linestyle = '-.'
                    alpha = 0.5
                    label_text = '$\\sigma^{c}$'
                
                # Draw vertical line through the rectangle height
                line_y_start = rect_y_start-.15
                line_y_end = rect_y_start + rect_height+.15
                ax_rect.plot([x_pos, x_pos], [line_y_start, line_y_end],
                           color=color, linestyle=linestyle, linewidth=3, alpha=alpha, zorder=10)
                
                # Alternate label position: above or below, positioned at line endpoints
                if label_counter % 2 == 0:
                    label_y_pos = line_y_end + 0.05
                    va = 'bottom'
                else:
                    label_y_pos = line_y_start - 0.05
                    va = 'top'
                
                # Position label directly at the line (ha='center' centers it on the line)
                ax_rect.text(x_pos, label_y_pos, label_text,
                            fontsize=rect_label_font_size, color=color, fontweight='bold',
                            ha='center', va=va)
                
                label_counter += 1
            
            # Set axis limits and labels
            ax_rect.set_xlim(0, rect_x_start + rect_total_width )
            # Use tighter y-limits when show_bif_labels is False to reduce whitespace
            if show_bif_labels:
                ax_rect.set_ylim(0, 4)
            else:
                # Tighter bounds: rect goes from y=1 to y=1.4, labels extend slightly above/below
                ax_rect.set_ylim(0.5, 2.0)
            ax_rect.set_xlabel('Sigma ($\\sigma$) - Agent Reach Parameter', fontsize=default_font_size)
            # Adjust title position based on layout
            if show_bif_labels:
                ax_rect.set_title(r'Bifurcation Regions on $\sigma$', fontsize=title_font_size)
            else:
                # Place title just above the rectangle, centered
                title_x = rect_x_start + rect_total_width / 2  # Center of rectangle
                title_y = rect_y_start + rect_height + 0.5  # Just above rectangle top
                ax_rect.text(title_x, title_y, r'Bifurcation Regions on $\sigma$',
                            fontsize=title_font_size, ha='center', va='bottom', fontweight='bold')
            if show_bif_labels:
                # Add sigma value labels below rectangle
                sigma_step = (sigma_max - sigma_min) / 10
                sigma_labels_vals = [sigma_min + i * sigma_step for i in range(11)]
                for sigma_val in sigma_labels_vals:
                    x_pos = sigma_to_x(sigma_val)
                    ax_rect.text(x_pos, rect_y_start - 0.45, f'{sigma_val:.2f}',
                                ha='center', va='top', fontsize=rect_sigma_font_size, color='black')
            
            # Create legend with sorted bifurcation values (right to left numbering)
            # First, sort each type by reach_param descending to match the rectangle labels
            type1_sorted = sorted([(rp, 'red', '--') for rp in type1_bifurcations if sigma_min <= rp <= sigma_max], 
                                  key=lambda x: x[0], reverse=True)
            type2_sorted = sorted([(rp, 'blue', ':') for rp in type2_bifurcations if sigma_min <= rp <= sigma_max], 
                                  key=lambda x: x[0], reverse=True)
            
            legend_data = []
            
            # Add type1 with right-to-left numbering
            for i, (reach_param, color, linestyle) in enumerate(type1_sorted):
                legend_data.append((reach_param, color, linestyle, f'$\\sigma_{i+1}^1 = {reach_param:.4f}$'))
            
            # Add type2 with right-to-left numbering
            for j, (reach_param, color, linestyle) in enumerate(type2_sorted):
                legend_data.append((reach_param, color, linestyle, f'$\\sigma_{j+1}^2 = {reach_param:.4f}$'))
            
            if len(cycle_reach_params) > 0:
                cycle_end_param = max(cycle_reach_params)
                if sigma_min <= cycle_end_param <= sigma_max:
                    legend_data.append((cycle_end_param, 'purple', '-.', f'$\\sigma^{{c}} = {cycle_end_param:.4f}$'))
            
            # Sort by sigma value in descending order for final legend display
            legend_data.sort(key=lambda x: x[0], reverse=True)
            
            # Create legend elements
            legend_elements = []
            for reach_param, color, linestyle, label in legend_data:
                legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=label))
            if show_bif_labels == True:
                # Add legend
                ax_rect.legend(handles=legend_elements, loc='upper right', fontsize=rect_label_font_size, framealpha=0.9)
            else:
                # When show_bif_labels is False, use manual positioning for compact layout
                # Position elements below the rectangle plot with adequate spacing to avoid overlap
                
                # Calculate dynamic positioning based on number of legend items
                num_sigma_items = len(legend_elements)
                num_region_items = len(ordered_region_labels)
                
                # Estimate heights needed with more generous sizing (0.035 per item, plus padding)
                sigma_legend_height = max(0.15, 0.035 * num_sigma_items + 0.06)
                region_legend_height = max(0.15, 0.035 * (num_region_items // 3 + 1) + 0.06)
                
                # Position sigma values legend below rectangle plot
                # Rectangle is at [0.52, 0.65, 0.45, 0.25], so its bottom is at 0.65
                # Use larger gap (0.05) to prevent overlap
                sigma_legend_bottom = 0.65 - sigma_legend_height - 0.05
                ax_sigma_legend = fig.add_axes([0.52, sigma_legend_bottom, 0.45, sigma_legend_height])
                ax_sigma_legend.set_axis_off()
                
                # Position sigma legend centered in the dedicated area
                ax_sigma_legend.legend(handles=legend_elements, 
                              loc='center',
                              ncol=1,
                              fontsize=rect_label_font_size, 
                              title='Bifurcation Values', 
                              title_fontsize=title_font_size - 2,
                              framealpha=0.9)
                
                # Position region legend below sigma legend with adequate gap (0.05)
                region_legend_bottom = sigma_legend_bottom - region_legend_height - 0.05
                ax_legend = fig.add_axes([0.52, region_legend_bottom, 0.45, region_legend_height])
                ax_legend.set_axis_off()
                
                # Create legend handles for region colors
                from matplotlib.patches import Patch as LegendPatch
                region_legend_elements = []
                for label_text in ordered_region_labels:
                    color = label_to_color.get(label_text, '#CCCCCC')
                    region_legend_elements.append(LegendPatch(facecolor=color, edgecolor='black', 
                                                              alpha=0.7, label=label_text))
                
                # Add envelope legend items if they exist
                if envelope_labels:
                    region_legend_elements.append(Line2D([0], [0], color='orange', linestyle='--', 
                                                         linewidth=2, label='Upper envelope'))
                    region_legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                                         linewidth=2, label='Lower envelope'))
                
                # Position region legend centered in the dedicated area
                ax_legend.legend(handles=region_legend_elements, 
                              loc='center',
                              ncol=min(3, len(region_legend_elements)),
                              fontsize=legend_font_size, 
                              title='Region Legend', 
                              title_fontsize=title_font_size - 4,
                              framealpha=0.9)
            
            # Display settings
            ax_rect.set_box_aspect(.5)

    
            

    
    # Plot features following project patterns
    if plot_type == "heat":
        max_extremes = int(density_matrix.max()) if 'density_matrix' in locals() else num_agents
        tick_levels = np.arange(0, max_extremes + 2, 1)
        
        if axis_return == False:
            if 'norm' in locals():
                cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
            else:
                cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
            
            # Updated label to reflect unique extreme positions
            cbar.set_label('Number of Agents', fontsize=cbar_font_size)
            
            if cbar_center_labels and len(tick_levels) > 1:
                centered_positions = []
                for i in range(len(tick_levels) - 1):
                    centered_positions.append((tick_levels[i] + tick_levels[i + 1]) / 2)
                
                if len(centered_positions) > 0:
                    cbar.set_ticks(centered_positions)
                    centered_labels = [str(i) for i in range(len(centered_positions))]
                    cbar.set_ticklabels(centered_labels)
            else:
                cbar.set_ticks(tick_levels)
                cbar.set_ticklabels([str(int(level)) for level in tick_levels])

    # Optional vertical lines following project patterns
    if optional_vline is not None:
        for vline_id in range(len(optional_vline)):
            ax.vlines(x=optional_vline[vline_id], ymin=0, ymax=1, colors='black', 
                     linestyles='dashed', 
                     label=r'$\sigma^*_' + str(vline_id + 1) + r'=$' + str(np.around(optional_vline[vline_id], decimals=4)))
    


    non_equal=torch.where(torch.round(extreme_positions['max'][:,1],decimals=1)!=0.5)[0]
    max_equal=torch.where(extreme_positions['max'][:,1]==extreme_positions['max'][:,0])[0]
    max_equal2=torch.where(extreme_positions['min'][:,1]==extreme_positions['min'][:,2])[0]
    
    # Find values that exist in all three tensors
    mask1 = torch.isin(non_equal, max_equal)
    mask2 = torch.isin(non_equal, max_equal2)
    if show_pred==True:
        # Safe handling when masks might be empty
        if torch.any(mask1) and torch.any(mask2):
            param_int = torch.max(torch.max(non_equal[mask1]), torch.max(non_equal[mask2]))
            ax.vlines(x=reach_parameters[param_int][0], ymin=0, ymax=1, colors='blue', 
                    linestyles='dashed', 
                    label='$\sigma^*_' + str(vline_id + 2) + r'=$' + str(np.around(reach_parameters[param_int][0].item(), decimals=4)))
        elif torch.any(mask1):
            param_int = torch.max(non_equal[mask1])
            ax.vlines(x=reach_parameters[param_int][0], ymin=0, ymax=1, colors='blue', 
                    linestyles='dashed', 
                    label='$\sigma^*_' + str(vline_id + 2) + r'=$' + str(np.around(reach_parameters[param_int][0].item(), decimals=4)))
        elif torch.any(mask2):
            param_int = torch.max(non_equal[mask2])
            ax.vlines(x=reach_parameters[param_int][0], ymin=0, ymax=1, colors='blue', 
                    linestyles='dashed', 
                    label='$\sigma^*_' + str(vline_id + 2) + r'=$' + str(np.around(reach_parameters[param_int][0].item(), decimals=4)))
        else:
            # No valid intersection found - skip adding this vline or use a default
            print("Warning: No valid parameter intersection found for critical value line")
            # Optionally, you could add a fallback or just skip the vline entirely

    # Legend handling following project patterns - skip if combined legend already created in show_pred==False branch
    if show_pred == True:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower center')
    
    # Title formatting following project patterns
    if short_title == True:
        title = 'Adaptive Agents Envelope'
    else:
        title = str(num_agents) + ' Agents'

    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + " " + title_addition
    
    ax.set_title(title, fontsize=title_font_size)
    
    if infl_type == 'gaussian':
        ax.set_xlabel(r"$\sigma$ (std)")
    else:
        ax.set_xlabel(r"$\sigma$")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Agent Position")

    plt.close()
    
    if axis_return:
        return ax
    else:
        return fig
    
def equilibrium_bifurcation_envelope_plot_1d_COMPLETE(num_agents: int,
                                    bin_points: np.ndarray,
                                    resource_distribution: np.ndarray,
                                    infl_type: str,
                                    reach_parameters: List[float],
                                    matrix_list: Dict[str, torch.Tensor],
                                    reach_start: float,
                                    reach_end: float,
                                    refinements: int,
                                    plot_type: str,
                                    title_ads: Optional[List[str]],
                                    short_title: bool = False,
                                    norm: bool = True,
                                    infl_cshift: bool = False,
                                    cmaps: dict = {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys', 'envelope': '#FF6B6B'},
                                    font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                    cbar_config: dict = {'center_labels': True, 'label_alignment': 'center', 'shrink': 0.8},
                                    axis_return: bool = False,
                                    show_pred: bool = False,
                                    optional_vline: List[float] = None,
                                    envelope_alpha: float = 0.3,
                                    show_bif_labels: bool = True,
                                    bifurcation_key_tolerance: int = 3
                                    ) -> matplotlib.figure.Figure:
    r"""
    Plot complete equilibrium bifurcation envelope with multiple trajectory data in a 1D domain.
    
    Advanced version that combines envelope data with multiple position matrices to create
    a comprehensive visualization showing both the envelope of extreme positions and detailed
    trajectory evolution. Includes density heatmap generation from multiple equilibrium searches.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param bin_points: Discretized points defining resource allocation regions.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource values at each bin point.
    :type resource_distribution: np.ndarray
    :param infl_type: Type of influence kernel ('gaussian', 'beta', 'multi_gaussian', etc.).
    :type infl_type: str
    :param reach_parameters: Array of reach parameter values to test.
    :type reach_parameters: List[float]
    :param matrix_list: Dictionary containing multiple position matrices and envelope data.
    :type matrix_list: Dict[str, torch.Tensor]
    :param reach_start: Starting value of reach parameter range.
    :type reach_start: float
    :param reach_end: Ending value of reach parameter range.
    :type reach_end: float
    :param refinements: Number of refinements for critical value estimation.
    :type refinements: int
    :param plot_type: Type of plot ('line', 'envelope', or 'heat').
    :type plot_type: str
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param short_title: Use abbreviated title format.
    :type short_title: bool
    :param norm: Normalize heatmap values.
    :type norm: bool
    :param infl_cshift: Whether influence uses center shift.
    :type infl_cshift: bool
    :param cmaps: Color map configuration dictionary with keys 'heat', 'trajectory', 'crit', 'envelope'.
    :type cmaps: dict
    :param font: Font configuration dictionary.
    :type font: dict
    :param cbar_config: Colorbar configuration dictionary.
    :type cbar_config: dict
    :param axis_return: If True, return axes object; if False, return figure object.
    :type axis_return: bool
    :param show_pred: Show predicted critical values (only for Gaussian kernels).
    :type show_pred: bool
    :param optional_vline: Optional vertical lines to add to plot.
    :type optional_vline: Optional[List[float]]
    :param envelope_alpha: Transparency level for envelope fill (0-1).
    :type envelope_alpha: float
    :param show_bif_labels: Whether to show bifurcation labels on the plot.
    :type show_bif_labels: bool
    :param bifurcation_key_tolerance: Minimum key distance between bifurcations to include both. Bifurcations with keys closer than this tolerance to the previous one will be ignored.
    :type bifurcation_key_tolerance: int
    
    :return: The generated matplotlib figure or axes object.
    :rtype: matplotlib.figure.Figure
    """
    
    # Extract configuration values following project patterns
    crit_cmap = cmaps.get('crit', 'Greys')
    trajectory_cmap = cmaps.get('trajectory', '#851321')
    heat_cmap = cmaps.get('heat', 'Blues')
    envelope_cmap = cmaps.get('envelope', '#FF6B6B')
    
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size = font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    table_font_size = font.get('table_size', 10)
    rect_label_font_size = font.get('rect_label_size', 16)
    rect_sigma_font_size = font.get('rect_sigma_size', 12)
    cbar_center_labels = cbar_config.get('center_labels', True)
    cbar_label_alignment = cbar_config.get('label_alignment', 'center')
    cbar_shrink = cbar_config.get('shrink', 1)
    vline_id = 0
    
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    # Create figure with GridSpec for main plot and rectangle subplot
    # Use same layout as equilibrium_bifurcation_envelope_plot_1d
    if show_bif_labels:
        fig = plt.figure(figsize=(28, 16))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
    else:
        # When show_bif_labels is False, use manual positioning for right column elements
        # to minimize whitespace and avoid overlap
        fig = plt.figure(figsize=(28, 16))
        # Create a simple 1x2 grid for the main plot, right column will use manual positioning
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)
    
    # Support both list (from equilibrium_bifurcation_complete) and dict (e.g. string-keyed)
    if isinstance(matrix_list, dict):
        keys_sorted = sorted(
            (k for k in matrix_list.keys() if str(k).isdigit()),
            key=int
        )
        matrix_list = [matrix_list[k] for k in keys_sorted]
    
    # Preserve original 2D reach_parameters for bifurcation_type_helper
    reach_parameters_2d = reach_parameters.clone() if hasattr(reach_parameters, 'clone') else reach_parameters.copy()
    reach_parameters = reach_parameters[:,0]
    # First pass: determine global position range from all matrices for consistent binning
    all_positions = []
    extreme_positions = matrix_list[1]
    max_positions = extreme_positions['max']
    min_positions = extreme_positions['min']
    non_equal=torch.where(torch.round(extreme_positions['max'][:,1],decimals=1)!=0.5)[0]
    max_equal=torch.where(extreme_positions['max'][:,1]==extreme_positions['max'][:,0])[0]
    max_equal2=torch.where(extreme_positions['min'][:,1]==extreme_positions['min'][:,2])[0]
    
    # Find values that exist in all three tensors
    mask1 = torch.isin(non_equal, max_equal)
    mask2 = torch.isin(non_equal, max_equal2)
    
    run_bifurcation_block = not (infl_type == 'gaussian' and show_pred)
    valid_param_int = False
    param_int_val = 0
    try:
        m1 = non_equal[mask1]
        m2 = non_equal[mask2]
        if len(m1) > 0 and len(m2) > 0:
            param_int_val = int(torch.max(torch.max(m1), torch.max(m2)).item())
            valid_param_int = True
    except Exception:
        pass
    param_int = param_int_val
    
    for matrix_id in range(len(matrix_list)):
        if matrix_id == 1:  # Envelope data
            extreme_positions = matrix_list[matrix_id]
            max_positions = extreme_positions['max']
            min_positions = extreme_positions['min']
            
            # Convert to numpy following project patterns
            if hasattr(max_positions, 'numpy'):
                max_pos_np = max_positions.cpu().numpy()
                min_pos_np = min_positions.cpu().numpy()
            else:
                max_pos_np = np.array(max_positions)
                min_pos_np = np.array(min_positions)
            
            # Collect all valid positions
            valid_max = max_pos_np[~np.isnan(max_pos_np)]
            valid_min = min_pos_np[~np.isnan(min_pos_np)]
            all_positions.extend(valid_max.flatten())
            all_positions.extend(valid_min.flatten())
        else:
            # Regular position matrix data
            final_pos_matrix = matrix_list[matrix_id]
            
            if isinstance(final_pos_matrix, tuple):
                pos_data = final_pos_matrix[1] if len(final_pos_matrix) > 1 else final_pos_matrix[0]
            else:
                pos_data = final_pos_matrix
            
            if hasattr(pos_data, 'numpy'):
                positions = pos_data.cpu().numpy()
            elif isinstance(pos_data, (list, tuple)):
                positions = np.array(pos_data)
            else:
                positions = pos_data
            
            if positions.ndim == 2:
                valid_pos = positions[~np.isnan(positions)]
                all_positions.extend(valid_pos.flatten())
    
    # Create global position bins for consistent alignment
    if len(all_positions) > 0:
        global_min_pos = max(0, np.min(all_positions) - 0.05)
        global_max_pos = min(1, np.max(all_positions) + 0.05)
        position_bins = np.linspace(global_min_pos, global_max_pos, 100)
    else:
        position_bins = np.linspace(0, 1, 100)
    
    # Ensure reach_parameters is properly formatted
    if hasattr(reach_parameters, 'numpy'):
        reach_params_np = reach_parameters.cpu().numpy()
    else:
        reach_params_np = np.array(reach_parameters)
    
    if reach_params_np.ndim > 1:
        reach_params_np = reach_params_np.flatten()
    reach_params_np = np.atleast_1d(reach_params_np).flatten()
    
    # Initialize global density matrix
    density_matrix = np.zeros((len(position_bins)-1, len(reach_params_np)))
    # Track reach indices where new info is added beyond matrix 1 baseline
    completion_mask = np.zeros(len(reach_params_np), dtype=bool)
    baseline_density = None
    
    # Process each matrix with consistent binning (envelope first)
    if len(matrix_list) > 1:
        matrix_order = [1] + [i for i in range(len(matrix_list)) if i != 1]
    else:
        matrix_order = list(range(len(matrix_list)))
    for matrix_id in matrix_order:
        if matrix_id == 1:
            # Handle envelope data (dictionary with 'max' and 'min' keys)
            extreme_positions = matrix_list[matrix_id]
            
            # Extract extreme positions following project patterns
            max_positions = extreme_positions['max']
            min_positions = extreme_positions['min']
            
            # Convert to numpy following project patterns
            if hasattr(max_positions, 'numpy'):
                max_pos_np = max_positions.cpu().numpy()
                min_pos_np = min_positions.cpu().numpy()
            else:
                max_pos_np = np.array(max_positions)
                min_pos_np = np.array(min_positions)
            
            if plot_type == "heat":
                # Create density matrix using envelope positions with consistent binning
                # Count each agent once per parameter (avoid double-counting max/min)
                combined_positions = []
                for i in range(max_pos_np.shape[0]):
                    max_row = max_pos_np[i, :]
                    min_row = min_pos_np[i, :]
                    
                    agent_positions = []
                    for agent_id in range(num_agents):
                        max_val = max_row[agent_id] 
                        min_val = min_row[agent_id]
                        
                        if np.isnan(max_val) and np.isnan(min_val):
                            continue
                        
                        if not np.isnan(max_val) and not np.isnan(min_val):
                            representative = (max_val + min_val) / 2.0
                        else:
                            representative = max_val if not np.isnan(max_val) else min_val
                        
                        agent_positions.append(round(representative, 6))
                    
                    combined_positions.append(np.array(agent_positions))
                
                # Create density matrix using global position bins
                density_matrix_iter = np.zeros((len(position_bins)-1, len(reach_params_np)))
                
                for i in range(min(len(reach_params_np), len(combined_positions))):
                    agent_positions = combined_positions[i]
                    valid_pos = agent_positions[~np.isnan(agent_positions)]
                    
                    if len(valid_pos) > 0:
                        counts, _ = np.histogram(valid_pos, bins=position_bins)
                        density_matrix_iter[:, i] = counts
                
                baseline_density = density_matrix_iter.copy()
                
                # Add to global density matrix
                difference_matrix = np.clip(density_matrix_iter - density_matrix, 0, None)
                density_matrix += difference_matrix
                
                # Add envelope overlay using original extreme positions data
                locs = torch.where(torch.round(extreme_positions['min'][:,0], decimals=2) != 
                                 torch.round(extreme_positions['max'][:,0], decimals=2))
                
                if len(locs[0]) > 0:
                    envelope_params = reach_parameters[locs]
                    max_pos_envelope = extreme_positions['max'][locs]
                    min_pos_envelope = extreme_positions['min'][locs]
                    
                    if hasattr(envelope_params, 'numpy'):
                        envelope_params_np = envelope_params.cpu().numpy()
                        max_pos_envelope_np = max_pos_envelope.cpu().numpy()
                        min_pos_envelope_np = min_pos_envelope.cpu().numpy()
                    else:
                        envelope_params_np = envelope_params
                        max_pos_envelope_np = max_pos_envelope
                        min_pos_envelope_np = min_pos_envelope
                    
                    for i in range(num_agents):
                        if i == 0:
                            ax.plot(envelope_params_np, max_pos_envelope_np[:, i], 
                                color='orange', linewidth=2, alpha=1, linestyle='--', label='Upper')
                            ax.plot(envelope_params_np, min_pos_envelope_np[:, i], 
                                color='red', linewidth=2, linestyle='--', alpha=1, label='Lower')
                        else:
                            ax.plot(envelope_params_np, max_pos_envelope_np[:, i], 
                                color='orange', linewidth=2, linestyle='--', alpha=1)
                            ax.plot(envelope_params_np, min_pos_envelope_np[:, i], 
                                color='red', linewidth=2, linestyle='--', alpha=1)
                

        else:
            # Handle regular position matrix data with consistent binning
            final_pos_matrix = matrix_list[matrix_id]
            
            if isinstance(final_pos_matrix, tuple):
                pos_data = final_pos_matrix[1] if len(final_pos_matrix) > 1 else final_pos_matrix[0]
            else:
                pos_data = final_pos_matrix
            
            if hasattr(pos_data, 'numpy'):
                positions = pos_data.numpy()
            elif isinstance(pos_data, (list, tuple)):
                positions = np.array(pos_data)
            else:
                positions = pos_data
            
            if positions.ndim == 2:
                # Create density matrix using global position bins
                density_matrix_iter = np.zeros((len(position_bins)-1, min(positions.shape[0], len(reach_params_np))))
                for i in range(param_int): #min(positions.shape[0], len(reach_params_np))
                    agent_positions = positions[i, :]
                    valid_positions = agent_positions[~np.isnan(agent_positions)]
                    if len(valid_positions) > 0:
                        counts, _ = np.histogram(valid_positions, bins=position_bins)
                        density_matrix_iter[:, i] = counts
                
                # Add to global density matrix (only additive contributions)
                difference_matrix = np.clip(density_matrix_iter - density_matrix, 0, None)
                density_matrix += difference_matrix
                
                # Mark reach indices where this matrix adds new info beyond baseline
                if baseline_density is not None and difference_matrix.size > 0:
                    baseline_diff = np.clip(density_matrix_iter - baseline_density, 0, None)
                    new_cols = np.any(baseline_diff > 0, axis=0)
                    completion_mask[:len(new_cols)] |= new_cols
                
                
                
    
    # Create the final heatmap with consistent binning
    if plot_type == "heat":
        max_extremes = int(density_matrix.max())
        if max_extremes > 0:
            levels = np.arange(0, max_extremes + 2, 1)
            norm = mpl.colors.BoundaryNorm(levels, ncolors=256)
            
            im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                        norm=norm, origin='lower',
                        extent=[reach_params_np[0], reach_params_np[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')
        else:
            im = ax.imshow(density_matrix, aspect='auto', cmap=heat_cmap, 
                        origin='lower',
                        extent=[reach_params_np[0], reach_params_np[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')
        
        # Add colorbar
        max_extremes = int(density_matrix.max())
        tick_levels = np.arange(0, max_extremes + 2, 1)
        
        if axis_return == False:
            cbar = plt.colorbar(im, ax=ax, shrink=cbar_shrink, ticks=tick_levels)
            cbar.set_label('Number of Agents', fontsize=cbar_font_size)
            
            if cbar_center_labels and len(tick_levels) > 1:
                centered_positions = []
                for i in range(len(tick_levels) - 1):
                    centered_positions.append((tick_levels[i] + tick_levels[i + 1]) / 2)
                
                if len(centered_positions) > 0:
                    cbar.set_ticks(centered_positions)
                    centered_labels = [str(i) for i in range(len(centered_positions))]
                    cbar.set_ticklabels(centered_labels)
            else:
                cbar.set_ticks(tick_levels)
                cbar.set_ticklabels([str(int(level)) for level in tick_levels])
    
    # Add bifurcation critical values following project patterns
    if infl_type == 'gaussian' and show_pred == True:
        _, means, crit_stds = one_utils.critical_values_plot(
            num_agents=num_agents, bin_points=bin_points, 
            resource_distribution=resource_distribution, axis=ax, 
            reach_start=reach_start, reach_end=reach_end, 
            refinements=refinements, crit_cs=crit_cmap)
    else:
        # Use matrix_id == 1 (envelope data) to determine bifurcation information
        extreme_positions = matrix_list[1]
        # Use 2D reach_parameters for bifurcation_type_helper (expects [item_id][0] indexing)
        bifurcation_types = one_utils.bifurcation_type_helper(matrix=extreme_positions, reach_parameters=reach_parameters_2d)
        locs = torch.where(torch.round(extreme_positions['min'][:,0], decimals=2) != torch.round(extreme_positions['max'][:,0], decimals=2))
        
        # Filter out isolated elements - keep only sequences of 2 or more consecutive values
        if len(locs[0]) > 0:
            locs_list = locs[0].tolist()
            locs_list.sort()
            
            filtered_locs = []
            for i, val in enumerate(locs_list):
                # Check if value is part of a sequence (has consecutive neighbor)
                has_prev = (i > 0 and locs_list[i-1] == val - 1)
                has_next = (i < len(locs_list) - 1 and locs_list[i+1] == val + 1)
                
                # Keep value if it's part of a sequence (has at least one consecutive neighbor)
                if has_prev or has_next:
                    filtered_locs.append(val)
            
            locs = (torch.tensor(filtered_locs),)
        
        # Convert locs to a set of excluded indices
        excluded_indices = set(locs[0].tolist()) if len(locs[0]) > 0 else set()

        # Separate bifurcations by type and get classifications
        type1_bifurcations = []
        type2_bifurcations = []
        bifurcation_info = []

        # First, collect all valid bifurcations sorted by key
        all_bifurcations_by_key = []
        for key, value in bifurcation_types.items():
            if int(key) in excluded_indices:
                continue
            all_bifurcations_by_key.append((int(key), key, value))
        
        # Sort by key to process in order
        all_bifurcations_by_key.sort(key=lambda x: x[0])
        
        # Keys that form the left/right boundary of a cycle region must never be filtered out,
        # otherwise we lose the vertical lines at cycle start/end.
        cycle_boundary_keys = set()
        if excluded_indices:
            exc_sorted = sorted(excluded_indices)
            runs = []
            curr = [exc_sorted[0]]
            for j in range(1, len(exc_sorted)):
                if exc_sorted[j] == curr[-1] + 1:
                    curr.append(exc_sorted[j])
                else:
                    runs.append(curr)
                    curr = [exc_sorted[j]]
            runs.append(curr)
            for r in runs:
                k = min(r) - 1
                if k >= 0:
                    cycle_boundary_keys.add(k)
                k_right = max(r) + 1
                if k_right >= 0 and str(k_right) in bifurcation_types:
                    cycle_boundary_keys.add(k_right)
        
        # Filter out bifurcations that are too close to the previous one
        # Never filter cycle-boundary keys (vertical lines at cycle start/end).
        last_accepted_key = None
        for int_key, key, value in all_bifurcations_by_key:
            is_cycle_boundary = int_key in cycle_boundary_keys
            if not is_cycle_boundary and last_accepted_key is not None and (int_key - last_accepted_key) <= bifurcation_key_tolerance:
                continue
            
            last_accepted_key = int_key
            reach_param = value['reach_parameter']
            bif_type = value['type']
            classification = value['classification_new']
            
            bifurcation_info.append({
                'reach': reach_param,
                'type': bif_type,
                'classification': classification,
                'key': key
            })
            
            if bif_type == '1':
                type1_bifurcations.append(reach_param)
            else:
                type2_bifurcations.append(reach_param)

        # Sort bifurcation_info by reach parameter (left to right)
        bifurcation_info.sort(key=lambda x: x['reach'])

        xlim_start, xlim_end = reach_start, reach_end

        # Define color palette for shaded regions
        region_colors = plt.cm.Pastel1(np.linspace(0, 1, 9))

        # Create shaded regions from right to left
        boundaries = [xlim_start] + [info['reach'] for info in bifurcation_info] + [xlim_end]

        # Get reach parameters for cycle indices
        cycle_reach_params = []
        if len(excluded_indices) > 0:
            for idx in excluded_indices:
                cycle_reach_params.append(reach_parameters[idx].item() if hasattr(reach_parameters[idx], 'item') else reach_parameters[idx])

        # Shade regions and track legend patches
        from matplotlib.patches import Patch

        # First pass: collect all unique labels and assign colors
        label_to_color = {}
        all_labels = []
        final_boundaries = []
        final_labels = []

        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            
            # Check if any cycle reach values fall within this region
            cycles_in_region = []
            cycle_indices = []
            for idx, cycle_reach in enumerate(cycle_reach_params):
                if x_start < cycle_reach < x_end:
                    cycles_in_region.append(cycle_reach)
                    cycle_indices.append(idx)
            
            # Get the original region's classification
            if i == len(boundaries) - 2:
                original_label = f'$({num_agents})$'
            elif i < len(bifurcation_info):
                original_label = bifurcation_info[i]['classification']
            else:
                original_label = ''
            
            if cycles_in_region:
                sorted_pairs = sorted(zip(cycles_in_region, cycle_indices))
                cycles_in_region = [val for val, idx in sorted_pairs]
                cycle_indices = [idx for val, idx in sorted_pairs]
                
                min_idx = cycle_indices[0]
                max_idx = cycle_indices[-1]
                
                cycle_start = x_start
                cycle_end_raw = cycle_reach_params[min(len(cycle_reach_params) - 1, max_idx + 2)] if max_idx + 2 < len(cycle_reach_params) else cycle_reach_params[-1]
                cycle_end = min(x_end, cycle_end_raw)
                
                final_boundaries.append((cycle_start, cycle_end))
                final_labels.append('Cycles')
                
                if cycle_end < x_end:
                    final_boundaries.append((cycle_end, x_end))
                    final_labels.append(original_label)
            else:
                final_boundaries.append((x_start, x_end))
                final_labels.append(original_label)

        # Assign colors to labels with specific colors for certain patterns
        import matplotlib.colors as mcolors

        specific_colors = {
            f'$({num_agents})$': '#87CEEB',
            'Cycles': '#FFD700',
            '(2,1,1,2)': '#FF6B6B',
            '(1,1,1,1,1,1)': '#9370DB'
        }

        additional_colors = ['#98D8C8', '#F7B7A3', '#EA5F89', '#9D84B7', '#A8E6CF', 
                            '#FFD3B6', '#FFAAA5', '#FF8B94', '#C7CEEA', '#B5EAD7']

        color_index = 0
        for label in final_labels:
            if label and label not in label_to_color:
                if label in specific_colors:
                    label_to_color[label] = specific_colors[label]
                else:
                    label_to_color[label] = additional_colors[color_index % len(additional_colors)]
                    color_index += 1

        # Draw regions with consistent colors
        region_legend_items = []
        
        for i, (x_start, x_end) in enumerate(final_boundaries):
            label = final_labels[i]
            color = label_to_color.get(label, region_colors[0])
            
            zorder = 1 if label == 'Cycles' else 0
            span = ax.axvspan(x_start, x_end, alpha=0.1, color=color, zorder=zorder)
            
            if label:
                region_legend_items.append((i, label, span))
        
        # Add hatched regions only where new info is added beyond matrix 1 baseline
        # Determine which trajectory was completed based on the first region's equilibrium label
        completed_segments = []
        if np.any(completion_mask):
            true_indices = np.where(completion_mask)[0]
            run_start = true_indices[0]
            run_prev = true_indices[0]
            for idx in true_indices[1:]:
                if idx == run_prev + 1:
                    run_prev = idx
                else:
                    seg_end_idx = min(run_prev + 1, len(reach_params_np) - 1)
                    completed_segments.append((reach_params_np[run_start], reach_params_np[seg_end_idx]))
                    run_start = idx
                    run_prev = idx
            seg_end_idx = min(run_prev + 1, len(reach_params_np) - 1)
            completed_segments.append((reach_params_np[run_start], reach_params_np[seg_end_idx]))
        
        # Get the first region's label to determine which was completed
        first_region_label = final_labels[0] if len(final_labels) > 0 else ''
        
        import matplotlib.patches as mpatches
        
        # Determine which hatch to show based on the equilibrium label
        if '2,1' in first_region_label or '(2,1)' in first_region_label:
            # If equilibrium is (2,1), then (1,2) initial condition completed to this
            completed_label = 'Completed'
            completed_color = 'purple'
            completed_hatch = '\\\\'
            completed_patch_label = '(1,2) Completed'
        elif '1,2' in first_region_label or '(1,2)' in first_region_label:
            # If equilibrium is (1,2), then (2,1) initial condition completed to this
            completed_label = 'Completed'
            completed_color = 'green'
            completed_hatch = '/'
            completed_patch_label = '(2,1) Completed'
        else:
            # Default case - no specific completion pattern identified
            completed_label = None
            completed_color = None
            completed_hatch = None
            completed_patch_label = None
        
        # Draw hatch patches only on segments that add new info
        if completed_label is not None and completed_segments:
            for seg_idx, (seg_start, seg_end) in enumerate(completed_segments):
                if seg_end <= seg_start:
                    continue
                hatch_patch = mpatches.Rectangle(
                    (seg_start, 0),
                    seg_end - seg_start,
                    1,
                    fill=False,
                    hatch=completed_hatch * 2,
                    edgecolor=completed_color,
                    linewidth=0.5,
                    zorder=2,
                    label=completed_patch_label if seg_idx == 0 else None
                )
                ax.add_patch(hatch_patch)
        
        # Create ordered, deduplicated legend entries
        seen_labels_ordered = {}
        for order_idx, label, handle in region_legend_items:
            seen_labels_ordered[label] = (order_idx, handle)
        
        sorted_region_items = sorted(seen_labels_ordered.items(), key=lambda x: x[1][0], reverse=True)
        
        ordered_region_handles = []
        ordered_region_labels = []
        for label, (order_idx, handle) in sorted_region_items:
            ordered_region_handles.append(handle)
            ordered_region_labels.append(label)

        # Plot bifurcation lines
        for i, reach_param in enumerate(type1_bifurcations):
            ax.axvline(x=reach_param, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=10,
                        label=f'$\\sigma_{i+1}^1 = {reach_param:.4f}$')

        for j, reach_param in enumerate(type2_bifurcations):
            ax.axvline(x=reach_param, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=10,
                        label=f'$\\sigma_{j+1}^2 = {reach_param:.4f}$')

        if len(cycle_reach_params) > 0:
            cycle_end_param = max(cycle_reach_params)
            ax.axvline(x=cycle_end_param, color='purple', linestyle='-.', linewidth=2, alpha=0.5, zorder=10,
                        label=f'$\\sigma^{{cycle}} = {cycle_end_param:.4f}$')

        # Simplified legend handling
        handles, labels = ax.get_legend_handles_labels()

        bifurcation_items = []
        envelope_items = []
        completed_items = []

        for handle, label in zip(handles, labels):
            if 'sigma' in label.lower():
                bifurcation_items.append((handle, label))
            elif label in ['Upper envelope', 'Lower envelope', 'Upper', 'Lower']:
                envelope_items.append((handle, label))
            elif 'Completed' in label:
                completed_items.append((handle, label))

        envelope_handles = [h for h, l in envelope_items]
        envelope_labels = [l for h, l in envelope_items]
        
        completed_handles = [h for h, l in completed_items]
        completed_labels = [l for h, l in completed_items]

        combined_handles = ordered_region_handles + envelope_handles + completed_handles
        combined_labels = ordered_region_labels + envelope_labels + completed_labels

        if show_bif_labels == True:
            ax.legend(handles=combined_handles, labels=combined_labels, 
                        loc='upper right', 
                        fontsize=legend_font_size, title='Legend', framealpha=0.9)

        ax.set_xlim(reach_start, reach_end)

        # Create rectangle bifurcation plot in separate subplot
        if len(bifurcation_items) > 0:
            import matplotlib.patches as patches
            from matplotlib.lines import Line2D
            
            if show_bif_labels:
                ax_rect = fig.add_subplot(gs[1])
            else:
                ax_rect = fig.add_axes([0.52, 0.65, 0.45, 0.25])
            ax_rect.set_axis_off()
            
            rect_height = 0.4
            rect_y_start = 1
            sigma_min = reach_start
            sigma_max = reach_end
            rect_x_start = 0
            rect_total_width = 8.0
            
            def sigma_to_x(sigma):
                return rect_x_start + (sigma - sigma_min) / (sigma_max - sigma_min) * rect_total_width
            
            # Draw colored segments
            for i, (x_start_sigma, x_end_sigma) in enumerate(final_boundaries):
                segment_x_start = sigma_to_x(x_start_sigma)
                segment_x_end = sigma_to_x(x_end_sigma)
                segment_width = segment_x_end - segment_x_start
                
                label = final_labels[i]
                color = label_to_color.get(label, '#CCCCCC')
                
                rectangle = patches.Rectangle(
                    (segment_x_start, rect_y_start),
                    segment_width,
                    rect_height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax_rect.add_patch(rectangle)
            
            # Add hatched overlay for completed sections based on new info segments
            if completed_label is not None and completed_segments:
                for seg_start, seg_end in completed_segments:
                    if seg_end <= seg_start:
                        continue
                    segment_x_start = sigma_to_x(seg_start)
                    segment_x_end = sigma_to_x(seg_end)
                    segment_width = segment_x_end - segment_x_start
                    
                    hatch_rect = patches.Rectangle(
                        (segment_x_start, rect_y_start),
                        segment_width,
                        rect_height,
                        fill=False,
                        hatch=completed_hatch * 3,  # Repeat hatch pattern for visibility
                        edgecolor=completed_color,
                        linewidth=1,
                        zorder=5
                    )
                    ax_rect.add_patch(hatch_rect)
            
            # Combine all bifurcations for rectangle
            all_bifurcations = []
            for i, reach_param in enumerate(type1_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    all_bifurcations.append(('type1', i, reach_param))
            
            for j, reach_param in enumerate(type2_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    all_bifurcations.append(('type2', j, reach_param))
            
            if len(cycle_reach_params) > 0:
                cycle_end_param = max(cycle_reach_params)
                if sigma_min <= cycle_end_param <= sigma_max:
                    all_bifurcations.append(('cycle', 0, cycle_end_param))
            
            all_bifurcations.sort(key=lambda x: x[2], reverse=True)
            
            label_counter = 0
            type1_label_counter = 1
            type2_label_counter = 1
            for bif_type, idx, reach_param in all_bifurcations:
                x_pos = sigma_to_x(reach_param)
                
                if bif_type == 'type1':
                    color = 'red'
                    linestyle = '--'
                    alpha = 0.9
                    label_text = f'$\\sigma_{type1_label_counter}^1$'
                    type1_label_counter += 1
                elif bif_type == 'type2':
                    color = 'blue'
                    linestyle = ':'
                    alpha = 0.9
                    label_text = f'$\\sigma_{type2_label_counter}^2$'
                    type2_label_counter += 1
                else:
                    color = 'purple'
                    linestyle = '-.'
                    alpha = 0.5
                    label_text = '$\\sigma^{c}$'
                
                line_y_start = rect_y_start - 0.15
                line_y_end = rect_y_start + rect_height + 0.15
                ax_rect.plot([x_pos, x_pos], [line_y_start, line_y_end],
                           color=color, linestyle=linestyle, linewidth=3, alpha=alpha, zorder=10)
                
                if label_counter % 2 == 0:
                    label_y_pos = line_y_end + 0.05
                    va = 'bottom'
                else:
                    label_y_pos = line_y_start - 0.05
                    va = 'top'
                
                ax_rect.text(x_pos, label_y_pos, label_text,
                            fontsize=rect_label_font_size, color=color, fontweight='bold',
                            ha='center', va=va)
                
                label_counter += 1
            
            ax_rect.set_xlim(0, rect_x_start + rect_total_width)
            if show_bif_labels:
                ax_rect.set_ylim(0, 4)
            else:
                ax_rect.set_ylim(0.5, 2.0)
            ax_rect.set_xlabel('Sigma ($\\sigma$) - Agent Reach Parameter', fontsize=default_font_size)
            
            if show_bif_labels:
                ax_rect.set_title(r'Bifurcation Regions on $\sigma$', fontsize=title_font_size)
            else:
                title_x = rect_x_start + rect_total_width / 2
                title_y = rect_y_start + rect_height + 0.5
                ax_rect.text(title_x, title_y, r'Bifurcation Regions on $\sigma$',
                            fontsize=title_font_size, ha='center', va='bottom', fontweight='bold')
            
            if show_bif_labels:
                sigma_step = (sigma_max - sigma_min) / 10
                sigma_labels_vals = [sigma_min + i * sigma_step for i in range(11)]
                for sigma_val in sigma_labels_vals:
                    x_pos = sigma_to_x(sigma_val)
                    ax_rect.text(x_pos, rect_y_start - 0.45, f'{sigma_val:.2f}',
                                ha='center', va='top', fontsize=rect_sigma_font_size, color='black')
            
            # Create legend with sorted bifurcation values
            type1_sorted = sorted([(rp, 'red', '--') for rp in type1_bifurcations if sigma_min <= rp <= sigma_max], 
                                  key=lambda x: x[0], reverse=True)
            type2_sorted = sorted([(rp, 'blue', ':') for rp in type2_bifurcations if sigma_min <= rp <= sigma_max], 
                                  key=lambda x: x[0], reverse=True)
            
            legend_data = []
            
            for i, (reach_param, color, linestyle) in enumerate(type1_sorted):
                legend_data.append((reach_param, color, linestyle, f'$\\sigma_{i+1}^1 = {reach_param:.4f}$'))
            
            for j, (reach_param, color, linestyle) in enumerate(type2_sorted):
                legend_data.append((reach_param, color, linestyle, f'$\\sigma_{j+1}^2 = {reach_param:.4f}$'))
            
            if len(cycle_reach_params) > 0:
                cycle_end_param = max(cycle_reach_params)
                if sigma_min <= cycle_end_param <= sigma_max:
                    legend_data.append((cycle_end_param, 'purple', '-.', f'$\\sigma^{{c}} = {cycle_end_param:.4f}$'))
            
            legend_data.sort(key=lambda x: x[0], reverse=True)
            
            legend_elements = []
            for reach_param, color, linestyle, label in legend_data:
                legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=label))
            
            if show_bif_labels == True:
                ax_rect.legend(handles=legend_elements, loc='upper right', fontsize=rect_label_font_size, framealpha=0.9)
            else:
                num_sigma_items = len(legend_elements)
                num_region_items = len(ordered_region_labels)
                
                sigma_legend_height = max(0.15, 0.035 * num_sigma_items + 0.06)
                region_legend_height = max(0.15, 0.035 * (num_region_items // 3 + 1) + 0.06)
                
                sigma_legend_bottom = 0.65 - sigma_legend_height - 0.05
                ax_sigma_legend = fig.add_axes([0.52, sigma_legend_bottom, 0.45, sigma_legend_height])
                ax_sigma_legend.set_axis_off()
                
                ax_sigma_legend.legend(handles=legend_elements, 
                              loc='center',
                              ncol=1,
                              fontsize=rect_label_font_size, 
                              title='Bifurcation Values', 
                              title_fontsize=title_font_size - 2,
                              framealpha=0.9)
                
                region_legend_bottom = sigma_legend_bottom - region_legend_height - 0.05
                ax_legend = fig.add_axes([0.52, region_legend_bottom, 0.45, region_legend_height])
                ax_legend.set_axis_off()
                
                from matplotlib.patches import Patch as LegendPatch
                region_legend_elements = []
                for label_text in ordered_region_labels:
                    color = label_to_color.get(label_text, '#CCCCCC')
                    region_legend_elements.append(LegendPatch(facecolor=color, edgecolor='black', 
                                                              alpha=0.7, label=label_text))
                
                if envelope_labels:
                    region_legend_elements.append(Line2D([0], [0], color='orange', linestyle='--', 
                                                         linewidth=2, label='Upper envelope'))
                    region_legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                                         linewidth=2, label='Lower envelope'))
                
                # Add completed section hatching legend entry (only one based on equilibrium)
                if completed_label is not None:
                    region_legend_elements.append(LegendPatch(facecolor='none', edgecolor=completed_color, 
                                                              hatch=completed_hatch, alpha=0.7, label=completed_label))
                
                ax_legend.legend(handles=region_legend_elements, 
                              loc='center',
                              ncol=min(3, len(region_legend_elements)),
                              fontsize=legend_font_size, 
                              title='Region Legend', 
                              title_fontsize=title_font_size - 4,
                              framealpha=0.9)
            
            ax_rect.set_box_aspect(.5)
    
    # Optional vertical lines
    if optional_vline is not None:
        for vline_id, vline_val in enumerate(optional_vline):
            ax.axvline(x=vline_val, ymin=0, ymax=1, color='black', 
                      linestyle='dashed', alpha=0.7,
                      label=r'$\sigma^*_' + str(vline_id + 1) + r'=$' + str(np.around(vline_val, decimals=4)))
    
    ax.vlines(x=reach_parameters[param_int], ymin=0, ymax=1, colors='blue', linestyles='dashed',
              label='$\sigma^*_' + str(vline_id + 2) + r'=$' + str(np.around(reach_parameters[param_int].item(), decimals=4)))


    # Legend handling following project patterns - skip if combined legend already created
    if show_pred == True:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower center')
    
    # Title formatting following project patterns
    if short_title == True:
        title = 'Adaptive Agents Envelope'
    else:
        title = str(num_agents) + f' Adaptive Agents \n Complete'

    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + " " + title_addition
    
    ax.set_title(title, fontsize=title_font_size)
    
    if infl_type == 'gaussian':
        ax.set_xlabel(r"$\sigma$ (std)")
    else:
        ax.set_xlabel(r"$\sigma$")
    
    plt.ylim(0, 1)
    plt.ylabel("Agent Position")
    plt.tight_layout()
    plt.close()
    
    if axis_return:
        return ax
    else:
        return fig

def final_position_histogram_1d(num_agents: int,
                                domain_bounds: Tuple[float, float],
                                current_alpha: float,
                                reach_parameter: float,
                                final_pos_vector: np.ndarray,
                                title_ads: Optional[List[str]],
                                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}
                                ) -> matplotlib.figure.Figure:
    r"""
    Plot histogram of agent final positions at equilibrium in a 1D domain.
    
    Creates a histogram showing the distribution of agent equilibrium positions for
    a specific reach parameter and resource configuration. Useful for analyzing
    clustering patterns and position distributions at equilibrium.

    :param num_agents: Number of agents in the simulation.
    :type num_agents: int
    :param domain_bounds: Minimum and maximum bounds of the 1D domain.
    :type domain_bounds: Tuple[float, float]
    :param current_alpha: Current resource parameter value (e.g., mode separation :math:`\\alpha`).
    :type current_alpha: float
    :param reach_parameter: Influence reach parameter value (e.g., :math:`\\sigma`).
    :type reach_parameter: float
    :param final_pos_vector: Vector of final equilibrium positions for all agents.
    :type final_pos_vector: np.ndarray
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: Optional[List[str]]
    :param font: Font configuration dictionary with keys: 'default_size', 'cbar_size', 'title_size', 'legend_size', 'font_family'.
    :type font: dict
    
    :return: The generated matplotlib figure.
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
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.histplot(final_pos_vector,binwidth=.05)
    plt.ylabel('Number of players')
    plt.xlabel('Position')
    plt.xlim(domain_bounds[0],domain_bounds[1])
    title=str(num_agents)+r' Player count in positions for $\alpha=$'+str(current_alpha)+r',$\sigma=$'+str(reach_parameter)
    if len(title_ads)>0:
        for title_addition in title_ads:
            title=title+" "+title_addition
    plt.title(title, fontsize=title_font_size)
    plt.close()
    return fig

def plot_equilibrium_heatmap_1d(unique_results,
                             num_agents: int,
                             stability_analysis=None,
                             title_ads: List[str] = [],
                             font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'table_size':15,'label_size':10,'font_family': 'sans-serif',},
                             )-> matplotlib.figure.Figure:
    """Generate a heatmap showing equilibrium positions with player positions as axes and color."""
    
    
    font['font.family'] = font.get('font_family', 'sans-serif')
    cbar_font_size= font.get('cbar_size', 12)
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    table_font_size = font.get('table_size',12)
    label_font_size = font.get('label_size',10)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
   
    
    
    
    unique_positions = unique_results
    
    if len(unique_positions) == 0:
        print("No equilibrium positions to plot")
        return None
    
    # Extract positions for each player
    player_positions = []
    for equilibrium in unique_positions:
        if len(equilibrium) >= 3:  # Need at least 3 players for x, y, and color
            player_positions.append(equilibrium)
    
    if len(player_positions) == 0:
        print("Need at least 3 players for heatmap visualization")
        return None
    
    # Convert to numpy array for easier manipulation
    positions_array = np.array(player_positions)
    num_players = positions_array.shape[1]
    
    # Sort equilibria by Player 2 position (column 1), then by Player 1 position (column 0)
    # This creates consistent ordering for visualization and table display
    sort_indices = np.lexsort((positions_array[:, 0], positions_array[:, 1]))
    positions_array = positions_array[sort_indices]
    
    # Reorder unique_positions to match the sorted array
    unique_positions = [unique_positions[i] for i in sort_indices]
    
    # Create figure with subplots - heatmap on left, table on right
    fig = plt.figure(figsize=(20, 8))  # Increased width for stability column
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.05, wspace=0.15)
    
    # Create heatmap subplot
    ax_plot = fig.add_subplot(gs[0])
    
    # Use first player as x-axis, last player as y-axis, second player as color
    x_positions = positions_array[:, 0]  # Player 1
    y_positions = positions_array[:, -1]  # Last player
    color_values = positions_array[:, 1]  # Player 2 
    
    # Create scatter plot with color mapping
    scatter = ax_plot.scatter(x_positions, y_positions, c=color_values, 
                        cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=1, vmin=0, vmax=1)
    ax_plot.plot([0, 1], [0, 1], 'k--', alpha=0.2, label=rf'$x_1=x_{num_players}$')  # Diagonal line for reference
    ax_plot.plot([0, 1], [1, 0], '--', c='blue', alpha=0.2, label=rf'$x_1=1-x_{num_players}$')  # Anti-diagonal line for reference
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_plot)
    cbar.set_label('Player 2 Position', rotation=270, labelpad=15, fontsize=cbar_font_size)
    
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator
    cbar.update_ticks()

    # Customize plot
    ax_plot.set_xlabel('Player 1 Position', fontsize=default_font_size)
    ax_plot.set_ylabel(f'Player {num_players} Position', fontsize=default_font_size)
    ax_plot.set_title(f'Equilibrium Positions Heatmap\n({len(unique_positions)} equilibria found)', fontsize=title_font_size)
    ax_plot.tick_params(axis='both', which='major', labelsize=8)
    ax_plot.set_xlim(0, 1)
    ax_plot.set_ylim(0, 1)
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_aspect('equal')
    ax_plot.legend(fontsize=legend_font_size)
    
   
    # Add text annotations with improved positioning
    label_positions = []
    for i, (x, y, color_val) in enumerate(zip(x_positions, y_positions, color_values)):
        # Find good label position offset from data point
        label_x, label_y = one_utils.find_label_position(x, y, label_positions)
        label_positions.append((label_x, label_y))
        
        # Always draw connecting line since label is offset
        ax_plot.plot([x, label_x], [y, label_y], 'k-', alpha=0.4, linewidth=0.8)
        
        # Add label with smaller font and tighter bbox
        ax_plot.annotate(f'E{i+1}', (label_x, label_y), 
                   fontsize=label_font_size, fontweight='bold', color='white', 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8, edgecolor='none'))
        
    
    # Create table subplot
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    # Prepare table headers dynamically based on number of players
    if stability_analysis is not None:
        headers = ['ID'] + [f'P{j+1}' for j in range(num_players)] + ['Stability']
    else:
        headers = ['ID'] + [f'P{j+1}' for j in range(num_players)]
    
    # Prepare table data with stability analysis
    table_data = []
    for i, pos in enumerate(unique_positions):
        row_data = [f'E{i+1}']
        for j in range(num_players):
            row_data.append(f'{pos[j]:.4f}')
        
        # Add stability analysis if available
        if stability_analysis is not None:
            # Use original index to look up stability analysis before sorting
            original_index = sort_indices[i]
            equilibrium_key = f'E{original_index+1}'
            if equilibrium_key in stability_analysis:
                stability_type = stability_analysis[equilibrium_key]['stability_type']
                # Abbreviate long stability types for table display
                if stability_type == 'line-stable':
                    stability_abbrev = 'Line'
                elif 'stable' in stability_type.lower() and '(' in stability_type:
                    # Handle new format like '(2,1) stable' or '(1,1,1) stable'
                    stability_abbrev = stability_type.replace(' stable', '')
                elif stability_type == 'stable':
                    stability_abbrev = 'Stable'
                elif stability_type == 'unstable':
                    stability_abbrev = 'Unstable'
                else:
                    stability_abbrev = stability_type
                row_data.append(stability_abbrev)
            else:
                row_data.append('N/A')
        
        table_data.append(row_data)
    
    # Create table with adjusted sizing
    table = ax_table.table(cellText=table_data,
                          colLabels=headers,
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 0.95])  # Reduce table height for title space
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(table_font_size)  # Smaller font for better fit
    table.scale(1, 1.5)  # Compact scaling
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors for better readability
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Color-code stability column if present
    if stability_analysis is not None:
        stability_col_idx = len(headers) - 1
        for i in range(1, len(table_data) + 1):
            cell_text = table[(i, stability_col_idx)].get_text().get_text()
            if cell_text == 'Stable':
                table[(i, stability_col_idx)].set_facecolor('#c8e6c9')  # Light green
            elif cell_text in [f'(1,{num_agents-2},1)']:
                table[(i, stability_col_idx)].set_facecolor('#fff9c4')  # Light yellow
            elif '(' in cell_text:
                table[(i, stability_col_idx)].set_facecolor('#ffccbc')  # Light orange
            elif cell_text == 'Line':
                table[(i, stability_col_idx)].set_facecolor('#ffebee')  # Light red
            elif cell_text == 'Unstable':
                table[(i, stability_col_idx)].set_facecolor('#ffcdd2')  # Very light red
    
    # Position title with proper spacing
    ax_table.set_title('Equilibrium Values & Stability', 
                      fontsize=title_font_size, pad=5, y=1.)
    
    plt.tight_layout()
    plt.close()  
        
    return fig

def bifurication_rewards_stacked_rectangle_plot(idx,
                                                matrix,
                                                reward_bifurcation_matrix,
                                                max_reward=None,
                                                space=None,
                                                box_width=0.025,
                                                title_ads=[],
                                                font = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                                                show_sigma=False,
                                                hide_text: bool = False,
                                                show_column_labels: bool = True,
                                                show_outline: bool = True,
                                                show_total_outline: bool = True,
                                                show_label_box: bool = True,
                                                aspect: float = 1
                                                ) -> matplotlib.figure.Figure:
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    label_font_size = font.get('label_size',10)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(aspect)
    reach_parameter=reward_bifurcation_matrix['reach_parameters'][idx].item()
    if reward_bifurcation_matrix['same'][idx] == 1:
        # No bifurcation case - use max positions
        positions_array = torch.round(matrix['max'][idx], decimals=2).numpy()
        rewards_array = reward_bifurcation_matrix['max'][idx].numpy()
        
        # Group agents by position
        from collections import defaultdict
        position_groups = defaultdict(list)
        for agent_idx, (pos, reward) in enumerate(zip(positions_array, rewards_array)):
            position_groups[pos].append(reward)
        
        # Draw stacked rectangles for each position
        colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Color palette for different agents
        box_width = box_width
        total_reward_list = []
        column_info = []  # Store (pos, total_reward, n_agents) for label placement
        
        for pos_idx, (pos, rewards) in enumerate(sorted(position_groups.items())):
            n_agents = len(rewards)
            total_reward = sum(rewards)
            total_reward_list.append(total_reward)
            column_info.append((pos, total_reward, n_agents))
            segment_height = total_reward / n_agents
            
            # Draw each agent's segment
            for i, reward in enumerate(rewards):
                bottom = i * segment_height
                height = segment_height
                
                # Create rectangle
                rect = plt.Rectangle((pos - box_width/2, bottom), box_width, height,
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='black' if show_outline else 'none',
                                    linewidth=1.5 if show_outline else 0,
                                    alpha=0.8)
                ax.add_patch(rect)
            
            # Draw outline box around all segments
            if show_total_outline:
                rect_outline = plt.Rectangle((pos - box_width/2, 0), box_width, total_reward,
                                            facecolor='none',
                                            edgecolor='darkblue',
                                            linewidth=2.5)
                ax.add_patch(rect_outline)
        
        # Add labels with connecting lines and overlap avoidance
        calculated_max_reward = max(total_reward_list)
        
        if show_column_labels:
            label_positions = []  # Track placed label bounding boxes (x, y, width, height)
            
            # Helper function to estimate label dimensions based on text content and font
            def estimate_label_dimensions(text, fontsize, bbox_pad=0.3):
                """Estimate label box dimensions in data coordinates based on text content."""
                char_width = 0.008 * (fontsize / 10.0)
                line_height = 0.025 * (fontsize / 10.0)
                text_width = len(text) * char_width
                text_height = line_height
                pad_data = bbox_pad * 0.015 * (fontsize / 10.0)
                total_width = text_width + 2 * pad_data
                total_height = text_height + 2 * pad_data
                return total_width, total_height
            
            # Label box styling
            label_bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='darkblue', linewidth=1.5, alpha=0.9)
            label_bbox_pad = 0.3  # Must match the pad value in label_bbox_props
            
            # Pre-calculate sigma box position to avoid overlaps with it
            sigma_box_x = 0.98
            sigma_box_y = (max_reward if max_reward is not None else calculated_max_reward) * .95
            # Estimate sigma box dimensions based on its text content
            sigma_text = f'$\\sigma = {reach_parameter:.3f}$'
            sigma_box_width, sigma_box_height = estimate_label_dimensions(sigma_text, default_font_size, bbox_pad=0.5)
            # Add sigma box to label_positions to avoid overlaps
            label_positions.append((sigma_box_x, sigma_box_y, sigma_box_width, sigma_box_height))
            
            for label_idx, (pos, total_reward, n_agents) in enumerate(column_info):
                # Calculate initial label position at top of column
                if space is not None:
                    base_y = total_reward + space
                else:
                    base_y = total_reward * 1.10
                
                # Calculate this label's dimensions based on its text content
                label_text = f'n={n_agents}'
                current_label_width, current_label_height = estimate_label_dimensions(
                    label_text, label_font_size, bbox_pad=label_bbox_pad
                )
                
                # Find non-overlapping position for label
                label_x = pos
                label_y = base_y
                
                # Check for overlaps and adjust position (try vertical first, then horizontal)
                overlap_found = True
                attempts = 0
                max_attempts = 50
                y_step = current_label_height * 0.8
                # Alternate direction based on label index: even goes right first, odd goes left first
                if label_idx % 2 == 0:
                    x_shifts = [0.05, 0.10, -0.05, -0.10]  # Right first
                else:
                    x_shifts = [-0.05, -0.10, 0.05, 0.10]  # Left first
                x_shift_idx = 0
                
                while overlap_found and attempts < max_attempts:
                    overlap_found = False
                    
                    # Calculate current label's bounding box edges
                    curr_left = label_x - current_label_width / 2
                    curr_right = label_x + current_label_width / 2
                    curr_bottom = label_y  # va='bottom' means y is the bottom edge
                    curr_top = label_y + current_label_height
                    
                    for (lx, ly, lw, lh) in label_positions:
                        exist_left = lx - lw / 2
                        exist_right = lx + lw / 2
                        exist_bottom = ly
                        exist_top = ly + lh
                        
                        padding = 0.1
                        
                        x_overlap = (curr_left - padding) < exist_right and exist_left < (curr_right + padding)
                        y_overlap = (curr_bottom - padding) < exist_top and exist_bottom < (curr_top + padding)
                        
                        if x_overlap or y_overlap:
                            overlap_found = True
                            x_shift_idx = (x_shift_idx + 1) % len(x_shifts)
                            label_x = pos + x_shifts[x_shift_idx]
                            label_y = base_y + (attempts // len(x_shifts)) * y_step
                            
                    attempts += 1
                
                # Store this label's position with its actual dimensions
                label_positions.append((label_x, label_y, current_label_width, current_label_height))
                
                # Draw connecting line from label to top of column
                if show_label_box:
                    line_end_y = total_reward + (space * 0.1 if space else total_reward * 0.02)
                    ax.plot([pos, label_x], [line_end_y, label_y], 
                           color='darkblue', linewidth=1, alpha=0.6, linestyle='-')
                    
                    # Add the label with text box
                    ax.text(label_x, label_y, f'n={n_agents}',
                        ha='center', va='bottom', fontsize=label_font_size, color='darkblue', 
                        fontweight='bold', bbox=label_bbox_props)
        
        # Build title with type of bifurcation
        title=f'Agent Reward Distribution'
        if len(title_ads)>0:
            for title_addition in title_ads:
                title=title+" "+title_addition
        ax.set_title(title, fontsize=title_font_size)
        
        # Add sigma parameter box centered under the title, inside the plot
        if show_sigma:
            from matplotlib.patches import FancyBboxPatch
            textstr = f'$\\sigma = {reach_parameter:.3f}$'
            props = dict(boxstyle='round,pad=0.5', facecolor='wheat', edgecolor='black', linewidth=2, alpha=0.9)
            sigma_y_pos = (max_reward if max_reward is not None else calculated_max_reward) * 0.92
            ax.text(0.5, sigma_y_pos, textstr, transform=ax.transData,
                    fontsize=default_font_size, verticalalignment='top', horizontalalignment='center',
                    bbox=props)
        ax.set_xlabel('Agent Position', fontsize=default_font_size)
        ax.set_ylabel('Total Reward (Stacked)', fontsize=default_font_size)
        ax.set_xlim(-0.05, 1.05)
        if max_reward is not None:
            ax.set_ylim(0, max_reward)
        else:
            ax.set_ylim(0, calculated_max_reward*1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Bifurcation case - show both max and min equilibria
        positions_max = torch.round(matrix['max'][idx], decimals=2).numpy()
        rewards_max = reward_bifurcation_matrix['max'][idx].numpy()
        positions_min = torch.round(matrix['min'][idx], decimals=2).numpy()
        rewards_min = reward_bifurcation_matrix['min'][idx].numpy()
        
        # Group agents by position for both equilibria
        from collections import defaultdict
        position_groups_max = defaultdict(list)
        for pos, reward in zip(positions_max, rewards_max):
            position_groups_max[pos].append(reward)
        
        position_groups_min = defaultdict(list)
        for pos, reward in zip(positions_min, rewards_min):
            position_groups_min[pos].append(reward)
        
        # Draw stacked rectangles
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        box_width = 0.012
        offset = box_width * 0.6
        total_reward_list = []
        max_column_info = []  # Store (actual_x, total_reward, n_agents, color) for label placement
        min_column_info = []
        
        # Max equilibrium (left side)
        for pos, rewards in sorted(position_groups_max.items()):
            n_agents = len(rewards)
            total_reward = sum(rewards)
            total_reward_list.append(total_reward)
            actual_x = pos - offset
            max_column_info.append((actual_x, total_reward, n_agents, 'blue'))
            segment_height = total_reward / n_agents
            
            for i, reward in enumerate(rewards):
                bottom = i * segment_height
                height = segment_height
                
                rect = plt.Rectangle((pos - offset - box_width/2, bottom), box_width, height,
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='blue' if show_outline else 'none',
                                    linewidth=1.2 if show_outline else 0,
                                    alpha=0.7)
                ax.add_patch(rect)
                
                ax.text(pos - offset, bottom + height/2, f'{reward:.3f}',
                    ha='center', va='center', fontsize=7, color='darkblue')
            
            # Outline
            if show_total_outline:
                rect_outline = plt.Rectangle((pos - offset - box_width/2, 0), box_width, total_reward,
                                            facecolor='none',
                                            edgecolor='blue',
                                            linewidth=2)
                ax.add_patch(rect_outline)
        
        # Min equilibrium (right side)
        for pos, rewards in sorted(position_groups_min.items()):
            n_agents = len(rewards)
            total_reward = sum(rewards)
            segment_height = total_reward / n_agents
            total_reward_list.append(total_reward)
            actual_x = pos + offset
            min_column_info.append((actual_x, total_reward, n_agents, 'red'))
            
            for i, reward in enumerate(rewards):
                bottom = i * segment_height
                height = segment_height
                
                rect = plt.Rectangle((pos + offset - box_width/2, bottom), box_width, height,
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='red' if show_outline else 'none',
                                    linewidth=1.2 if show_outline else 0,
                                    alpha=0.7)
                ax.add_patch(rect)
                
                ax.text(pos + offset, bottom + height/2, f'{reward:.3f}',
                    ha='center', va='center', fontsize=7, color='darkred')
            
            # Outline
            if show_total_outline:
                rect_outline = plt.Rectangle((pos + offset - box_width/2, 0), box_width, total_reward,
                                            facecolor='none',
                                            edgecolor='red',
                                            linewidth=2)
                ax.add_patch(rect_outline)
        
        # Add labels with connecting lines and overlap avoidance
        calculated_max_reward = max(total_reward_list)
        
        if show_column_labels:
            label_positions = []  # Track placed label bounding boxes (x, y, width, height)
            
            # Helper function to estimate label dimensions based on text content and font
            def estimate_label_dimensions(text, fontsize, bbox_pad=0.2):
                """Estimate label box dimensions in data coordinates based on text content."""
                char_width = 0.008 * (fontsize / 10.0)
                line_height = 0.025 * (fontsize / 10.0)
                text_width = len(text) * char_width
                text_height = line_height
                pad_data = bbox_pad * 0.015 * (fontsize / 10.0)
                total_width = text_width + 2 * pad_data
                total_height = text_height + 2 * pad_data
                return total_width, total_height
            
            label_bbox_pad = 0.2  # Must match the pad value in label_bbox_props
            
            # Pre-calculate sigma box position to avoid overlaps with it
            sigma_box_x = 0.98
            sigma_box_y = (max_reward if max_reward is not None else calculated_max_reward) * .95
            sigma_text = f'$\\sigma = {reach_parameter:.3f}$'
            sigma_box_width, sigma_box_height = estimate_label_dimensions(sigma_text, 20, bbox_pad=0.5)
            label_positions.append((sigma_box_x, sigma_box_y, sigma_box_width, sigma_box_height))
            
            # Combine all columns for label placement
            all_columns = max_column_info + min_column_info
            
            for label_idx, (actual_x, total_reward, n_agents, color) in enumerate(all_columns):
                base_y = -0.04
                
                label_bbox_props = dict(boxstyle='round,pad=0.2', facecolor='white', 
                                        edgecolor=color, linewidth=1.5, alpha=0.9)
                
                label_text = f'n={n_agents}'
                current_label_width, current_label_height = estimate_label_dimensions(
                    label_text, 9, bbox_pad=label_bbox_pad
                )
                
                label_x = actual_x
                label_y = base_y
                
                overlap_found = True
                attempts = 0
                max_attempts = 50
                y_step = current_label_height * 0.8
                if label_idx % 2 == 0:
                    x_shifts = [0, 0.03, 0.06, -0.03, -0.06]
                else:
                    x_shifts = [0, -0.03, -0.06, 0.03, 0.06]
                x_shift_idx = 0
                
                while overlap_found and attempts < max_attempts:
                    overlap_found = False
                    
                    curr_left = label_x - current_label_width / 2
                    curr_right = label_x + current_label_width / 2
                    curr_top = label_y
                    curr_bottom = label_y - current_label_height
                    
                    for (lx, ly, lw, lh) in label_positions:
                        exist_left = lx - lw / 2
                        exist_right = lx + lw / 2
                        exist_top = ly
                        exist_bottom = ly - lh
                        
                        padding = 0.02
                        
                        x_overlap = (curr_left - padding) < exist_right and exist_left < (curr_right + padding)
                        y_overlap = (curr_bottom - padding) < exist_top and exist_bottom < (curr_top + padding)
                        
                        if x_overlap and y_overlap:
                            overlap_found = True
                            if attempts < 8:
                                label_y -= y_step
                            else:
                                x_shift_idx = (x_shift_idx + 1) % len(x_shifts)
                                label_x = actual_x + x_shifts[x_shift_idx]
                                label_y = base_y - (attempts // len(x_shifts)) * y_step
                            break
                    attempts += 1
                
                label_positions.append((label_x, label_y, current_label_width, current_label_height))
                
                if show_label_box:
                    ax.plot([actual_x, label_x], [0, label_y], 
                           color=color, linewidth=1, alpha=0.6, linestyle='-')
                    
                    ax.text(label_x, label_y, f'n={n_agents}',
                        ha='center', va='top', fontsize=9, color=color, fontweight='bold',
                        bbox=label_bbox_props)
        
        # Build title with type of bifurcation
        title = f'Agent Reward Distribution'
        if len(title_ads) > 0:
            for title_addition in title_ads:
                title = title + " " + title_addition
        ax.set_title(title, fontsize=title_font_size)
        
        # Add sigma parameter box centered under the title, inside the plot
        if show_sigma:
            from matplotlib.patches import FancyBboxPatch
            textstr = f'$\\sigma = {reach_parameter:.3f}$'
            props = dict(boxstyle='round,pad=0.5', facecolor='wheat', edgecolor='black', linewidth=2, alpha=0.9)
            sigma_y_pos = (max_reward if max_reward is not None else calculated_max_reward) * 0.92
            ax.text(0.5, sigma_y_pos, textstr, transform=ax.transData,
                    fontsize=default_font_size, verticalalignment='top', horizontalalignment='center',
                    bbox=props)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='white', edgecolor='blue', linewidth=2, label='Max Equilibrium'),
                        Patch(facecolor='white', edgecolor='red', linewidth=2, label='Min Equilibrium')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_font_size)
        ax.set_xlabel('Agent Position', fontsize=default_font_size)
        ax.set_ylabel('Total Reward (Stacked)', fontsize=default_font_size)
        ax.set_xlim(-0.05, 1.05)
        if max_reward is not None:
            ax.set_ylim(0, max_reward)
        else:
            ax.set_ylim(0, calculated_max_reward*1.2)
        ax.grid(True, alpha=0.3, axis='y')

    if hide_text:
        import matplotlib.text as _mtext
        for _t in fig.findobj(match=_mtext.Text):
            _t.set_alpha(0)
    plt.tight_layout()
    plt.close()
    return fig

def pos_rewards_stacked_rectangle_plot(idx,
                                       matrix,
                                       reward_matrix,
                                       max_reward=None,
                                       space=None,
                                       box_width=0.025,
                                       title_ads: List[str] = [],
                                       font={'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}):
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    label_font_size = font.get('label_size',10)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    positions_array = torch.round(matrix[idx], decimals=2).numpy()
    rewards_array = reward_matrix[idx].numpy()
    
    # Group agents by position
    from collections import defaultdict
    position_groups = defaultdict(list)
    for agent_idx, (pos, reward) in enumerate(zip(positions_array, rewards_array)):
        position_groups[pos].append(reward)
    
    # Draw stacked rectangles for each position
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Color palette for different agents
    box_width = box_width
    total_reward_list=[]    
    for pos_idx, (pos, rewards) in enumerate(sorted(position_groups.items())):
        n_agents = len(rewards)
        total_reward = sum(rewards)
        total_reward_list.append(total_reward)
        segment_height = total_reward / n_agents
        
        # Draw each agent's segment
        for i, reward in enumerate(rewards):
            bottom = i * segment_height
            height = segment_height
            
            # Create rectangle
            rect = plt.Rectangle((pos - box_width/2, bottom), box_width, height,
                                facecolor=colors[i % len(colors)],
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=0.8)
            ax.add_patch(rect)
            
            
        
        # Add position label at top
        if space is not None:
            y=total_reward+space
        else:
            y=total_reward+.05*total_reward
        ax.text(pos, y, f'n={n_agents}',
            ha='center', va='top', fontsize=label_font_size, color='darkblue', fontweight='bold')
        
        # Draw outline box around all segments
        rect_outline = plt.Rectangle((pos - box_width/2, 0), box_width, total_reward,
                                    facecolor='none',
                                    edgecolor='darkblue',
                                    linewidth=2.5)
        ax.add_patch(rect_outline)
    
    # Add parameter box in upper right corner
    from matplotlib.patches import FancyBboxPatch
    calculated_max_reward = max(total_reward_list)
    #textstr = f'$\\sigma = {reach_parameter:.3f}$'
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', edgecolor='black', linewidth=2, alpha=0.9)
    # Use provided max_reward if available, otherwise use calculated
    text_y_pos = (max_reward if max_reward is not None else calculated_max_reward) * .95
    #ax.text(0.98, text_y_pos, textstr, transform=ax.transData,
    #        fontsize=20, verticalalignment='top', horizontalalignment='right',
    #        bbox=props)
    title=f'Agent Reward Distribution'
    if len(title_ads)>0:
        for title_addition in title_ads:
            title=title+" "+title_addition
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel('Agent Position', fontsize=default_font_size)
    ax.set_ylabel('Total Reward (Stacked)', fontsize=default_font_size)
    ax.set_xlim(-0.05, 1.05)
    if max_reward is not None:
        ax.set_ylim(0, max_reward)
    else:
        ax.set_ylim(0, calculated_max_reward*1.2)
    ax.grid(True, alpha=0.3, axis='y')


    plt.tight_layout()
    plt.close()
    return fig

def bifurcation_tree_plot_with_images(main_matrix,
                                        left_matrices,
                                        right_matrices, 
                                        num_agents,
                                        reach_parameters, 
                                        reach_start,
                                        reach_end,
                                        node_images=None,
                                        label_to_color=None, 
                                        figsize=(20, 24),
                                        font={'default_size': 12, 'title_size': 16, 'font_family': 'sans-serif'},
                                        image_zoom=0.15,
                                        show_labels=True,
                                        image_offset=(0, 0),
                                        branch_spacing=1.5,
                                        label_offset=0.7,
                                        hide_text: bool = False):
    """
    NetworkX-based hierarchical tree visualization for bifurcation structures
    with support for placing custom figures/images on top of nodes.
    
    Parameters:
    -----------
    main_matrix : dict
        Main bifurcation matrix containing 'max', 'min', etc.
    left_matrices : list of dict
        List of matrices for left branches
    right_matrices : list of dict
        List of matrices for right branches
    reach_parameters : torch.Tensor
        Reach parameters for each matrix
    num_agents : int
        Number of agents in the system
    reach_start : float
        Starting sigma value
    reach_end : float
        Ending sigma value
    node_images : dict, optional
        Dictionary mapping node labels to image paths or matplotlib figures.
        Keys can be:
        - Exact node names (e.g., '$(6)$_m', '(3,3)_l0')
        - Display labels (e.g., '$(6)$', '(3,3)') - will apply to all nodes with that label
        - Branch-specific: ('$(6)$', 'main'), ('(3,3)', 'left'), etc.
        Values can be:
        - String path to an image file
        - matplotlib.figure.Figure object
        - numpy array (image data)
        - PIL Image object
    label_to_color : dict, optional
        Mapping of classification labels to colors
    figsize : tuple
        Figure size (width, height)
    font_size : int
        Font size for labels
    image_zoom : float
        Zoom factor for images (default 0.15)
    show_labels : bool
        Whether to show text labels below images (default True)
    image_offset : tuple
        (x, y) offset for image placement relative to node center
    branch_spacing : float
        Horizontal distance between branches (default 1.5)
    label_offset : float
        Horizontal offset for text labels to the left of nodes (default 0.7)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    node_positions : dict mapping node names to (x, y) positions for further customization
    """
    
    # Process all matrices - get both labels and boundaries (sigma ranges)
    main_proc = one_utils.process_matrix_tree(main_matrix, num_agents=num_agents, reach_parameters=reach_parameters, reach_start=reach_start, reach_end=reach_end)
    left_proc_list = [one_utils.process_matrix_tree(mat, num_agents=num_agents, reach_parameters=reach_parameters, reach_start=reach_start, reach_end=reach_end) for mat in left_matrices]
    right_proc_list = [one_utils.process_matrix_tree(mat, num_agents=num_agents, reach_parameters=reach_parameters, reach_start=reach_start, reach_end=reach_end) for mat in right_matrices]
    
    # Extract labels and boundaries (reversed for tree display)
    main_labels = main_proc['labels'][::-1]
    main_boundaries = main_proc['boundaries'][::-1]
    left_labels_list = [proc['labels'][::-1] for proc in left_proc_list]
    left_boundaries_list = [proc['boundaries'][::-1] for proc in left_proc_list]
    right_labels_list = [proc['labels'][::-1] for proc in right_proc_list]
    right_boundaries_list = [proc['boundaries'][::-1] for proc in right_proc_list]
    
    # Build directed graph
    G = nx.DiGraph()
    
    # Track label counts per branch to handle duplicates
    label_counts = {}
    
    
    
    # Create the main branch
    id = 0
    prev_main_node = None
    main_node_list = []
    for label in main_labels:
        main_node = one_utils.make_unique_node_name(label, '_m', label_counts)
        # Get sigma range for this node
        sigma_range = main_boundaries[id] if id < len(main_boundaries) else (None, None)
        G.add_node(main_node, labels=main_labels[id], branch_type="main", display_label=label, sigma_range=sigma_range)
        main_node_list.append(main_node)
        if prev_main_node is not None:
            G.add_edge(prev_main_node, main_node)
        prev_main_node = main_node
        id += 1
    
    # Add left branches
    prev_labels = main_labels
    left_node_lists = []  # Track nodes per left branch
    for i, left_labels in enumerate(left_labels_list):
        id = 0
        new_labels, j = one_utils.get_new_labels(prev_labels, left_labels)
        
        # Find the parent node - it should be on main branch (for i=0) or previous left branch
        if j > 0:
            parent_label = prev_labels[j-1]
            if i == 0:
                # Parent should be on main branch
                prev_node = one_utils.find_node_by_label(parent_label,G, ['main'])
            else:
                # Parent should be on previous left branch, or main if not found
                prev_node = one_utils.find_node_by_label(parent_label,G, ['left', 'main'])
            
            if prev_node is None:
                print(f"Warning: Could not find parent node for label '{parent_label}' when building left branch {i}")
                # Try to find ANY node with this label
                prev_node = one_utils.find_node_by_label(parent_label,G)
                if prev_node is None:
                    print(f"  Creating dummy parent node")
                    prev_node = f'{parent_label}_m'
                    G.add_node(prev_node, labels=parent_label, branch_type="main", display_label=parent_label)
        elif j == -1 and len(left_labels) > 0:
            # Identical sequences – add branch-specific node for the last label so it uses
            # the branch matrix's own image rather than inheriting the main branch's image.
            last_label = left_labels[-1]
            if i == 0:
                prev_node = one_utils.find_node_by_label(last_label, G, ['main'])
            else:
                prev_node = one_utils.find_node_by_label(last_label, G, ['left', 'main'])
            if prev_node is None:
                print(f"Warning: Could not find parent for identical left branch {i}, skipping")
                prev_labels = left_labels
                left_node_lists.append([])
                continue
        else:
            # No divergence point found (j == 0), skip this branch
            print(f"Warning: No divergence found for left branch {i}, skipping")
            prev_labels = left_labels
            left_node_lists.append([])
            continue

        # Get boundaries for this left branch
        left_boundaries = left_boundaries_list[i] if i < len(left_boundaries_list) else []

        # Include the shared parent label as the FIRST node of the new branch so each
        # branch generates its own image for the branch-point state from its own matrix,
        # preventing cross-branch image inheritance.
        if j > 0 and j <= len(left_labels):
            branch_labels = [prev_labels[j-1]] + list(new_labels)
            start_boundary_offset = j - 1
        elif j == -1 and len(left_labels) > 0:
            branch_labels = [left_labels[-1]]
            start_boundary_offset = len(left_labels) - 1
        else:
            branch_labels = list(new_labels)
            start_boundary_offset = j

        branch_nodes = []
        for k, label in enumerate(branch_labels):
            node = one_utils.make_unique_node_name(label, f'_l{i}', label_counts)
            # Get sigma range for this node from left boundaries
            boundary_idx = start_boundary_offset + k
            sigma_range = left_boundaries[boundary_idx] if boundary_idx < len(left_boundaries) else (None, None)
            # First node of a j>0 branch is the shared parent label — mark as branch-point
            # so it renders tiny and gets covered by the next (unique) node at the same position.
            is_bp = (j > 0 and k == 0)
            G.add_node(node, labels=label, branch_type="left", display_label=label, sigma_range=sigma_range, is_branch_point_node=is_bp)
            branch_nodes.append(node)
            G.add_edge(prev_node, node)
            prev_node = node
            id += 1

        left_node_lists.append(branch_nodes)
        prev_labels = left_labels
    
    # Add right branches
    prev_labels = main_labels
    right_node_lists = []  # Track nodes per right branch
    for i, right_labels in enumerate(right_labels_list):
        id = 0
        new_labels, j = one_utils.get_new_labels(prev_labels, right_labels)
        
        # Find the parent node - it should be on main branch (for i=0) or previous right branch
        if j > 0:
            parent_label = prev_labels[j-1]
            if i == 0:
                # Parent should be on main branch
                prev_node = one_utils.find_node_by_label(parent_label,G, ['main'])
            else:
                # Parent should be on previous right branch, or main if not found
                prev_node = one_utils.find_node_by_label(parent_label,G, ['right', 'main'])
            
            if prev_node is None:
                print(f"Warning: Could not find parent node for label '{parent_label}' when building right branch {i}")
                # Try to find ANY node with this label
                prev_node = one_utils.find_node_by_label(parent_label,G)
                if prev_node is None:
                    print(f"  Creating dummy parent node")
                    prev_node = f'{parent_label}_m'
                    G.add_node(prev_node, labels=parent_label, branch_type="main", display_label=parent_label)
        elif j == -1 and len(right_labels) > 0:
            # Identical sequences – add branch-specific node for the last label so it uses
            # the branch matrix's own image rather than inheriting the main branch's image.
            last_label = right_labels[-1]
            if i == 0:
                prev_node = one_utils.find_node_by_label(last_label, G, ['main'])
            else:
                prev_node = one_utils.find_node_by_label(last_label, G, ['right', 'main'])
            if prev_node is None:
                print(f"Warning: Could not find parent for identical right branch {i}, skipping")
                prev_labels = right_labels
                right_node_lists.append([])
                continue
        else:
            # No divergence point found (j == 0), skip this branch
            print(f"Warning: No divergence found for right branch {i}, skipping")
            prev_labels = right_labels
            right_node_lists.append([])
            continue

        # Get boundaries for this right branch
        right_boundaries = right_boundaries_list[i] if i < len(right_boundaries_list) else []

        # Include the shared parent label as the FIRST node of the new branch so each
        # branch generates its own image for the branch-point state from its own matrix,
        # preventing cross-branch image inheritance.
        if j > 0 and j <= len(right_labels):
            branch_labels = [prev_labels[j-1]] + list(new_labels)
            start_boundary_offset = j - 1
        elif j == -1 and len(right_labels) > 0:
            branch_labels = [right_labels[-1]]
            start_boundary_offset = len(right_labels) - 1
        else:
            branch_labels = list(new_labels)
            start_boundary_offset = j

        branch_nodes = []
        for k, label in enumerate(branch_labels):
            node = one_utils.make_unique_node_name(label, f'_r{i}', label_counts)
            # Get sigma range for this node from right boundaries
            boundary_idx = start_boundary_offset + k
            sigma_range = right_boundaries[boundary_idx] if boundary_idx < len(right_boundaries) else (None, None)
            # First node of a j>0 branch is the shared parent label — mark as branch-point
            # so it renders tiny and gets covered by the next (unique) node at the same position.
            is_bp = (j > 0 and k == 0)
            G.add_node(node, labels=label, branch_type="right", display_label=label, sigma_range=sigma_range, is_branch_point_node=is_bp)
            branch_nodes.append(node)
            G.add_edge(prev_node, node)
            prev_node = node
            id += 1

        right_node_lists.append(branch_nodes)
        prev_labels = right_labels
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    
    # Manual hierarchical layout with vertical main branches
    pos = {}
    
    # Get all nodes by branch type
    main_nodes = [n for n in G.nodes() if G.nodes[n].get('branch_type') == 'main']
    left_nodes = [n for n in G.nodes() if G.nodes[n].get('branch_type') == 'left']
    right_nodes = [n for n in G.nodes() if G.nodes[n].get('branch_type') == 'right']
    
    # Position all main nodes vertically down the center
    for i, node in enumerate(main_nodes):
        pos[node] = (0, -i * 2)
    
    # Build adjacency list for quick parent lookup
    parent_map = {}
    children_map = {}
    for edge in G.edges():
        parent_map[edge[1]] = edge[0]
        if edge[0] not in children_map:
            children_map[edge[0]] = []
        children_map[edge[0]].append(edge[1])
    
    # Use topological sort to ensure parents are always positioned before children
    # Process all nodes that still need positions using iterative approach
    all_branch_nodes = left_nodes + right_nodes
    
    # Track y-offset per branch to stack nodes vertically within same branch
    branch_y_offsets = {}
    
    # Keep iterating until all nodes are positioned
    max_iterations = len(all_branch_nodes) * 2  # Safety limit
    iteration = 0
    
    while len(pos) < len(G.nodes()) and iteration < max_iterations:
        iteration += 1
        positioned_this_round = False
        
        for node in all_branch_nodes:
            if node in pos:
                continue
                
            parent = parent_map.get(node)
            if parent and parent in pos:
                parent_x, parent_y = pos[parent]
                
                if '_l' in node:
                    # Extract branch index - handle cases like '(4,2)_l1' or '(4,2)_1_l1'
                    parts = node.rsplit('_l', 1)
                    branch_idx = int(parts[-1])
                    x_offset = -branch_spacing - (branch_idx * branch_spacing)
                    branch_key = ('left', branch_idx)
                elif '_r' in node:
                    parts = node.rsplit('_r', 1)
                    branch_idx = int(parts[-1])
                    x_offset = branch_spacing + (branch_idx * branch_spacing)
                    branch_key = ('right', branch_idx)
                else:
                    # Fallback
                    x_offset = 0
                    branch_key = ('unknown', 0)
                
                # Track vertical position within this branch.
                # Branch-point nodes (tiny, covered by next sibling) share the parent y so
                # the full-size covering node is initialised to the same position.
                if G.nodes[node].get('is_branch_point_node'):
                    # Sits at the parent's y level; prime the offset so the NEXT node
                    # (which covers this one) also lands at parent_y.
                    pos[node] = (x_offset, parent_y)
                    branch_y_offsets[branch_key] = parent_y + 2  # first decrement → parent_y
                elif branch_key not in branch_y_offsets:
                    branch_y_offsets[branch_key] = parent_y - 2
                    pos[node] = (x_offset, branch_y_offsets[branch_key])
                else:
                    branch_y_offsets[branch_key] -= 2
                    pos[node] = (x_offset, branch_y_offsets[branch_key])
                positioned_this_round = True
        
        if not positioned_this_round:
            # No progress made, might have orphan nodes - position them anyway
            for node in all_branch_nodes:
                if node not in pos:
                    print(f"Warning: Could not find parent for node {node}, positioning at fallback")
                    pos[node] = (0, -len(pos) * 1.5)
            break  # Exit the while loop after handling orphans
    
    # Ensure ALL nodes have positions (including any edge sources that might be missing)
    for edge in G.edges():
        if edge[0] not in pos:
            print(f"Warning: Edge source {edge[0]} not in pos, adding fallback position")
            pos[edge[0]] = (0, -len(pos) * 1.5)
        if edge[1] not in pos:
            print(f"Warning: Edge target {edge[1]} not in pos, adding fallback position")
            pos[edge[1]] = (0, -len(pos) * 1.5)

    # Shift the whole tree down so column labels (M, L*, R*) sit above the top images.
    # AnnotationBbox images extend above node centers; clearance scales with zoom.
    _label_branch_gap_scale = 0.5  # keep 1/2 of prior label-to-top-branch spacing
    top_clearance = (2.0 + max(0.0, image_zoom) * 10.0) * _label_branch_gap_scale
    for node in pos:
        x, y = pos[node]
        pos[node] = (x, y - top_clearance)
    
    # Draw edges - with safety check
    for edge in G.edges():
        if edge[0] not in pos or edge[1] not in pos:
            print(f"Skipping edge {edge} - nodes not positioned")
            continue
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        edge_color = "black"
        
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color=edge_color, lw=3,
                                 connectionstyle="arc3,rad=0.1"),
                   zorder=3)
    
    # Draw nodes with images
    node_images = node_images or {}
    placed_figures = []  # Track (gid_str, img_data) tuples — one per placed AnnotationBbox
    _gid_counter = 0

    def _label_text_bbox(edge_color):
        """White backing box so grouping/sigma text is not obscured by tree edges."""
        return dict(
            boxstyle='round,pad=0.35',
            facecolor='white',
            edgecolor=edge_color,
            linewidth=1.0,
            alpha=1.0,
        )
    
    for node in G.nodes():
        x, y = pos[node]
        node_data = G.nodes[node]
        display_label = node_data.get('display_label', node.split('_')[0])
        branch_type = node_data.get('branch_type', 'main')
        
        # Determine node styling
        if branch_type == "main":
            box_color = "lightgray"
            edge_color = "black"
            edge_width = 4
        else:
            box_color = "lightgray"
            edge_color = "black"
            edge_width = 3
        
        # Check if we have an image for this node
        img_data = None
        
        # Strict exact node-name lookup only — no cross-branch fallbacks
        if node in node_images:
            img_data = node_images[node]
        
        is_branch_point_node = node_data.get('is_branch_point_node', False)
        effective_zoom = image_zoom * 0.3 if is_branch_point_node else image_zoom

        if img_data is not None:
            # Load and place image
            try:
                img_array = one_utils.load_image(img_data)
                imagebox = OffsetImage(img_array, zoom=effective_zoom)
                ab = AnnotationBbox(imagebox, (x + image_offset[0], y + image_offset[1]),
                                   frameon=True, 
                                   bboxprops=dict(boxstyle='round,pad=0.1', 
                                                 facecolor='white', 
                                                 edgecolor=edge_color, 
                                                 linewidth=edge_width),
                                   pad=0.3)
                ab.set_zorder(5 if is_branch_point_node else 10)
                _gid_str = f'node_subfig_{_gid_counter}'
                _gid_counter += 1
                ab.set_gid(_gid_str)
                ax.add_artist(ab)
                placed_figures.append((_gid_str, img_data))  # Track (gid, figure) for SVG compositing
                
                # Add label left of the image if show_labels is True.
                # Skip branch-point helper nodes because they intentionally sit on top of
                # mother nodes and would otherwise duplicate/overlap text (group + sigma).
                if show_labels and not is_branch_point_node:
                    # Get sigma range for this node
                    sigma_range = node_data.get('sigma_range', (None, None))
                    # Format sigma text - use the end of the range (more specific sigma value)
                    if sigma_range and sigma_range[1] is not None:
                        sigma_val = sigma_range[1]
                        if hasattr(sigma_val, 'item'):
                            sigma_val = sigma_val.item()
                        sigma_text = f'$\\sigma={sigma_val:.3f}$'
                        # Draw classification label
                        ax.text(x + image_offset[0] - label_offset, y + image_offset[1] + 0.15, display_label, 
                               fontsize=font['default_size'], ha='center', va='center',
                               fontweight='bold', color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
                        # Draw sigma below the classification label
                        ax.text(x + image_offset[0] - label_offset, y + image_offset[1] - 0.15, sigma_text, 
                               fontsize=font['default_size']-2, ha='center', va='center',
                               color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
                    else:
                        ax.text(x + image_offset[0] - label_offset, y + image_offset[1], display_label, fontsize=font['default_size'], ha='center', va='center',
                               fontweight='bold', color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
            except Exception as e:
                print(f"Warning: Could not load image for node {node}: {e}")
                # Fall back to box
                box = FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=box_color, edgecolor=edge_color, 
                                    linewidth=edge_width, transform=ax.transData,
                                    zorder=10)
                ax.add_patch(box)
                if show_labels and not is_branch_point_node:
                    # Get sigma range for this node
                    sigma_range = node_data.get('sigma_range', (None, None))
                    if sigma_range and sigma_range[1] is not None:
                        sigma_val = sigma_range[1]
                        if hasattr(sigma_val, 'item'):
                            sigma_val = sigma_val.item()
                        sigma_text = f'$\\sigma={sigma_val:.3f}$'
                        ax.text(x - label_offset, y + 0.1, display_label, fontsize=font['default_size']+2, ha='center', va='center',
                               fontweight='bold', color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
                        ax.text(x - label_offset, y - 0.1, sigma_text, fontsize=font['default_size'], ha='center', va='center',
                               color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
                    else:
                        ax.text(x - label_offset, y, display_label, fontsize=font['default_size']+2, ha='center', va='center',
                               fontweight='bold', color=edge_color, zorder=20,
                               bbox=_label_text_bbox(edge_color))
        else:
            # Draw fancy box (no image)
            box = FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor=box_color, edgecolor=edge_color, 
                                linewidth=edge_width, transform=ax.transData,
                                zorder=10)
            ax.add_patch(box)
            if show_labels and not is_branch_point_node:
                # Get sigma range for this node
                sigma_range = node_data.get('sigma_range', (None, None))
                if sigma_range and sigma_range[1] is not None:
                    sigma_val = sigma_range[1]
                    if hasattr(sigma_val, 'item'):
                        sigma_val = sigma_val.item()
                    sigma_text = f'$\\sigma={sigma_val:.3f}$'
                    ax.text(x - label_offset, y + 0.1, display_label, fontsize=font['default_size']+2, ha='center', va='center',
                           fontweight='bold', color=edge_color, zorder=20,
                           bbox=_label_text_bbox(edge_color))
                    ax.text(x - label_offset, y - 0.1, sigma_text, fontsize=font['default_size'], ha='center', va='center',
                           color=edge_color, zorder=20,
                           bbox=_label_text_bbox(edge_color))
                else:
                    ax.text(x - label_offset, y, display_label, fontsize=font['default_size']+2, ha='center', va='center',
                           fontweight='bold', color=edge_color, zorder=20,
                           bbox=_label_text_bbox(edge_color))
    
    # Calculate actual bounds from positioned nodes
    all_y_values = [p[1] for p in pos.values()]
    all_x_values = [p[0] for p in pos.values()]
    actual_y_min = min(all_y_values) if all_y_values else -2
    actual_y_max = max(all_y_values) if all_y_values else 0
    actual_x_min = min(all_x_values) if all_x_values else -3
    actual_x_max = max(all_x_values) if all_x_values else 3
    
    # Get position of first main node for title centering (with image_offset applied)
    first_main_pos = pos[main_nodes[0]] if main_nodes else (0, 0)
    title_x = first_main_pos[0] + image_offset[0]
    # Space above top node row: image half-height + padding for column labels
    image_top_extent = (0.8 + max(0.0, image_zoom) * 6.0) * _label_branch_gap_scale
    header_y = actual_y_max + image_top_extent + 0.35 * _label_branch_gap_scale
    title_y = header_y + 1.0
    
    # Add title centered above first main node (accounting for image_offset)
    ax.text(title_x, title_y, "Bifurcation Tree Structure", 
           fontsize=font['title_size'], ha='center', va='center',
           fontweight='bold', transform=ax.transData)

    # Add compact column headers for main and branch columns.
    # This mirrors the notebook expectation: M, L1.. and R1..
    header_font = max(font['default_size'], font['title_size'] - 4)
    ax.text(title_x, header_y, "M",
            fontsize=header_font, ha='center', va='center',
            fontweight='bold', color='black', transform=ax.transData)

    if left_node_lists:
        for i, branch_nodes in enumerate(left_node_lists):
            if not branch_nodes:
                continue
            branch_x = pos[branch_nodes[0]][0] + image_offset[0]
            ax.text(branch_x, header_y, f"L{i+1}",
                    fontsize=header_font, ha='center', va='center',
                    fontweight='bold', color='darkblue', transform=ax.transData)

    if right_node_lists:
        for i, branch_nodes in enumerate(right_node_lists):
            if not branch_nodes:
                continue
            branch_x = pos[branch_nodes[0]][0] + image_offset[0]
            ax.text(branch_x, header_y, f"R{i+1}",
                    fontsize=header_font, ha='center', va='center',
                    fontweight='bold', color='darkred', transform=ax.transData)
    
    # Set limits dynamically based on actual node positions with minimal padding
    padding = 1.5  # Padding around nodes
    y_min = actual_y_min - padding
    y_max = title_y + 0.8  # Padding above title and column headers
    x_range = max(abs(actual_x_min), abs(actual_x_max)) + padding + abs(image_offset[0])
    
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(y_min, y_max)
    
    # Make aspect square
    ax.set_box_aspect(1)
    
    if hide_text:
        import matplotlib.text as _mtext
        for _t in fig.findobj(match=_mtext.Text):
            _t.set_alpha(0)
    
    plt.close()
    
    return fig, ax, pos, placed_figures

def bifurcation_rectangle_plot(main_matrix,
                                left_matrices,
                                right_matrices, 
                                reach_parameters,
                                num_agents, 
                                reach_start,
                                reach_end, 
                                label_to_color=None, 
                                figsize=(20, 24),
                                rect_width=0.8,
                                horizontal_spacing=2.5,
                                box_height=10,
                                font_size=14,
                                show_labels=False):
    """
    Internal function: Rectangle display mode for bifurcation tree.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    
    # Starting y position for rectangles
    rect_y_start = 1
    
    # Process main matrix
    main_segments = one_utils.process_matrix(main_matrix,num_agents=num_agents,reach_parameters=reach_parameters,reach_start=reach_start,reach_end=reach_end)
    
    # Process left matrices
    left_segments_list = [one_utils.process_matrix(mat, num_agents=num_agents,reach_parameters=reach_parameters,reach_start=reach_start,reach_end=reach_end) for mat in left_matrices]
    
    # Process right matrices
    right_segments_list = [one_utils.process_matrix(mat, num_agents=num_agents,reach_parameters=reach_parameters,reach_start=reach_start,reach_end=reach_end) for mat in right_matrices]
    

    # Set default colors if not provided - ensure ALL labels get colors
    specific_colors = {
        f'$({num_agents})$': '#87CEEB',
        'Cycles': '#FFD700',
        '(2,1,1,2)': '#FF6B6B',
        '(1,1,1,1,1,1)': '#9370DB'
    }
    additional_colors = ['#98D8C8', '#F7B7A3', '#EA5F89', '#9D84B7', '#A8E6CF', 
                        '#FFD3B6', '#FFAAA5', '#FF8B94', '#C7CEEA', '#B5EAD7']
    
    if label_to_color is None:
        label_to_color = {}
    
    # Collect all unique labels from all matrices
    all_labels = set()
    all_labels.update(main_segments['labels'])
    for seg in left_segments_list:
        all_labels.update(seg['labels'])
    for seg in right_segments_list:
        all_labels.update(seg['labels'])
    
    # Remove empty strings and None from labels
    all_labels.discard('')
    all_labels.discard(None)
    
    # Assign colors to any missing labels
    color_index = len([c for c in label_to_color.values() if c in additional_colors])
    for label in sorted(all_labels):  # Sort for consistency
        if label not in label_to_color:
            if label in specific_colors:
                label_to_color[label] = specific_colors[label]
            else:
                label_to_color[label] = additional_colors[color_index % len(additional_colors)]
                color_index += 1
    
    # Helper function to convert sigma to y coordinate (vertical rectangles)
    sigma_min = reach_start.item() if hasattr(reach_start, 'item') else reach_start
    sigma_max = reach_end.item() if hasattr(reach_end, 'item') else reach_end
    
    # Calculate x positions
    main_x = 0
    left_x_positions = [-horizontal_spacing * (i + 1) for i in range(len(left_matrices))]
    right_x_positions = [horizontal_spacing * (i + 1) for i in range(len(right_matrices))]
    
    
    # Draw main rectangle (full)
    one_utils.draw_rectangle(main_segments, main_x, label_to_color, rect_width, edge_color='black', edge_width=3)
    
    # Track first_diff values for each branch to use in legends
    left_first_diffs = []
    right_first_diffs = []
    # Draw left branches (only from first difference point)
    for i, (left_seg, left_x) in enumerate(zip(left_segments_list, left_x_positions)):
        # Determine reference: main for first, previous left for others
        if i == 0:
            first_diff = one_utils.find_first_difference_and_draw(main_segments, left_seg, main_x, left_x, rect_width, branch_color='blue')
        else:
            first_diff = one_utils.find_first_difference_and_draw(left_segments_list[i-1], left_seg, left_x_positions[i-1], left_x, rect_width,branch_color='blue')
        
        left_first_diffs.append(first_diff)
        
        # Draw rectangle only from first difference point downward
        one_utils.draw_rectangle(left_seg, left_x, label_to_color, rect_width, edge_color='darkblue', edge_width=2, start_from_sigma=first_diff)
    
    # Draw right branches (only from first difference point)
    right_ys=[]
    for i, (right_seg, right_x) in enumerate(zip(right_segments_list, right_x_positions)):
        # Determine reference: main for first, previous right for others
        if i == 0:
            first_diff = one_utils.find_first_difference_and_draw(main_segments, right_seg, main_x, right_x,rect_width, branch_color='red')
        else:
            first_diff = one_utils.find_first_difference_and_draw(right_segments_list[i-1], right_seg, right_x_positions[i-1], right_x, rect_width, branch_color='red')
        
        right_first_diffs.append(first_diff)
        
        # Draw rectangle only from first difference point downward
        one_utils.draw_rectangle(right_seg, right_x, label_to_color, rect_width, edge_color='darkred', edge_width=2, start_from_sigma=first_diff)
    
    # Add sigma scale on the left
    sigma_step = (sigma_max - sigma_min) / 10
    sigma_labels_vals = [sigma_min + i * sigma_step for i in range(11)]
    leftmost_x = min(left_x_positions) if left_x_positions else main_x
    scale_x = leftmost_x - rect_width - 0.5
    
    for sigma_val in sigma_labels_vals:
        y_pos = one_utils.sigma_to_y(sigma_val, box_height, rect_y_start, sigma_min, sigma_max)
        ax.text(scale_x, y_pos, f'{sigma_val:.2f}',
                ha='right', va='center', fontsize=font_size-2, color='black')
    
    ax.text(scale_x - 0.5, rect_y_start + box_height/2, 
            'Sigma ($\\sigma$) - Agent Reach Parameter',
            ha='center', va='center', fontsize=font_size+2, fontweight='bold',
            rotation=90)
    
    # Add compact column labels at the top (M, L1.., R1..)
    ax.text(main_x + rect_width/2, rect_y_start - 0.5, 'M',
            ha='center', va='bottom', fontsize=font_size, 
            fontweight='bold', color='black')
    
    for i, left_x in enumerate(left_x_positions):
        ax.text(left_x + rect_width/2, rect_y_start - 0.5, f'L{i+1}',
                ha='center', va='bottom', fontsize=font_size, 
                fontweight='bold', color='darkblue')
    
    for i, right_x in enumerate(right_x_positions):
        ax.text(right_x + rect_width/2, rect_y_start - 0.5, f'R{i+1}',
                ha='center', va='bottom', fontsize=font_size, 
                fontweight='bold', color='darkred')
    
    # Set axis limits
    leftmost = min(left_x_positions + [main_x]) if left_x_positions else main_x
    rightmost = max(right_x_positions + [main_x]) if right_x_positions else main_x
    ax.set_xlim(leftmost - rect_width - 2, rightmost + rect_width + 1)
    ax.set_ylim(0, rect_y_start + box_height + 1)
    
    # Apply tight_layout BEFORE creating legends to avoid hiding them
    plt.tight_layout()
    
    # Create legends for each branch showing classifications in order of appearance (top to bottom)
   
    
    # Get ordered labels for main branch
    main_labels = one_utils.get_ordered_labels(main_segments)
    
    # Create legend for main branch
    if main_labels:
        legend_elements = [Patch(facecolor=label_to_color.get(label, '#CCCCCC'), 
                                edgecolor='black', linewidth=2, label=label)
                          for label in main_labels]
        # Position legend using same x-coordinate as the 'Main' label
        main_center_x = main_x + rect_width / 2
        main_y_pos = rect_y_start + box_height + 0.5
        main_leg = ax.legend(handles=legend_elements, loc='lower center', 
                 bbox_to_anchor=(main_center_x, main_y_pos), bbox_transform=ax.transData,
                 fontsize=font_size-4, title='M', framealpha=0.9,
                 title_fontsize=font_size-3, ncol=1, labelspacing=0.7, borderpad=0.6)
        main_leg.set_clip_on(False)
        ax.add_artist(main_leg)
        
        # Add title centered above the main legend
        # Estimate legend height and add title above it
        legend_height_estimate = len(main_labels) * 0.4 + 0.8  # Rough estimate
        title_y_pos = main_y_pos + legend_height_estimate
        ax.text(main_center_x, title_y_pos*.9, 
                'Bifurcation Tree: Main Rectangle with Left/Right Branches',
                ha='center', va='bottom', fontsize=font_size+6, fontweight='bold',
                transform=ax.transData)
        
    # Create legends for left branches
    for i, (left_seg, left_x) in enumerate(zip(left_segments_list, left_x_positions)):
        # Get labels filtered by the first_diff threshold for this branch
        left_labels = one_utils.get_ordered_labels(left_seg, start_from_sigma=left_first_diffs[i])
        if left_labels:
            legend_elements = [Patch(facecolor=label_to_color.get(label, '#CCCCCC'), 
                                    edgecolor='darkblue', linewidth=2, label=label)
                              for label in left_labels]
            # Position legend using same x-coordinate as the 'Left' label
            left_center_x = left_x + rect_width / 2
            if left_first_diffs[i] is not None:
                first_diff_y = one_utils.sigma_to_y(left_first_diffs[i])
                left_y_pos = first_diff_y + 0.5
            else:
                left_y_pos = rect_y_start + box_height + 0.5
            leg = ax.legend(handles=legend_elements, loc='lower center',
                           bbox_to_anchor=(left_center_x, left_y_pos), bbox_transform=ax.transData,
                           fontsize=font_size-4, title=f'L{i+1}', framealpha=0.9,
                           title_fontsize=font_size-3, ncol=1, labelspacing=0.9, borderpad=0.7)
            leg.set_clip_on(False)
            ax.add_artist(leg)
    
    # Create legends for right branches
    for i, (right_seg, right_x) in enumerate(zip(right_segments_list, right_x_positions)):
        # Get labels filtered by the first_diff threshold for this branch
        right_labels = one_utils.get_ordered_labels(right_seg, start_from_sigma=right_first_diffs[i])
        if right_labels:
            legend_elements = [Patch(facecolor=label_to_color.get(label, '#CCCCCC'), 
                                    edgecolor='darkred', linewidth=2, label=label)
                              for label in right_labels]
            # Position legend using same x-coordinate as the 'Right' label
            right_center_x = right_x + rect_width / 2
            if right_first_diffs[i] is not None:
                first_diff_y = one_utils.sigma_to_y(right_first_diffs[i])
                # Small per-branch vertical staggering helps prevent text collisions for
                # the first legend entries in nearby right columns.
                right_y_pos = first_diff_y + 0.5 + (0.2 * i)
            else:
                right_y_pos = rect_y_start + box_height + 0.5 + (0.2 * i)
            leg = ax.legend(handles=legend_elements, loc='lower center',
                           bbox_to_anchor=(right_center_x, right_y_pos), bbox_transform=ax.transData,
                           fontsize=font_size-4, title=f'R{i+1}', framealpha=0.9,
                           title_fontsize=font_size-3, ncol=1, labelspacing=1.0, borderpad=0.8)
            leg.set_clip_on(False)
            ax.add_artist(leg)
    
    return fig, ax

#do not use , kept as legacy
def create_gradient_vector_field_plot_clipped(vis, grid_resolution=20, figsize=(24, 12), 
                                            max_magnitude=None, normalize_arrows=True):
    r"""
    Create gradient vector field plot with clipped magnitudes for 1D domain visualization.
    
    .. deprecated::
       This function is kept as legacy code and should not be used in new development.
    
    Generates a 2D vector field plot showing gradient directions and magnitudes at grid points
    in the projected 2D plane. Includes magnitude clipping for better visualization of large
    gradients and optional normalization to show pure direction information.
    
    Following Influencer Games patterns:
    
    - Use torch tensor operations for autograd compatibility
    - Preserve original state using .clone()
    - Handle 1d domain type properly
    - Return matplotlib figure for visualization

    :param vis: Visualization Shell instance containing the field and parameters.
    :type vis: Shell
    :param grid_resolution: Number of grid points per dimension (default: 20).
    :type grid_resolution: int
    :param figsize: Figure size as (width, height) in inches (default: (24, 12)).
    :type figsize: tuple
    :param max_magnitude: Maximum gradient magnitude for clipping; if None, auto-clips at 95th percentile.
    :type max_magnitude: Optional[float]
    :param normalize_arrows: If True, normalize all arrows to same length for direction visualization.
    :type normalize_arrows: bool
    
    :return: Tuple of (matplotlib figure, dictionary with computed data including positions, gradients, statistics).
    :rtype: Tuple[matplotlib.figure.Figure, Dict]
    """
    # Preserve original state following project patterns
    original_pos = vis.field.agents_pos.clone()
    original_params = vis.field.parameters.clone()
    original_pos_matrix = vis.field.pos_matrix.clone()
    original_grad_matrix = vis.field.grad_matrix.clone()
    
    # Generate 2D grid using torch meshgrid to match your coordinate grid approach
    sqrt2_inv = 1.0 / np.sqrt(2)
    
    # Create coordinate grid using torch to match the coordinate grid behavior
    y_coords = torch.linspace(-sqrt2_inv, sqrt2_inv, grid_resolution, dtype=torch.float32)
    x_coords = torch.linspace(-sqrt2_inv, sqrt2_inv, grid_resolution, dtype=torch.float32)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')  # This matches your mgrid order
    
    # Flatten to match  a1, a2 assignment pattern
    a1 = X.flatten()  # x coordinates
    a2 = Y.flatten()  # y coordinates
    
    # Stack coordinates to create 2D points
    grid_points_2d = torch.stack([a1, a2], dim=1)  # Shape: (grid_resolution^2, 2)
    
    # Filter points satisfying the constraint |x-y| ≤ 1/√2
    x_coords = grid_points_2d[:, 0]
    y_coords = grid_points_2d[:, 1]
    diff_constraint = torch.abs(x_coords - y_coords) <= sqrt2_inv
    
    # Apply constraint filtering
    valid_grid_points = grid_points_2d[diff_constraint]
    
    print(f"Generated {len(grid_points_2d)} total grid points")
    print(f"Found {len(valid_grid_points)} points satisfying constraint |x-y| ≤ 1/√2")
    
    # Project to 3D coordinates within domain bounds
    grid_3d = one_utils.projection_to_3d_auto_constrained(
        valid_grid_points, 
        target_bounds=(0.0, 1.0)
    )
    
    # Calculate gradients at each grid point
    gradient_vectors = []
    position_points = []
    print(grid_3d)
    
    print(f"Computing gradients for {len(grid_3d)} points...")
    
    for i, pos_3d in enumerate(grid_3d):
        try:
            # Set field state for gradient computation following project patterns
            vis.field.agents_pos = pos_3d.clone()
            
            # Compute gradient using the field's gradient function
            gradient_3d = vis.field.gradient(
                parameter_instance=vis.parameters
            )
            
            # Project gradient back to 2D plane coordinates
            gradient_2d = one_utils.projection_to_plane_coordinates(gradient_3d)
            
            # Store results
            gradient_vectors.append(gradient_2d)
            position_points.append(valid_grid_points[i])
            
            if (i + 1) % max(1, len(grid_3d) // 10) == 0:
                progress = (i + 1) / len(grid_3d) * 100
                print(f"Progress: {i+1}/{len(grid_3d)} ({progress:.1f}%)")
                
        except Exception as e:
            print(f"Error computing gradient at point {i}: {str(e)}")
            # Use zero gradient as fallback
            gradient_vectors.append(torch.zeros(2, dtype=torch.float32))
            position_points.append(valid_grid_points[i])
    
    # Convert to numpy arrays for matplotlib
    positions = torch.stack(position_points).numpy()
    gradients = torch.stack(gradient_vectors).numpy()
    
    # Calculate original magnitudes
    gradient_magnitudes = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
    
    # Clip magnitudes if specified, otherwise auto-clip to 95th percentile
    if max_magnitude is None:
        max_magnitude = np.percentile(gradient_magnitudes[gradient_magnitudes > 0], 95)
        print(f"Auto-clipping magnitudes at 95th percentile: {max_magnitude:.4f}")
    
    # Create clipped gradients for arrow display
    clipped_gradients = gradients.copy()
    large_magnitude_mask = gradient_magnitudes > max_magnitude
    
    # For vectors with large magnitudes, normalize and scale to max_magnitude
    if np.any(large_magnitude_mask):
        large_gradients = gradients[large_magnitude_mask]
        large_mags = gradient_magnitudes[large_magnitude_mask]
        # Normalize and scale
        clipped_gradients[large_magnitude_mask] = (large_gradients / large_mags[:, np.newaxis]) * max_magnitude
    
    # Optionally normalize all arrows to same length for direction visualization
    if normalize_arrows:
        nonzero_mask = gradient_magnitudes > 1e-10
        if np.any(nonzero_mask):
            normalized_gradients = gradients.copy()
            normalized_gradients[nonzero_mask] = (gradients[nonzero_mask] / 
                                                gradient_magnitudes[nonzero_mask, np.newaxis]) * max_magnitude * 0.5
        else:
            normalized_gradients = clipped_gradients
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)

    def add_constraint_boundaries(ax):
        sqrt2_inv = 1.0 / np.sqrt(2)
        ax.axhline(y=sqrt2_inv, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(y=-sqrt2_inv, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=sqrt2_inv, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=-sqrt2_inv, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add diagonal constraint lines: x-y = ±1/√2
        x_line = np.linspace(-sqrt2_inv, sqrt2_inv, 100)
        ax.plot(x_line, x_line - sqrt2_inv, 'r--', alpha=0.5, linewidth=2, label='Boundary')
        ax.plot(x_line, x_line + sqrt2_inv, 'r--', alpha=0.5, linewidth=2)

    if normalize_arrows:
        quiver2 = ax.quiver(positions[:, 0], positions[:, 1], 
                           normalized_gradients[:, 0], normalized_gradients[:, 1],
                           gradient_magnitudes, cmap='viridis',
                           angles='xy', scale_units='xy', scale=1, 
                           alpha=0.8, width=0.002, headwidth=1.5, headlength=2.5)  # Much smaller arrows
        add_constraint_boundaries(ax)
        ax.set_xlabel('Projected X Coordinate', fontsize=14)
        ax.set_ylabel('Projected Y Coordinate', fontsize=14)
        ax.set_title('Gradient Directions (Normalized Arrows)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=12)
        
        # Add colorbar for normalized plot
        cbar2 = plt.colorbar(quiver2, ax=ax, shrink=0.8)
        cbar2.set_label('Original Magnitude', fontsize=12)
   
    else:    
        # Plot: Original vector field (clipped magnitudes) with smaller arrow heads
        quiver1 = ax.quiver(positions[:, 0], positions[:, 1], 
                           clipped_gradients[:, 0], clipped_gradients[:, 1],
                           gradient_magnitudes, cmap='plasma',
                           angles='xy', scale_units='xy', scale=1, 
                           alpha=0.8, width=0.002, headwidth=1.5, headlength=2.5)  # Much smaller arrows
        add_constraint_boundaries(ax)
        ax.set_xlabel('Projected X Coordinate', fontsize=14)
        ax.set_ylabel('Projected Y Coordinate', fontsize=14)
        ax.set_title(f'Gradient Vector Field (Clipped at {max_magnitude:.3f})', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim([-sqrt2_inv-0.1, sqrt2_inv+0.1])
        ax.set_ylim([-sqrt2_inv-0.1, sqrt2_inv+0.1])
        ax.tick_params(labelsize=12)
        
        # Add colorbar for clipped plot
        cbar1 = plt.colorbar(quiver1, ax=ax, shrink=0.8)
        cbar1.set_label('Original Magnitude', fontsize=12)
    
    # Adjust spacing
    plt.tight_layout(pad=3.0)
    
    # Restore original state following project patterns
    vis.field.agents_pos = original_pos
    vis.field.parameters = original_params
    vis.field.pos_matrix = original_pos_matrix
    vis.field.grad_matrix = original_grad_matrix
    
    # Print statistics
    print(f"Vector field plot completed with {len(positions)} points")
    print(f"Grid efficiency: {len(valid_grid_points)}/{len(grid_points_2d)} ({len(valid_grid_points)/len(grid_points_2d)*100:.1f}%)")
    print(f"Magnitude statistics:")
    print(f"  Min: {gradient_magnitudes.min():.6f}")
    print(f"  Max: {gradient_magnitudes.max():.6f}")
    print(f"  Mean: {gradient_magnitudes.mean():.6f}")
    print(f"  95th percentile: {np.percentile(gradient_magnitudes, 95):.6f}")
    print(f"  Vectors clipped: {np.sum(large_magnitude_mask)}/{len(gradient_magnitudes)}")
    
    return fig, {
        'positions': positions,
        'gradients': gradients,
        'clipped_gradients': clipped_gradients,
        'normalized_gradients': normalized_gradients if normalize_arrows else None,
        'magnitudes': gradient_magnitudes,
        'max_magnitude': max_magnitude,
        'grid_points_2d': grid_points_2d,
        'valid_grid_points': valid_grid_points,
        'grid_efficiency': len(valid_grid_points)/len(grid_points_2d)
    }


# =============================================================================
# Combined Bifurcation Figure Utilities
# =============================================================================

def transfer_axis_artists(source_ax, target_ax):
    """
    Transfer all visual elements from source axis to target axis by recreating them.

    Parameters
    ----------
    source_ax : matplotlib.axes.Axes
        Source axis containing elements to transfer.
    target_ax : matplotlib.axes.Axes
        Target axis to receive transferred elements.
    """
    # Transfer lines
    for line in source_ax.get_lines():
        target_ax.plot(
            line.get_xdata(), line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            label=line.get_label() if not line.get_label().startswith('_') else ''
        )

    # Transfer collections (fill_between → PolyCollection; scatter → PathCollection)
    for coll in source_ax.collections:
        if isinstance(coll, PolyCollection):
            verts = [path.vertices for path in coll.get_paths()]
            new_coll = PolyCollection(
                verts,
                facecolors=coll.get_facecolors(),
                edgecolors=coll.get_edgecolors(),
                linewidths=coll.get_linewidths(),
                linestyles=coll.get_linestyle(),
                alpha=coll.get_alpha()
            )
            if coll.get_hatch():
                new_coll.set_hatch(coll.get_hatch())
            new_coll.set_zorder(coll.get_zorder())
            target_ax.add_collection(new_coll)
        elif isinstance(coll, PathCollection):
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                target_ax.scatter(
                    offsets[:, 0], offsets[:, 1],
                    c=coll.get_facecolors(),
                    s=coll.get_sizes(),
                    alpha=coll.get_alpha()
                )

    # Transfer patches
    for patch in source_ax.patches:
        if isinstance(patch, FancyBboxPatch):
            bbox = patch.get_bbox()
            new_fancybox = FancyBboxPatch(
                (bbox.x0, bbox.y0), bbox.width, bbox.height,
                boxstyle=patch.get_boxstyle(),
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                alpha=patch.get_alpha(),
                transform=target_ax.transData if patch.get_transform() == source_ax.transData else patch.get_transform(),
                zorder=patch.get_zorder()
            )
            if patch.get_hatch():
                new_fancybox.set_hatch(patch.get_hatch())
            target_ax.add_patch(new_fancybox)
        elif isinstance(patch, Wedge):
            new_wedge = Wedge(
                patch.center, patch.r, patch.theta1, patch.theta2,
                width=patch.width,
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth()
            )
            target_ax.add_patch(new_wedge)
        elif isinstance(patch, Rectangle):
            new_rect = Rectangle(
                patch.get_xy(), patch.get_width(), patch.get_height(),
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth()
            )
            if patch.get_hatch():
                new_rect.set_hatch(patch.get_hatch())
            target_ax.add_patch(new_rect)
        elif isinstance(patch, Polygon):
            new_poly = Polygon(
                patch.get_xy(),
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth()
            )
            if patch.get_hatch():
                new_poly.set_hatch(patch.get_hatch())
            target_ax.add_patch(new_poly)

    # Transfer text elements
    for text in source_ax.texts:
        bbox_props = None
        if text.get_bbox_patch() is not None:
            bbox_patch = text.get_bbox_patch()
            bbox_props = dict(
                boxstyle=bbox_patch.get_boxstyle(),
                facecolor=bbox_patch.get_facecolor(),
                edgecolor=bbox_patch.get_edgecolor(),
                linewidth=bbox_patch.get_linewidth(),
                alpha=bbox_patch.get_alpha()
            )
            if bbox_patch.get_hatch():
                bbox_props['hatch'] = bbox_patch.get_hatch()

        if text.get_transform() == source_ax.transData:
            transform = target_ax.transData
        elif text.get_transform() == source_ax.transAxes:
            transform = target_ax.transAxes
        else:
            transform = text.get_transform()

        target_ax.text(
            text.get_position()[0], text.get_position()[1],
            text.get_text(),
            fontsize=text.get_fontsize(),
            ha=text.get_ha(),
            va=text.get_va(),
            color=text.get_color(),
            fontweight=text.get_fontweight(),
            rotation=text.get_rotation(),
            transform=transform,
            bbox=bbox_props,
            zorder=text.get_zorder()
        )

    # Transfer axis properties
    target_ax.set_xlabel(source_ax.get_xlabel(), fontsize=source_ax.xaxis.label.get_fontsize())
    target_ax.set_ylabel(source_ax.get_ylabel(), fontsize=source_ax.yaxis.label.get_fontsize())
    target_ax.set_title(source_ax.get_title(), fontsize=source_ax.title.get_fontsize())

    target_ax.set_xticks(source_ax.get_xticks())
    target_ax.set_yticks(source_ax.get_yticks())
    target_ax.set_xticklabels(
        source_ax.get_xticklabels(),
        rotation=source_ax.xaxis.get_ticklabels()[0].get_rotation() if source_ax.xaxis.get_ticklabels() else 0
    )
    target_ax.set_yticklabels(source_ax.get_yticklabels())

    has_wedges = any(isinstance(patch, Wedge) for patch in source_ax.patches)
    if has_wedges:
        target_ax.set_frame_on(False)
        target_ax.set_xticks([])
        target_ax.set_yticks([])
        target_ax.set_xlabel('')
        target_ax.set_ylabel('')

    # Transfer legend
    if source_ax.get_legend():
        source_legend = source_ax.get_legend()
        try:
            legend_handles = source_legend.legend_handles
        except AttributeError:
            try:
                legend_handles = source_legend.legendHandles
            except AttributeError:
                legend_handles, _ = source_ax.get_legend_handles_labels()

        legend_labels = [t.get_text() for t in source_legend.get_texts()]
        new_handles = []
        for handle in legend_handles:
            if isinstance(handle, mpatches.Patch):
                new_handles.append(mpatches.Patch(
                    facecolor=handle.get_facecolor(),
                    edgecolor=handle.get_edgecolor(),
                    linewidth=handle.get_linewidth(),
                    linestyle=handle.get_linestyle(),
                    alpha=handle.get_alpha(),
                    hatch=handle.get_hatch(),
                    label=handle.get_label()
                ))
            elif isinstance(handle, Line2D):
                new_handles.append(Line2D(
                    [], [],
                    color=handle.get_color(),
                    linestyle=handle.get_linestyle(),
                    linewidth=handle.get_linewidth(),
                    marker=handle.get_marker(),
                    markersize=handle.get_markersize(),
                    label=handle.get_label()
                ))
            else:
                new_handles.append(handle)

        target_ax.legend(
            handles=new_handles, labels=legend_labels,
            fontsize=source_legend._fontsize,
            loc=source_legend._loc
        )

    if source_ax.get_aspect() != 'auto' and has_wedges:
        target_ax.set_aspect(source_ax.get_aspect())

    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())


def transfer_multi_subplot_figure(source_fig, target_ax, label: Optional[str] = None):
    """
    Transfer content from a multi-subplot figure into a single target axis region.

    Parameters
    ----------
    source_fig : matplotlib.figure.Figure
        Source figure whose subplots are transferred.
    target_ax : matplotlib.axes.Axes
        Target axis that defines the bounding region.
    label : str, optional
        Region label to draw in the upper-right corner (e.g. 'a', 'b').
    """
    source_axes = source_fig.get_axes()

    if len(source_axes) == 0:
        return

    if len(source_axes) == 1:
        transfer_axis_artists(source_axes[0], target_ax)
        return

    target_bbox = target_ax.get_position()
    cols = len(source_axes)

    target_ax.set_visible(False)
    target_ax.set_frame_on(False)
    target_ax.set_xticks([])
    target_ax.set_yticks([])

    fig = target_ax.figure
    left = target_bbox.x0
    bottom = target_bbox.y0
    width = target_bbox.width / cols
    height = target_bbox.height
    spacing = 0.02

    for i, source_ax in enumerate(source_axes):
        if i == 0:
            new_ax = fig.add_axes([left + i * width, bottom, width * 0.95, height])
        else:
            new_ax = fig.add_axes([left + i * width + spacing, bottom, width * 0.95, height])
        transfer_axis_artists(source_ax, new_ax)

    if label is not None:
        label_x = target_bbox.x0 + target_bbox.width - 0.38
        label_y = target_bbox.y0 + target_bbox.height - 0.01
        box = FancyBboxPatch(
            (label_x, label_y), 0.007, 0.007,
            boxstyle="round,pad=0.005",
            edgecolor='black', facecolor='white',
            linewidth=1, transform=fig.transFigure, zorder=1000
        )
        fig.patches.append(box)
        fig.text(
            label_x + 0.0035, label_y + 0.0035, label,
            ha='center', va='center',
            fontsize=28, fontweight='bold',
            transform=fig.transFigure, zorder=1001
        )


def generate_combined_bifurcation_figure(
        vis,
        alpha,
        test_processed: dict,
        results_list: List,
        search,
        figsize: Tuple[float, float] = (32, 22),
        width_ratios: List[float] = [1, 1],
        height_ratios: List[float] = [1, 1, 1],
        hspace: float = 0.4,
        wspace: float = 0.12,
        region_labels: List[str] = ['a', 'b', 'c', 'd'],
        save: bool = False,
        save_types: List[str] = ['.png', '.svg'],
        paper_figure: dict = {'paper': True, 'section': '3_2_6', 'figure_id': 'fig_combined'},
        font: dict = {'default_size': 32, 'cbar_size': 16, 'title_size': 40,
                      'legend_size': 12, 'font_family': 'sans-serif'}
) -> matplotlib.figure.Figure:
    """
    Generate the standardized combined bifurcation and equilibrium analysis figure.

    Parameters
    ----------
    vis : Shell
        Configured Shell instance (used for ``first_order_bifurcation_plot`` and metadata).
    alpha : array-like
        Alpha parameter values corresponding to the bifurcation data.
    test_processed : dict
        Pre-processed bifurcation data (output of bifurcation pipeline).
    results_list : List[dict]
        List of four region results dicts
        (``[results1, results2, results3, results4]``).
    search : search_env
        Configured ``monte_search.search_env`` instance used to plot equilibrium analyses.
    figsize : Tuple[float, float]
        Overall figure size in inches.
    width_ratios : List[float]
        Width ratios for the two-column GridSpec layout.
    height_ratios : List[float]
        Height ratios for the three-row GridSpec layout.
    hspace : float
        Vertical spacing between subplots.
    wspace : float
        Horizontal spacing between subplots.
    region_labels : List[str]
        Labels (a–d) placed on the bifurcation diagram and sub-panels.
    save : bool
        Whether to save the figure.
    save_types : List[str]
        File extensions for saving.
    paper_figure : dict
        Paper-figure metadata with keys ``'paper'``, ``'section'``, ``'figure_id'``.
    font : dict
        Font configuration dict.

    Returns
    -------
    matplotlib.figure.Figure
    """
    font = dict(font)  # avoid mutating caller's dict
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    num_agents = vis.num_agents
    infl_configs = vis.infl_configs

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=width_ratios, height_ratios=height_ratios,
                  hspace=hspace, wspace=wspace)

    ax_main_new = fig.add_subplot(gs[:2, 0])
    ax4_new = fig.add_subplot(gs[2, 0])
    ax1_new = fig.add_subplot(gs[0, 1])
    ax2_new = fig.add_subplot(gs[1, 1])
    ax3_new = fig.add_subplot(gs[2, 1])

    # --- Main bifurcation plot ---
    fig_temp, ax_temp = vis.first_order_bifurcation_plot(
        alpha_st=0,
        alpha_end=alpha[-1],
        processed_data=test_processed,
        alpha_values=alpha,
        save=False,
        font=font,
        paper_figure={
            'paper': paper_figure['paper'],
            'section': paper_figure['section'],
            'figure_id': f"{paper_figure.get('figure_id', 'bifurcation_equilibrium_combined')}_main"
        }
    )
    transfer_axis_artists(ax_temp, ax_main_new)
    plt.close(fig_temp)

    # --- Region labels on the bifurcation diagram ---
    main_bbox = ax_main_new.get_position()
    y_start = main_bbox.y0 + main_bbox.height * 0.7

    if infl_configs['infl_type'] == 'beta' and num_agents == 6:
        y_start = main_bbox.y0 + main_bbox.height * 0.82
        center_x = [main_bbox.x0 + main_bbox.width / 2 + 0.135] * 4
        y_spacing = [0, main_bbox.height * 0.20, main_bbox.height * 0.40, main_bbox.height * 0.6]
    elif infl_configs['infl_type'] == 'beta' and num_agents == 3:
        y_start = main_bbox.y0 + main_bbox.height * 0.55
        center_x = [main_bbox.x0 + main_bbox.width / 2 + 0.12] * 4
        y_spacing = [0, main_bbox.height * 0.1, main_bbox.height * 0.33, main_bbox.height * 0.45]
    elif num_agents == 4:
        center_x = ([main_bbox.x0 + main_bbox.width / 2 - 0.0075] * 3
                    + [main_bbox.x0 + main_bbox.width / 2 + 0.11])
        y_spacing = [0, main_bbox.height * 0.15, main_bbox.height * 0.41, main_bbox.height * 0.12]
    elif num_agents >= 4:
        y_start = main_bbox.y0 + main_bbox.height * 0.45
        center_x = ([main_bbox.x0 + main_bbox.width / 2 - 0.0075] * 2
                    + [main_bbox.x0 + main_bbox.width / 2 + 0.11] * 2)
        y_spacing = [0, main_bbox.height * 0.2, main_bbox.height * -0.05, main_bbox.height * 0.07]
    else:
        center_x = [main_bbox.x0 + main_bbox.width / 2 - 0.0075] * 4
        y_spacing = [0, main_bbox.height * 0.15, main_bbox.height * 0.30, main_bbox.height * 0.55]

    for idx, label in enumerate(region_labels):
        label_y = y_start - y_spacing[idx]
        label_x = center_x[idx]
        box = FancyBboxPatch(
            (label_x, label_y), 0.007, 0.007,
            boxstyle="round,pad=0.005",
            edgecolor='black', facecolor='white',
            linewidth=1, transform=fig.transFigure, zorder=1000
        )
        fig.patches.append(box)
        fig.text(
            label_x + 0.0035, label_y + 0.0035, label,
            ha='center', va='center',
            fontsize=28, fontweight='bold',
            transform=fig.transFigure, zorder=1001
        )

    # --- Region equilibrium analysis panels ---
    target_axes = [ax1_new, ax2_new, ax3_new, ax4_new]
    for idx, (results_dict, target_ax, label) in enumerate(
            zip(results_list, target_axes, region_labels)):
        fig_temp, axes_temp = search.plot_equilibrium_analysis(
            results_dict=results_dict,
            plot_types=['convergence', 'distribution'],
            save=False,
            font={'default_size': 20, 'title_size': 27, 'label_size': 14,
                  'tick_size': 12, 'legend_size': 12},
            paper_figure={
                'paper': paper_figure['paper'],
                'section': paper_figure['section'],
                'figure_id': f"{paper_figure.get('figure_id', 'bifurcation_equilibrium_combined')}_region{idx + 1}"
            },
            name_ads=[f'region{idx + 1}']
        )
        transfer_multi_subplot_figure(fig_temp, target_ax, label=label)
        plt.close(fig_temp)

    plt.tight_layout()

    if save:
        file_names = data_management.data_final_name(
            {
                'data_type': 'plot',
                'plot_type': 'bifurcation_equilibrium_combined',
                'domain_type': '1d',
                'num_agents': vis.num_agents,
                'section': paper_figure['section'],
                'figure_id': paper_figure.get('figure_id', 'bifurcation_equilibrium_combined')
            },
            name_ads=[], save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            fig.savefig(file_name, bbox_inches='tight')

    plt.show()
    return fig

