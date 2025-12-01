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

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict, Optional
import matplotlib.figure

import InflGame.utils.general as general
import InflGame.domains.one_d.one_utils as one_utils


def pos_plot_1d(num_agents: int,
                pos_matrix: torch.Tensor,
                domain_bounds: Tuple[float, float],
                title_ads: Optional[List[str]] = [],
                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12, 'font_family': 'sans-serif'},
                axis_return: Optional[bool] = False
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
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    for a_id in range(num_agents):
        ax.plot(domain,pos_matrix[:,a_id].numpy(),label='Agent '+ str(a_id+1))
    #ax.axhline(y=self.mean,color='r', linestyle='--',label='Mean')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Influencer location')
    plt.xlim(0,num_points)
    plt.ylim(domain_bounds[0],domain_bounds[1])
    plt.legend()
    title="Agent Positions"
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
    X_np = X.numpy()
    Y_np = Y.numpy()
    U_np = U.detach().numpy()
    V_np = V.detach().numpy()
    
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
        ax0.plot(domain,pos_matrix[:,a_id].numpy(),color=cm(1.*a_id/NUM_COLORS))
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
        ax1.plot(bin_points,infl_dist[agent_id].numpy(),color=cm(1.*agent_id/NUM_COLORS),label='Player '+str(agent_id))
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
            positions = pos_data.numpy()
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
        
        # Add individual agent trajectory lines
        if positions.ndim == 2:
            
            num_agents = positions.shape[1]
            for agent_id in range(num_agents):
                # Get positions for this agent across all parameters
                agent_trajectory = positions[:len(reach_parameters), agent_id]
                
                # Remove NaN values but keep track of valid indices
                valid_mask = ~np.isnan(agent_trajectory)
                valid_params = reach_parameters[valid_mask]
                valid_positions = agent_trajectory[valid_mask]
                
                # Plot the trajectory line
                ax.plot(valid_params, valid_positions, 
                        color=trajectory_cmap,
                        linestyle='--', 
                        linewidth=2, 
                        alpha=1)
                
    
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
                                    envelope_alpha: float = 0.3
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
    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)
    
    # Extract extreme positions following project patterns
    max_positions = extreme_positions['max']
    min_positions = extreme_positions['min']

    
    
    # Convert to numpy if needed following project patterns
    if hasattr(max_positions, 'numpy'):
        max_pos_np = max_positions.numpy()
    else:
        max_pos_np = np.array(max_positions)
        
    if hasattr(min_positions, 'numpy'):
        min_pos_np = min_positions.numpy()
    else:
        min_pos_np = np.array(min_positions)
    
    # Ensure reach_parameters is numpy array and properly 1D
    if hasattr(reach_parameters, 'numpy'):
        reach_params_np = reach_parameters.numpy()
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

        # Create density matrix using extreme positions
        # Count unique extreme positions per agent to avoid double-counting overlaps
        
        # For each parameter point, count extreme positions per agent individually
        combined_positions = []
        for i in range(max_pos_np.shape[0]):  # For each parameter value
            # Get max and min positions for this parameter
            max_row = max_pos_np[i, :]
            min_row = min_pos_np[i, :]
            
            # For each agent, collect their unique extreme positions
            agent_extremes = []
            for agent_id in range(num_agents):
                max_val = max_row[agent_id] 
                min_val = min_row[agent_id]
                
                # Create set of unique positions for this agent
                agent_unique_positions = set()
                
                # Only add valid (non-NaN) positions
                if not np.isnan(max_val):
                    # Round to avoid floating point precision issues
                    agent_unique_positions.add(round(max_val, 6))
                if not np.isnan(min_val):
                    agent_unique_positions.add(round(min_val, 6))
                
                # Convert to sorted list and extend agent_extremes
                agent_extremes.extend(sorted(list(agent_unique_positions)))
            
            # Store all agent extremes for this parameter point
            combined_positions.append(np.array(agent_extremes))
        
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
                
                # Create discrete colormap for extreme position count per agent
                max_extremes = int(density_matrix.max())
                if max_extremes > 0:
                    # The maximum should be 2*num_agents (if all agents have different min/max)
                    # But could be lower if some agents have min == max
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
                envelope_params_np = envelope_params[:, 0].numpy()
            else:
                envelope_params_np = envelope_params[:, 0]
                
            if hasattr(max_pos_envelope, 'numpy'):
                max_pos_envelope_np = max_pos_envelope.numpy()
                min_pos_envelope_np = min_pos_envelope.numpy()
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

        for key, value in bifurcation_types.items():
            # Skip if this key is in the excluded indices
            if int(key) in excluded_indices:
                continue
                
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
                cycle_end = cycle_reach_params[min(len(cycle_reach_params) - 1, max_idx + 2)]
                
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
            span = ax.axvspan(x_start, x_end, alpha=0.1, color=color, zorder=0)
            
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

        # Create legend in upper right corner of main plot
        ax.legend(handles=combined_handles, labels=combined_labels, 
                    loc='upper right', 
                    fontsize=legend_font_size, title='Legend', framealpha=0.9)

        ax.set_xlim(reach_start,reach_end)

        
        # Create rectangle bifurcation plot in separate subplot
        if len(bifurcation_items) > 0:
            import matplotlib.patches as patches
            from matplotlib.lines import Line2D
            
            # Create rectangle plot axes
            ax_rect = fig.add_subplot(gs[1])
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
            
            # Sort by position
            all_bifurcations.sort(key=lambda x: x[2])
            
            # Draw vertical bifurcation lines with alternating labels
            label_counter = 0
            for bif_type, idx, reach_param in all_bifurcations:
                x_pos = sigma_to_x(reach_param)
                
                if bif_type == 'type1':
                    color = 'red'
                    linestyle = '--'
                    alpha = 0.9
                    label_text = f'$\\sigma_{idx+1}^1$'
                elif bif_type == 'type2':
                    color = 'blue'
                    linestyle = ':'
                    alpha = 0.9
                    label_text = f'$\\sigma_{idx+1}^2$'
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
            ax_rect.set_ylim(0, 4)
            ax_rect.set_xlabel('Sigma ($\\sigma$) - Agent Reach Parameter', fontsize=default_font_size)
            ax_rect.set_title(r'Bifurcation Regions on $\sigma$', fontsize=title_font_size)
            
            # Add sigma value labels below rectangle
            sigma_step = (sigma_max - sigma_min) / 10
            sigma_labels_vals = [sigma_min + i * sigma_step for i in range(11)]
            for sigma_val in sigma_labels_vals:
                x_pos = sigma_to_x(sigma_val)
                ax_rect.text(x_pos, rect_y_start - 0.45, f'{sigma_val:.2f}',
                            ha='center', va='top', fontsize=rect_sigma_font_size, color='black')
            
            # Create legend with sorted bifurcation values
            legend_data = []
            
            for i, reach_param in enumerate(type1_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    legend_data.append((reach_param, 'red', '--', f'$\\sigma_{i+1}^1 = {reach_param:.4f}$'))
            
            for j, reach_param in enumerate(type2_bifurcations):
                if sigma_min <= reach_param <= sigma_max:
                    legend_data.append((reach_param, 'blue', ':', f'$\\sigma_{j+1}^2 = {reach_param:.4f}$'))
            
            if len(cycle_reach_params) > 0:
                cycle_end_param = max(cycle_reach_params)
                if sigma_min <= cycle_end_param <= sigma_max:
                    legend_data.append((cycle_end_param, 'purple', '-.', f'$\\sigma^{{c}} = {cycle_end_param:.4f}$'))
            
            # Sort by sigma value in descending order
            legend_data.sort(key=lambda x: x[0], reverse=True)
            
            # Create legend elements
            legend_elements = []
            for reach_param, color, linestyle, label in legend_data:
                legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=label))
            
            # Add legend
            ax_rect.legend(handles=legend_elements, loc='upper right', fontsize=rect_label_font_size, framealpha=0.9)
            
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
            
            if cbar_label_alignment == 'center':
                cbar.ax.tick_params(axis='y', which='major', pad=5)
                for label in cbar.ax.get_yticklabels():
                    label.set_horizontalalignment('center')

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
        title = str(num_agents) + ' Adaptive Agents\' Envelope of Closed orbits'

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
                                    envelope_alpha: float = 0.3
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
    cbar_center_labels = cbar_config.get('center_labels', True)
    cbar_label_alignment = cbar_config.get('label_alignment', 'center')
    cbar_shrink = cbar_config.get('shrink', 1)
    
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_box_aspect(1)
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
    
    param_int=torch.max(torch.max(non_equal[mask1]),torch.max(non_equal[mask2]))
    for matrix_id in range(len(matrix_list)):
        if matrix_id == 1:  # Envelope data
            extreme_positions = matrix_list[matrix_id]
            max_positions = extreme_positions['max']
            min_positions = extreme_positions['min']
            
            # Convert to numpy following project patterns
            if hasattr(max_positions, 'numpy'):
                max_pos_np = max_positions.numpy()
                min_pos_np = min_positions.numpy()
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
                positions = pos_data.numpy()
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
        reach_params_np = reach_parameters.numpy()
    else:
        reach_params_np = np.array(reach_parameters)
    
    if reach_params_np.ndim > 1:
        reach_params_np = reach_params_np.flatten()
    reach_params_np = np.atleast_1d(reach_params_np).flatten()
    
    # Initialize global density matrix
    density_matrix = np.zeros((len(position_bins)-1, len(reach_params_np)))
    
    # Process each matrix with consistent binning
    for matrix_id in range(len(matrix_list)):
        if matrix_id == 1:
            # Handle envelope data (dictionary with 'max' and 'min' keys)
            extreme_positions = matrix_list[matrix_id]
            
            # Extract extreme positions following project patterns
            max_positions = extreme_positions['max']
            min_positions = extreme_positions['min']
            
            # Convert to numpy following project patterns
            if hasattr(max_positions, 'numpy'):
                max_pos_np = max_positions.numpy()
                min_pos_np = min_positions.numpy()
            else:
                max_pos_np = np.array(max_positions)
                min_pos_np = np.array(min_positions)
            
            if plot_type == "heat":
                # Create density matrix using extreme positions with consistent binning
                combined_positions = []
                for i in range(max_pos_np.shape[0]):
                    max_row = max_pos_np[i, :]
                    min_row = min_pos_np[i, :]
                    
                    agent_extremes = []
                    for agent_id in range(num_agents):
                        max_val = max_row[agent_id] 
                        min_val = min_row[agent_id]
                        
                        agent_unique_positions = set()
                        if not np.isnan(max_val):
                            agent_unique_positions.add(round(max_val, 6))
                        if not np.isnan(min_val):
                            agent_unique_positions.add(round(min_val, 6))
                        
                        agent_extremes.extend(sorted(list(agent_unique_positions)))
                    
                    combined_positions.append(np.array(agent_extremes))
                
                # Create density matrix using global position bins
                density_matrix_iter = np.zeros((len(position_bins)-1, len(reach_params_np)))
                
                for i in range(min(len(reach_params_np), len(combined_positions))):
                    agent_positions = combined_positions[i]
                    valid_pos = agent_positions[~np.isnan(agent_positions)]
                    
                    if len(valid_pos) > 0:
                        counts, _ = np.histogram(valid_pos, bins=position_bins)
                        density_matrix_iter[:, i] = counts
                
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
                        envelope_params_np = envelope_params.numpy()
                        max_pos_envelope_np = max_pos_envelope.numpy()
                        min_pos_envelope_np = min_pos_envelope.numpy()
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
                
                # Add individual agent trajectory lines
                num_agents_matrix = positions.shape[1]
                for agent_id in range(num_agents_matrix):
                    #agent_trajectory = positions[:len(reach_params_np), agent_id]
                    #valid_mask = ~np.isnan(agent_trajectory)
                    #valid_params = reach_params_np[valid_mask]
                    #valid_positions = agent_trajectory[valid_mask]
                    
                    #ax.plot(valid_params, valid_positions, color=trajectory_cmap,
                    #       linestyle='--', linewidth=2, alpha=1)
                    ax.plot(reach_parameters[:param_int], positions[:param_int, agent_id], color=trajectory_cmap,
                           linestyle='--', linewidth=2, alpha=1)
    
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
    
    # Optional vertical lines
    if optional_vline is not None:
        for vline_id, vline_val in enumerate(optional_vline):
            ax.axvline(x=vline_val, ymin=0, ymax=1, color='black', 
                      linestyle='dashed', alpha=0.7,
                      label=r'$\sigma^*_' + str(vline_id + 1) + r'=$' + str(np.around(vline_val, decimals=4)))
    
    ax.vlines(x=reach_parameters[param_int], ymin=0, ymax=1, colors='blue', linestyles='dashed',label='$\sigma^*_' + str(vline_id + 2) + r'=$' + str(np.around(reach_parameters[param_int].item(), decimals=4)))


    # Legend handling following project patterns
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower center')
    
    # Title formatting following project patterns
    if short_title == True:
        title = 'Adaptive Agents Envelope'
    else:
        title = str(num_agents) + ' Adaptive Agents\' Envelope of Closed orbits'

    if len(title_ads) > 0:
        for title_addition in title_ads:
            title = title + " " + title_addition
    
    plt.title(title, fontsize=title_font_size)
    
    if infl_type == 'gaussian':
        plt.xlabel(r"$\sigma$ (std)")
    else:
        plt.xlabel(r"$\sigma$")
    
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

