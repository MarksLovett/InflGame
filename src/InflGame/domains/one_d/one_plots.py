"""
.. module:: one_plots
   :synopsis: Provides 1D visualization tools for analyzing agent dynamics and resource distributions in influencer games.

1D Visualization Module
=======================

This module provides visualization tools for analyzing and understanding the dynamics of agents and resource distributions in 1D domains for influencer games.
It includes utilities for plotti        # Create discrete colormap for agent count
        max_agents = int(density_matrix.max())
        if max_agents > 0:
            # Define discrete levels based on agent count
            levels = np.arange(0, max_agents + 2, 1)  # 0, 1, 2, ..., max_agents+1
            norm = mpl.colors.BoundaryNorm(levels, ncolors=256)
            
            # Create the heatmap with discrete colormap
            im = ax.imshow(density_matrix, aspect='auto', cmap='viridis', norm=norm, origin='lower',
                        extent=[reach_parameters[0], reach_parameters[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')  # Use 'nearest' for discrete appearance
        else:
            # Fallback for empty data
            im = ax.imshow(density_matrix, aspect='auto', cmap='viridis', origin='lower',
                        extent=[reach_parameters[0], reach_parameters[-1], 
                                position_bins[0], position_bins[-1]],
                        interpolation='nearest')ent positions, gradients, influence distributions, and bifurcation dynamics in 1D domains.

The module is designed to work with the `InflGame.adaptive` package and supports creating visual representations of agent behaviors and resource distributions in 1D environments.

Dependencies:
-------------
- InflGame.utils
- InflGame.domains

Usage:
------
The functions in this module can be used to visualize agent dynamics and resource distributions in 1D domains. For example, the `pos_plot_1d` function
can be used to plot agent positions over time, while the `dist_and_pos_plot_1d` function can visualize both agent positions and influence distributions.


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
    Plots the positions of agents over time in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param pos_matrix: Matrix of agent positions over time.
    :type pos_matrix: torch.Tensor
    :param domain_bounds: Bounds of the domain (min, max).
    :type domain_bounds: Tuple[float, float]
    :param title_ads: Additional strings to append to the plot title, defaults to [].
    :type title_ads: list, optional
    :return: The generated plot figure.
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
    Plots the gradients of agents over time in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param grad_matrix: Matrix of agent gradients over time.
    :type grad_matrix: torch.Tensor
    :param title_ads: Additional strings to append to the plot title, defaults to [].
    :type title_ads: list, optional
    :return: The generated plot figure.
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
    """
    Plots the probability distribution of influence for agents in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: List of agent positions.
    :type agents_pos: list
    :param bin_points: Points defining the bins for the probability distribution.
    :type bin_points: numpy.ndarray
    :param domain_bounds: Bounds of the domain (min, max).
    :type domain_bounds: list[float, float]
    :param prob: Probability distributions for each agent.
    :type prob: list
    :param voting_configs: Configuration for voting types.
    :type voting_configs: dict
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list, optional
    :return: The generated plot figure.
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
    Plots the dynamics of three players in a 3D space.

    :param pos_matrix: Matrix of player positions over time.
    :type pos_matrix: numpy.ndarray
    :param x_star: Reference point (e.g., mean position).
    :type x_star: float
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list
    :return: The generated plot figure.
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
    Plots a vector field for two players in a 1D domain.

    :param ids: List of player IDs.
    :type ids: list
    :param gradient: Gradient matrix for the vector field.
    :type gradient: torch.Tensor
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list, optional
    :return: The generated plot figure.
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
    Plots the resource distribution and agent positions over time in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Points defining the bins for the resource distribution.
    :type bin_points: numpy.ndarray
    :param resource_distribution: Resource distribution across the domain.
    :type resource_distribution: numpy.ndarray
    :param pos_matrix: Matrix of agent positions over time.
    :type pos_matrix: torch.Tensor
    :param len_grad_matrix: Length of the gradient matrix.
    :type len_grad_matrix: int
    :param infl_dist: Influence distributions for each agent.
    :type infl_dist: list
    :param cm: Colormap for plotting.
    :type cm: matplotlib.colors.Colormap
    :param NUM_COLORS: Number of colors to use in the colormap.
    :type NUM_COLORS: int
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list, optional
    :return: The generated plot figure.
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
                                    optional_vline: float = None
                                    ) -> matplotlib.figure.Figure:
    """
    Plots the equilibrium bifurcation for agents in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Points defining the bins for the resource distribution.
    :type bin_points: numpy.ndarray
    :param resource_distribution: Resource distribution across the domain.
    :type resource_distribution: numpy.ndarray
    :param infl_type: Type of influence (e.g., 'gaussian').
    :type infl_type: str
    :param reach_parameters: Parameters defining the reach of influence.
    :type reach_parameters: list
    :param final_pos_matrix: Final positions of agents.
    :type final_pos_matrix: numpy.ndarray
    :param reach_start: Start of the reach parameter range.
    :type reach_start: float
    :param reach_end: End of the reach parameter range.
    :type reach_end: float
    :param refinements: Number of refinements for critical value calculations.
    :type refinements: int
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list, optional
    :return: The generated plot figure.
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
                std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_'+str(len(crit_stds)-std_loc_id-1)+r'$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]])
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
            ax.vlines(x=optional_vline, ymin=0, ymax=1, colors='black', linestyles='dashed', label=r'$\sigma^*_2=$'+str(np.around(optional_vline,decimals=4)))
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




def final_position_histogram_1d(num_agents: int,
                                domain_bounds: Tuple[float, float],
                                current_alpha: float,
                                reach_parameter: float,
                                final_pos_vector: np.ndarray,
                                title_ads: Optional[List[str]],
                                font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'}
                                ) -> matplotlib.figure.Figure:
    """
    Plots a histogram of the final positions of agents in a 1D domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param domain_bounds: Bounds of the domain (min, max).
    :type domain_bounds: Tuple[float, float]
    :param current_alpha: Current alpha parameter.
    :type current_alpha: float
    :param reach_parameter: Current reach parameter.
    :type reach_parameter: float
    :param final_pos_vector: Final positions of agents.
    :type final_pos_vector: numpy.ndarray
    :param title_ads: Additional strings to append to the plot title.
    :type title_ads: list, optional
    :return: The generated plot figure.
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

