
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

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pylab
import matplotlib.figure
import matplotlib as mpl

def dist_and_pos_plot_2d_simple(num_agents: int,
                                bin_points: np.ndarray,
                                rect_X: np.ndarray,
                                rect_Y: np.ndarray,
                                cmap1,
                                cmap2,
                                pos_matrix: torch.Tensor,
                                infl_dist: torch.Tensor,
                                resource_type: str,
                                resources: str = 0,
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


    NUM_COLORS = num_agents+1
    cm = pylab.get_cmap(cmap1)
    fig = plt.figure(figsize=(19, 7))
    gs = GridSpec(nrows=num_agents, ncols=2,width_ratios=[1, 1],wspace=0.0, hspace=0.2, top=1, bottom=0.05, left=0.17, right=0.845)

    ax0 = fig.add_subplot(gs[:, 0])
    for a_id in range(num_agents):
        new_coor=pos_matrix[:,a_id]
        y=new_coor.detach().numpy()
        ax0.scatter(y[-1,0],y[-1,1],s=70,color=cm(1.*a_id/NUM_COLORS),linewidth=0.3,label='Agent '+str(a_id))
        ax0.scatter(y[0,0],y[0,1],s=70,facecolors='none',edgecolors=cm(1.*a_id/NUM_COLORS),linewidth=1)
        ax0.plot(y[:,0],y[:,1],color=cm(1.*a_id/NUM_COLORS))
    if resource_type in  ["multi_modal_gaussian_distribution_2D",'multi_modal_gaussian_distribution_2D_square','multi_modal_gaussian_distribution_2D_square''multi_modal_gaussian_distribution_2D_triangle']:
        pval=resources.reshape(len(rect_Y),len(rect_X))
        im = ax0.pcolormesh(rect_X,rect_Y, pval,alpha=.2)
        

    else:
        for b_id in range(len(bin_points)):
            new_coor=torch.tensor(bin_points[b_id])
            y=new_coor.detach().numpy()
            if b_id==0:
                ax0.scatter(y[0],y[1],color=cm(1.*(a_id+1)/NUM_COLORS),s=70,linewidth=0.3,label='Resource point')
            else:
                ax0.scatter(y[0],y[1],color=cm(1.*(a_id+1)/NUM_COLORS),s=70,linewidth=0.3)
    plt.title('Agents positions in time',y=1.05,fontsize=title_font_size)
    plt.legend(title="End pos", bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

    for a_id in range(num_agents):
        ax1 = fig.add_subplot(gs[a_id, 1])
        pvals=infl_dist[a_id].numpy()
        pvals=pvals.reshape(len(rect_Y),len(rect_X))
        pcm = ax1.pcolormesh(rect_X,rect_Y, pvals,cmap=cmap2)
        ax1.axis('equal')
        ax1.axis('off')
        plt.title("Player "+str(a_id),x=.15,y=0.5,fontsize=sub_title_font_size)

    ax2 = fig.add_subplot(gs[:, 1])
    fig.colorbar(pcm,ax=ax2, extend='max')
    plt.title('Players\' influence distributions',x=.65,y=1.05,fontsize=sub_title_font_size)
    ax2.axis('off')
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
