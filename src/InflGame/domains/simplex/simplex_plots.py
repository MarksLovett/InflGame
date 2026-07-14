"""
.. module:: simplex_plots
   :synopsis: Provides visualization tools for analyzing agent dynamics and resource distributions on simplex domains in influencer games.

Simplex Visualization Module
============================

This module provides visualization tools for analyzing and understanding the dynamics of agents and resource distributions on simplex domains for influencer games.
It includes utilities for plotting agent positions, influence distributions, and bifurcation dynamics on simplex domains.

The module is designed to work with the `InflGame.adaptive` subpackage and supports creating visual representations of agent behaviors and resource distributions in simplex environments.

Dependencies:
-------------
- numpy
- torch
- matplotlib
- InflGame.domains.simplex.simplex_utils

Usage:
------
The functions in this module can be used to visualize agent dynamics and resource distributions on simplex domains. For example, the `pos_plot_simplex` function
can be used to plot agent positions on a simplex, while the `dist_and_pos_plot_simplex` function can visualize both agent positions and influence distributions.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec
import pylab
import matplotlib.figure
import matplotlib as mpl
from typing import Tuple

import InflGame.domains.simplex.simplex_utils as simplex_utils

def pos_plot_simplex(num_agents: int,
                     bin_points: np.ndarray,
                     corners: np.ndarray,
                     triangle,
                     pos_matrix: torch.Tensor,
                     font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif','sub_title_size':12},
                     fig_size:Tuple=(18, 18)
                     ) -> matplotlib.figure.Figure:
    """
    Plots the positions of agents and bin points on a simplex.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    bin_points : numpy.ndarray
        Array of bin points in barycentric coordinates.
    corners : numpy.ndarray
        Coordinates of the simplex corners.
    triangle : matplotlib.tri.Triangulation
        Triangulation object for the simplex.
    pos_matrix : torch.Tensor
        Position matrix of agents in barycentric coordinates.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig,ax=plt.subplots(figsize=fig_size)
    ax.set_box_aspect(1)
    ax.triplot(triangle,linewidth=0.8,color="black")
    for a_id in range(num_agents):
        new_coor=simplex_utils.ba2xy(pos_matrix[:,a_id],corners=corners)
        y=new_coor.detach().cpu().numpy()
        ax.scatter(y[0,0],y[0,1],s=70,linewidth=0.3)
        ax.plot(y[:,0],y[:,1])
    if len(bin_points)<=10:
        for b_id in range(len(bin_points)):
            new_coor=simplex_utils.ba2xy(torch.tensor(bin_points[b_id]),corners=corners)
            y=new_coor.detach().cpu().numpy()
            ax.scatter(y[0],y[1],c="red",s=70,linewidth=0.3)
    plt.title('Agents positions in time',y=1.05, fontsize=title_font_size)
    return fig

def dist_plot_simplex(agent_id: int,
                      r2: list[float],
                      corners: np.ndarray,
                      triangle,
                      trimesh,
                      infl_dist: torch.Tensor,
                      cmap,
                      typelabels: list[str],
                      margin: float = .01,
                      font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif'},
                      **kwargs):
    """
    Plots the influence distribution of a single agent on a simplex.

    Parameters
    ----------
    agent_id : int
        ID of the agent.
    r2 : list[float]
        Range of the simplex.
    corners : numpy.ndarray
        Coordinates of the simplex corners.
    triangle : matplotlib.tri.Triangulation
        Triangulation object for the simplex.
    trimesh : matplotlib.tri.Triangulation
        Triangulation mesh for contour plotting.
    infl_dist : torch.Tensor
        Influence distribution tensor.
    cmap : matplotlib.colors.Colormap
        Colormap for the plot.
    typelabels : list[str]
        Labels for the simplex corners.
    margin : float
        Margin for the plot axes.
    kwargs
        Additional arguments for contour plotting.

    Returns
    -------
    None
    """

    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig,ax= plt.subplots(figsize=(12, 8))
    ax.triplot(triangle,linewidth=0.8,color="black")
    # Convert to numpy for matplotlib (handle GPU tensors)
    pvals = infl_dist[agent_id].cpu().numpy() if torch.is_tensor(infl_dist[agent_id]) else infl_dist[agent_id]
    pcm=ax.tricontourf(trimesh, pvals, alpha=0.8,levels=100, cmap=cmap,**kwargs)
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(ymin=-margin,ymax=r2[1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)

    #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
    
    ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
    fig.colorbar(pcm, ax=ax, extend='max')
    plt.title('Agent '+str(agent_id)+'\'s'+' influence distribution',fontsize=title_font_size,y=1.05)

def dist_and_pos_plot_simplex(num_agents: int,
                              bin_points: np.ndarray,
                              r2: list[float],
                              corners,
                              triangle,
                              trimesh,
                              typelabels,
                              cmap1,
                              cmap2,
                              pos_matrix: torch.Tensor,
                              infl_dist: torch.Tensor,
                              resource_type: str,
                              resources: np.ndarray = 0,
                              font: dict = {'default_size': 12, 'cbar_size': 12, 'title_size': 14, 'subtitle_size': 12, 'legend_size': 12,'font_family': 'sans-serif'},
                              ) -> matplotlib.figure.Figure:
    """
    Plots both the positions of agents and their influence distributions on a simplex.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    bin_points : numpy.ndarray
        Array of bin points in barycentric coordinates.
    r2 : list[float]
        Range of the simplex.
    corners : numpy.ndarray
        Coordinates of the simplex corners.
    triangle : matplotlib.tri.Triangulation
        Triangulation object for the simplex.
    trimesh : matplotlib.tri.Triangulation
        Triangulation mesh for contour plotting.
    typelabels : list[str]
        Labels for the simplex corners.
    cmap1 : matplotlib.colors.Colormap
        Colormap for agent positions.
    cmap2 : matplotlib.colors.Colormap
        Colormap for influence distributions.
    pos_matrix : torch.Tensor
        Position matrix of agents in barycentric coordinates.
    infl_dist : torch.Tensor
        Influence distribution tensor.
    resource_type : str
        Type of resource distribution.
    resources : numpy.ndarray
        Resource distribution values.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """

    
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    subtitle_font_size = font.get('subtitle_size', 12)
    legend_font_size = font.get('legend_size', 12)
    cbar_font_size = font.get('cbar_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size

    NUM_COLORS = num_agents+1
    cm = pylab.get_cmap(cmap1)
    fig = plt.figure(figsize=(19, 7))
    gs = GridSpec(nrows=num_agents, ncols=2,width_ratios=[1, 1],wspace=0.0, hspace=0.2, top=1, bottom=0.05, left=0.17, right=0.845)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.triplot(triangle,linewidth=0.8,color="black")
    for a_id in range(num_agents):
        new_coor=simplex_utils.ba2xy_vectorized(pos_matrix[:,a_id],corners=corners)
        y=new_coor.detach().cpu().numpy()
        ax0.scatter(y[-1,0],y[-1,1],s=70,color=cm(1.*a_id/NUM_COLORS),linewidth=0.3,label='Agent '+str(a_id))
        ax0.scatter(y[0,0],y[0,1],s=70,facecolors='none',edgecolors=cm(1.*a_id/NUM_COLORS),linewidth=1)
        ax0.plot(y[:,0],y[:,1],color=cm(1.*a_id/NUM_COLORS))
    if resource_type in  ["multi_modal_gaussian_distribution_2D","multi_modal_gaussian_distribution_2D_square","multi_modal_gaussian_distribution_2D_triangle","dirichlet_distribution"]:
        ax0.triplot(triangle,linewidth=0.8,color="black")
        im=ax0.tricontourf(trimesh, resources, alpha=0.3,levels=100)
        ax0.axis('equal')
        ax0.axis('off')
        margin=0.01

        ax0.set_ylim(ymin=-margin,ymax=r2[1]+margin)
        ax0.set_xlim(xmin=-margin,xmax=1.+margin)
        ax0.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
        ax0.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
        ax0.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')

    else:
        for b_id in range(len(bin_points)):
            new_coor=simplex_utils.ba2xy(torch.tensor(bin_points[b_id]),corners=corners)
            y=new_coor.detach().cpu().numpy()
            if b_id==0:
                ax0.scatter(y[0],y[1],color=cm(1.*(a_id+1)/NUM_COLORS),s=70,linewidth=0.3,label='Resource point')
            else:
                ax0.scatter(y[0],y[1],color=cm(1.*(a_id+1)/NUM_COLORS),s=70,linewidth=0.3)
        
        ax0.axis('equal')
        ax0.axis('off')
        margin=0.01

        ax0.set_ylim(ymin=-margin,ymax=r2[1]+margin)
        ax0.set_xlim(xmin=-margin,xmax=1.+margin)
        ax0.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
        ax0.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
        ax0.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')

        
    plt.title('Agents positions in time',y=1.05)
    plt.legend(title="End pos")

    for a_id in range(num_agents):
        ax1 = fig.add_subplot(gs[a_id, 1])
        pvals = infl_dist[a_id].cpu().numpy() if torch.is_tensor(infl_dist[a_id]) else infl_dist[a_id]
        ax1.triplot(triangle,linewidth=0.8,color="black")
        pcm=ax1.tricontourf(trimesh, pvals, alpha=0.8,levels=100, cmap=cmap2)
        ax1.axis('equal')
        ax1.axis('off')
        margin=0.01
        ax1.set_ylim(ymin=-margin,ymax=r2[1]+margin)
        ax1.set_xlim(xmin=-margin,xmax=1.+margin)

        #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)

        ax1.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
        ax1.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
        ax1.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
        plt.title("Agent "+str(a_id),x=.25,y=0.25,fontsize=subtitle_font_size)
    ax2 = fig.add_subplot(gs[:, 1])
    fig.colorbar(pcm,ax=ax2, extend='max')
    plt.title('Agents\' influence distributions',x=.65,y=1.05,fontsize=title_font_size)
    ax2.axis('off')
    plt.close()
    return fig



def equalibirium_bifurication_plot_simplex(num_agents: int,
                                           r2: list[float],
                                           corners,
                                           triangle,
                                           final_pos_matrix: torch.Tensor,
                                           reach_num_points: int,
                                           title_ads: list[str],
                                           type_labels: list[str] = None,
                                           font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif','sub_title_size':12},
                                           ) -> matplotlib.figure.Figure:
    """
    Plots the equilibrium bifurcation of agents on a simplex.

    Parameters
    ----------
    num_agents : int
        Number of agents.
    r2 : list[float]
        Range of the simplex.
    corners : numpy.ndarray
        Coordinates of the simplex corners.
    triangle : matplotlib.tri.Triangulation
        Triangulation object for the simplex.
    final_pos_matrix : torch.Tensor
        Final positions of agents in barycentric coordinates.
    reach_num_points : int
        Number of points reached.
    title_ads : list[str]
        Additional strings for the plot title.
    type_labels : list[str], optional
        Labels for the simplex corners. Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size
    
    fig, ax = plt.subplots()
    if type_labels==None:
        type_labels=["A","B","C"]
    ax.set_box_aspect(1)
    c=range(reach_num_points)
    ax.triplot(triangle,linewidth=0.8,color="black")
    for agent_id in range(num_agents): 
        ax.scatter(final_pos_matrix[:,agent_id][:,0].cpu().numpy(),final_pos_matrix[:,agent_id][:,1].cpu().numpy(),c=c, cmap='rainbow')
    ax.annotate(type_labels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(type_labels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(type_labels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
    title="Bifurcation of " +str(num_agents)+r" players with varying $\sigma$ values"
    if len(title_ads)>0:
        for title_additon in title_ads:
            title=title+" "+title_additon
    ax.set_title(title,fontsize=title_font_size)
    margin=.2
    ax.set_ylim(ymin=-margin,ymax=r2[1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)
    plt.close()
    return fig

def simplex_plot_resources(domain_bounds: tuple,
                           resources: np.ndarray) -> matplotlib.figure.Figure:
    """
    Plots the resource distribution on a simplex.

    Parameters
    ----------
    domain_bounds : tuple
        Bounds of the simplex domain.
    resources : numpy.ndarray
        Resource distribution values.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """
    typelabels=["A","B","C"]
    fig,ax = plt.subplots() 
    ax.triplot(domain_bounds[2],linewidth=0.8,color="black")
    pcm=ax.tricontourf(domain_bounds[3], resources, alpha=0.8,levels=100)
    ax.axis('equal')
    ax.axis('off')
    margin=0.01


    ax.set_ylim(ymin=-margin,ymax=domain_bounds[0][1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)
    ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[2],domain_bounds[1][2],xytext=domain_bounds[1][2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')



##incomplete

# def vector_plot_simplex():
    
#     fig,ax=plt.subplots()
    
#     ax.triplot(self.triangle,linewidth=0.8,color="black")
#     pcm=ax.tricontourf(self.trimesh, self.pvals,norm=colors.LogNorm(vmin=np.array(self.pvals).min(), vmax=np.array(self.pvals).max()), alpha=0.8, cmap=cmap,**kwargs)

#     #arrow plot options:
#     # Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],self.pvals,angles='xy',pivot='mid',  cmap=cmap)#pivot='tail',
#     Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],angles='xy',pivot='mid')#pivot='tail')#
#     # Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction.T[0],self.direction.T[1],angles='xy',pivot='mid')#pivot='tail')#


#     ax.axis('equal')
#     ax.axis('off')
#     margin=0.01
#     ax.set_ylim(ymin=-margin,ymax=self.r2[1]+margin)
#     ax.set_xlim(xmin=-margin,xmax=1.+margin)

#     #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
    
#     ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
#     ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
#     ax.annotate(typelabels[2],self.corners[2],xytext=self.corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
#     fig.colorbar(pcm, ax=ax, extend='max')
#     plt.title('Agent '+str(agent_id)+'\'s'+' gradient vector self')


def agent_density_3d_simplex(
    pos_matrix: np.ndarray,
    num_agents: int,
    domain_bounds: tuple,
    bins: int = 10,
    distance_threshold: float = 0.05,
    cmap: str = 'viridis',
    font: dict = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
    figsize: Tuple = (20, 16),
    xlabel: str = r'$x$',
    ylabel: str = r'$y$',
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
    Create a 3D histogram showing agent density at final positions for simplex domain.
    
    Parameters
    ----------
    pos_matrix : np.ndarray or torch.Tensor
        Position matrix of shape (time_steps, num_agents, 3) in barycentric coordinates.
    num_agents : int
        Number of agents.
    domain_bounds : tuple
        Simplex domain bounds tuple containing (r2, corners, triangle, trimesh).
    bins : int
        Number of bins in each dimension.
    distance_threshold : float
        Distance threshold for clustering nearby agents.
    cmap : str
        Colormap name.
    font : dict
        Font configuration dictionary.
    figsize : tuple
        Figure size as (width, height).
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    zlabel : str
        Label for z-axis.
    axis_return : bool
        If True, return axes object; if False, return figure object.
    edgecolor : str
        Color of outlines around bars.
    linewidth : float
        Width of bar edge lines.
    alpha : float
        Bar transparency.
    title_ads : list
        Additional titles for the plot.
    save : bool
        Whether to save the plot.
    name_ads : list
        Additional names for saved files.
    save_types : list
        File types to save the plot.
    paper_figure : dict
        Dictionary for paper figure naming.
    id : int
        Identifier for file naming.
    cap_z_axis : bool
        If True, cap the z-axis maximum at num_agents.
    integer_ticks : bool
        If True, only show integer ticks on the z-axis.

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
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
    
    # Extract corners from domain_bounds
    if isinstance(domain_bounds, (list, tuple)):
        corners = domain_bounds[1]
    else:
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
    
    if torch.is_tensor(corners):
        corners = corners.cpu().numpy()
    
    # Get final positions
    if pos_matrix.ndim == 3:
        final_positions_bary = pos_matrix[-1, :, :]
    else:
        final_positions_bary = pos_matrix
    
    # Convert barycentric to Cartesian
    final_positions = final_positions_bary @ corners
    
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
    
    x_final = cluster_centers[:, 0]
    y_final = cluster_centers[:, 1]
    
    # Define bounds
    x_min, x_max = corners[:, 0].min() - 0.05, corners[:, 0].max() + 0.05
    y_min, y_max = corners[:, 1].min() - 0.05, corners[:, 1].max() + 0.05
    
    # Create histogram with cluster weights
    h, xedges, yedges = np.histogram2d(
        x_final, y_final,
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
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
    ax.set_ylabel(ylabel, labelpad=15)
    ax.set_zlabel(zlabel, labelpad=15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if cap_z_axis:
        ax.set_zlim(0, num_agents)
    if integer_ticks:
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_title(title, fontsize=title_font_size)

    # Draw triangle outline
    triangle_x = np.append(corners[:, 0], corners[0, 0])
    triangle_y = np.append(corners[:, 1], corners[0, 1])
    triangle_z = np.zeros_like(triangle_x)
    ax.plot(triangle_x, triangle_y, triangle_z, 'k-', linewidth=2, alpha=0.7)

    if save:
        file_names = data_management.data_final_name(
            {'data_type': 'plot', 'plot_type': 'agent_density_3d', 'section': paper_figure['section'], 
             'figure_id': paper_figure.get('figure_id', 'agent_density_3d'), "num_agents": num_agents, 'domain_type': 'simplex'},
            name_ads=name_ads + [f'id_{id}'] if id else name_ads,
            save_types=save_types,
            paper_figure=paper_figure['paper']
        )
        for file_name in file_names:
            if file_name.lower().endswith('.svg'):
                plt.rcParams['svg.fonttype'] = 'none'
            fig.savefig(file_name, dpi=600 if file_name.lower().endswith('.svg') else 300, bbox_inches='tight')
    
    return ax if axis_return else fig
