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

import InflGame.domains.simplex.simplex_utils as simplex_utils

def pos_plot_simplex(num_agents: int,
                     bin_points: np.ndarray,
                     corners: np.ndarray,
                     triangle,
                     pos_matrix: torch.Tensor,
                     font: dict = {'default_size': 12, 'title_size': 14, 'legend_size': 12,'font_family': 'sans-serif','sub_title_size':12},
                     ) -> matplotlib.figure.Figure:
    """
    Plots the positions of agents and bin points on a simplex.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Array of bin points in barycentric coordinates.
    :type bin_points: numpy.ndarray
    :param corners: Coordinates of the simplex corners.
    :type corners: numpy.ndarray
    :param triangle: Triangulation object for the simplex.
    :type triangle: matplotlib.tri.Triangulation
    :param pos_matrix: Position matrix of agents in barycentric coordinates.
    :type pos_matrix: torch.Tensor
    :return: The generated plot figure.
    :rtype: matplotlib.figure.Figure
    """
    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig,ax=plt.subplots(figsize=(12, 8))
    ax.set_box_aspect(1)
    ax.triplot(triangle,linewidth=0.8,color="black")
    for a_id in range(num_agents):
        new_coor=simplex_utils.ba2xy(pos_matrix[:,a_id],corners=corners)
        y=new_coor.detach().numpy()
        ax.scatter(y[0,0],y[0,1],s=70,linewidth=0.3)
        ax.plot(y[:,0],y[:,1])
    if len(bin_points)<=10:
        for b_id in range(len(bin_points)):
            new_coor=simplex_utils.ba2xy(torch.tensor(bin_points[b_id]),corners=corners)
            y=new_coor.detach().numpy()
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

    :param agent_id: ID of the agent.
    :type agent_id: int
    :param r2: Range of the simplex.
    :type r2: list[float]
    :param corners: Coordinates of the simplex corners.
    :type corners: numpy.ndarray
    :param triangle: Triangulation object for the simplex.
    :type triangle: matplotlib.tri.Triangulation
    :param trimesh: Triangulation mesh for contour plotting.
    :type trimesh: matplotlib.tri.Triangulation
    :param infl_dist: Influence distribution tensor.
    :type infl_dist: torch.Tensor
    :param cmap: Colormap for the plot.
    :type cmap: matplotlib.colors.Colormap
    :param typelabels: Labels for the simplex corners.
    :type typelabels: list[str]
    :param margin: Margin for the plot axes.
    :type margin: float
    :param kwargs: Additional arguments for contour plotting.
    :return: None
    """

    font['font.family'] = font.get('font_family', 'sans-serif')
    default_font_size = font.get('default_size', 12)
    title_font_size = font.get('title_size', 14)
    legend_font_size = font.get('legend_size', 12)
    mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
    mpl.rcParams['legend.fontsize'] = legend_font_size


    fig,ax= plt.subplots(figsize=(12, 8))
    ax.triplot(triangle,linewidth=0.8,color="black")
    pvals=infl_dist[agent_id].numpy()
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

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Array of bin points in barycentric coordinates.
    :type bin_points: numpy.ndarray
    :param r2: Range of the simplex.
    :type r2: list[float]
    :param corners: Coordinates of the simplex corners.
    :type corners: numpy.ndarray
    :param triangle: Triangulation object for the simplex.
    :type triangle: matplotlib.tri.Triangulation
    :param trimesh: Triangulation mesh for contour plotting.
    :type trimesh: matplotlib.tri.Triangulation
    :param typelabels: Labels for the simplex corners.
    :type typelabels: list[str]
    :param cmap1: Colormap for agent positions.
    :type cmap1: matplotlib.colors.Colormap
    :param cmap2: Colormap for influence distributions.
    :type cmap2: matplotlib.colors.Colormap
    :param pos_matrix: Position matrix of agents in barycentric coordinates.
    :type pos_matrix: torch.Tensor
    :param infl_dist: Influence distribution tensor.
    :type infl_dist: torch.Tensor
    :param resource_type: Type of resource distribution.
    :type resource_type: str
    :param resources: Resource distribution values.
    :type resources: numpy.ndarray
    :return: The generated plot figure.
    :rtype: matplotlib.figure.Figure
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
        y=new_coor.detach().numpy()
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
            y=new_coor.detach().numpy()
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
        pvals=infl_dist[a_id].numpy()
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

    :param num_agents: Number of agents.
    :type num_agents: int
    :param r2: Range of the simplex.
    :type r2: list[float]
    :param corners: Coordinates of the simplex corners.
    :type corners: numpy.ndarray
    :param triangle: Triangulation object for the simplex.
    :type triangle: matplotlib.tri.Triangulation
    :param final_pos_matrix: Final positions of agents in barycentric coordinates.
    :type final_pos_matrix: torch.Tensor
    :param reach_num_points: Number of points reached.
    :type reach_num_points: int
    :param title_ads: Additional strings for the plot title.
    :type title_ads: list[str]
    :param type_labels: Labels for the simplex corners. Defaults to None.
    :type type_labels: list[str], optional
    :return: The generated plot figure.
    :rtype: matplotlib.figure.Figure
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
        ax.scatter(final_pos_matrix[:,agent_id][:,0].numpy(),final_pos_matrix[:,agent_id][:,1].numpy(),c=c, cmap='rainbow')
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

    :param domain_bounds: Bounds of the simplex domain.
    :type domain_bounds: tuple
    :param resources: Resource distribution values.
    :type resources: numpy.ndarray
    :return: The generated plot figure.
    :rtype: matplotlib.figure.Figure
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
