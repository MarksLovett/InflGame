import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec
import pylab

from influencer_games.utils.utilities import *
from influencer_games.domains.simplex.simplex_utlities import *

def pos_plot_simplex(num_agents:int,
                     bin_points:np.ndarray,
                     corners:np.ndarray,
                     triangle,
                     pos_matrix:torch.Tensor):
    fig,ax=plt.subplots()
    ax.set_box_aspect(1)
    ax.triplot(triangle,linewidth=0.8,color="black")
    for a_id in range(num_agents):
        new_coor=ba2xy(pos_matrix[:,a_id],corners=corners)
        y=new_coor.detach().numpy()
        ax.scatter(y[0,0],y[0,1],s=70,linewidth=0.3)
        ax.plot(y[:,0],y[:,1])
    for b_id in range(len(bin_points)):
        new_coor=ba2xy(torch.tensor(bin_points[b_id]),corners=corners)
        y=new_coor.detach().numpy()
        ax.scatter(y[0],y[1],c="red",s=70,linewidth=0.3)
    return fig

def dist_plot_simplex(agent_id:int,
                      r2:list[float],
                      corners:np.ndarray,
                      triangle,
                      trimesh,
                      infl_dist:torch.Tensor,
                      cmap,
                      typelabels:list[str],
                      margin:float =.01,
                      **kwargs):
    
    fig,ax=plt.subplots()
    ax.triplot(triangle,linewidth=0.8,color="black")
    pvals=infl_dist[agent_id].numpy()
    pcm=ax.tricontourf(trimesh, pvals, alpha=0.8,levels=40, cmap=cmap,**kwargs)
    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(ymin=-margin,ymax=r2[1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)

    #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
    
    ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
    fig.colorbar(pcm, ax=ax, extend='max')
    plt.title('Agent '+str(agent_id)+'\'s'+' influence distribution')

def dist_and_pos_plot_simplex(num_agents:int,
                              bin_points:np.ndarray,
                              r2:list[float],
                              corners,
                              triangle,
                              trimesh,
                              typelabels,
                              cmap1,
                              cmap2,
                              pos_matrix:torch.Tensor,
                              infl_dist:torch.Tensor,
                              resource_type:str,
                              resources:np.ndarray = 0
                              ):
        NUM_COLORS = num_agents+1
        cm = pylab.get_cmap(cmap1)
        fig = plt.figure(figsize=(19, 7))
        gs = GridSpec(nrows=num_agents, ncols=2,width_ratios=[1, 1],wspace=0.0, hspace=0.2, top=1, bottom=0.05, left=0.17, right=0.845)

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.triplot(triangle,linewidth=0.8,color="black")
        for a_id in range(num_agents):
            new_coor=ba2xy(pos_matrix[:,a_id],corners=corners)
            y=new_coor.detach().numpy()
            ax0.scatter(y[-1,0],y[-1,1],s=70,color=cm(1.*a_id/NUM_COLORS),linewidth=0.3,label='Agent '+str(a_id))
            ax0.scatter(y[0,0],y[0,1],s=70,facecolors='none',edgecolors=cm(1.*a_id/NUM_COLORS),linewidth=1)
            ax0.plot(y[:,0],y[:,1],color=cm(1.*a_id/NUM_COLORS))
        if resource_type in  ["multi_modal_gaussian_distribution_2D","multi_modal_gaussian_distribution_2D_square","multi_modal_gaussian_distribution_2D_triangle","dirichlet_distribution"]:
            ax0.triplot(triangle,linewidth=0.8,color="black")
            im=ax0.tricontourf(trimesh, resources, alpha=0.3,levels=40)
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
                new_coor=ba2xy(torch.tensor(bin_points[b_id]),corners=corners)
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
            pcm=ax1.tricontourf(trimesh, pvals, alpha=0.8,levels=40, cmap=cmap2)
            ax1.axis('equal')
            ax1.axis('off')
            margin=0.01
            ax1.set_ylim(ymin=-margin,ymax=r2[1]+margin)
            ax1.set_xlim(xmin=-margin,xmax=1.+margin)

            #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)

            ax1.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
            ax1.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
            ax1.annotate(typelabels[2],corners[2],xytext=corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
            plt.title("Agent "+str(a_id),x=.25,y=0.25)
        ax2 = fig.add_subplot(gs[:, 1])
        fig.colorbar(pcm,ax=ax2, extend='max')
        plt.title('Agents\' influence distributions',x=.65,y=1.05)
        ax2.axis('off')
        plt.close()
        return fig

def equalibirium_bifurication_plot_simplex(num_agents,r2,corners,triangle,final_pos_matrix,reach_num_points,title_ads,type_labels=None):
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
    ax.set_title(title)
    margin=.2
    ax.set_ylim(ymin=-margin,ymax=r2[1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)
    plt.close()
    return fig

def simplex_plot_resources(domain_bounds,resources):
    typelabels=["A","B","C"]
    fig,ax = plt.subplots() 
    ax.triplot(domain_bounds[2],linewidth=0.8,color="black")
    pcm=ax.tricontourf(domain_bounds[3], resources, alpha=0.8,levels=40)
    ax.axis('equal')
    ax.axis('off')
    margin=0.01


    ax.set_ylim(ymin=-margin,ymax=domain_bounds[0][1]+margin)
    ax.set_xlim(xmin=-margin,xmax=1.+margin)
    ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
    ax.annotate(typelabels[2],domain_bounds[1][2],xytext=domain_bounds[1][2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')