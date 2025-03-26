import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pylab

from influencer_games.utils.utilities import *
from influencer_games.domains.two_d.two_dimensional_utilities import *

def dist_and_pos_plot_2d_simple(num_agents:int,
                                                bin_points:np.ndarray,
                                                rect_X,
                                                rect_Y,
                                                cmap1,
                                                cmap2,
                                                pos_matrix:torch.Tensor,
                                                infl_dist:torch.Tensor,
                                                resource_type:str,
                                                resources:str = 0)->None:
        """
        Plots the agents' distributions and their postions for 2D rectangular domain
        
        Parameters:
            Agent_id: the id of the agent
  
        Returns:
            Plot: Plots the agents' postions with thier distributions

        """
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
        plt.title('Agents positions in time',y=1.05)
        plt.legend(title="End pos")

        for a_id in range(num_agents):
            ax1 = fig.add_subplot(gs[a_id, 1])
            pvals=infl_dist[a_id].numpy()
            pvals=pvals.reshape(len(rect_Y),len(rect_X))
            pcm = ax1.pcolormesh(rect_X,rect_Y, pvals,cmap=cmap2)
            ax1.axis('equal')
            ax1.axis('off')
            plt.title("Player "+str(a_id),x=.15,y=0.5)
            
        ax2 = fig.add_subplot(gs[:, 1])
        fig.colorbar(pcm,ax=ax2, extend='max')
        plt.title('Players\' influence distributions',x=.65,y=1.05)
        ax2.axis('off')
        plt.close()
        return fig

def dist_plot_2d(agent_id,infl_dist,rect_Y,rect_X):
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

def equalibirium_bifurication_plot_2d_simple(num_agents,domain_bounds,reach_num_points,final_pos_matrix,title_ads):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    c=range(reach_num_points)
    for agent_id in range(num_agents): 
        ax.scatter(final_pos_matrix[:,agent_id][:,0].numpy(),final_pos_matrix[:,agent_id][:,1].numpy(),c=c, cmap='rainbow')
        #lines=colored_line(final_pos_matrix[:,agent_id][:,0].numpy(),final_pos_matrix[:,agent_id][:,1].numpy(), sigmas, ax, linewidth=1, cmap="plasma")
    #fig.colorbar(lines)  # add a color legend
    ax.set_xlim(domain_bounds[0][0], domain_bounds[0][1])
    ax.set_ylim(domain_bounds[1][0], domain_bounds[1][1])
    title="bifurcation of " +str(num_agents)+r" players for different $\sigma$ values"
    if len(title_ads)>0:
        for title_additon in title_ads:
            title=title+" "+title_additon
    ax.set_title(title)

    plt.close()
    return fig
        