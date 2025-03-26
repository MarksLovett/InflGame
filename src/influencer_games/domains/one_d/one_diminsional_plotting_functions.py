import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


from influencer_games.utils.utilities import *
from influencer_games.domains.one_d.one_diminsional_utilities import *


def pos_plot_1d(num_agents,pos_matrix,domain_bounds,title_ads=[]):
    num_points=len(pos_matrix)
    domain=np.linspace(0,num_points,num_points)
    fig, ax  = plt.subplots()
    ax.set_box_aspect(1)
    for a_id in range(num_agents):
        ax.plot(domain,pos_matrix[:,a_id].numpy(),label='Player '+ str(a_id))
    #ax.axhline(y=self.mean,color='r', linestyle='--',label='Mean')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Influencer location')
    plt.xlim(0,num_points)
    plt.ylim(domain_bounds[0],domain_bounds[1])
    plt.legend()
    title="Player Postions"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title)
    plt.close()
    return fig

def gradient_plot_1d(num_agents,grad_matrix,title_ads=[]):
    num_points=len(grad_matrix)
    domain=np.linspace(0,num_points,num_points)
    
    fig, ax  = plt.subplots()
    ax.set_box_aspect(1)
    for a_id in range(num_agents):
        ax.plot(domain,grad_matrix[:,a_id],label='Player '+ str(a_id))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Influencer gradient')
    plt.xlim(0,num_points)
    plt.legend()
    title="Player Gradients"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title)
    plt.close()
    return fig


def prob_plot_1d(num_agents,agents_pos,bin_points,domain_bounds,prob,voting_configs,title_ads):
    fig, ax  = plt.subplots()
    ax.set_box_aspect(1)
    for agent_id in range(num_agents):
        ax.plot(bin_points,prob[agent_id],label=f'Player {agent_id}')
        ax.scatter(agents_pos[agent_id],0)

    if voting_configs['TYPE']=='fixed_party':
        ax.plot(bin_points,prob[num_agents],label=f'Fixed Party')
    elif voting_configs['TYPE']=='abstaing':
        ax.plot(bin_points,prob[num_agents],label=f'Abstaining')
    elif voting_configs['TYPE']=='fixed_party and abstaing':
        ax.plot(bin_points,prob[num_agents],label=f'Fixed party')
        ax.plot(bin_points,prob[num_agents+1],label=f'Abstaining')
    
    plt.legend()
    plt.xlim(domain_bounds[0],domain_bounds[1])
    plt.ylabel('Probability')
    plt.xlabel('Resource postion')
    title="Player probability of influence"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title)
    plt.close()
    return fig


            
    

def three_player_dynamcis(pos_matrix,
                        x_star,
                        title_ads,
                        ):
    new_pos=pos_matrix.T
    x=new_pos[0,:]
    y=new_pos[1,:]
    z=new_pos[2,:]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x, y, z, label='parametric curve')
    ax.scatter(x_star,x_star,x_star,label='mean')
    ax.set_zlim(0,1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.legend()
    ax.set_box_aspect((1,1,1))

    plt.close()
    return fig

def vector_plot_1d(ids,gradient,title_ads):
    Y, X = np.mgrid[0:1:100j, 0:1:100j]
    U,V = gradient[:,0].reshape((100, 100)),gradient[:,1].reshape((100, 100))
    fig,ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.streamplot(X, Y, U, V)
    plt.xlabel(f"Player {ids[0]}'s postion")
    plt.ylabel(f"Player {ids[1]}'s postion")
    title=f"Player {ids[0]} and {ids[1]}'s vector field"
    if len(title_ads)>0:
        for item in title_ads:
            title+=title+item
    plt.title(title)
    plt.close()
    return fig


def dist_and_pos_plot_1d(num_agents,bin_points,resource_distribution,pos_matrix,len_grad_matrix,infl_dist,cm,NUM_COLORS,title_ads):
    fig = plt.figure(figsize=(19, 7))
    num_points=len(pos_matrix)
    gs = GridSpec(nrows=num_agents, ncols=2,width_ratios=[1, 1],wspace=0.2, hspace=0.2, top=1, bottom=0.05, left=0.17, right=0.845)
    domain=np.linspace(0,num_points,num_points)
    ax0 = fig.add_subplot(gs[:, 1])
    for a_id in range(num_agents):
        ax0.scatter(0,pos_matrix[:,a_id][0],s=70,color=cm(1.*a_id/NUM_COLORS),linewidth=0.3,label='Player '+str(a_id))
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
    plt.title(title)
    plt.close()
    return fig

def equalibirium_bifurication_plot_1d(num_agents,bin_points,resource_distribution,infl_type,reach_parameters,final_pos_matrix,reach_start,reach_end,refinements,title_ads):
            fig,ax=plt.subplots()
            
            for agent_id in range(num_agents): 
                ax.plot(reach_parameters,final_pos_matrix[:,agent_id])

            #Bifurications critical values (works for gaussian only)
            if infl_type=='gaussian':
                _,means,crit_stds=critical_values_plot(num_agents=num_agents,bin_points=bin_points,resource_distribution=resource_distribution,axis=ax,reach_start=reach_start,reach_end=reach_end,refinements=refinements)
                ax.set_box_aspect(1)
                crit_stds=flatten_list(xss=crit_stds)
                crit_stds.sort()
                std_ticks = [float(reach_end),float(reach_end)]
                crit_means=np.around(means,decimals=3)
                mean_ticks= [0,1]+list(crit_means)
                mean_ticks.sort()

                std_tick_vals=np.unique(std_ticks+crit_stds)
                std_tick_vals.sort()
                crit_std_locs=[]
                for std_id in range(len(crit_stds)):
                    crit_std_locs.append(int(np.where(std_tick_vals==crit_stds[std_id])[0][0]))
                std_tick_labels=list(std_tick_vals.copy())
                for std_loc_id in range(len(crit_std_locs)):
                    if std_loc_id ==len(crit_std_locs)-1:
                        std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_*$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]]) 
                    else:
                        std_tick_labels[int(crit_std_locs[std_loc_id])]=r'$t_'+str(len(crit_std_locs)-std_loc_id-1)+r'$' #+r'='+str(std_tick_vals[crit_std_locs[std_loc_id]])
                ax.xaxis.set_ticks(std_tick_vals)
                ax.xaxis.set_ticklabels(std_tick_labels)
                ax.yaxis.set_ticks(crit_means)
                ax.yaxis.set_ticklabels(crit_means)
            
            #Plot features
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            title=str(num_agents)+' players bifurication of equalibria'
            if len(title_ads)>0:
                for title_additon in title_ads:
                    title=title+" "+title_additon
            plt.title(title)
            if infl_type=='gaussian':
                plt.xlabel(r"$\sigma$ (std)")
            else: 
                plt.xlabel(r"$\sigma$")
            plt.ylim(0,1)
            plt.ylabel("Player Postion")
            plt.close()
            return fig

def final_postion_historgram_1d(num_agents,domain_bounds,current_alpha,reach_parameter,final_pos_vector,title_ads):
    fig,ax=plt.subplots()
    ax.set_box_aspect(1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.histplot(final_pos_vector,binwidth=.05)
    plt.ylabel('Number of players')
    plt.xlabel('Position')
    plt.xlim(domain_bounds[0],domain_bounds[1])
    title=str(num_agents)+r' Player count in postions for $\alpha=$'+str(current_alpha)+r',$\sigma=$'+str(reach_parameter)
    if len(title_ads)>0:
        for title_additon in title_ads:
            title=title+" "+title_additon
    plt.title(title)
    plt.close()
    return fig

