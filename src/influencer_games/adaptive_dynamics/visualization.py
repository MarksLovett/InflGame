import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import pylab
import imageio.v2 as imageio #For the Gif 
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import scipy
from matplotlib.collections import LineCollection
import warnings

import influencer_games.adaptive_dynamics.grad_func_env as grad_func_env
from influencer_games.adaptive_dynamics.jacobian import *

from influencer_games.utils.utilities import *
from influencer_games.infl_kernels.gaussian_influence import *
from influencer_games.infl_kernels.Jones_influence import *
from influencer_games.infl_kernels.dirl_influence import *
from influencer_games.infl_kernels.MVG_influence import *


from influencer_games.domains.resource_distributions import *
from influencer_games.domains.simplex.simplex_utlities import *
from influencer_games.domains.simplex.simplex_plotting_functions import *
from influencer_games.domains.two_d.two_dimensional_utilities import *
from influencer_games.domains.two_d.two_dimensional_plotting_functions import *
from influencer_games.domains.one_d.one_diminsional_plotting_functions import *
from influencer_games.domains.one_d.one_diminsional_utilities import *


class Shell:
    def __init__(self,
                 num_agents:int,
                 agents_pos:list|np.ndarray,
                 parameters:torch.Tensor,
                 resource_distribution:torch.Tensor,
                 bin_points:list|np.ndarray,
                 mean:int= None,
                 infl_configs:dict = {'infl_type':'gaussian'},
                 lr_type:str = 'cosine',
                 learning_rate:list = [.0001,.01,15],
                 time_steps:int = 100,
                 fp:int = 0,
                 infl_cshift:bool = False,
                 cshift:torch.tensor = None,
                 infl_fshift:bool = False,
                 Q:int = None,
                 domain_type:str ='1d',
                 domain_bounds:list[float]|torch.Tensor = [0,1],
                 resource_type:float = 'na',
                 domain_refinement:int = 10,
                 tolarance:float = 10**-5,
                 tolarated_agents: int = None):
        
        self.num_agents=num_agents
        self.agents_pos=agents_pos
        self.infl_type=infl_configs['infl_type']
        self.infl_configs=infl_configs
        self.parameters=parameters
        self.resource_distribution=resource_distribution
        self.bin_points=np.array(bin_points)
        self.learning_rate=learning_rate
        self.time_steps=time_steps
        self.mean=mean
        self.fixed_pa=fp
        self.lr_type=lr_type
        self.infl_cshift=infl_cshift
        self.cshift=cshift
        self.infl_fshift=infl_fshift
        self.Q=Q
        self.domain_type= domain_type
        self.domain_bounds=domain_bounds
        self.resource_type=resource_type
        self.tolarance=tolarance
        if tolarated_agents == None:
            self.tolarated_agents=num_agents
        else:
            self.tolarated_agents=tolarated_agents
        if domain_type=='simplex':
            self.r2=domain_bounds[0]
            self.corners=domain_bounds[1]
            self.triangle=domain_bounds[2]
            self.trimesh=domain_bounds[3]
        if domain_type=='2d':
            self.rect_X,self.rect_Y,self.rect_postions=two_dimensional__rectangle_setup(domain_bounds,domain_refinement=domain_refinement)

    def setup_adaptive_env(self):
        self.field=grad_func_env.AdapativeEnv(num_agents=self.num_agents,agents_pos=self.agents_pos,parameters=self.parameters,resource_distribution=self.resource_distribution,bin_points=self.bin_points,
                                    infl_configs=self.infl_configs,lr_type=self.lr_type,learning_rate=self.learning_rate,time_steps=self.time_steps,fp=self.fixed_pa,infl_cshift=self.infl_cshift,cshift=self.cshift,
                                    infl_fshift=self.infl_fshift,Q=self.Q,domain_type=self.domain_type,domain_bounds=self.domain_bounds,tolarance=self.tolarance,tolarated_agents=self.tolarated_agents)
    
    ## Plots
    def pos_plot(self,
                 title_ads:list=[],
                 save:bool = False,
                 name_ads:list=[],
                 save_types:list=['.png','.svg'],
                 )->None:
        
        """
        plots the agents position through the gradient accent steps
        
        Parameters:
            Agent_id: the id of the agent

        Returns:
            Plot: 2-d plot either simplex or plot in 2d cartiesian plot

        """
        if self.domain_type=='1d':
            fig=pos_plot_1d(num_agents=self.num_agents,pos_matrix=self.field.pos_matrix,title_ads=title_ads,domain_bounds=self.domain_bounds)

        elif self.domain_type=='2d':
            print('do')

        elif self.domain_type=='simplex':
            fig=pos_plot_simplex(num_agents=self.num_agents,bin_points=self.bin_points,corners=self.corners,triangle=self.triangle,pos_matrix=self.field.pos_matrix)
        
        if save==True:
            file_names=figure_final_name(np.array([self.domain_type,'Postion',self.num_agents]),name_ads=name_ads,save_types=save_types)
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig

    def grad_plot(self,
                  title_ads:list=[],
                  save:bool = False,
                  name_ads:list=[],
                  save_types:list=['.png','.svg']):
        

        if self.domain_type=='1d':
            fig=gradient_plot_1d(num_agents=self.num_agents,grad_matrix=self.field.grad_matrix,title_ads=title_ads)
        else:
            print("grad_plot only applies to 1d domains")

        if save==True:
            file_names=figure_final_name(np.array([self.domain_type,'Gradient',self.num_agents]),name_ads=name_ads,save_types=save_types)
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def prob_plot(self,
                  postion:np.ndarray = [],
                  parameters:np.ndarray = [],
                  voting_configs:dict = {"TYPE":None},
                  title_ads:list=[],
                  save:bool = False,
                  name_ads:list=[],
                  save_types:list=['.png','.svg']):
        og_postion=self.field.agents_pos.copy()
        og_parameters=self.field.parameters.copy()
        if len(postion) != 0:
            self.field.agents_pos=postion
        else:
            postion=self.agents_pos
        if len(parameters) != 0:
            self.field.parameters=parameters
        else:
            parameters=self.parameters
            

        prob=self.field.prob_matrix(parameter_instance=parameters)
        if self.domain_type=='1d':
            fig=prob_plot_1d(num_agents=self.num_agents,agents_pos=postion,bin_points=self.bin_points,domain_bounds=self.domain_bounds,prob=prob,voting_configs=voting_configs,title_ads=title_ads)

        if save==True:  
            file_names=figure_final_name(np.array([self.domain_type,'prob_plot']),name_ads=name_ads,save_types=save_types)    
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')

        self.field.parameters=og_parameters
        self.field.agents_pos=og_postion
        return fig
    


    def three_players(self,
                      x_star:float=None,
                      title_ads:list=[],
                      name_ads:list=[],
                      save:bool = False,
                      save_types:list=['.png','.svg']):
        if self.domain_type=='1d' or self.num_agents!=3:
            if self.infl_type=="gaussian":
                fig=three_player_dynamcis(self.field.pos_matrix,x_star=discrete_mean(self.bin_points,resource_distribution=self.resource_distribution),title_ads=title_ads)
            
            else:
                if x_star==None:
                    print("no x_star")
                else:   
                    fig=three_player_dynamcis(self.field.pos_matrix,x_star=discrete_mean(self.bin_points,resource_distribution=self.resource_distribution))

            if save==True:  
                file_names=figure_final_name(np.array([self.domain_type,'3_player_dynamics']),name_ads=name_ads,save_types=save_types)    
                for file_name in file_names:
                    fig.savefig(file_name,bbox_inches='tight')
            return fig
        else:
            print("Three player dynamics are feasablily visible only for 1D domains and 3 players")

    def vect_plot(self,
                  agent_id:int,
                  parameter_instance:list|np.ndarray|torch.Tensor=0,
                  cmap:str='viridis',
                  typelabels:list[str] = ["A","B","C"],
                  ids:list[int] = [0,1],
                  pos:torch.Tensor = None,
                  title_ads:list=[],
                  save:bool = False,
                  name_ads:list=[],
                  save_types:list=['.png','.svg'],
                  **kwargs,
                  )->None:
        """
        Plots the vector plot of the agents' gradient
        
        Parameters:
            Agent_id: the id of the agent
        
        Returns:
            Vector Plot: agents' gradient

        """
        
        if pos is None:
            pos=player_optimal_postion_setup(num_agents=self.num_agents,agents_pos=self.agents_pos,infl_type=self.infl_type,mean=self.mean,domain_type=self.domain_type,ids=ids)

        if self.domain_type=='1d':
            grads=self.calc_direction_and_strength(parameter_instance=parameter_instance,agent_id=agent_id,ids=ids,pos=pos)
            fig=vector_plot_1d(gradient=grads,ids=ids,title_ads=title_ads)
            return fig

        elif self.domain_type=='simplex':
            fig,ax=plt.subplots()
            self.calc_direction_and_strength(parameter_instance,agent_id=agent_id)
            ax.triplot(self.triangle,linewidth=0.8,color="black")
            pcm=ax.tricontourf(self.trimesh, self.pvals,norm=colors.LogNorm(vmin=np.array(self.pvals).min(), vmax=np.array(self.pvals).max()), alpha=0.8, cmap=cmap,**kwargs)

            #arrow plot options:
            # Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],self.pvals,angles='xy',pivot='mid',  cmap=cmap)#pivot='tail',
            Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],angles='xy',pivot='mid')#pivot='tail')#
            # Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction.T[0],self.direction.T[1],angles='xy',pivot='mid')#pivot='tail')#


            ax.axis('equal')
            ax.axis('off')
            margin=0.01
            ax.set_ylim(ymin=-margin,ymax=self.r2[1]+margin)
            ax.set_xlim(xmin=-margin,xmax=1.+margin)

            #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
            
            ax.annotate(typelabels[0],(0,0),xytext=(-0.0,-0.02),horizontalalignment='center',va='top')
            ax.annotate(typelabels[1],(1,0),xytext=(1.0,-0.02),horizontalalignment='center',va='top')
            ax.annotate(typelabels[2],self.corners[2],xytext=self.corners[2]+np.array([0.0,0.02]),horizontalalignment='center',va='bottom')
            fig.colorbar(pcm, ax=ax, extend='max')
            plt.title('Agent '+str(agent_id)+'\'s'+' gradient vector self')

        elif self.domain_type=='2d':
            self.two_a=True
            self.calc_direction_and_strength(parameter_instance=parameter_instance,agent_id=agent_id,ids=ids,pos=pos)
            fig,ax = plt.subplots()
            ax.set_box_aspect(1)
            Y, X = self.rect_Y,self.rect_X
            U,V = self.direction[:,0].reshape((10,10)),self.direction[:,1].reshape((10,10))
            strm = ax.streamplot(X, Y, U, V, **kwargs)
        
        if save==True:
            file_names=figure_final_name(np.array([self.domain_type,'vect',self.num_agents]),name_ads=name_ads,save_types=save_types)
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def dist_plot(self,
                  agent_id:int,
                  parameter_instance:list|np.ndarray|torch.Tensor=0,
                  cmap:str = 'viridis',
                  typelabels:list[str] = ["A","B","C"],
                  **kwargs,
                  )->None:
        """
        Plots the agents' distributions
        
        Parameters:
            Agent_id: the id of the agent 
  
        Returns:
            Plot: agents' influence distributions
        
        """
        self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)

        if self.domain_type=='1d':
            fig, ax = plt.subplots()
            for agent_id in range(self.num_agents):
                ax.plot(self.bin_points,self.infl_dist[agent_id].numpy(),label='Agent '+str(agent_id))
            plt.legend()
            plt.title('Agent '+str(agent_id)+'\'s'+' influence distribution')

        elif self.domain_type=='simplex':
            fig=dist_plot_simplex(agent_id=agent_id,r2=self.r2,corners=self.corners,triangle=self.triangle,trimesh=self.trimesh,infl_dist=self.infl_dist,cmap=cmap,typelabels=typelabels,margin=0.01,**kwargs)

        elif self.domain_type=='2d':
            fig=dist_plot_2d(agent_id=agent_id,infl_dist=self.infl_dist,rect_X=self.rect_X,rect_Y=self.rect_Y)
        
        return fig
            
    def dist_pos_plot(self,
                      parameter_instance:list|np.ndarray|torch.Tensor,
                      typelabels:list[str] = ["A","B","C"],
                      cmap1:str = 'twilight',
                      cmap2:str = 'viridis',
                      title_ads=[],
                      save:bool = False,
                      name_ads:list=[],
                      save_types:list=['.png','.svg']
                      )->None:
        """
        Plots the agents' distributions and their postions based on domain type
        
        Parameters:
            Agent_id: the id of the agent
  
        Returns:
            Plot: Plots the agents' postions with thier distributions

        """

        NUM_COLORS = self.num_agents+1
        cm = pylab.get_cmap(cmap1)
        if self.domain_type=='1d':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.field.pos_matrix[-1].numpy())
            fig=dist_and_pos_plot_1d(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=self.resource_distribution,pos_matrix=self.field.pos_matrix,len_grad_matrix=len(self.field.grad_matrix),infl_dist=self.infl_dist,cm=cm,NUM_COLORS=NUM_COLORS,title_ads=title_ads)
        
        elif self.domain_type=='2d':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)
            fig=dist_and_pos_plot_2d_simple(num_agents=self.num_agents,bin_points=self.bin_points,rect_X=self.rect_X,rect_Y=self.rect_Y,cmap1=cmap1,cmap2=cmap2,pos_matrix=self.field.pos_matrix,infl_dist=self.infl_dist,resource_type=self.resource_type,resources=self.resource_distribution)
        
        elif self.domain_type=='simplex':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)
            fig=dist_and_pos_plot_simplex(num_agents=self.num_agents,bin_points=self.bin_points,r2=self.r2,corners=self.corners,triangle=self.triangle,trimesh=self.trimesh,typelabels=typelabels,cmap1=cmap1,cmap2=cmap2,pos_matrix=self.field.pos_matrix,infl_dist=self.infl_dist,resource_type=self.resource_type,resources=self.resource_distribution)
        
        if save==True:
            file_names=figure_final_name(np.array([self.domain_type,'dist_pos',self.num_agents]),name_ads=name_ads,save_types=save_types)
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')    
        
        return fig
    
    def dist_pos_gif(self,
                     max_frames:int,
                     )->None:

        """
        Plots the agents' distributions and their postions into a gif
        
        Parameters:
        -Agent_id: the id of the agent
  
        Returns:
        -Plot: vector plot for the agents gradient

        """
        pos_len=len(self.field.pos_matrix)
        times=range(1,pos_len,int(np.round(pos_len/max_frames)))
        filenames=[]
        og_pos=self.agents_pos.copy()
        og_pos_matrix=self.field.pos_matrix
        if max_frames==1:
            times=range(1,pos_len,int(np.round(pos_len/2))+1)
            self.field.pos_matrix=self.field.pos_matrix[0:]
            self.agents_pos=self.field.pos_matrix.numpy()[-1]
            f=self.dist_pos_plot(self.parameters,typelabels=["A","B","C"])
            self.agents_pos=og_pos.copy()
            self.field.pos_matrix=og_pos_matrix.clone()
            return f
        else:
            for i in range(len(times)):
                self.field.pos_matrix=self.field.pos_matrix[0:times[i]]
                self.agents_pos=self.field.pos_matrix.numpy()[times[i]-1]
                f=self.dist_pos_plot(self.parameters,typelabels=["A","B","C"])
                filename = f'{i}.png'
                filenames.append(filename)
                # save frame
                f.savefig(filename,bbox_inches="tight")
                self.agents_pos=og_pos.copy()
                self.field.pos_matrix=og_pos_matrix
            with imageio.get_writer('timelapse_test.gif', mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                # Remove files
                for filename in set(filenames):
                    os.remove(filename)

    def equalibirum_bifurcation_plot(self,
                                reach_start:float = .03,
                                reach_end:float = .3,
                                reach_num_points:int = 30,
                                num_interations:int = 100,
                                intitial_pos:list|np.ndarray = 0,
                                current_alpha:float = .5,
                                tolarance:float = None,
                                tolarated_agents: int = None,
                                refinements=2,
                                title_ads:list=[],
                                name_ads:list=[],
                                save:bool = False,
                                save_types:list=['.png','.svg'],
                                return_matrix:bool = False,
                                )->torch.Tensor|None:
        """
        Calculates the final postion of every player after they have completed their gradient accent
        
        Parameters:
            reach_start: intial reach parameter
            reach_end: final reach
            reach_num_points: Number of reach points we test
            num_interations: Numver of gradient accent iterations
  
        Returns:
            Plot:  The equalibira given an intial postion. 

        """

        og_iterations=self.time_steps
        og_pos=self.agents_pos.copy()
        self.field.time_steps=num_interations
        self.field.agents_pos=intitial_pos.copy()
        self.agents_pos=np.array(intitial_pos).copy()
        
        if tolarated_agents==None:
            tolarated_agents=self.tolarated_agents
        if tolarance==None:
            tolarance=self.tolarance

        if self.domain_type=="1d":
            reach_parameters=player_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters,tolarance=tolarance,tolarated_agents=tolarated_agents)
            fig=equalibirium_bifurication_plot_1d(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=self.resource_distribution,infl_type=self.infl_type,reach_parameters=reach_parameters,final_pos_matrix=final_pos_matrix,reach_start=reach_start,reach_end=reach_end,refinements=refinements,title_ads=title_ads)
        
        elif self.domain_type=='2d':
            reach_parameters=player_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters=reach_parameters.copy(),tolarance=tolarance,tolarated_agents=tolarated_agents)
            fig=equalibirium_bifurication_plot_2d_simple(num_agents=self.num_agents,domain_bounds=self.domain_bounds,reach_num_points=reach_num_points,final_pos_matrix=final_pos_matrix,title_ads=title_ads)
            
        elif self.domain_type=='simplex':
            reach_parameters=player_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters=reach_parameters.copy(),tolarance=tolarance,tolarated_agents=tolarated_agents)
            fig=equalibirium_bifurication_plot_simplex(num_agents=self.num_agents,r2=self.r2,corners=self.corners,triangle=self.triangle,final_pos_matrix=final_pos_matrix,reach_num_points=reach_num_points,type_labels=None,title_ads=title_ads)
        
        self.field.time_steps=og_iterations
        self.field.agents_pos=og_pos.copy()
        self.agents_pos=og_pos.copy()
        if save==True:
            file_names=figure_final_name(np.array([self.domain_type,'equalibirum_bifurcation',self.num_agents,current_alpha]),name_ads=name_ads,save_types=save_types)
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        if return_matrix==True:
            return fig,final_pos_matrix
        else: 
            return fig

    def first_order_bifurication_plot(self,
                                        player_parameter_instance,
                                        resource_distribution_type,
                                        resource_entropy = False,
                                        infl_entropy = False,
                                        alpha_current=.5,
                                        alpha_st=0,
                                        alpha_end=1,
                                        varying_paramter_type='mean',
                                        fixed_parameters_lst = None,
                                        name_ads=[],
                                        title_ads=[],
                                        save_types=['.png','.svg'],
                                        ):
        """
        Calculates the final postion of every player after they have completed their gradient accent
        
        Parameters:
            reach_parameter: sigma value
            num_iterations: the number of gradient accent steps
            intial_pos: where the agents start
            current_alpha: resource parameter.
  
        Returns:
            Plot: 

        """
        resource_parameters,alpha=resource_parameter_setup(resource_distribution_type=resource_distribution_type,varying_paramter_type=varying_paramter_type,alpha_st=alpha_st, alpha_end=alpha_end, fixed_parameters_lst=fixed_parameters_lst)
        y=self.jacobian_stability_fast(player_parameter_instance=player_parameter_instance,resource_distribution_type=resource_distribution_type,resource_parameters=resource_parameters,resource_entropy=resource_entropy,infl_entropy=infl_entropy)[0]
        fig,ax=plt.subplots()
        ax.set_box_aspect(1)
        
        #plots the line where the e-values of the jacobian becomes postive
        ax.plot(alpha,y)
        ax.fill_between(alpha, 0, y, where=(alpha >= alpha_st), color='red', alpha=0.3)
        ax.fill_between(alpha, y,max(max(y),.3), where=(alpha <= alpha_end), color='blue', alpha=0.3)
        ax.vlines([alpha_current],ymin=max(min(y),0),ymax=max(max(y),.3),label=r'Current $\alpha$ value= '+str(alpha_current),colors='black',linestyles='--' )
        
        plt.ylim(max(min(y),0),max(max(y),.3))
        red_p = mpatches.Patch(color='red', label='unstable')
        blue_p = mpatches.Patch(color='blue', label='stable')
        plt.legend()
        old_handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=[red_p, blue_p]+old_handles)
        plt.ylabel("$\sigma$ (reach)")
        plt.xlabel(r"$\alpha$ (mode distance)")
        
        title=r"$(\alpha,\sigma)$ $x^{*}$ stability bifurication for "+str(self.num_agents)+" players"
        if len(title_ads)>0:
            for title_additon in title_ads:
                title=title+" "+title_additon
        plt.title(title)
        plt.xlim(alpha_st,alpha_end) 
        plt.close()
        
        file_names=figure_final_name(np.array([self.domain_type,'stability_bifurcation_plot_fast',self.num_agents]),name_ads,save_types)
        for file_name in file_names:
            fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def postion_at_equalibirum_histogram(self,
                                reach_parameter:float =.5,
                                num_interations:int = 100,
                                intitial_pos:list|np.ndarray = 0,
                                current_alpha:float = .5,
                                tolarance:float = None,
                                tolarated_agents: int = None,
                                title_ads:list=[],
                                name_ads:list=[],
                                save:bool = False,
                                save_types:list=['.png','.svg'],
                                return_pos:bool = False,
                                )->None:
        """
        Calculates the final postion of every player after they have completed their gradient accent
        
        Parameters:
            reach_parameter: sigma value
            num_iterations: the number of gradient accent steps
            intial_pos: where the agents start
            current_alpha: resource parameter.
  
        Returns:
            Plot:  The histogram of the number of players in a certian point.

        """
        og_iterations=self.time_steps
        og_pos=self.agents_pos.copy()
        self.field.time_steps=num_interations
        self.field.agents_pos=intitial_pos.copy()
        self.agents_pos=np.array(intitial_pos).copy()
        
        if tolarated_agents==None:
            tolarated_agents=self.tolarated_agents
        if tolarance==None:
            tolarance=self.tolarance


        if self.domain_type=="1d":
            reach_parameters=player_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_parameter,reach_end = reach_parameter,reach_num_points = 1)
            final_pos_vector=self.final_pos_over_reach(reach_parameters,tolarance=tolarance,tolarated_agents=tolarated_agents).numpy()
            fig=final_postion_historgram_1d(num_agents=self.num_agents,domain_bounds=self.domain_bounds,current_alpha=current_alpha,reach_parameter=reach_parameter,final_pos_vector=final_pos_vector,title_ads=title_ads)
        else:
            "No histogram for domains other then simple 1d domains"
            
        self.field.time_steps=og_iterations
        self.field.agents_pos=og_pos.copy()
        self.agents_pos=og_pos.copy()

        file_names=figure_final_name(np.array([self.domain_type,'postional_histogram',self.num_agents]),name_ads,save_types)
        if save==True:
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        if return_pos==True:
            return fig,final_pos_vector
        else:
            return fig



    #Utils integrated into the class for simplicity
    def calc_infl_dist(self,
                       pos:torch.Tensor,
                       parameter_instance:list|np.ndarray|torch.Tensor,
                       )->torch.Tensor:
        """
        Calculates the influence distribution over the support for every agent
        
        Parameters:
            pos: array of 3-dim ba coordinates
            parameter_instance: parameter(s) unique to your influence distribution
        
        Returns:
            infl_dist: the distribution of influence over the support

        """

        infl_dist=0
        og_alpha=self.field.alpha_matrix
        og_pos=self.field.agents_pos.copy()
        og_binpoints=self.bin_points.copy()
        self.field.agents_pos=pos
        
        if self.domain_type=='simplex':
            if self.infl_type=='dirl':
                alpha_matrix=dirl_parm(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.field.agents_pos,fixed_pa=self.fixed_pa)
            else: 
                alpha_matrix=0
            self.field.alpha_matrix=alpha_matrix
            self.field.bin_points=np.array([xy2ba(x,y,corners=self.corners)  for x,y in zip(self.trimesh.x, self.trimesh.y)])
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        elif self.domain_type=='2d':
            self.field.bin_points=self.rect_postions
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        else:
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        self.field.alpha_matrix=og_alpha
        self.field.agents_pos=og_pos.copy()
        self.field.bin_points=og_binpoints.copy()
        self.infl_dist=infl_dist
    
    def calc_direction_and_strength(self,
                                    parameter_instance:list|np.ndarray|torch.Tensor = 0,
                                    agent_id:int = 0,
                                    ids:list = [0,1],
                                    pos:torch.Tensor = None,
                                    )->torch.Tensor:
        """
        Calculates the direction and strength of vectors on the all the points on the grid 
        
        Parameters:
            parameter_instance: parameter(s) unique to your influence distribution
            Agent id: the id of the agent 
            ids: for two agent diagram the ids of the agents of intrest
            pos: pos vector for agents
        
        Returns:
            direction: the direction of the gradient vectors
            pvals: the magnitude of the gradient vectors

        """
        
        if self.domain_type=='simplex':
            direction= np.array([self.field.gradient_function(np.array([xy2ba(x,y,corners=self.corners)+.001]*self.num_agents),parameter_instance,agent_id).nan_to_num().detach().numpy() for x,y in zip(self.trimesh.x, self.trimesh.y)])
            self.direction_norm=np.array([ba2xy(torch.tensor(v),corners=self.corners).detach().numpy()/np.linalg.norm(v) if np.linalg.norm(v)>0 else np.array([0,0]) for v in direction[:,agent_id]])
            
            #print(direction_ba_norm)
            self.pvals =[np.linalg.norm(v)+.0001 for v in direction[:,agent_id]]
            self.direction=np.array([ba2xy(torch.tensor(v),corners=self.corners).detach().numpy() for v in direction[:,agent_id]])
        elif self.domain_type=='2d':
            direction= np.array([self.field.gradient_function(np.array(list(value)*self.num_agents),parameter_instance,agent_id).nan_to_num().detach().numpy() for value in self.rect_postions])
            self.direction_norm=np.array([v/np.linalg.norm(v) if np.linalg.norm(v)>0 else np.array([0,0]) for v in direction[:,agent_id]])
            #print(direction_ba_norm)
            self.pvals =[np.linalg.norm(v) for v in direction[:,agent_id]]
            self.direction=np.array([v for v in direction[:,agent_id]])

        elif self.domain_type=="1d":
            if self.num_agents>2:
                two_a=False
            else:
                two_a=True
            grads=direction_strength_1d(self.field.gradient_function,parameter_instance=parameter_instance,two_a=two_a,ids=ids,pos=pos)
            two_a=True
            return grads
    
    def final_pos_over_reach(self,reach_parameters,tolarance,tolarated_agents):
        if self.domain_type in ['1d']:
            og_parameters=self.parameters
        elif self.domain_type in ['2d','simplex']: 
            og_parameters=self.parameters.clone()
        og_pos=self.agents_pos.copy()
        og_lr=self.learning_rate.copy()
        og_tolarance=self.tolarance
        og_tolarated_agents=self.tolarated_agents
        self.field.tolarance=tolarance
        self.field.tolarated_agents=tolarated_agents
        final_pos_matrix=0
        for parameter_id in range(len(reach_parameters)):
            self.field.pos_matrix=0
            self.field.agents_pos=og_pos.copy()
            self.agents_pos=og_pos.copy()
            if self.domain_type in ['1d']:
                self.field.learning_rate=[10**(-1*(max(3,5*(parameter_id+1)/len(reach_parameters)))),1/10000,500]
                self.field.parameters=np.array(reach_parameters[parameter_id])
            elif self.domain_type in ['2d','simplex']:
                self.field.parameters=torch.tensor(reach_parameters[parameter_id]).clone()
                
            self.field.gradient_accent(show_out=False) 
            if self.domain_type == 'simplex':
                final_pos_row=ba2xy(x=self.field.pos_matrix[-1].clone(),corners=self.corners)
            else:
                final_pos_row=self.field.pos_matrix[-1].clone()
            
            self.field.agents_pos=og_pos.copy()
            final_pos_matrix=matrix_builder(row_id=parameter_id,row=final_pos_row,matrix=final_pos_matrix)
        self.field.tolarated_agents=og_tolarated_agents
        self.field.tolarance=og_tolarance 
        self.field.agents_pos=og_pos.copy()
        self.field.learning_rate=og_lr.copy()
        self.agents_pos=og_pos.copy()
        self.field.parameters=og_parameters
        return final_pos_matrix
    
    def jacobian_stability_fast(self,player_parameter_instance,resource_distribution_type,resource_parameters,resource_entropy = False,infl_entropy = False):
        og_pos=self.agents_pos.copy()
        parameter_star_list=[]
        entropy_ls=[]
        ag_ent_ls=[]
        for parameter_id in range(len(resource_parameters)):
            resource_distribution=resource_distribution_choice(bin_points=self.bin_points,resource_type=resource_distribution_type,resource_parameters=resource_parameters[parameter_id])
            if self.infl_type=="gaussian":
                x_star=np.dot(self.bin_points,resource_distribution)/np.sum(resource_distribution)
                self.field.agents_pos=np.array([x_star]*self.num_agents)
                e_s=(player_parameter_instance[0]**2)*self.field.d_lnf_matrix(player_parameter_instance)[0]
                parameter_star=gaussian_symmetric_stability(num_agents=self.num_agents,d_values=e_s,resource_distribution=resource_distribution)
            elif self.infl_type=="multi_gaussian":
                #only for the special case of scaler multiple of 2x2 Id matrix for covariance matrix
                x_star = gaussian_symmetric_stability_2d_x_star_special_case(bin_points=self.bin_points,resource_distribution=resource_distribution)
                self.field.agents_pos=np.array(x_star*self.num_agents)
                parameter_star=gaussian_symmetric_stability_2d_test(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=resource_distribution)
            if resource_entropy == True:
                entropy_ls.append(scipy.stats.entropy(resource_distribution))
            if infl_entropy == True: 
                infl_dist=self.influence_matrix([parameter_star.item()]*self.num_agents)
                infl_dist_ent=scipy.stats.entropy(infl_dist[0])
                ag_ent_ls.append(infl_dist_ent)

            parameter_star_list.append(parameter_star)
        
        self.agents_pos=og_pos.copy()
        return parameter_star_list,entropy_ls,ag_ent_ls
    

    ##Outdated need to remove or archive (too slow but generalizable if you know x_star)
    def jocaobian_classifier_alt(self,
                                 param_list):
        row=[]
        for param in param_list:
            infl_matrix=self.field.influence_matrix(param)
            prob_matrix=self.field.prob_matrix(param)
            d_lnf_matrix=self.field.d_lnf_matrix(param)
            jacobian=jacobian_matrix(num_agents=self.num_agents,parameter_instance=param,agents_pos=self.agents_pos,bin_points=self.bin_points,resource_distribution=self.resource_distribution,
                          infl_type=self.infl_type,infl_fshift=self.infl_fshift,Q=self.Q,infl_matrix=infl_matrix,prob_matrix=prob_matrix,d_lnf_matrix=d_lnf_matrix)
            evals=torch.linalg.eigvals(jacobian)  
            if torch.sum(torch.sign(torch.real(evals)))==-1*len(evals):
                row.append(1)

            elif torch.all(evals==0):
                row.append(0)
            else:
                row.append(-1)
        
        return row

    def jocobian_cl_iter_alt(self,
                             param_list,
                             resource_params,
                             sig = .1):
        og_re_dist=self.resource_distribution
        bin_points=self.bin_points
        j_class_mat=0
        for parameter_id in range(len(resource_params)):
            re_param=resource_params[parameter_id]
            resource_distribution=1/(3*sig*np.sqrt(2*np.pi))*(np.exp(-(bin_points-.5-re_param/2)**2/(2*(sig)**2))+np.exp(-(bin_points-.5+1/2*re_param)**2/(2*sig**2)))
            self.resource_distribution=resource_distribution
            jc_class_row=torch.tensor(self.jocaobian_classifier_alt(param_list))
            j_class_mat=matrix_builder(row_id=parameter_id,row=jc_class_row,matrix=j_class_mat)
        self.resource_distribution=og_re_dist
        return j_class_mat

    def mean_stability_birfurcation_rs(self,r_st = .1,
                                       r_end = .5,
                                       r_points = 200,
                                       s_st = 0,
                                       s_end = 1,
                                       s_points = 100):
        agents_pos_og=self.agents_pos.copy()
        self.agents_pos=np.array([.5]*self.num_agents)
        start=[r_st]*self.num_agents
        end=[r_end]*self.num_agents
        params=np.linspace(start,end,r_points)
        params2=np.linspace(s_end,s_st,s_points)
        t=self.jocobian_cl_iter_alt(params,params2)
        
        #colors
        cmap = mpl.colormaps["viridis"]
        newcolors = cmap(np.linspace(0, 1, 100))
        newcolors[90:]= np.array([0.8, 0.8, 0.8,1])
        newcolors[:10]= np.array([0,0,0,1])
        newcolors[20] = mpl.colors.to_rgb('tab:orange') + (1,)
        newcmap = mpl.colors.ListedColormap(newcolors)

        #plot
        ax=sns.heatmap(t,xticklabels=False,yticklabels=False,cbar=False,vmin=0, vmax=1,cmap=newcmap)

        num_ticks_y = 11
        num_ticks_x = 9

        # the index of the position of yticks
        yticks = np.linspace(0, len(params2) - 1, num_ticks_y)
        xticks = np.linspace(0, len(params[:,0]) - 1, num_ticks_x)
        # the content of labels of these yticks
        yticklabel = [np.round(params2[int(idx)],decimals=1) for idx in yticks]
        xticklabel = [np.round(params[int(idx),0],decimals=2) for idx in xticks]

        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabel)
        ax.set_yticklabels(yticklabel)
        

        black_p = mpatches.Patch(color='black', label='unstable')
        white_p = mpatches.Patch(color='grey', label='stable')

        plt.legend(handles=[black_p, white_p])



        plt.title('Mean Stability Bifurcation Map')
        plt.xlabel(r"$\sigma$"+' (Reach parameter)' )
        plt.ylabel(r'$\alpha$'+' (Seperation parameter)')
        self.agents_pos=agents_pos_og.copy()

    