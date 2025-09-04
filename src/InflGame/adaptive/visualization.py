"""
.. module:: visualization
   :synopsis: Provides visualization tools for analyzing and understanding the dynamics of adaptive environments and agent interactions for influencer games.


Visualization Module
====================



This module provides visualization tools for analyzing and understanding the dynamics of adaptive environments and agent interactions for influencer games.
It includes plotting utilities for various domains (1D, 2D, and simplex) and supports visualizing agent positions, gradients, 
influence distributions, and bifurcation dynamics.

The module is designed to work with the `AdaptiveEnv` class and provides a framework for creating visual representations of agent behaviors 
in influencer game environments.

Dependencies:
-------------
- InflGame.utils
- InflGame.kernels
- InflGame.domains

Usage:
------
The `Shell` class can be used to visualize the results of simulations performed using the `AdaptiveEnv` class. It supports various visualization types, including position plots, gradient plots, probability plots, and bifurcation plots.

Example:
--------

.. code-block:: python
    
    from InflGame.adaptive.visualization import Shell
    import torch
    import numpy as np

    # Initialize the Shell
    shell = Shell(
        num_agents=3,
        agents_pos=np.array([0.2, 0.5, 0.8]),
        parameters=torch.tensor([1.0, 1.0, 1.0]),
        resource_distribution=torch.tensor([10.0, 20.0, 30.0]),
        bin_points=np.array([0.1, 0.4, 0.7]),
        infl_configs={'infl_type': 'gaussian'},
        learning_rate_type='cosine_annealing',
        learning_rate=[0.0001, 0.01, 15],
        time_steps=100,
        domain_type='1d',
        domain_bounds=[0, 1]
    )

    # Set up the adaptive environment
    shell.setup_adaptive_env()

    # Plot agent positions
    fig = shell.pos_plot()
    fig.show()
"""

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
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import sys
import time
import copy
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import traceback
import copy

import matplotlib.figure
from typing import Union, List, Dict, Optional, Tuple

import InflGame.adaptive.grad_func_env as grad_func_env
#import InflGame.adaptive_dynamics.jacobian as jacobian

import InflGame.utils.general as general
from InflGame.utils import data_management
import InflGame.utils.validation as validation
import InflGame.utils.plot_utils as plot_utils

import InflGame.kernels.gauss as gauss 
import InflGame.kernels.diric as diric
import InflGame.kernels.MV_gauss as MV_gauss
#import InflGame.kernels.jones as jones


import InflGame.domains.rd as rd

import InflGame.domains.one_d.one_utils as one_utils
import InflGame.domains.one_d.one_plots as one_plots


import InflGame.domains.two_d.two_utils as two_utils
import InflGame.domains.two_d.two_plots as two_plots


import InflGame.domains.simplex.simplex_utils as simplex_utils
import InflGame.domains.simplex.simplex_plots as simplex_plots


# Import the new stability analysis module
from InflGame.utils.general import agent_parameter_setup
import InflGame.adaptive.jacobian as jc



class Shell:
    """
    The Shell class provides a framework for simulating and visualizing adaptive dynamics
    in various domains (1D, 2D, and simplex). It supports gradient ascent, influence distribution
    calculations, and plotting utilities for analyzing agent behaviors in resource distribution environments.
    """

    def __init__(self,
                 num_agents: int,
                 agents_pos: Union[List[float], np.ndarray],
                 parameters: torch.Tensor,
                 resource_distribution: torch.Tensor,
                 bin_points: Union[List[float], np.ndarray],
                 infl_configs: Dict[str, str] = {'infl_type': 'gaussian'},
                 learning_rate_type: str = 'cosine_annealing',
                 learning_rate: List[float] = [.0001, .01, 15],
                 time_steps: int = 100,
                 fp: int = 0,
                 infl_cshift: bool = False,
                 cshift: Optional[torch.Tensor] = None,
                 infl_fshift: bool = False,
                 Q: Optional[int] = None,
                 domain_type: str = '1d',
                 domain_bounds: Union[List[float], torch.Tensor] = [0, 1],
                 resource_type: float = 'na',
                 domain_refinement: int = 10,
                 tolerance: float = 10**-5,
                 tolerated_agents: Optional[int] = None,
                 ignore_zero_infl: bool = False) -> None:
        """
        Initialize the Shell class with simulation parameters.

        :param num_agents: Number of agents in the simulation.
        :type num_agents: int
        :param agents_pos: Initial positions of agents.
        :type agents_pos: Union[List[float], np.ndarray]
        :param parameters: Parameters for the influence function.
        :type parameters: torch.Tensor
        :param resource_distribution: Resource distribution over the domain.
        :type resource_distribution: torch.Tensor
        :param bin_points: Discretized points in the domain.
        :type bin_points: Union[List[float], np.ndarray]
        :param mean: Mean value for certain influence functions.
        :type mean: Optional[int]
        :param infl_configs: Configuration for influence kernels.
            - ``infl_type`` (str): The type of influence kernel (e.g., "gaussian", "multi_gaussian", "Jones_M", "dirichlet", "custom").
            - ``custom_influence`` (callable): Function for a custom influence (see guides).
        :type infl_configs: Dict[str, str]
        :param learning_rate_type: Learning rate type (e.g., 'cosine_annealing').
        :type learning_rate_type: str
        :param learning_rate: Learning rate parameters.
        :type learning_rate: List[float]
        :param time_steps: Number of gradient ascent steps.
        :type time_steps: int
        :param fp: Fixed parameter for influence function.
        :type fp: int
        :param infl_cshift: Whether to apply a center shift to influence.
        :type infl_cshift: bool
        :param cshift: Center shift tensor.
        :type cshift: Optional[torch.Tensor]
        :param infl_fshift: Whether to apply a fixed shift to influence.
        :type infl_fshift: bool
        :param Q: Additional parameter for influence function.
        :type Q: Optional[int]
        :param domain_type: Type of domain ('1d', '2d', or 'simplex').
        :type domain_type: str
        :param domain_bounds: Bounds of the domain.
        :type domain_bounds: Union[List[float], torch.Tensor]
        :param resource_type: Type of resource distribution.
        :type resource_type: float
        :param domain_refinement: Refinement level for 2D domains.
        :type domain_refinement: int
        :param tolerance: Tolerance for convergence.
        :type tolerance: float
        :param tolerated_agents: Number of agents allowed to tolerate deviations.
        :type tolerated_agents: Optional[int]
        """
        validated=validation.validate_adaptive_config(
            num_agents=num_agents,
            agents_pos=agents_pos,
            parameters=parameters,
            resource_distribution=resource_distribution,
            bin_points=bin_points,
            infl_configs=infl_configs,
            learning_rate_type=learning_rate_type,
            learning_rate=learning_rate,
            time_steps=time_steps,
            fp=fp,
            infl_cshift=infl_cshift,
            cshift=cshift,
            infl_fshift=infl_fshift,
            Q=Q,
            domain_type=domain_type,
            domain_bounds=domain_bounds,
            tolerance=tolerance,
            tolerated_agents=tolerated_agents
        )
        self.num_agents = validated['num_agents']
        self.agents_pos = validated['agents_pos']
        self.infl_type = validated['infl_type']
        self.infl_configs = validated['infl_configs']
        self.parameters = validated['parameters']
        self.resource_distribution = validated['resource_distribution']
        self.bin_points = validated['bin_points']
        self.learning_rate = validated['learning_rate']
        self.time_steps = validated['time_steps']
        self.fp = validated['fp']
        self.learning_rate_type = validated['learning_rate_type']
        self.infl_cshift = validated['infl_cshift']
        self.cshift = validated['cshift']
        self.infl_fshift = validated['infl_fshift']
        self.Q = validated['Q']
        self.domain_type = validated['domain_type']
        self.domain_bounds = validated['domain_bounds']
        self.sigma_inv = 0
        self.tolerance = validated['tolerance']
        self.tolerated_agents = validated['tolerated_agents']
        self.resource_type = resource_type
        self.ignore_zero_infl=ignore_zero_infl
        # Set up the domain based on the type
        if domain_type == 'simplex':
            self.r2 = domain_bounds[0]
            self.corners = domain_bounds[1]
            self.triangle = domain_bounds[2]
            self.trimesh = domain_bounds[3]
        if domain_type == '2d':
            self.rect_X, self.rect_Y, self.rect_positions = two_utils.two_dimensional_rectangle_setup(domain_bounds, domain_refinement=domain_refinement)

    def setup_adaptive_env(self) -> None:
        """
        Set up the adaptive environment for the simulation. This initializes the
        gradient function environment with the provided parameters.
        """
        self.field=grad_func_env.AdaptiveEnv(num_agents=self.num_agents,agents_pos=self.agents_pos,parameters=self.parameters,
                                             resource_distribution=self.resource_distribution,bin_points=self.bin_points,
                                             infl_configs=self.infl_configs,learning_rate_type=self.learning_rate_type,learning_rate=self.learning_rate,time_steps=self.time_steps,fp=self.fp,infl_cshift=self.infl_cshift,cshift=self.cshift,
                                             infl_fshift=self.infl_fshift,Q=self.Q,domain_type=self.domain_type,domain_bounds=self.domain_bounds,tolerance=self.tolerance,tolerated_agents=self.tolerated_agents,ignore_zero_infl=self.ignore_zero_infl)
    
    ## Plots
    def pos_plot(self,
                 title_ads: List[str] = [],
                 save: bool = False,
                 name_ads: List[str] = [],
                 font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                 save_types: List[str] = ['.png', '.svg'],
                 paper_figure: dict= {'paper':False,'section':'A','figure_id':'pos_plot'}) -> matplotlib.figure.Figure:
        r"""
        Plots the positions of agents over gradient ascent steps. The positions of players are calculated via the 
        results of :func:`InflGame.adaptive.grad_func_env.gradient_ascent`  


        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        
        if self.domain_type=='1d':
            fig=one_plots.pos_plot_1d(num_agents=self.num_agents,pos_matrix=self.field.pos_matrix,title_ads=title_ads,domain_bounds=self.domain_bounds,font=font)

        elif self.domain_type=='2d':
            ValueError("Not yet complete for 2d systems")

        elif self.domain_type=='simplex':
            fig=simplex_plots.pos_plot_simplex(num_agents=self.num_agents,bin_points=self.bin_points,corners=self.corners,triangle=self.triangle,pos_matrix=self.field.pos_matrix,font=font)
        
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'position','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','pos_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig

    def grad_plot(self,
                  title_ads: List[str] = [],
                  save: bool = False,
                  name_ads: List[str] = [],
                  font={'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                  save_types: List[str] = ['.png', '.svg'],
                  paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'grad_plot'}) -> matplotlib.figure.Figure:
        r"""
        Plots the gradients of agents as the lines versus time.  

        .. math::
                \frac{\partial}{\partial x_{(i,l)}}u_i(x)=\sum_{k=1}^{K}G_{i,k}(x_i,b_k)\frac{\partial}{\partial x_{(i,l)}}ln(f_{i}(x_i,b_k))
        
        Where :math:`x_i` is the :math:`i` th players position, :math:`b_k\in \mathbb{B}` are the bin/resource points, :math:`B(b_k)` is the resource value at :math:`b_k`, and :math:`G_i(x_i,b_k)` 
        probability of player :math:`i` influencing the bin point :math:`b`. The gradients are calculated via :func:`InflGame.adaptive.grad_func_env.gradient`.
        

        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        

        if self.domain_type=='1d':
            fig=one_plots.gradient_plot_1d(num_agents=self.num_agents,grad_matrix=self.field.grad_matrix,title_ads=title_ads,font=font)
        else:
            raise ValueError("Gradient plot is only available for 1D domains.")
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'gradient','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','grad_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def prob_plot(self,
                  position: np.ndarray = [],
                  parameters: np.ndarray = [],
                  title_ads: List[str] = [],
                  save: bool = False,
                  name_ads: List[str] = [],
                  font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                  save_types: List[str] = ['.png', '.svg'],
                  paper_figure: dict = {'paper': False, 'section': 'A','figure_id': 'prob_plot'}) -> matplotlib.figure.Figure:
        r"""
        Plot the probability distribution of agents' influence over a bin/resource point via their relative influence over that point  
        
        .. math::
                G_{i,k}(\mathbf{x},b_k)=\frac{f_{i}(x_i,b_k)}{\sum_{j=1}^{N}f_{j}(x_j,b_k)}.
        
        where :math:`f_{i}(x_i,b_k)` is the :math:`i` th players influence. The probabilities are calculated via :func:`InflGame.adaptive.grad_func_env.prob_matrix` .
        
        .. figure:: examples/probability.png
            :scale: 75 %

            This is an example of the a three player game with agents in position :math:`[.1,.45,.9]` . 
            Here the the influence kernels are symmetric gaussian with with parameter (reach) :math:`\sigma=0.25` .
        
            
        There can is also the option to have a fixed third party (infl_csift==True) and/or abstaining voters if (infl_fshift==True). 
        
        :param position: Positions of agents.
        :type position: np.ndarray
        :param parameters: Parameters for the influence function.
        :type parameters: np.ndarray
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        og_position=self.field.agents_pos.clone()
        og_parameters=self.field.parameters.clone()
        if len(position) != 0:
            self.field.agents_pos=position
        else:
            position=self.agents_pos
        if len(parameters) != 0:
            self.field.parameters=parameters
        else:
            parameters=self.parameters
            

        prob=self.field.prob_matrix(parameter_instance=parameters)
        if self.domain_type=='1d':
            voting_configs={'fixed_party':self.infl_cshift,'abstain':self.infl_fshift}
            fig=one_plots.prob_plot_1d(num_agents=self.num_agents,agents_pos=position,bin_points=self.bin_points,domain_bounds=self.domain_bounds,prob=prob,voting_configs=voting_configs,title_ads=title_ads,font=font)
        else:
            raise ValueError("Probability plot is only available for 1D domains.")
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'probability','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure['figure_id']},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')

        self.field.parameters=og_parameters
        self.field.agents_pos=og_position
        return fig
    
    def three_agent_pos_3d(self,
                      x_star: Optional[float] = None,
                      title_ads: List[str] = [],
                      name_ads: List[str] = [],
                      save: bool = False,
                      font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                      save_types: List[str] = ['.png', '.svg'],
                      paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'three_players'}) -> matplotlib.figure.Figure:
        """
        Visualize dynamics for three players: demonstrates the players positions changing in time in 3-d space. Demonstrating the instability of 3-players in influencer games. Given that this function 
        is in 3-d space for players with 1-d strategies there isn't a way to visualize the 3 player dynamics in games with more then 1d strategy spaces. The positions of players are calculated via the 
        results of :func:`InflGame.adaptive.grad_func_env.gradient_ascent` .

        .. figure:: examples/three_player.png
            :scale: 75 %

            This is an example of the three player dynamics via their positions in time`.

        :param x_star: Equilibrium position.
        :type x_star: Optional[float]
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        if self.domain_type=='1d' or self.num_agents!=3:
            if self.infl_type=="gaussian":
                fig=one_plots.three_agent_dynamics(self.field.pos_matrix,x_star=general.discrete_mean(self.bin_points,resource_distribution=self.resource_distribution),title_ads=title_ads,font=font)
            
            else:
                if x_star==None:
                    raise ValueError("x_star must be provided for non-Gaussian influence types.")
                else:   
                    fig=one_plots.three_agent_dynamics(self.field.pos_matrix,x_star=general.discrete_mean(self.bin_points,resource_distribution=self.resource_distribution),font=font)

            if save==True:
                file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'3d_space','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','three_players')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
                for file_name in file_names:
                    fig.savefig(file_name,bbox_inches='tight')
            return fig
        else:
            raise ValueError("Three player dynamics is only available for 1D domains.")

    def vect_plot(self,
                  agent_id: int= None,
                  parameter_instance: Union[List[float], np.ndarray, torch.Tensor] = None,
                  cmap: str = 'viridis',
                  typelabels: List[str] = ["A", "B", "C"],
                  ids: List[int] = [0, 1],
                  pos: Optional[torch.Tensor] = None,
                  title_ads: List[str] = [],
                  save: bool = False,
                  name_ads: List[str] = [],
                  save_types: List[str] = ['.png', '.svg'],
                  paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'vect_plot'},
                  font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                  alt_form : bool = False,
                  **kwargs) -> matplotlib.figure.Figure:
        """
        Plot the vector field of gradients for a specific agents calculated by the function :func:`calc_direction_and_strength` . **Currently only supports 1d domains**

        .. figure:: examples/vector_field.png
            :scale: 75 %

            This is an example of the vector field for a three player game with only players 1 and 2 dynamics shown (player 3 is fixed)`.

        :param agent_id: ID of the agent.
        :type agent_id: int
        :param parameter_instance: Parameters for the influence function.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param cmap: Colormap for the plot.
        :type cmap: str
        :param typelabels: Labels for agent types.
        :type typelabels: List[str]
        :param ids: IDs of agents of interest.
        :type ids: List[int]
        :param pos: Positions of agents.
        :type pos: Optional[torch.Tensor]
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]
        :param kwargs: Additional arguments for plotting.

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        
        
        if pos is None:
            mean=general.discrete_mean(self.bin_points,resource_distribution=self.resource_distribution)
            pos=general.agent_optimal_position_setup(num_agents=self.num_agents,agents_pos=self.agents_pos,infl_type=self.infl_type,mean=mean,domain_type=self.domain_type,ids=ids)
        else:
            if not torch.is_tensor(pos):
                pos=torch.tensor(pos, dtype=torch.float32)
        if parameter_instance is None:
            parameter_instance=self.parameters
        else:
            if not torch.is_tensor(parameter_instance):
                parameter_instance=torch.tensor(parameter_instance, dtype=torch.float32)
        if self.domain_type=='1d':
            grads=self.calc_direction_and_strength(parameter_instance=parameter_instance,agent_id=agent_id,ids=ids,pos=pos,alt_form=alt_form)
            fig=one_plots.vector_plot_1d(gradient=grads,ids=ids,title_ads=title_ads,font=font,**kwargs)

        elif self.domain_type=='simplex':  
            ValueError("Vector plots currently only available for 1D domains.")
            #self.calc_direction_and_strength(parameter_instance,agent_id=agent_id)

        elif self.domain_type=='2d':
            ValueError("Vector plots currently only available for 1D domains.")
            #self.two_a=True
            #self.calc_direction_and_strength(parameter_instance=parameter_instance,agent_id=agent_id,ids=ids,pos=pos)
            
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'vector','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','vect_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def dist_plot(self,
                  agent_id: int,
                  parameter_instance: Union[List[float], np.ndarray, torch.Tensor] = 0,
                  cmap: str = 'viridis',
                  typelabels: List[str] = ["A", "B", "C"],
                  save: bool = False,
                  name_ads: List[str] = [],
                  save_types: List[str] = ['.png', '.svg'],
                  paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'dist_plot'},
                  font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                  **kwargs) -> matplotlib.figure.Figure:
        r"""
        Plot the agents' influence distributions distributions via the function :func:`calc_infl_dist` .
        

        :param agent_id: ID of the agent.
        :type agent_id: int
        :param parameter_instance: Parameters for the influence function.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param cmap: Colormap for the plot.
        :type cmap: str
        :param typelabels: Labels for agent types.
        :type typelabels: List[str]
        :param kwargs: Additional arguments for plotting.

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)

        if self.domain_type=='1d':
            fig, ax = plt.subplots()
            for agent_id in range(self.num_agents):
                ax.plot(self.bin_points,self.infl_dist[agent_id],label='Agent '+str(agent_id))
            plt.legend()
            plt.title('Agent influence distribution')
            plt.close()

        elif self.domain_type=='simplex':
            fig=simplex_plots.dist_plot_simplex(agent_id=agent_id,r2=self.r2,corners=self.corners,triangle=self.triangle,trimesh=self.trimesh,infl_dist=self.infl_dist,cmap=cmap,typelabels=typelabels,margin=0.01,font=font,**kwargs)

        elif self.domain_type=='2d':
            fig=two_plots.dist_plot_2d(agent_id=agent_id,infl_dist=self.infl_dist,rect_X=self.rect_X,rect_Y=self.rect_Y,font=font)
        
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'distribution','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','dist_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        
        return fig
            
    def dist_pos_plot(self,
                      parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                      typelabels: List[str] = ["A", "B", "C"],
                      cmap1: str = 'twilight',
                      cmap2: str = 'viridis',
                      title_ads: List[str] = [],
                      save: bool = False,
                      name_ads: List[str] = [],
                      save_types: List[str] = ['.png', '.svg'],
                      paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'dist_pos_plot'},
                      font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                      ) -> matplotlib.figure.Figure:
        r"""
        Plots the agents' influence distributions with their their positions. The influence distributions are calculated from :func:`calc_infl_dist` and the position vectors are calculated 
        via :func:`InflGame.adaptive.grad_func_env.gradient_ascent` . This function works with all 3 domain types '1d','2d', and 'simplex'.

        ** In one dimension**

        .. figure:: examples/dist_pos_1d.png
            :scale: 75 %

            This is a 3 player influence distribution and position plot for a 1d domain`.

        **For a simplex**

        
        .. figure:: examples/dist_pos.png
            :scale: 75 %

            This is a 2 player influence distribution and position plot for a simplex domain`.


        :param parameter_instance: Parameters for the influence function.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param typelabels: Labels for agent types.
        :type typelabels: List[str]
        :param cmap1: Colormap for the plot.
        :type cmap1: str
        :param cmap2: Colormap for the plot.
        :type cmap2: str
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """

        NUM_COLORS = self.num_agents+1
        cm = pylab.get_cmap(cmap1)
        if self.domain_type=='1d':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.field.pos_matrix[-1])
            fig=one_plots.dist_and_pos_plot_1d(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=self.resource_distribution,pos_matrix=self.field.pos_matrix,len_grad_matrix=len(self.field.grad_matrix),infl_dist=self.infl_dist,cm=cm,NUM_COLORS=NUM_COLORS,title_ads=title_ads,font=font)
        
        elif self.domain_type=='2d':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)
            fig=two_plots.dist_and_pos_plot_2d_simple(num_agents=self.num_agents,bin_points=self.bin_points,rect_X=self.rect_X,rect_Y=self.rect_Y,cmap1=cmap1,cmap2=cmap2,pos_matrix=self.field.pos_matrix,infl_dist=self.infl_dist,resource_type=self.resource_type,resources=self.resource_distribution,font=font)
        
        elif self.domain_type=='simplex':
            self.calc_infl_dist(parameter_instance=parameter_instance,pos=self.agents_pos)
            fig=simplex_plots.dist_and_pos_plot_simplex(num_agents=self.num_agents,bin_points=self.bin_points,r2=self.r2,corners=self.corners,triangle=self.triangle,trimesh=self.trimesh,typelabels=typelabels,cmap1=cmap1,cmap2=cmap2,pos_matrix=self.field.pos_matrix,infl_dist=self.infl_dist,resource_type=self.resource_type,resources=self.resource_distribution,font=font)
        
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'dist_pos','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','dist_pos_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig   
        
    def dist_pos_gif(self,
                     max_frames: int,
                     output_filename: str = 'timelapse_test.gif',
                     optimize_memory: bool = True,
                     dpi: int = 100,
                     fps: int = 10,
                     quality: int = 8,
                     verbose: bool = False) -> matplotlib.figure.Figure:
        r"""
        Plot the agents' influence distributions and their positions into a gif. Works for all domain types  '1d', '2d', 'simplex'. This is done by 
        putting together frames of :func:`dist_pos_plot` .
        
        **Optimized Performance Features:**
        - Direct memory writing without intermediate files
        - Matplotlib figure recycling for memory efficiency
        - Optimized frame sampling
        - Configurable quality vs speed trade-offs
        
        **Simplex example** 

        .. figure:: examples/timelapse.gif
            :scale: 75 %

            This is a 3 player influence distribution and position plot gif for a simplex domain`.


        :param max_frames: Maximum number of frames for the gif.
        :type max_frames: int
        :param output_filename: Name of the output GIF file.
        :type output_filename: str
        :param optimize_memory: Whether to use memory optimization techniques.
        :type optimize_memory: bool
        :param dpi: DPI for the frames (lower = faster, higher = better quality).
        :type dpi: int
        :param fps: Frames per second for the GIF.
        :type fps: int
        :param quality: GIF compression quality (1-10, lower = smaller file).
        :type quality: int
        :param verbose: Whether to print progress information.
        :type verbose: bool

        :return: The generated gif figure.
        :rtype: matplotlib.figure.Figure
        """
        import io
        from PIL import Image
        
        pos_len = len(self.field.pos_matrix)
        
        # Handle single frame case
        if max_frames == 1:
            times = range(1, pos_len, int(np.round(pos_len/2)) + 1)
            og_pos = self.agents_pos.clone()
            og_pos_matrix = self.field.pos_matrix.clone()
            
            self.field.pos_matrix = self.field.pos_matrix[0:]
            self.agents_pos = self.field.pos_matrix[-1]
            f = self.dist_pos_plot(self.parameters, typelabels=["A","B","C"])
            
            # Restore original state
            self.agents_pos = og_pos
            self.field.pos_matrix = og_pos_matrix
            return f
        
        # Calculate optimized frame indices
        if max_frames >= pos_len:
            times = list(range(1, pos_len))
        else:
            # Smart sampling: ensure we get key frames including start and end
            step = max(1, int(np.floor(pos_len / max_frames)))
            times = list(range(1, pos_len, step))
            if times[-1] != pos_len - 1:
                times.append(pos_len - 1)  # Ensure we include the final frame
        
        # Store original state
        og_pos = self.agents_pos.clone()
        og_pos_matrix = self.field.pos_matrix.clone()
        
        # Prepare for optimized GIF creation
        frames = []
        
        if optimize_memory:
            # Memory-optimized approach: direct buffer writing
            if verbose:
                print(f"Creating {len(times)} frames with memory optimization...")
            
            # Create a single figure and reuse it
            fig = None
            
            try:
                with imageio.get_writer(output_filename, mode='I', fps=fps, 
                                      quantizer='nq', palettesize=256) as writer:
                    
                    for i, time_step in enumerate(times):
                        # Update positions
                        self.field.pos_matrix = og_pos_matrix[:time_step]
                        self.agents_pos = og_pos_matrix[time_step - 1]
                        
                        # Create or update plot
                        if fig is not None:
                            plt.close(fig)  # Close previous figure to avoid accumulation
                        
                        # Suppress matplotlib output when not in verbose mode
                        if verbose:
                            fig = self.dist_pos_plot(self.parameters, typelabels=["A","B","C"])
                        else:
                            # Redirect stdout/stderr to suppress matplotlib figure output
                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                fig = self.dist_pos_plot(self.parameters, typelabels=["A","B","C"])
                        
                        fig.set_dpi(dpi)
                        
                        # Convert to image buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                        buf.seek(0)
                        
                        # Load and append frame
                        image = Image.open(buf)
                        image_array = np.array(image)
                        writer.append_data(image_array)
                        
                        buf.close()
                        
                        # Progress indicator
                        if i % max(1, len(times) // 10) == 0 and verbose:
                            print(f"Progress: {i+1}/{len(times)} frames ({100*(i+1)/len(times):.1f}%)")
                    
                    if fig is not None:
                        plt.close(fig)
                        
            except Exception as e:
                if verbose:
                    print(f"Memory optimization failed: {e}")
                    print("Falling back to file-based approach...")
                optimize_memory = False
        
        if not optimize_memory:
            # Traditional file-based approach with optimizations
            import tempfile
            
            if verbose:
                print(f"Creating {len(times)} frames with file optimization...")
            
            # Use temporary directory for better performance
            with tempfile.TemporaryDirectory() as temp_dir:
                filenames = []
                
                for i, time_step in enumerate(times):
                    # Update positions
                    self.field.pos_matrix = og_pos_matrix[:time_step]
                    self.agents_pos = og_pos_matrix[time_step - 1]
                    
                    # Create plot
                    f = self.dist_pos_plot(self.parameters, typelabels=["A","B","C"])
                    
                    # Save to temporary file
                    filename = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    filenames.append(filename)
                    f.savefig(filename, dpi=dpi, bbox_inches="tight", 
                             facecolor='white', edgecolor='none')
                    plt.close(f)  # Free memory immediately
                    
                    # Progress indicator
                    if i % max(1, len(times) // 10) == 0 and verbose:
                        print(f"Progress: {i+1}/{len(times)} frames ({100*(i+1)/len(times):.1f}%)")
                
                # Create GIF from files
                if verbose:
                    print("Writing GIF...")
                with imageio.get_writer(output_filename, mode='I', fps=fps,
                                      quantizer='nq', palettesize=256) as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
        
        # Restore original state
        self.agents_pos = og_pos
        self.field.pos_matrix = og_pos_matrix
        
        if verbose:
            print(f"GIF created successfully: {output_filename}")
        
        # Return the final frame for display
        self.agents_pos = og_pos_matrix[-1]
        final_fig = self.dist_pos_plot(self.parameters, typelabels=["A","B","C"])
        self.agents_pos = og_pos
        
        return final_fig

    def three_agent_pos_3A1d(self,
                             x_star: Optional[float] = None,
                             title_ads: List[str] = [],
                             save: bool = False,
                             name_ads: List[str] = [],
                             fontL = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                             fontR = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                             fontmain= {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                             save_types: List[str] = ['.png', '.svg'],
                             paper_figure: dict= {'paper':False,'section':'A','figure_id':'pos_plot'}) -> matplotlib.figure.Figure:
        
        if self.domain_type=='1d':
            if x_star is None:
                x_star=general.discrete_mean(self.bin_points,resource_distribution=self.resource_distribution)
            ax1=one_plots.three_agent_dynamics(pos_matrix=self.field.pos_matrix, x_star=x_star,axis_return=True,title_ads=[],font=fontL)
            ax2=one_plots.pos_plot_1d(num_agents=self.num_agents, pos_matrix=self.field.pos_matrix, domain_bounds=self.domain_bounds, title_ads=[], font=fontR, axis_return=True)
            fig=plot_utils.side_by_side_plots(ax1,ax2,title_ads=title_ads,font=fontmain,title_main='Three Agent dynamics')

            if save==True:
                file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'3a_all','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','pos_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
                for file_name in file_names:
                    fig.savefig(file_name,bbox_inches='tight')
            return fig

    def equilibrium_bifurcation_plot(self,
                                     reach_start: float = .03,
                                     reach_end: float = .3,
                                     reach_num_points: int = 30,
                                     time_steps: int = 100,
                                     initial_pos: Union[List[float], np.ndarray] = 0,
                                     current_alpha: float = .5,
                                     tolerance: Optional[float] = None,
                                     tolerated_agents: Optional[int] = None,
                                     refinements: int = 2,
                                     plot_type: str = "normal",
                                     title_ads: List[str] = [],
                                     name_ads: List[str] = [],
                                     save: bool = False,
                                     save_types: List[str] = ['.png', '.svg'],
                                     return_matrix: bool = False,
                                     parallel_configs: Dict[str, Union[bool, int]] = None,
                                     cmaps: dict = {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys'},
                                     font: dict = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12,'font_family': 'sans-serif'},
                                     cbar_config: dict = {'center_labels': True, 'label_alignment': 'center', 'shrink': 0.8},
                                     paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'equilibrium_bifurcation_plot'},
                                     show_pred: bool = False,
                                     optional_vline: float = None) -> Union[torch.Tensor, matplotlib.figure.Figure]:
        r"""
        Plots the equilibrium bifurcation for agents over a range of reach parameters. As :math:`\sigma` or as variance goes to zero for players' influence kernels, 
        the players begin to bifuricate. This plotting function computes the gradient ascent alogorithm repetively for varying parameter values util the players reach an equalbiirum.
        
        I.e. each player will have a vector of final positions :math:`X_i=[x_1,x_2,\dots,x_A]` where :math:`A` is the number of test parameters and
        :math:`x_i` is the final postion of the :math:`i` th player. Then the plot plots each players final postion.

        This is done via the function :func:`final_pos_over_reach`.

        **1d example**

        .. figure:: examples/bifurcation.png
            :scale: 75 %

            This is a 5 player positions bifurication plot for a 1d domain`.
             
        the critical values are estimated via :func:`InflGame.domains.one_d.utils.critical_values_plot`. 

        **2d example**

        .. figure:: examples/2d_bifurcation.png
            :scale: 75 %

            This is a 6 player positions bifurication plot for a 2d domain`.

        **Simplex example**

        .. figure:: examples/simplex_bifurcation.png
            :scale: 75 %

            This is a 3 player positions bifurication plot for a simplex`.

        :param reach_start: Starting value of reach parameter.
        :type reach_start: float
        :param reach_end: Ending value of reach parameter.
        :type reach_end: float
        :param reach_num_points: Number of points in the reach parameter range.
        :type reach_num_points: int
        :param num_interations: Number of gradient ascent steps.
        :type num_interations: int
        :param intitial_pos: Initial positions of agents.
        :type intitial_pos: Union[List[float], np.ndarray]
        :param current_alpha: Current alpha value.
        :type current_alpha: float
        :param tolerance: Tolerance for convergence.
        :type tolerance: Optional[float]
        :param tolerated_agents: Number of agents allowed to tolerate deviations.
        :type tolerated_agents: Optional[int]
        :param refinements: Refinement level for plotting.
        :type refinements: int
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param save_types: File types to save the plot.
        :type save_types: List[str]
        :param return_matrix: Whether to return the final position matrix.
        :type return_matrix: bool
        :param parallel: Whether to use parallel processing.
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers (defaults to CPU count).
        :type max_workers: Optional[int]
        :param batch_size: Batch size for processing (auto-calculated if None).
        :type batch_size: Optional[int]

        :return: The generated plot figure or final position matrix.
        :rtype: Union[torch.Tensor, matplotlib.figure.Figure]
        """
        og_iterations=self.time_steps
        og_pos=self.agents_pos.clone()
        self.field.time_steps=time_steps
        self.agents_pos=initial_pos.clone()

        if parallel_configs is None:
            parallel_configs = {'parallel': False, 'max_workers': 4, 'batch_size': 2}
        
        parallel = parallel_configs.get('parallel', True)
        max_workers = parallel_configs.get('max_workers', 4)
        batch_size = parallel_configs.get('batch_size', 2)
        if tolerated_agents==None:
            tolerated_agents=self.tolerated_agents
        if tolerance==None:
            tolerance=self.tolerance

        if self.domain_type=="1d":
            reach_parameters=general.agent_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters, tolerance=tolerance, tolerated_agents=tolerated_agents, parallel=parallel, max_workers=max_workers, batch_size=batch_size, time_steps=time_steps)
            fig=one_plots.equilibrium_bifurcation_plot_1d(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=self.resource_distribution,infl_type=self.infl_type,infl_cshift=self.infl_cshift,reach_parameters=reach_parameters,final_pos_matrix=final_pos_matrix,reach_start=reach_start,reach_end=reach_end,refinements=refinements,plot_type=plot_type,title_ads=title_ads,font=font,cmaps=cmaps,cbar_config=cbar_config,show_pred=show_pred,optional_vline=optional_vline)

        elif self.domain_type=='2d':
            reach_parameters=general.agent_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters.clone(), tolerance=tolerance, tolerated_agents=tolerated_agents, parallel=parallel, max_workers=max_workers, batch_size=batch_size, time_steps=time_steps)
            fig=two_plots.equilibrium_bifurcation_plot_2d_simple(num_agents=self.num_agents,domain_bounds=self.domain_bounds,reach_num_points=reach_num_points,final_pos_matrix=final_pos_matrix,title_ads=title_ads,font=font)
            
        elif self.domain_type=='simplex':
            reach_parameters=general.agent_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
            final_pos_matrix=self.final_pos_over_reach(reach_parameters.clone(), tolerance=tolerance, tolerated_agents=tolerated_agents, parallel=parallel, max_workers=max_workers, batch_size=batch_size, time_steps=time_steps)
            fig=simplex_plots.equalibirium_bifurication_plot_simplex(num_agents=self.num_agents,r2=self.r2,corners=self.corners,triangle=self.triangle,final_pos_matrix=final_pos_matrix,reach_num_points=reach_num_points,type_labels=None,title_ads=title_ads)
        
        self.field.time_steps=og_iterations
        self.field.agents_pos=og_pos.clone()
        self.agents_pos=og_pos.clone()
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'bifurcation','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','equilibrium_bifurcation_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        if return_matrix==True:
            return fig,final_pos_matrix
        else: 
            return fig
        
    def first_order_bifurcation_plot(self,
                                      agent_parameter_instance: Union[List[float], np.ndarray],
                                      resource_distribution_type: str,
                                      resource_entropy: bool = False,
                                      infl_entropy: bool = False,
                                      alpha_current: float = .5,
                                      alpha_st: float = 0,
                                      alpha_end: float = 1,
                                      varying_paramter_type: str = 'mean',
                                      fixed_parameters_lst: Optional[List[float]] = None,
                                      name_ads: List[str] = [],
                                      title_ads: List[str] = [],
                                      save_types: List[str] = ['.png', '.svg'],
                                      paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'first_order_bifurcation_plot'},
                                      font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'}) -> matplotlib.figure.Figure:
        r"""
        Plots the first-order bifurcation for agents over a range of alpha values (resource parameters) via func:`jacobian_stability_fast`. Currently only works for 
        Gaussian influence and multi-variate Gaussian influence kernels. 

        **Gaussian example**

        .. figure:: examples/first_order.png
            :scale: 75 %

            This is a first order bifurcations plot for 5 players using symmetric Gaussian influence kernels.
             

        :param agent_parameter_instance: Parameters for the influence function.
        :type agent_parameter_instance: Union[List[float], np.ndarray]
        :param resource_distribution_type: Type of resource distribution.
        :type resource_distribution_type: str
        :param resource_entropy: Whether to calculate resource entropy.
        :type resource_entropy: bool
        :param infl_entropy: Whether to calculate influence entropy.
        :type infl_entropy: bool
        :param alpha_current: Current alpha value.
        :type alpha_current: float
        :param alpha_st: Starting value of alpha.
        :type alpha_st: float
        :param alpha_end: Ending value of alpha.
        :type alpha_end: float
        :param varying_paramter_type: Type of varying parameter (e.g., 'mean').
        :type varying_paramter_type: str
        :param fixed_parameters_lst: List of fixed parameters.
        :type fixed_parameters_lst: Optional[List[float]]
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param save_types: File types to save the plot.
        :type save_types: List[str]

        :return: The generated plot figure.
        :rtype: matplotlib.figure.Figure
        """
        resource_parameters,alpha=general.resource_parameter_setup(resource_distribution_type=resource_distribution_type,varying_paramter_type=varying_paramter_type,alpha_st=alpha_st, alpha_end=alpha_end, fixed_parameters_lst=fixed_parameters_lst)
        y=self.jacobian_stability_fast(agent_parameter_instance=agent_parameter_instance,resource_distribution_type=resource_distribution_type,resource_parameters=resource_parameters,resource_entropy=resource_entropy,infl_entropy=infl_entropy)[0]
        fig,ax=plt.subplots()
        ax.set_box_aspect(1)
        
        # Apply font settings
        plt.rcParams.update({'font.size': font['default_size'], 'font.family': font['font_family']})
        
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
        plt.ylabel("$\sigma$ (reach)", fontsize=font['default_size'])
        plt.xlabel(r"$\alpha$ (mode distance)", fontsize=font['default_size'])
        
        title=r"$(\alpha,\sigma)$ $x^{*}$ stability bifurication for "+str(self.num_agents)+" players"
        if len(title_ads)>0:
            for title_additon in title_ads:
                title=title+" "+title_additon
        plt.title(title, fontsize=font['title_size'])
        plt.xlim(alpha_st,alpha_end) 
        plt.close()
        
        file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'stability_bifurcation_plot_fast','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','first_order_bifurcation_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
        for file_name in file_names:
            fig.savefig(file_name,bbox_inches='tight')
        return fig
        
    def position_at_equalibirum_histogram(self,
                                         reach_parameter: float = .5,
                                         num_interations: int = 100,
                                         intitial_pos: Union[List[float], np.ndarray] = 0,
                                         current_alpha: float = .5,
                                         tolerance: Optional[float] = None,
                                         tolerated_agents: Optional[int] = None,
                                         title_ads: List[str] = [],
                                         name_ads: List[str] = [],
                                         save: bool = False,
                                         save_types: List[str] = ['.png', '.svg'],
                                         return_pos: bool = False,
                                         parallel: bool = True,
                                         max_workers: Optional[int] = None,
                                         batch_size: Optional[int] = None,
                                         paper_figure: dict = {'paper': False, 'section': 'A', 'figure_id': 'position_at_equalibirum_histogram'},
                                         font = {'default_size': 15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'}) -> Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, np.ndarray]]:
        """
        Plot the histogram of agents' positions at equilibrium for a given reach parameter via the function :func:`InflGame.adaptive.grad_func_env.gradient_ascent` .
        This only works for '1d' domains.
        
        .. figure:: examples/histogram.png
            :scale: 75 %

            This is the histogram of player postions at equalibirum for 5 player game.
             
           
        

        :param reach_parameter: Reach parameter value.
        :type reach_parameter: float
        :param num_interations: Number of gradient ascent steps.
        :type num_interations: int
        :param intitial_pos: Initial positions of agents.
        :type intitial_pos: Union[List[float], np.ndarray]
        :param current_alpha: Current alpha value.
        :type current_alpha: float
        :param tolerance: Tolerance for convergence.
        :type tolerance: Optional[float]
        :param tolerated_agents: Number of agents allowed to tolerate deviations.
        :type tolerated_agents: Optional[int]
        :param title_ads: Additional titles for the plot.
        :type title_ads: List[str]
        :param name_ads: Additional names for saved files.
        :type name_ads: List[str]
        :param save: Whether to save the plot.
        :type save: bool
        :param save_types: File types to save the plot.
        :type save_types: List[str]
        :param return_pos: Whether to return the final positions.
        :type return_pos: bool
        :param parallel: Whether to use parallel processing.
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers (defaults to CPU count).
        :type max_workers: Optional[int]
        :param batch_size: Batch size for processing (auto-calculated if None).
        :type batch_size: Optional[int]

        :return: The generated plot figure or final positions.
        :rtype: Union[matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, np.ndarray]]
        """
        og_iterations=self.time_steps
        og_pos=self.agents_pos.clone()
        self.field.time_steps=num_interations
        
        if not torch.is_tensor(intitial_pos):
            intitial_pos=torch.tensor(intitial_pos)
        self.agents_pos=intitial_pos.clone()
        self.field.agents_pos=intitial_pos.clone()
        
        if tolerated_agents==None:
            tolerated_agents=self.tolerated_agents
        if tolerance==None:
            tolerance=self.tolerance


        if self.domain_type=="1d":
            reach_parameters=general.agent_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_parameter,reach_end = reach_parameter,reach_num_points = 1)
            final_pos_vector=self.final_pos_over_reach(reach_parameters, tolerance=tolerance, tolerated_agents=tolerated_agents, parallel=True,time_steps=num_interations)
            fig=one_plots.final_position_histogram_1d(num_agents=self.num_agents,domain_bounds=self.domain_bounds,current_alpha=current_alpha,reach_parameter=reach_parameter,final_pos_vector=final_pos_vector,title_ads=title_ads,font=font)
        else:
            ValueError("Histogram is limited to 1d domains")
            
        self.field.time_steps=og_iterations
        self.field.agents_pos=og_pos.clone()
        self.agents_pos=og_pos.clone()

        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'positional_histogram','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','position_at_equalibirum_histogram')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        if return_pos==True:
            return fig,final_pos_vector
        else:
            return fig

    def bifurcation_plot_AD_MARL(self,
                                  parameters_AD:dict,
                                  parameters_MARL:dict,
                                  fontmain= {'default_size': 15, 'axis_size':15, 'cbar_size': 16, 'title_size': 18, 'legend_size': 12, 'font_family': 'sans-serif'},
                                  save: bool = False,
                                  name_ads: List[str] = [],
                                  save_types: List[str] = ['.png', '.svg'],
                                  paper_figure: dict= {'paper':False,'section':'A','figure_id':'pos_plot'},
                                  ) -> matplotlib.figure.Figure:



        import hickle as hkl  


        # AD bifurcation plot

        reach_start = parameters_AD.get('reach_start', 0.03)
        reach_end = parameters_AD.get('reach_end', 0.3)
        reach_num_points = parameters_AD.get('reach_num_points', 200)
        time_steps = parameters_AD.get('time_steps', 10000)
        tolerance = parameters_AD.get('tolerance', None)
        tolerated_agents = parameters_AD.get('tolerated_agents', None)
        plot_type = parameters_AD.get('plot_type', 'heat')
        refinements = parameters_AD.get('refinements', 10)
        title_ads = parameters_AD.get('title_ads', [])
        cmaps = parameters_AD.get('cmaps', {'heat': 'Blues', 'trajectory': '#851321', 'crit': 'Greys'})
        cbar_config = parameters_AD.get('cbar_config', {'center_labels': True, 'label_alignment': 'center', 'shrink': 1.0})
        parallel_configs = parameters_AD.get('parallel_configs', {})
        parallel = parallel_configs.get('parallel', True)
        max_workers = parallel_configs.get('max_workers', 6)
        batch_size = parallel_configs.get('batch_size', 4)
        font= parameters_AD.get('font', fontmain)
        
        if tolerated_agents==None:
            tolerated_agents=self.tolerated_agents
        if tolerance==None:
            tolerance=self.tolerance



        reach_parameters=general.agent_parameter_setup(num_agents=self.num_agents,infl_type=self.infl_type,setup_type="parameter_space",reach_start = reach_start,reach_end = reach_end,reach_num_points = reach_num_points)
        final_pos_matrix=self.final_pos_over_reach(reach_parameters, tolerance=tolerance, tolerated_agents=tolerated_agents, parallel=parallel, max_workers=max_workers, batch_size=batch_size, time_steps=time_steps)
        ax1=one_plots.equilibrium_bifurcation_plot_1d(num_agents=self.num_agents,
                                                      bin_points=self.bin_points,
                                                      resource_distribution=self.resource_distribution,
                                                      infl_type=self.infl_type,
                                                      infl_cshift=self.infl_cshift,
                                                      reach_parameters=reach_parameters,
                                                      final_pos_matrix=final_pos_matrix,
                                                      reach_start=reach_start,
                                                      reach_end=reach_end,
                                                      refinements=refinements,
                                                      plot_type=plot_type,
                                                      title_ads=title_ads,
                                                      short_title=True,
                                                      font=font,
                                                      cmaps=cmaps,
                                                      cbar_config=cbar_config,
                                                      axis_return=True,)

        # MARL bifurcation plot this is only done if the MARL data is available

        step = parameters_MARL.get('step_size', 0.1)
        resource = parameters_MARL.get('resource', "gauss_mix_2m")
        

        env_config_main = {"num_agents":self.num_agents,"parameters":[self.parameters[0].item()],"step_size":step}
        configs={"env_config_main":env_config_main}
        params=data_management.data_parameters(configs=configs,data_type="configs",resource_type=resource)
        config_name=data_management.data_final_name(data_parameters=params,name_ads=[])
        params=data_management.data_parameters(configs=configs,data_type="final_positions",resource_type=resource)
        data_name=data_management.data_final_name(data_parameters=params,name_ads=["test"])
        params=data_management.data_parameters(configs=configs,data_type="final_mad",resource_type=resource)
        data_name2=data_management.data_final_name(data_parameters=params,name_ads=["test"])
        mean=hkl.load(data_name[0])
        mean=torch.tensor(mean)
        mad=hkl.load(data_name2[0])
        mad=torch.tensor(mad)


        reach_start_marl = parameters_MARL.get('reach_start', 0.03)
        reach_end_marl = parameters_MARL.get('reach_end', 0.3)
        refinements_marl = parameters_MARL.get('refinements', 10)
        from InflGame.MARL.MARL_plots import bifurcation_over_parameters
        ax2=bifurcation_over_parameters(positions=mean,
                                    reach_parameters=parameters_MARL.get('reach_parameters', np.linspace(reach_start_marl, reach_end_marl, 96)),
                                    num_agents=self.num_agents,
                                    bin_points=self.bin_points,
                                    resource_distribution=self.resource_distribution,
                                    refinements=refinements_marl,
                                    plot_type=parameters_MARL.get('plot_type', 'gaussian'),
                                    infl_cshift=self.infl_cshift,
                                    infl_type=parameters_MARL.get('infl_type', 'gaussian'),
                                    name_ads=[],
                                    title_ads=parameters_MARL.get('title_ads', []),
                                    short_title=True,
                                    font=parameters_MARL.get('font', fontmain),
                                    cbar_config=parameters_MARL.get('cbar_config', {'center_labels': True, 'label_alignment': 'center', 'shrink': 1.0}),
                                    save=False,
                                    axis_return=True,
                                    )
        # Combine plots
        fig=plot_utils.side_by_side_plots(ax1,ax2,title_ads=[f'{self.num_agents} Agents']+title_ads,font=fontmain,title_main='Adaptive vs MARL Dynamcis Bifurcations',cbar_params={'common_cbar':True,'cbar_title':'Number of Agents'},axis_params={'common_axis':True,'axis_ylabel':'Agent Position','axis_xlabel':r'$\sigma$ (reach)'})

        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'bif_adp_marl','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','pos_plot')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
        return fig




    # 3d visualization for traces 
    def _process_point_worker(self, args):
        """Worker function for processing gradient ascent from a starting point."""
        point, color, shell_obj, time_steps = args
        try:
            # Set the starting position
            shell_obj.agents_pos = point.clone()
            shell_obj.field.agents_pos = point.clone()
            
            # Debug print to verify position was set
            print(f"Processing point {point} (color: {color})")
            
            # Run gradient ascent with the specified max iterations
            shell_obj.field.gradient_ascent()
            
            # Check if pos_matrix exists and has data
            if not hasattr(shell_obj.field, 'pos_matrix') or shell_obj.field.pos_matrix is None:
                print(f"Error: pos_matrix is None for point {point}")
                return None
                
            if len(shell_obj.field.pos_matrix) == 0:
                print(f"Error: pos_matrix is empty for point {point}")
                return None
            
            # Get the path data
            pos_matrix = shell_obj.field.pos_matrix.numpy()
            converged = len(shell_obj.field.pos_matrix) < time_steps
            
            print(f"Success: Generated path with {len(pos_matrix)} points for {point}")
            
            return {
                'path': pos_matrix,
                'color': color,
                'converged': converged,
                'start': pos_matrix[0],
                'end': pos_matrix[-1]
            }
        except Exception as e:
            print(f"Error processing point {point}: {str(e)}")
            traceback.print_exc()  # Print the full stack trace
            return None

    def simple_diagonal_test_point(self, point=None):
        """Test a single point to verify gradient ascent works."""
        if point is None:
            point = torch.tensor([0.3, 0.5, 0.7])
        
        print(f"\nDIAGNOSTIC TEST with point: {point}")
        
        # Set position
        original_pos = self.agents_pos.clone()
        self.agents_pos = point.clone()
        
        print(f"Starting gradient ascent from {self.agents_pos}")
        
        # Run gradient ascent directly
        self.field.gradient_ascent()
        
        # Check result
        if hasattr(self.field, 'pos_matrix') and self.field.pos_matrix is not None and len(self.field.pos_matrix) > 0:
            print(f"Success! Path generated with {len(self.field.pos_matrix)} points")
            print(f"Start: {self.field.pos_matrix[0]}")
            print(f"End: {self.field.pos_matrix[-1]}")
            result = True
        else:
            print("Failed to generate path!")
            result = False
        
        # Restore original position
        self.agents_pos = original_pos
        return result

    def plot_3d_fixed_diagonal_view(self, 
                             resolution=5,
                             time_steps=10000,
                             show_planes=True,
                             figsize=(12, 10),
                             seed=42,
                             include_boundaries=True,
                             elev=10,  # Lower elevation to look down diagonal
                             azim=45,
                             num_workers=4,
                             use_parallel=True):
        """
        Creates a 3D plot with fixed view looking down the x=y=z diagonal line,
        with optional parallel processing for improved performance.
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Sample fixed points for better reproducibility
        def get_fixed_test_points():
            """Generate fixed test points that cover all regions of interest."""
            fixed_points = [
                # Main regions - one point from each
                torch.tensor([0.3, 0.5, 0.7]),  # x < .5 < z (red)
                torch.tensor([0.7, 0.3, 0.5]),  # y < .5 < x (green)
                torch.tensor([0.5, 0.7, 0.3]),  # z < .5 < y (blue)
                torch.tensor([0.5, 0.3, 0.7]),  # y < .5 < z (orange)
                torch.tensor([0.3, 0.7, 0.5]),  # x < .5 < y (purple)
                torch.tensor([0.7, 0.5, 0.3]),  # z < .5 < x (cyan)

                # Additional points for thorough coverage
                torch.tensor([0.2, 0.25, 0.6]),  # x < y < z (red)
                torch.tensor([0.25, 0.6, 0.2]),  # y < z < x (green)
                torch.tensor([0.25, 0.2, 0.6]),  # z < x < y (blue)
                torch.tensor([0.6, 0.25, 0.2]),  # y < x < z (orange)
                torch.tensor([0.2, 0.6, 0.25]),  # x < y = z (purple)
                torch.tensor([0.6, 0.2, 0.25]),  # z < y < x (cyan)

                # Boundary points
                torch.tensor([0.3, 0.3, 0.7]),  # x = y < z (grey)
                torch.tensor([0.7, 0.7, 0.3]),  # x = y > z (grey)
                torch.tensor([0.7, 0.3, 0.3]),  # x > y = z (black)
                torch.tensor([0.3, 0.7, 0.7]),  # x < y = z (black)
                torch.tensor([0.3, 0.7, 0.3]),  # x = z < y (brown)
                torch.tensor([0.7, 0.3, 0.7]),  # y < x = z (cyan)

                # Diagonal points
                torch.tensor([0.3, 0.3, 0.3]),  # x = y = z (diagonal)
                torch.tensor([0.7, 0.7, 0.7])   # x = y = z (diagonal)
            ]
            return fixed_points
        
        # Pre-calculate region colors for points
        def get_region_colors(points):
            colors = []
            for point in points:
                x, y, z = point
                
                # Check for equality (boundary regions)
                if torch.isclose(x, y, rtol=1e-5) and torch.isclose(y, z, rtol=1e-5):
                    colors.append('black')  # Diagonal x=y=z
                elif torch.isclose(x, y, rtol=1e-5):
                    colors.append('grey')
                elif torch.isclose(y, z, rtol=1e-5):
                    colors.append('darkblue')
                elif torch.isclose(x, z, rtol=1e-5):
                    colors.append('brown')
                # Determine the ordering of the coordinates
                elif x < y < z:
                    colors.append('red')
                elif y < z < x:
                    colors.append('green')
                elif z < x < y:
                    colors.append('blue')
                elif y < x < z:
                    colors.append('orange')
                elif x < z < y:
                    colors.append('purple')
                elif z < y < x:
                    colors.append('cyan')
                else:
                    colors.append('gray')  # Default fallback
            return colors
        
        # Get fixed test points and their colors
        point_list = get_fixed_test_points()
        colors = get_region_colors(point_list)
        
        print(f"Processing {len(point_list)} test points...")
        
        # Start timing
        start_time = time.time()
        
        # Store original state
        original_pos = self.agents_pos.clone()
        results = []
        
        if use_parallel and len(point_list) > 1:
            # Parallel processing
            if num_workers is None:
                num_workers = min(mp.cpu_count(), len(point_list))
            
            print(f"Using parallel processing with {num_workers} workers...")
            
            # Create deep copies of the shell object for each worker
            def create_worker_args():
                worker_args = []
                for point, color in zip(point_list, colors):
                    # Create a serializable shell copy for this worker
                    shell_copy = copy.deepcopy(self)
                    worker_args.append((point, color, shell_copy, time_steps))
                return worker_args
            
            # Prepare worker arguments
            worker_args = create_worker_args()
            
            try:
                # Use ProcessPoolExecutor for better resource management
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    future_to_args = {
                        executor.submit(self._process_point_worker, args): i 
                        for i, args in enumerate(worker_args)
                    }
                    
                    # Collect results as they complete
                    completed_results = {}
                    completed_count = 0
                    
                    for future in as_completed(future_to_args):
                        try:
                            result = future.result()
                            if result is not None:
                                arg_index = future_to_args[future]
                                completed_results[arg_index] = result
                            completed_count += 1
                            
                            # Progress update
                            if completed_count % max(1, len(point_list) // 4) == 0:
                                print(f"Parallel progress: {completed_count}/{len(point_list)} points")
                                
                        except Exception as e:
                            arg_index = future_to_args[future]
                            print(f"Worker {arg_index} failed: {str(e)}")
                    
                    # Convert results back to ordered list
                    for i in range(len(point_list)):
                        if i in completed_results:
                            results.append(completed_results[i])
                            
            except Exception as e:
                print(f"Parallel processing failed: {str(e)}")
                print("Falling back to sequential processing...")
                use_parallel = False
        
        if not use_parallel or len(results) == 0:
            # Sequential processing (fallback or by choice)
            print("Using sequential processing...")
            results = []
            
            for i, (point, color) in enumerate(zip(point_list, colors)):
                print(f"\nProcessing point {i+1}/{len(point_list)}: {point}")
                try:
                    # Set position for this run
                    self.agents_pos = point.clone()
                    self.field.agents_pos = point.clone()
                    
                    # Run gradient ascent
                    self.field.gradient_ascent()
                    
                    # Check if pos_matrix exists and has data
                    if not hasattr(self.field, 'pos_matrix') or self.field.pos_matrix is None:
                        print(f"Error: pos_matrix is None for point {point}")
                        continue
                        
                    if len(self.field.pos_matrix) == 0:
                        print(f"Error: pos_matrix is empty for point {point}")
                        continue
                    
                    # Get the path data
                    pos_matrix = self.field.pos_matrix.numpy()
                    converged = len(self.field.pos_matrix) < time_steps
                    
                    print(f"Success: Generated path with {len(pos_matrix)} points for {point}")
                    
                    results.append({
                        'path': pos_matrix,
                        'color': color,
                        'converged': converged,
                        'start': pos_matrix[0],
                        'end': pos_matrix[-1]
                    })
                except Exception as e:
                    print(f"Error processing point {point}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Always restore the original position after each run
                    self.agents_pos = original_pos.clone()
                    self.field.agents_pos = original_pos.clone()
        
        # End timing and report
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Generated {len(results)} valid paths out of {len(point_list)} points")
        
        # Create matplotlib figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Add the equality planes if requested
        if show_planes:
            # Create a grid for the planes (reduced resolution for speed)
            grid_points = np.linspace(0, 1, 8)
            X, Y = np.meshgrid(grid_points, grid_points)
            
            # x=y plane (grey)
            ax.plot_surface(X, X, Y, color='grey', alpha=0.15)
            
            # y=z plane (black)
            ax.plot_surface(X, Y, Y, color='black', alpha=0.15)
            
            # x=z plane (brown)
            ax.plot_surface(X, Y, X, color='brown', alpha=0.15)
        
        # Dictionary to track paths by color for legend
        paths_by_color = {}
        
        # Plot all paths
        if not results:
            print("WARNING: No valid paths to plot!")
        else:
            print(f"Plotting {len(results)} paths...")
            
            for result in results:
                # Add the path
                line = ax.plot(result['path'][:, 0], result['path'][:, 1], result['path'][:, 2], 
                        color=result['color'], linewidth=2.5, alpha=0.8)[0]
                
                # Track this path for legend
                if result['color'] not in paths_by_color:
                    paths_by_color[result['color']] = line
                
                # Add starting point marker (red)
                ax.scatter(result['start'][0], result['start'][1], result['start'][2],
                        color='red', s=60, alpha=1.0, zorder=10)
                
                # If converged, add endpoint marker (green)
                if result['converged']:
                    ax.scatter(result['end'][0], result['end'][1], result['end'][2],
                            color='green', s=100, alpha=1.0, zorder=10)
        
        # Add diagonal line x=y=z (make it thicker and more visible)
        diag = np.linspace(0, 1, 100)
        ax.plot(diag, diag, diag, 'k--', linewidth=4, alpha=1.0, label='x=y=z')
        
        # Set labels and title
        ax.set_xlabel('Agent 1 Position', fontweight='bold', fontsize=12)
        ax.set_ylabel('Agent 2 Position', fontweight='bold', fontsize=12)
        ax.set_zlabel('Agent 3 Position', fontweight='bold', fontsize=12)
        ax.set_title('Gradient Ascent Paths (View Along Diagonal)', fontweight='bold', fontsize=14)
        
        # Set view to look directly down the x=y=z line
        ax.view_init(elev=elev, azim=azim)
        
        # Region descriptions for legend
        region_descriptions = {
            'red': 'Agent 1 < Agent 2 < Agent 3',
            'green': 'Agent 2 < Agent 3 < Agent 1',
            'blue': 'Agent 3 < Agent 1 < Agent 2',
            'orange': 'Agent 2 < Agent 1 < Agent 3',
            'purple': 'Agent 1 < Agent 3 < Agent 2',
            'cyan': 'Agent 3 < Agent 2 < Agent 1',
            'grey': 'Agent 1 = Agent 2',
            'darkblue': 'Agent 2 = Agent 3',
            'brown': 'Agent 1 = Agent 3',
            'black': 'x = y = z (Diagonal)'
        }
        
        # Build legend - move it outside the plot area
        legend_elements = []
        for color, line in paths_by_color.items():
            legend_elements.append(line)
        
        # Add legend with better positioning
        if legend_elements:
            ax.legend(legend_elements, 
                    [region_descriptions.get(line.get_color(), line.get_color()) for line in legend_elements],
                    loc='center left', 
                    bbox_to_anchor=(1.05, 0.5),  # Position legend outside plot
                    fontsize=10)
        
        # Set limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Add text box with current parameter value
        try:
            # Get the current sigma parameter value
            if hasattr(self.field, 'parameters') and self.field.parameters is not None:
                if torch.is_tensor(self.field.parameters):
                    if len(self.field.parameters.shape) > 0:
                        sigma_value = self.field.parameters[0].item()
                    else:
                        sigma_value = self.parameters.item()
                else:
                    sigma_value = self.field.parameters[0] if isinstance(self.field.parameters, (list, np.ndarray)) else self.field.parameters
            elif hasattr(self.field, 'parameters') and self.field.parameters is not None:
                if torch.is_tensor(self.field.parameters):
                    if len(self.field.parameters.shape) > 0:
                        sigma_value = self.field.parameters[0].item()
                    else:
                        sigma_value = self.field.parameters.item()
                else:
                    sigma_value = self.field.parameters[0] if isinstance(self.field.parameters, (list, np.ndarray)) else self.field.parameters
            else:
                sigma_value = "N/A"
            
            # Format the parameter value for display
            if isinstance(sigma_value, (int, float)):
                param_text = f'σ = {sigma_value:.4f}'
            else:
                param_text = f'σ = {sigma_value}'
            
            # Add text box in the upper left corner
            ax.text2D(0.02, 0.98, param_text, 
                     transform=ax.transAxes, 
                     fontsize=14, 
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='lightblue', 
                              alpha=0.8,
                              edgecolor='black'))
        except Exception as e:
            print(f"Warning: Could not display parameter value: {e}")
        
        # Add padding to right side for legend
        plt.subplots_adjust(right=0.8)
        
        return fig


    # 3d visualization for traces gif
    def plot_3d_fixed_diagonal_view_gif(self,
                                        max_frames: int,
                                        output_filename: str = '3d_diagonal_traces.gif',
                                        resolution: int = 5,
                                        time_steps: int = 10000,
                                        show_planes: bool = True,
                                        figsize: tuple = (12, 10),
                                        seed: int = 42,
                                        include_boundaries: bool = True,
                                        elev: int = 10,
                                        azim: int = 45,
                                        num_workers: int = 4,
                                        use_parallel: bool = True,
                                        reach_start: float = 0.08,
                                        reach_end: float = 0.25,
                                        optimize_memory: bool = True,
                                        dpi: int = 100,
                                        fps: int = 10,
                                        quality: int = 8,
                                        verbose: bool = False) -> matplotlib.figure.Figure:
        r"""
        Create a 3D diagonal view GIF showing gradient ascent traces over varying reach parameters.
        Similar to dist_pos_gif but for 3D visualization. Works by varying the reach parameters
        and creating frames of :func:`plot_3d_fixed_diagonal_view` .
        
        **Optimized Performance Features:**
        - Direct memory writing without intermediate files
        - Matplotlib figure recycling for memory efficiency
        - Optimized frame sampling
        - Configurable quality vs speed trade-offs
        
        :param max_frames: Maximum number of frames for the gif.
        :type max_frames: int
        :param output_filename: Name of the output GIF file.
        :type output_filename: str
        :param resolution: Resolution for point sampling in 3D plot.
        :type resolution: int
        :param time_steps: Number of gradient ascent steps.
        :type time_steps: int
        :param show_planes: Whether to show equality planes in 3D plot.
        :type show_planes: bool
        :param figsize: Figure size for each frame.
        :type figsize: tuple
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param include_boundaries: Whether to include boundary points in 3D plot.
        :type include_boundaries: bool
        :param elev: Elevation angle for 3D view.
        :type elev: int
        :param azim: Azimuth angle for 3D view.
        :type azim: int
        :param num_workers: Number of parallel workers for 3D processing.
        :type num_workers: int
        :param use_parallel: Whether to use parallel processing for 3D plots.
        :type use_parallel: bool
        :param reach_start: Starting reach parameter value.
        :type reach_start: float
        :param reach_end: Ending reach parameter value.
        :type reach_end: float
        :param optimize_memory: Whether to use memory optimization techniques.
        :type optimize_memory: bool
        :param dpi: DPI for the frames (lower = faster, higher = better quality).
        :type dpi: int
        :param fps: Frames per second for the GIF.
        :type fps: int
        :param quality: GIF compression quality (1-10, lower = smaller file).
        :type quality: int
        :param verbose: Whether to print progress information.
        :type verbose: bool

        :return: The final frame figure for display.
        :rtype: matplotlib.figure.Figure
        """
        import io
        from PIL import Image
        import os
        import tempfile
        from contextlib import redirect_stdout, redirect_stderr
        
        # Generate the parameters to be used
        reach_parameters = general.agent_parameter_setup(
            num_agents=self.num_agents,
            infl_type=self.infl_type,
            setup_type="parameter_space",
            reach_start=reach_start,
            reach_end=reach_end,
            reach_num_points=max_frames
        )
        
        # Handle single frame case
        if max_frames == 1:
            og_parameters = self.field.parameters.clone()
            self.field.parameters = reach_parameters[0]
            final_fig = self.plot_3d_fixed_diagonal_view(
                resolution=resolution, time_steps=time_steps, show_planes=show_planes,
                figsize=figsize, seed=seed, include_boundaries=include_boundaries,
                elev=elev, azim=azim, num_workers=num_workers, use_parallel=use_parallel
            )
            self.field.parameters = og_parameters
            return final_fig
        
        # Store original parameters
        og_parameters = self.field.parameters.clone()
        
        # Calculate optimized frame indices
        num_params = len(reach_parameters)
        if max_frames >= num_params:
            param_indices = list(range(num_params))
        else:
            # Smart sampling: ensure we get key frames including start and end
            step = max(1, int(np.floor(num_params / max_frames)))
            param_indices = list(range(0, num_params, step))
            if param_indices[-1] != num_params - 1:
                param_indices.append(num_params - 1)  # Ensure we include the final frame
        
        # Prepare for optimized GIF creation
        frames = []
        
        if optimize_memory:
            # Memory-optimized approach: direct buffer writing
            if verbose:
                print(f"Creating {len(param_indices)} frames with memory optimization...")
            
            # Create a single figure and reuse it
            fig = None
            
            try:
                with imageio.get_writer(output_filename, mode='I', fps=fps, 
                                      quantizer='nq', palettesize=256, loop=0) as writer:
                    
                    for i, param_idx in enumerate(param_indices):
                        # Update parameters
                        self.field.parameters = reach_parameters[param_idx]
                        
                        # Create or update plot
                        if fig is not None:
                            plt.close(fig)  # Close previous figure to avoid accumulation
                        
                        # Suppress matplotlib output when not in verbose mode
                        if verbose:
                            fig = self.plot_3d_fixed_diagonal_view(
                                resolution=resolution, time_steps=time_steps, 
                                show_planes=show_planes, figsize=figsize, seed=seed,
                                include_boundaries=include_boundaries, elev=elev, azim=azim,
                                num_workers=num_workers, use_parallel=use_parallel
                            )
                        else:
                            # Redirect stdout/stderr to suppress matplotlib figure output
                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                fig = self.plot_3d_fixed_diagonal_view(
                                    resolution=resolution, time_steps=time_steps,
                                    show_planes=show_planes, figsize=figsize, seed=seed,
                                    include_boundaries=include_boundaries, elev=elev, azim=azim,
                                    num_workers=num_workers, use_parallel=use_parallel
                                )
                        
                        fig.set_dpi(dpi)
                        
                        # Convert to image buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                        buf.seek(0)
                        
                        # Load and append frame
                        image = Image.open(buf)
                        image_array = np.array(image)
                        writer.append_data(image_array)
                        
                        buf.close()
                        
                        # Progress indicator
                        if i % max(1, len(param_indices) // 10) == 0 and verbose:
                            reach_val = reach_parameters[param_idx][0].item() if torch.is_tensor(reach_parameters[param_idx]) else reach_parameters[param_idx][0]
                            print(f"Progress: {i+1}/{len(param_indices)} frames ({100*(i+1)/len(param_indices):.1f}%) - reach={reach_val:.4f}")
                    
                    if fig is not None:
                        plt.close(fig)
                        
            except Exception as e:
                if verbose:
                    print(f"Memory optimization failed: {e}")
                    print("Falling back to file-based approach...")
                optimize_memory = False
        
        if not optimize_memory:
            # Traditional file-based approach with optimizations
            if verbose:
                print(f"Creating {len(param_indices)} frames with file optimization...")
            
            # Use temporary directory for better performance
            with tempfile.TemporaryDirectory() as temp_dir:
                filenames = []
                
                for i, param_idx in enumerate(param_indices):
                    # Update parameters
                    self.field.parameters = reach_parameters[param_idx]
                    
                    # Create plot
                    fig = self.plot_3d_fixed_diagonal_view(
                        resolution=resolution, time_steps=time_steps,
                        show_planes=show_planes, figsize=figsize, seed=seed,
                        include_boundaries=include_boundaries, elev=elev, azim=azim,
                        num_workers=num_workers, use_parallel=use_parallel
                    )
                    
                    # Save to temporary file
                    filename = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    filenames.append(filename)
                    fig.savefig(filename, dpi=dpi, bbox_inches="tight", 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)  # Free memory immediately
                    
                    # Progress indicator
                    if i % max(1, len(param_indices) // 10) == 0 and verbose:
                        reach_val = reach_parameters[param_idx][0].item() if torch.is_tensor(reach_parameters[param_idx]) else reach_parameters[param_idx][0]
                        print(f"Progress: {i+1}/{len(param_indices)} frames ({100*(i+1)/len(param_indices):.1f}%) - reach={reach_val:.4f}")
                
                # Create GIF from files
                if verbose:
                    print("Writing GIF...")
                with imageio.get_writer(output_filename, mode='I', fps=fps,
                                      quantizer='nq', palettesize=256, loop=0) as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
        
        # Restore original parameters
        self.field.parameters = og_parameters
        
        if verbose:
            print(f"3D diagonal traces GIF created successfully: {output_filename}")
        
        # Return the final frame for display
        self.field.parameters = reach_parameters[-1]
        final_fig = self.plot_3d_fixed_diagonal_view(
            resolution=resolution, time_steps=time_steps, show_planes=show_planes,
            figsize=figsize, seed=seed, include_boundaries=include_boundaries,
            elev=elev, azim=azim, num_workers=num_workers, use_parallel=use_parallel
        )
        self.field.parameters = og_parameters
        
        return final_fig
        

    #Utils integrated into the class for simplicity
    def calc_infl_dist(self,
                       pos: torch.Tensor,
                       parameter_instance: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the influence distribution for agents via thier influence kernels by interacting with the class :func:`InflGame.adaptive.grad_func_env` .

        :param pos: Positions of agents.
        :type pos: torch.Tensor
        :param parameter_instance: Parameters for the influence function.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]

        :return: The influence distribution.
        :rtype: torch.Tensor
        """
        infl_dist=0
        if self.domain_type=='simplex':
            og_alpha=self.field.alpha_matrix
        og_pos=self.agents_pos.clone()
        og_binpoints=self.bin_points.clone()
        self.field.agents_pos=pos
        
        if self.domain_type=='simplex':
            if self.infl_type=='dirichlet':
                alpha_matrix=diric.param(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.field.agents_pos,fixed_pa=self.fp)
            else: 
                alpha_matrix=0
            self.field.alpha_matrix=alpha_matrix
            self.field.bin_points=np.array([simplex_utils.xy2ba(x,y,corners=self.corners)  for x,y in zip(self.trimesh.x, self.trimesh.y)])
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        elif self.domain_type=='2d':
            self.field.bin_points=self.rect_positions
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        else:
            infl_dist=self.field.influence_matrix(parameter_instance=parameter_instance)
        if self.domain_type=='simplex':
            self.field.alpha_matrix=og_alpha
        self.field.agents_pos=og_pos.clone()
        self.field.bin_points=og_binpoints.clone()
        self.infl_dist=infl_dist
    
    def calc_direction_and_strength(self,
                                    parameter_instance: Union[List[float], np.ndarray, torch.Tensor] = 0,
                                    agent_id: int = 0,
                                    ids: List[int] = [0, 1],
                                    pos: Optional[torch.Tensor] = None,
                                    alt_form: bool = False) -> torch.Tensor:
        """
        Calculate the direction and strength of gradients for agents via interacting with :func:`InflGame.adaptive.grad_func_env` .

        :param parameter_instance: Parameters for the influence function.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param agent_id: ID of the agent.
        :type agent_id: int
        :param ids: IDs of agents of interest.
        :type ids: List[int]
        :param pos: Positions of agents.
        :type pos: Optional[torch.Tensor]

        :return: The calculated gradients.
        :rtype: torch.Tensor
        """
        
        if self.domain_type=='simplex':
            direction= np.array([self.field.gradient_function(np.array([simplex_utils.xy2ba(x,y,corners=self.corners)+.001]*self.num_agents),parameter_instance,agent_id).nan_to_num().detach() for x,y in zip(self.trimesh.x, self.trimesh.y)])
            self.direction_norm=np.array([simplex_utils.ba2xy(torch.tensor(v),corners=self.corners).detach()/np.linalg.norm(v) if np.linalg.norm(v)>0 else np.array([0,0]) for v in direction[:,agent_id]])
            
            #print(direction_ba_norm)
            self.pvals =[np.linalg.norm(v)+.0001 for v in direction[:,agent_id]]
            self.direction=np.array([simplex_utils.ba2xy(torch.tensor(v),corners=self.corners).detach() for v in direction[:,agent_id]])

        elif self.domain_type=='2d':
            direction= np.array([self.field.gradient_function(np.array(list(value)*self.num_agents),parameter_instance,agent_id).nan_to_num().detach() for value in self.rect_positions])
            self.direction_norm=np.array([v/np.linalg.norm(v) if np.linalg.norm(v)>0 else np.array([0,0]) for v in direction[:,agent_id]])
            #print(direction_ba_norm)
            self.pvals =[np.linalg.norm(v) for v in direction[:,agent_id]]
            self.direction=np.array([v for v in direction[:,agent_id]])

        elif self.domain_type=="1d":
            if self.num_agents>2:
                two_a=False
            else:
                two_a=True
            if alt_form==True:
                self.field.alt_form=True
                grads=one_utils.direction_strength_1d(self.field.gradient_function,parameter_instance=parameter_instance,two_a=two_a,ids=ids,pos=pos)
            else:
                grads=one_utils.direction_strength_1d(self.field.gradient_function,parameter_instance=parameter_instance,two_a=two_a,ids=ids,pos=pos)
            two_a=True
            return grads
    
    def _compute_single_parameter(self, parameter_data: Dict) -> Tuple[int, torch.Tensor]:
        """
        Helper function to compute final position for a single parameter.
        Designed to be used with multiprocessing.
        
        :param parameter_data: Dictionary containing parameter data and configuration
        :type parameter_data: Dict
        
        :return: Tuple of parameter_id and final position row
        :rtype: Tuple[int, torch.Tensor]
        """
        try:
            parameter_id = parameter_data['parameter_id']
            reach_param = parameter_data['reach_param']
            og_pos = parameter_data['og_pos']
            tolerance = parameter_data['tolerance']
            tolerated_agents = parameter_data['tolerated_agents']
            domain_type = parameter_data['domain_type']
            total_params = parameter_data['total_params']
            time_steps = parameter_data['time_steps']
            
            
            # Create a temporary field environment for this computation
            if domain_type == 'simplex':
                temp_field = grad_func_env.AdaptiveEnv(
                    num_agents=self.num_agents,
                    agents_pos=og_pos.clone(),
                    parameters=self.parameters,
                    resource_distribution=self.resource_distribution,
                    bin_points=self.bin_points,
                    infl_configs=self.infl_configs,
                    learning_rate_type=self.learning_rate_type,
                    learning_rate=self.learning_rate,
                    time_steps=time_steps,
                    fp=self.fp,
                    infl_cshift=self.infl_cshift,
                    cshift=self.cshift,
                    infl_fshift=self.infl_fshift,
                    Q=self.Q,
                    domain_type=domain_type,
                    tolerance=tolerance,
                    tolerated_agents=tolerated_agents,
                    ignore_zero_infl=self.ignore_zero_infl
                )
            else:
                temp_field = grad_func_env.AdaptiveEnv(
                    num_agents=self.num_agents,
                    agents_pos=og_pos.clone(),
                    parameters=self.parameters,
                    resource_distribution=self.resource_distribution,
                    bin_points=self.bin_points,
                    infl_configs=self.infl_configs,
                    learning_rate_type=self.learning_rate_type,
                    learning_rate=self.learning_rate,
                    time_steps=time_steps,
                    fp=self.fp,
                    infl_cshift=self.infl_cshift,
                    cshift=self.cshift,
                    infl_fshift=self.infl_fshift,
                    Q=self.Q,
                    domain_type=domain_type,
                    domain_bounds=self.domain_bounds,
                    tolerance=tolerance,
                    tolerated_agents=tolerated_agents,
                    ignore_zero_infl=self.ignore_zero_infl
                )
            
            # Set parameters based on domain type
            if domain_type in ['1d']:
                temp_field.learning_rate = [10**(-1*(max(3,5*(total_params-parameter_id)/total_params))), 1/10000, 500]
                temp_field.parameters = np.array(reach_param)
            elif domain_type in ['2d', 'simplex']:
                temp_field.parameters = torch.tensor(reach_param).clone()
            
            # Run gradient ascent
            temp_field.gradient_ascent(show_out=False)
            
           
            final_pos_row = temp_field.pos_matrix[-1].clone()
                
            return parameter_id, final_pos_row
            
        except Exception as e:
            logging.error(f"Error computing parameter {parameter_id}: {str(e)}")
            raise RuntimeError(f"Failed to compute parameter {parameter_id}: {str(e)}")

    def final_pos_over_reach(self, 
                           reach_parameters: Union[List[float], np.ndarray], 
                           tolerance: float, 
                           tolerated_agents: int,
                           parallel: bool = True,
                           max_workers: Optional[int] = None,
                           batch_size: Optional[int] = None,
                           time_steps: Optional[int] = None) -> torch.Tensor:
        """
        Calculate the final positions of agents over a range of reach parameters via repeated initiations of 
        :func:`InflGame.adaptive.grad_func_env.gradient_ascent` over a group of parameters.
        
        This method has been optimized with:
        - Vectorized operations where possible
        - Parallel processing support
        - Comprehensive error handling
        - Input validation
        - Progress logging

        :param reach_parameters: Reach parameters to iterate over
        :type reach_parameters: Union[List[float], np.ndarray]
        :param tolerance: Tolerance for convergence
        :type tolerance: float
        :param tolerated_agents: Number of agents allowed to tolerate deviations
        :type tolerated_agents: int
        :param parallel: Whether to use parallel processing
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers (defaults to CPU count)
        :type max_workers: Optional[int]
        :param batch_size: Batch size for processing (auto-calculated if None)
        :type batch_size: Optional[int]

        :return: The final positions of agents for each parameter
        :rtype: torch.Tensor
        
        :raises ValueError: If input parameters are invalid
        :raises RuntimeError: If computation fails
        """
        # Input validation
        if not isinstance(reach_parameters, (list, np.ndarray, torch.Tensor)):
            raise ValueError(f"reach_parameters must be list, numpy array, or torch tensor, got {type(reach_parameters)}")
        
        if isinstance(reach_parameters, list):
            reach_parameters = np.array(reach_parameters)
        elif isinstance(reach_parameters, torch.Tensor):
            reach_parameters = reach_parameters.numpy()
            
        if len(reach_parameters) == 0:
            raise ValueError("reach_parameters cannot be empty")
            
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
            
        if tolerated_agents < 0:
            raise ValueError(f"tolerated_agents must be non-negative, got {tolerated_agents}")
        
        # Validate domain type
        if self.domain_type not in ['1d', '2d', 'simplex']:
            raise ValueError(f"Unsupported domain_type: {self.domain_type}")
            
        logging.info(f"Computing final positions for {len(reach_parameters)} parameters using domain '{self.domain_type}'")
        if self.domain_type == 'simplex':
            self.field.domain_bounds=[0,1]
            self.domain_bounds=[0,1]
        # Store original state
        try:
            
            og_parameters = self.parameters.clone()   
            og_pos = self.agents_pos.clone()
            og_lr = self.learning_rate.clone()
            og_tolerance = self.tolerance
            og_tolerated_agents = self.tolerated_agents
            
            # Update field tolerance settings
            self.field.tolerance = tolerance
            self.field.tolerated_agents = tolerated_agents
            
            # Determine optimal batch size and workers
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(reach_parameters))
            
            if batch_size is None:
                batch_size = max(1, len(reach_parameters) // max_workers)
            
            # Prepare parameter data for parallel processing
            parameter_data_list = []
            for parameter_id, reach_param in enumerate(reach_parameters):
                parameter_data = {
                    'parameter_id': parameter_id,
                    'reach_param': reach_param,
                    'og_pos': og_pos,
                    'tolerance': tolerance,
                    'tolerated_agents': tolerated_agents,
                    'domain_type': self.domain_type,
                    'total_params': len(reach_parameters),
                    'time_steps': time_steps
                }
                parameter_data_list.append(parameter_data)
            
            # Initialize result storage
            final_pos_matrix = 0
            
            if parallel and len(reach_parameters) > 1:
                # Parallel processing
                logging.info(f"Using parallel processing with {max_workers} workers")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_param = {
                            executor.submit(self._compute_single_parameter, param_data): param_data['parameter_id']
                            for param_data in parameter_data_list
                        }
                        
                        # Collect results as they complete
                        results = {}
                        completed_count = 0
                        
                        for future in as_completed(future_to_param):
                            try:
                                parameter_id, final_pos_row = future.result()
                                results[parameter_id] = final_pos_row
                                completed_count += 1
                                
                                if completed_count % max(1, len(reach_parameters) // 10) == 0:
                                    logging.info(f"Completed {completed_count}/{len(reach_parameters)} parameters")
                                    
                            except Exception as e:
                                param_id = future_to_param[future]
                                logging.error(f"Parameter {param_id} failed: {str(e)}")
                                raise RuntimeError(f"Failed to compute parameter {param_id}: {str(e)}")
                        
                        # Build final matrix from results - vectorized approach
                        # Ensure results are stored in original parameter order
                        num_params = len(reach_parameters)
                        if len(results) != num_params:
                            raise RuntimeError(f"Expected {num_params} results, got {len(results)}")
                        
                        # Create a list to store results in original order
                        ordered_results = [None] * num_params
                        for param_id, result in results.items():
                            if param_id >= num_params:
                                raise RuntimeError(f"Invalid parameter ID {param_id}, expected 0-{num_params-1}")
                            ordered_results[param_id] = result
                        
                        # Check that we have all results
                        if None in ordered_results:
                            missing_ids = [i for i, res in enumerate(ordered_results) if res is None]
                            raise RuntimeError(f"Missing results for parameter IDs: {missing_ids}")
                        
                        # Vectorized matrix construction
                        if len(ordered_results) == 1:
                            final_pos_matrix = ordered_results[0]
                        else:
                            final_pos_matrix = torch.stack(ordered_results, dim=0)
                        
                        logging.info(f"Successfully built final position matrix with shape: {final_pos_matrix.shape}")
                            
                except Exception as e:
                    logging.error(f"Parallel processing failed: {str(e)}")
                    logging.info("Falling back to sequential processing")
                    parallel = False
            
            if not parallel:
                # Sequential processing (fallback or by choice)
                logging.info("Using sequential processing")
                
                # Pre-allocate results list for vectorized construction
                final_pos_results = []
                
                for parameter_id, reach_param in enumerate(reach_parameters):
                    try:
                        # Reset field state
                        self.field.pos_matrix = 0
                        self.field.agents_pos = og_pos.clone()
                        self.agents_pos = og_pos.clone()
                        
                        # Set parameters based on domain type
                        if self.domain_type in ['1d']:
                            self.field.learning_rate = [
                                10**(-1*(max(3, 5*(parameter_id+1)/len(reach_parameters)))), 
                                1/10000, 
                                500
                            ]
                            self.field.parameters = np.array(reach_param)
                        elif self.domain_type in ['2d', 'simplex']:
                            self.field.parameters = torch.tensor(reach_param).clone()
                        
                        # Run gradient ascent
                        self.field.gradient_ascent(show_out=False)
                        
                        # Extract final position
                        if self.domain_type == 'simplex':
                            final_pos_row = simplex_utils.ba2xy_vectorized(
                                barycentric_coords=self.field.pos_matrix[-1].clone(),
                                corners=self.corners
                            )
                        else:
                            final_pos_row = self.field.pos_matrix[-1].clone()
                        
                        # Reset agents position
                        self.field.agents_pos = og_pos.clone()
                        
                        # Store result in order
                        final_pos_results.append(final_pos_row)
                        
                        # Progress logging
                        if (parameter_id + 1) % max(1, len(reach_parameters) // 10) == 0:
                            logging.info(f"Completed {parameter_id + 1}/{len(reach_parameters)} parameters")
                            
                    except Exception as e:
                        logging.error(f"Error processing parameter {parameter_id}: {str(e)}")
                        raise RuntimeError(f"Failed to process parameter {parameter_id}: {str(e)}")
                
                # Vectorized matrix construction for sequential processing
                if len(final_pos_results) == 1:
                    final_pos_matrix = final_pos_results[0]
                else:
                    final_pos_matrix = torch.stack(final_pos_results, dim=0)
                
                logging.info(f"Successfully built final position matrix with shape: {final_pos_matrix.shape}")
            
            logging.info(f"Successfully computed final positions for all {len(reach_parameters)} parameters")
            
        except Exception as e:
            logging.error(f"Critical error in final_pos_over_reach: {str(e)}")
            raise
            
        finally:
            # Restore original state
            try:
                self.field.tolerated_agents = og_tolerated_agents
                self.field.tolerance = og_tolerance 
                self.field.agents_pos = og_pos.clone()
                self.field.learning_rate = og_lr.clone()
                self.agents_pos = og_pos.clone()
                self.field.parameters = og_parameters
            except Exception as e:
                logging.warning(f"Error restoring original state: {str(e)}")
        
        return final_pos_matrix
    
    def jacobian_stability_fast(self,
                                agent_parameter_instance: Union[List[float], np.ndarray],
                                resource_distribution_type: str,
                                resource_parameters: Union[List[float], np.ndarray],
                                resource_entropy: bool = False,
                                infl_entropy: bool = False) -> Tuple[List[torch.Tensor], List[float], List[float]]:
        """
        Calculate the stability of the symmetric Nash equalbirium via analytically calculating the maximum Eigenvalues of the Jacobian. This function only works for 
        multi-variate Gaussian influence and 1d Gaussian kernels. 

        :param agent_parameter_instance: Parameters for the influence function.
        :type agent_parameter_instance: Union[List[float], np.ndarray]
        :param resource_distribution_type: Type of resource distribution.
        :type resource_distribution_type: str
        :param resource_parameters: Resource parameters.
        :type resource_parameters: Union[List[float], np.ndarray]
        :param resource_entropy: Whether to calculate resource entropy.
        :type resource_entropy: bool
        :param infl_entropy: Whether to calculate influence entropy.
        :type infl_entropy: bool

        :return: The stability parameters, resource entropy, and influence entropy.
        :rtype: Tuple[List[torch.Tensor], List[float], List[float]]
        """
        og_pos=self.agents_pos.clone()
        parameter_star_list=[]
        entropy_ls=[]
        ag_ent_ls=[]
        for parameter_id in range(len(resource_parameters)):
            resource_distribution=rd.resource_distribution_choice(bin_points=self.bin_points,resource_type=resource_distribution_type,resource_parameters=resource_parameters[parameter_id])
            if self.infl_type=="gaussian":
                x_star=np.dot(self.bin_points,resource_distribution)/np.sum(resource_distribution)
                self.field.agents_pos=np.array([x_star]*self.num_agents)
                e_s=(agent_parameter_instance[0]**2)*self.field.d_lnf_matrix(agent_parameter_instance)[0]
                parameter_star=gauss.symmetric_nash_stability(num_agents=self.num_agents,d_values=e_s,resource_distribution=resource_distribution)
            elif self.infl_type=="multi_gaussian":
                #only for the special case of scaler multiple of 2x2 Id matrix for covariance matrix
                x_star = MV_gauss.symmetric_nash(bin_points=self.bin_points,resource_distribution=resource_distribution)
                self.field.agents_pos=np.array(x_star*self.num_agents)
                parameter_star=MV_gauss.symmetric_nash_stability(num_agents=self.num_agents,bin_points=self.bin_points,resource_distribution=resource_distribution)
            if resource_entropy == True:
                entropy_ls.append(scipy.stats.entropy(resource_distribution))
            if infl_entropy == True: 
                infl_dist=self.influence_matrix([parameter_star.item()]*self.num_agents)
                infl_dist_ent=scipy.stats.entropy(infl_dist[0])
                ag_ent_ls.append(infl_dist_ent)

            parameter_star_list.append(parameter_star)
        
        self.agents_pos=og_pos.clone()
        return parameter_star_list,entropy_ls,ag_ent_ls

    def find_zero_crossings(self,test_eval, parameters_list, threshold=5.0):
        """
        Identify parameter values where eigenvalue real parts are close to zero.
        
        Args:
            test_eval: Tensor of eigenvalues [n_params, n_agents]
            parameters_list: Tensor of parameter values [n_params, n_agents]
            threshold: Maximum absolute value to consider "close to zero"
            
        Returns:
            Tuple of (parameter indices, parameter values) near zero crossings
        """
        # Extract real parts from eigenvalues
        real_parts = test_eval.real
        
        # Find where any eigenvalue has real part close to zero
        zero_crossing_mask = torch.any(torch.abs(real_parts) < threshold, dim=1)
        
        # Get parameter indices and values where crossings occur
        zero_indices = torch.where(zero_crossing_mask)[0]
        zero_params = parameters_list[zero_indices, 0]  # Assuming first column is reach parameter
        
        return zero_indices, zero_params, real_parts

    def refine_parameter_space(self, zero_params, initial_positions=None, padding=0.01, num_points=50, time_steps=10000):
        """
        Create a refined parameter space around zero crossings for higher precision analysis.
        
        Args:
            vis: Shell visualization object
            zero_params: Tensor of parameter values where real parts are close to zero
            padding: Additional parameter range to include around zero crossings
            num_points: Number of points to sample in each refined region
            
        Returns:
            Refined parameter list and corresponding test_eval values
        """
        from InflGame.utils.general import agent_parameter_setup
        import InflGame.adaptive.jacobian as jc
        
        # Create boundaries for refined regions with padding
        min_vals = torch.clamp(zero_params.min() - padding, min=0.03)
        max_vals = torch.clamp(zero_params.max() + padding, max=0.3)
        
        # Generate refined parameter list
        refined_parameters = agent_parameter_setup(
            num_agents=3,
            infl_type="gaussian",
            setup_type="parameter_space",
            reach_start=min_vals.item(),
            reach_end=max_vals.item(),
            reach_num_points=num_points
        )
        
        # Preserve original state
        original_pos = self.field.agents_pos.clone()
        original_params = self.field.parameters.clone()
        original_time_steps = self.field.time_steps
        
        # Run gradient ascent with refined parameters
        self.field.time_steps = time_steps
        refined_positions = []
        
        # Initial positions from original run
        if initial_positions is None:
            if torch.is_tensor(self.field.agents_pos):
                initial_positions = self.field.agents_pos.clone()
            else:
                initial_positions = torch.tensor(self.field.agents_pos)
        else:
            if not torch.is_tensor(initial_positions):
                initial_positions = torch.tensor(initial_positions)
            else:
                initial_positions = initial_positions.clone()

        for parameters in refined_parameters:
            # Reset field state for each run
            self.field.agents_pos = initial_positions.clone()
            self.field.parameters = parameters.clone()
            self.field.gradient_ascent()
            
            # Get final position
            final_pos = self.field.pos_matrix[-1].clone()
            refined_positions.append(final_pos)
        
        # Compute eigenvalues for refined positions
        refined_eval_list = []
        for final_position, parameters in zip(refined_positions, refined_parameters):
            # Set field state for jacobian computation
            self.field.agents_pos = final_position.clone()
            self.agents_pos = final_position.clone()
            
            jacobian = jc.jacobian_matrix(
                num_agents=3,
                parameters=parameters,
                agents_pos=final_position,
                bin_points=self.bin_points,
                resource_distribution=self.resource_distribution,
                infl_type='gaussian',
                infl_fshift=False,
                Q=0,
                infl_matrix=self.field.influence_matrix(parameter_instance=parameters),
                prob_matrix=self.field.prob_matrix(parameter_instance=parameters),
                d_lnf_matrix=self.field.d_lnf_matrix(parameter_instance=parameters)
            )
            
            eigen_values = torch.linalg.eigvals(jacobian)
            refined_eval_list.append(eigen_values)
        
        # Restore original state
        self.field.agents_pos = original_pos
        self.agents_pos = original_pos
        self.field.parameters = original_params
        self.field.time_steps = original_time_steps
        
        # Stack results into tensors
        refined_eval = torch.stack(refined_eval_list)
        
        return refined_parameters, refined_eval

    def analyze_zero_crossings(self,refined_parameters, refined_eval, x_star=None):
        """
        Perform detailed analysis of zero crossings in the refined parameter space.
        
        Args:
            refined_parameters: Tensor of refined parameter values
            refined_eval: Tensor of eigenvalues at refined parameters
            x_star: Theoretical critical point (if available)
        
        Returns:
            Dictionary of analysis results and figure handles
        """
        # Extract parameter values and real/imaginary parts
        param_indices = refined_parameters[:, 0]
        real_parts = refined_eval.real
        imag_parts = refined_eval.imag
        
        # Find exact zero crossings for each eigenvalue
        zero_crossings = []
        for i in range(real_parts.shape[1]):
            # Get parameter values where sign changes (zero crossing)
            signs = torch.sign(real_parts[:-1, i] * real_parts[1:, i])
            crossing_indices = torch.where(signs < 0)[0]
            
            # Interpolate to find exact crossing point
            for idx in crossing_indices:
                p1, p2 = param_indices[idx].item(), param_indices[idx+1].item()
                r1, r2 = real_parts[idx, i].item(), real_parts[idx+1, i].item()
                
                # Linear interpolation to find where real part = 0
                t = -r1 / (r2 - r1) if r2 != r1 else 0.5
                crossing_param = p1 + t * (p2 - p1)
                
                zero_crossings.append({
                    'eigenvalue_idx': i,
                    'param_idx': idx,
                    'param_value': crossing_param,
                    'imag_part': imag_parts[idx, i].item() * (1-t) + imag_parts[idx+1, i].item() * t
                })
        
        # Sort crossings by parameter value
        zero_crossings.sort(key=lambda x: x['param_value'])
        
        # Create zoomed-in visualization of zero crossing regions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Detailed View of Eigenvalue Real Parts Near Zero', fontsize=16)
        
        colors = ['blue', 'red', 'green']
        eigenvalue_labels = ['Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3']
        
        for i in range(3):
            ax = axes[i]
            ax.plot(param_indices.numpy(), real_parts[:, i].numpy(), 
                color=colors[i], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            if x_star is not None:
                ax.axvline(x=x_star.item(), color='orange', linestyle='-', 
                        alpha=0.8, linewidth=2, label=f'x* = {x_star.item():.4f}')
            
            # Mark zero crossings
            for crossing in zero_crossings:
                if crossing['eigenvalue_idx'] == i:
                    ax.plot(crossing['param_value'], 0, 'ko', markersize=8)
                    ax.annotate(f"{crossing['param_value']:.5f}",
                            xy=(crossing['param_value'], 0),
                            xytext=(5, 10), textcoords='offset points')
            
            ax.set_xlabel('Parameter Value (Reach)')
            ax.set_ylabel('Real Part')
            ax.set_title(f'{eigenvalue_labels[i]} Near Zero')
            ax.grid(True, alpha=0.3)
            
            # Zoom in on y-axis around zero
            y_range = np.max(np.abs(real_parts[:, i].numpy())) * 1.5
            ax.set_ylim(-y_range, y_range)
        
        plt.tight_layout()
        
        # Create complex plane visualization focused on crossings
        fig2, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Plot unit circle and axes
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Focus on eigenvalues near zero real part
        for i in range(3):
            # Filter points where real part is close to zero
            mask = torch.abs(real_parts[:, i]) < 5.0
            if torch.any(mask):
                scatter = ax.scatter(
                    real_parts[mask, i].numpy(), 
                    imag_parts[mask, i].numpy(),
                    c=param_indices[mask].numpy(), 
                    cmap='viridis', 
                    s=80, 
                    alpha=0.7,
                    label=eigenvalue_labels[i]
                )
        
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title('Eigenvalues in Complex Plane Near Stability Transitions')
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Parameter Value (Reach)')
        
        # Compile analysis results
        results = {
            'zero_crossings': zero_crossings,
            'figures': [fig, fig2],
            'refined_parameters': refined_parameters,
            'refined_eval': refined_eval
        }
        
        return results
    

    # Add these methods to the Shell class
    def analyze_near_critical_point(self, x_star, padding=0.01, num_points=50):
        """
        Perform high-resolution analysis near the theoretical critical point x*.
        
        Args:
            x_star: Theoretical critical point
            padding: Range to examine on either side of x*
            num_points: Number of parameter points to examine
            
        Returns:
            Analysis results focused on the critical point
        """
        # Create parameter range centered on x*
        x_star_val = x_star.item()
        min_val = max(0.03, x_star_val - padding)
        max_val = min(0.3, x_star_val + padding)
        
        # Generate refined parameter list
        critical_parameters = agent_parameter_setup(
            num_agents=self.num_agents,
            infl_type="gaussian",
            setup_type="parameter_space",
            reach_start=min_val,
            reach_end=max_val,
            reach_num_points=num_points
        )
        
        # Preserve original state
        original_pos = self.field.agents_pos.clone()
        original_params = self.field.parameters.clone()
        original_time_steps = self.field.time_steps
        
        # Run gradient ascent with refined parameters
        self.field.time_steps = 10000
        critical_positions = []
        
        # Initial positions
        initial_positions = self.field.agents_pos.clone()
        
        # Process parameters to get equilibrium positions
        for parameters in critical_parameters:
            # Reset field state
            self.field.agents_pos = initial_positions.clone()
            self.field.parameters = parameters.clone()
            self.field.gradient_ascent()
            
            # Get final position
            final_pos = self.field.pos_matrix[-1].clone()
            critical_positions.append(final_pos)
        
        # Compute eigenvalues
        critical_eval_list = []
        for final_position, parameters in zip(critical_positions, critical_parameters):
            # Set field state for jacobian computation
            self.field.agents_pos = final_position.clone()
            self.agents_pos = final_position.clone()
            
            jacobian = jc.jacobian_matrix(
                num_agents=self.num_agents,
                parameters=parameters,
                agents_pos=final_position,
                bin_points=self.bin_points,
                resource_distribution=self.resource_distribution,
                infl_type='gaussian',
                infl_fshift=False,
                Q=0,
                infl_matrix=self.field.influence_matrix(parameter_instance=parameters),
                prob_matrix=self.field.prob_matrix(parameter_instance=parameters),
                d_lnf_matrix=self.field.d_lnf_matrix(parameter_instance=parameters)
            )
            
            eigen_values = torch.linalg.eigvals(jacobian)
            critical_eval_list.append(eigen_values)
        
        # Restore original state
        self.field.agents_pos = original_pos
        self.agents_pos = original_pos
        self.field.parameters = original_params
        self.field.time_steps = original_time_steps
        
        # Stack results into tensors
        critical_eval = torch.stack(critical_eval_list)
        
        # Create visualization focused on critical point
        param_indices = critical_parameters[:, 0]
        real_parts = critical_eval.real
        imag_parts = critical_eval.imag
        
        # Find exact x* index
        x_star_idx = torch.argmin(torch.abs(param_indices - x_star)).item()
        
        # Create detailed plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot real parts
        ax1 = axes[0]
        colors = ['blue', 'red', 'green']
        eigenvalue_labels = ['Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3']
        
        for i in range(3):
            ax1.plot(param_indices.numpy(), real_parts[:, i].numpy(), 
                color=colors[i], linewidth=2, label=eigenvalue_labels[i])
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Stability Boundary')
        ax1.axvline(x=x_star.item(), color='orange', linestyle='-', 
                    alpha=0.8, linewidth=2, label=f'x* = {x_star.item():.6f}')
        
        ax1.set_xlabel('Parameter Value (Reach)')
        ax1.set_ylabel('Real Part')
        ax1.set_title('Eigenvalue Real Parts Near Critical Point x*')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Set square aspect ratio for the plot
        ax1.set_box_aspect(1)
        
        # Mark the exact critical point eigenvalues
        for i in range(3):
            real_val = real_parts[x_star_idx, i].item()
            ax1.plot(x_star.item(), real_val, 'ko', markersize=8)
            ax1.annotate(f"{real_val:.4f}",
                        xy=(x_star.item(), real_val),
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot imaginary parts
        ax2 = axes[1]
        for i in range(3):
            ax2.plot(param_indices.numpy(), imag_parts[:, i].numpy(), 
                color=colors[i], linewidth=2, label=eigenvalue_labels[i])
        
        ax2.axvline(x=x_star.item(), color='orange', linestyle='-', 
                    alpha=0.8, linewidth=2, label=f'x* = {x_star.item():.6f}')
        
        ax2.set_xlabel('Parameter Value (Reach)')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title('Eigenvalue Imaginary Parts Near Critical Point x*')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Set square aspect ratio for the plot
        ax2.set_box_aspect(1)
        
        # Mark the exact critical point imaginary parts
        for i in range(3):
            imag_val = imag_parts[x_star_idx, i].item()
            ax2.plot(x_star.item(), imag_val, 'ko', markersize=8)
            ax2.annotate(f"{imag_val:.4f}",
                        xy=(x_star.item(), imag_val),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        # Calculate properties exactly at x*
        x_star_real = real_parts[x_star_idx].numpy()
        x_star_imag = imag_parts[x_star_idx].numpy()
        
        print(f"Eigenvalues at critical point x* = {x_star.item():.6f}:")
        for i in range(3):
            print(f"  Eigenvalue {i+1}: {x_star_real[i]:.6f} + {x_star_imag[i]:.6f}j")
        
        print(f"Stability at x*: {'Stable' if np.all(x_star_real < 0) else 'Unstable'}")
        
        results = {
            'critical_parameters': critical_parameters,
            'critical_eval': critical_eval,
            'x_star_idx': x_star_idx,
            'figure': fig
        }
        
        return results

    def find_and_analyze_zero_crossings(self, test_eval, parameters_list, x_star=None, threshold=5.0):
        """
        Complete workflow to find, refine, and analyze eigenvalue zero crossings.
        
        Args:
            test_eval: Tensor of eigenvalues
            parameters_list: Tensor of parameter values
            x_star: Theoretical critical point (if available)
            threshold: Maximum absolute value to consider "close to zero"
            
        Returns:
            Analysis results
        """
        # Use the imported function from stability_analysis
        zero_indices, zero_params, real_parts = self.find_zero_crossings(test_eval, parameters_list, threshold)
        print(f"Found {len(zero_indices)} parameter points with eigenvalues near zero")
        
        if len(zero_indices) == 0:
            print("No eigenvalues with real parts close to zero found.")
            return None
        
        # Print the parameter values where we found near-zero eigenvalues
        print("Parameter values with near-zero eigenvalues:")
        for i, param in zip(zero_indices, zero_params):
            print(f"  Index {i.item()}: reach = {param.item():.5f}")
        
        # Extract imaginary parts
        imag_parts = test_eval.imag
        
        # Find exact zero crossings for each eigenvalue
        zero_crossings = []
        param_indices = parameters_list[:, 0]
        
        for i in range(real_parts.shape[1]):
            # Get parameter values where sign changes (zero crossing)
            signs = torch.sign(real_parts[:-1, i] * real_parts[1:, i])
            crossing_indices = torch.where(signs < 0)[0]
            
            # Interpolate to find exact crossing point
            for idx in crossing_indices:
                p1, p2 = param_indices[idx].item(), param_indices[idx+1].item()
                r1, r2 = real_parts[idx, i].item(), real_parts[idx+1, i].item()
                
                # Linear interpolation to find where real part = 0
                t = -r1 / (r2 - r1) if r2 != r1 else 0.5
                crossing_param = p1 + t * (p2 - p1)
                
                zero_crossings.append({
                    'eigenvalue_idx': i,
                    'param_idx': idx,
                    'param_value': crossing_param,
                    'imag_part': imag_parts[idx, i].item() * (1-t) + imag_parts[idx+1, i].item() * t
                })
        
        # Sort crossings by parameter value
        zero_crossings.sort(key=lambda x: x['param_value'])
        print(f"Found {len(zero_crossings)} precise zero crossings")
        
        # Create zoomed-in visualization of zero crossing regions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Detailed View of Eigenvalue Real Parts Near Zero', fontsize=16)
        
        colors = ['blue', 'red', 'green']
        eigenvalue_labels = ['Eigenvalue 1', 'Eigenvalue 2', 'Eigenvalue 3']
        
        for i in range(3):
            ax = axes[i]
            ax.plot(param_indices.numpy(), real_parts[:, i].numpy(), 
                color=colors[i], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Set square aspect ratio for the plot
            ax.set_box_aspect(1)
            
            if x_star is not None:
                ax.axvline(x=x_star.item(), color='orange', linestyle='-', 
                        alpha=0.8, linewidth=2, label=f'x* = {x_star.item():.4f}')
            
            # Mark zero crossings
            for crossing in zero_crossings:
                if crossing['eigenvalue_idx'] == i:
                    ax.plot(crossing['param_value'], 0, 'ko', markersize=8)
                    ax.annotate(f"{crossing['param_value']:.5f}",
                            xy=(crossing['param_value'], 0),
                            xytext=(5, 10), textcoords='offset points')
            
            ax.set_xlabel('Parameter Value (Reach)')
            ax.set_ylabel('Real Part')
            ax.set_title(f'{eigenvalue_labels[i]} Near Zero')
            ax.grid(True, alpha=0.3)
            
            # Zoom in on y-axis around zero
            y_range = np.max(np.abs(real_parts[:, i].numpy())) * 1.5
            ax.set_ylim(-y_range, y_range)
        
        plt.tight_layout()
        
        # Print details of zero crossings
        print("\nPrecise zero crossings:")
        for i, crossing in enumerate(zero_crossings):
            print(f"  {i+1}. Eigenvalue {crossing['eigenvalue_idx']+1} crosses zero at reach = {crossing['param_value']:.6f}")
            print(f"     Imaginary part at crossing: {crossing['imag_part']:.6f}")
        
        # Compile analysis results
        results = {
            'zero_crossings': zero_crossings,
            'figure': fig,
            'zero_indices': zero_indices,
            'zero_params': zero_params
        }
        
        return results


    
    ##Outdated need to remove or archive (too slow but generalizable if you know x_star)

    # def jocaobian_classifier_alt(self,
    #                              param_list: List[Union[List[float], np.ndarray]]) -> List[int]:
    #     """
    #     Classify the stability of the symmetric Nash via numerical calculations assuming the symmetric Nash is known.

    #     :param param_list: List of parameters for the influence function.
    #     :type param_list: List[Union[List[float], np.ndarray]]

    #     :return: The stability classification.
    #     :rtype: List[int]
    #     """
    #     row=[]
    #     for param in param_list:
    #         infl_matrix=self.field.influence_matrix(param)
    #         prob_matrix=self.field.prob_matrix(param)
    #         d_lnf_matrix=self.field.d_lnf_matrix(param)
    #         jacobian_mtrx=jacobian.jacobian_matrix(num_agents=self.num_agents,parameter_instance=param,agents_pos=self.agents_pos,bin_points=self.bin_points,resource_distribution=self.resource_distribution,
    #                       infl_type=self.infl_type,infl_fshift=self.infl_fshift,Q=self.Q,infl_matrix=infl_matrix,prob_matrix=prob_matrix,d_lnf_matrix=d_lnf_matrix)
    #         evals=torch.linalg.eigvals(jacobian_mtrx)  
    #         if torch.sum(torch.sign(torch.real(evals)))==-1*len(evals):
    #             row.append(1)

    #         elif torch.all(evals==0):
    #             row.append(0)
    #         else:
    #             row.append(-1)
        
    #     return row

    # def jocobian_cl_iter_alt(self,
    #                          param_list: List[Union[List[float], np.ndarray]],
    #                          resource_params: List[float],
    #                          sig: float = .1) -> torch.Tensor:
    #     """
    #     Iterate the stability classification using the Jacobian matrix.

    #     :param param_list: List of parameters for the influence function.
    #     :type param_list: List[Union[List[float], np.ndarray]]
    #     :param resource_params: List of resource parameters.
    #     :type resource_params: List[float]
    #     :param sig: Standard deviation for resource distribution.
    #     :type sig: float

    #     :return: The stability classification matrix.
    #     :rtype: torch.Tensor
    #     """
    #     og_re_dist=self.resource_distribution
    #     bin_points=self.bin_points
    #     j_class_mat=0
    #     for parameter_id in range(len(resource_params)):
    #         re_param=resource_params[parameter_id]
    #         resource_distribution=1/(3*sig*np.sqrt(2*np.pi))*(np.exp(-(bin_points-.5-re_param/2)**2/(2*(sig)**2))+np.exp(-(bin_points-.5+1/2*re_param)**2/(2*(sig)**2)))
    #         self.resource_distribution=resource_distribution
    #         jc_class_row=torch.tensor(self.jocaobian_classifier_alt(param_list))
    #         j_class_mat=utilities.matrix_builder(row_id=parameter_id,row=jc_class_row,matrix=j_class_mat)
    #     self.resource_distribution=og_re_dist
    #     return j_class_mat

    # def mean_stability_birfurcation_rs(self,
    #                                    r_st: float = .1,
    #                                    r_end: float = .5,
    #                                    r_points: int = 200,
    #                                    s_st: float = 0,
    #                                    s_end: float = 1,
    #                                    s_points: int = 100) -> None:
    #     """
    #     Plot the mean stability bifurcation for agents over a range of reach and separation parameters.

    #     :param r_st: Starting value of reach parameter.
    #     :type r_st: float
    #     :param r_end: Ending value of reach parameter.
    #     :type r_end: float
    #     :param r_points: Number of points in the reach parameter range.
    #     :type r_points: int
    #     :param s_st: Starting value of separation parameter.
    #     :type s_st: float
    #     :param s_end: Ending value of separation parameter.
    #     :type s_end: float
    #     :param s_points: Number of points in the separation parameter range.
    #     :type s_points: int
    #     """
    #     agents_pos_og=self.agents_pos.clone()
    #     self.agents_pos=np.array([.5]*self.num_agents)
    #     start=[r_st]*self.num_agents
    #     end=[r_end]*self.num_agents
    #     params=np.linspace(start,end,r_points)
    #     params2=np.linspace(s_end,s_st,s_points)
    #     t=self.jocobian_cl_iter_alt(params,params2)
        
    #     #colors
    #     cmap = mpl.colormaps["viridis"]
    #     newcolors = cmap(np.linspace(0, 1, 100))
    #     newcolors[90:]= np.array([0.8, 0.8, 0.8,1])
    #     newcolors[:10]= np.array([0,0,0,1])
    #     newcolors[20] = mpl.colors.to_rgb('tab:orange') + (1,)
    #     newcmap = mpl.colors.ListedColormap(newcolors)

    #     #plot
    #     ax=sns.heatmap(t,xticklabels=False,yticklabels=False,cbar=False,vmin=0, vmax=1,cmap=newcmap)

    #     num_ticks_y = 11
    #     num_ticks_x = 9

    #     # the index of the position of yticks
    #     yticks = np.linspace(0, len(params2) - 1, num_ticks_y)
    #     xticks = np.linspace(0, len(params[:,0]) - 1, num_ticks_x)
    #     # the content of labels of these yticks
    #     yticklabel = [np.round(params2[int(idx)],decimals=1) for idx in yticks]
    #     xticklabel = [np.round(params[int(idx),0],decimals=2) for idx in xticks]

    #     ax.set_yticks(yticks)
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(xticklabel)
    #     ax.set_yticklabels(yticklabel)
        

    #     black_p = mpatches.Patch(color='black', label='unstable')
    #     white_p = mpatches.Patch(color='grey', label='stable')

    #     plt.legend(handles=[black_p, white_p])



    #     plt.title('Mean Stability Bifurcation Map')
    #     plt.xlabel(r"$\sigma$"+' (Reach parameter)' )
    #     plt.ylabel(r'$\alpha$'+' (Seperation parameter)')
    #     self.agents_pos=agents_pos_og.clone()
