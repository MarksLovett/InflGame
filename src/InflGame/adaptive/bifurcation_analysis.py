"""
.. module:: selfualization
   :synopsis: Provides selfualization tools for analyzing and understanding the dynamics of adaptive environments and agent interactions for influencer games.


Visualization Module
====================



This module provides selfualization tools for analyzing and understanding the dynamics of adaptive environments and agent interactions for influencer games.
It includes plotting utilities for various domains (1D, 2D, and simplex) and supports selfualizing agent positions, gradients, 
influence distributions, and bifurcation dynamics.

The module is designed to work with the `AdaptiveEnv` class and provides a framework for creating selfual representations of agent behaviors 
in influencer game environments.

Dependencies:
-------------
- InflGame.utils
- InflGame.kernels
- InflGame.domains

Usage:
------
The `Shell` class can be used to selfualize the results of simulations performed using the `AdaptiveEnv` class. It supports various selfualization types, including position plots, gradient plots, probability plots, and bifurcation plots.

Example:
--------

.. code-block:: python
    
    from InflGame.adaptive.selfualization import Shell
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
from multiprocessing import Pool, cpu_count
import traceback
import plotly.graph_objects as go
from scipy.optimize import fsolve, brentq
from scipy.special import digamma as psi




import matplotlib.figure
from typing import Union, List, Dict, Optional, Tuple

import InflGame.adaptive.grad_func_env as grad_func_env
from InflGame.adaptive.grad_func_env import AdaptiveEnv
#import InflGame.adaptive_dynamics.jacobian as jacobian

import InflGame.utils.general as general
from InflGame.utils import data_management
import InflGame.utils.validation as validation
import InflGame.utils.plot_utils as plot_utils

import InflGame.kernels.gauss as gauss 
import InflGame.kernels.diric as diric
import InflGame.kernels.MV_gauss as MV_gauss
import InflGame.kernels.beta as beta
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



class BifurcationEnv:
    """
    The bif class provides a framework for simulating and selfualizing adaptive dynamics
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
        self.matrix_results_complete=None
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

    def final_pos_over_reach_envelope(self, 
                                reach_parameters: Union[List[float], np.ndarray], 
                                tolerance: float, 
                                tolerated_agents: int,
                                percentage: float = 1.0,
                                parallel: bool = True,
                                max_workers: Optional[int] = None,
                                batch_size: Optional[int] = None,
                                time_steps: Optional[int] = None,
                                learning_rate:Optional[List] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate both the extreme (maximum and minimum) positions of agents over a range of reach parameters 
        via repeated initiations of :func:`InflGame.adaptive.grad_func_env.gradient_ascent` over a group of parameters.
        
        This method tracks the extreme positions achieved during the specified percentage of gradient ascent iterations,
        but ONLY when the dynamics did not converge. If convergence is achieved, the final position is returned instead.
        
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
        :param percentage: Percentage of trajectory to analyze (0.0-1.0, e.g., 0.5 for last 50%, 1.0 for entire trajectory)
        :type percentage: float
        :param parallel: Whether to use parallel processing
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers (defaults to CPU count)
        :type max_workers: Optional[int]
        :param batch_size: Batch size for processing (auto-calculated if None)
        :type batch_size: Optional[int]
        :param time_steps: Maximum number of gradient ascent steps
        :type time_steps: Optional[int]

        :return: Dictionary containing 'max' and 'min' extreme positions for each parameter
        :rtype: Dict[str, torch.Tensor]
        
        :raises ValueError: If input parameters are invalid
        :raises RuntimeError: If computation fails
        """
        # Input validation
        if tolerance == None:
            tolerance = self.tolerance
        if tolerated_agents == None:
            tolerated_agents = self.tolerated_agents
            
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
        
        if not (0.0 < percentage <= 1.0):
            raise ValueError(f"percentage must be between 0.0 and 1.0, got {percentage}")
        
        # Validate domain type
        if self.domain_type not in ['1d', '2d', 'simplex']:
            raise ValueError(f"Unsupported domain_type: {self.domain_type}")
            
        percentage_percent = int(percentage * 100)
        if percentage == 1.0:
            logging.info(f"Computing both max and min positions (entire trajectory, non-converged only) for {len(reach_parameters)} parameters using domain '{self.domain_type}'")
        else:
            logging.info(f"Computing both max and min positions (last {percentage_percent}% of iterations, non-converged only) for {len(reach_parameters)} parameters using domain '{self.domain_type}'")
        
        if self.domain_type == 'simplex':
            self.field.domain_bounds = [0, 1]
            self.domain_bounds = [0, 1]
        
        # Store original state following project patterns
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
                    'time_steps': time_steps,
                    'percentage': percentage
                }
                parameter_data_list.append(parameter_data)
            
            # Initialize result storage
            max_pos_matrix = 0
            min_pos_matrix = 0
            
            if parallel and len(reach_parameters) > 1:
                # Parallel processing
                logging.info(f"Using parallel processing with {max_workers} workers")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_param = {
                            executor.submit(self._compute_single_parameter_both_extremes_percentage, param_data): param_data['parameter_id']
                            for param_data in parameter_data_list
                        }
                        
                        # Collect results as they complete
                        max_results = {}
                        min_results = {}
                        completed_count = 0
                        
                        for future in as_completed(future_to_param):
                            try:
                                parameter_id, max_pos_row, min_pos_row = future.result()
                                max_results[parameter_id] = max_pos_row
                                min_results[parameter_id] = min_pos_row
                                completed_count += 1
                                
                                if completed_count % max(1, len(reach_parameters) // 10) == 0:
                                    logging.info(f"Completed {completed_count}/{len(reach_parameters)} parameters")
                                    
                            except Exception as e:
                                param_id = future_to_param[future]
                                logging.error(f"Parameter {param_id} failed: {str(e)}")
                                raise RuntimeError(f"Failed to compute parameter {param_id}: {str(e)}")
                        
                        # Build extreme matrices from results - vectorized approach
                        num_params = len(reach_parameters)
                        if len(max_results) != num_params or len(min_results) != num_params:
                            raise RuntimeError(f"Expected {num_params} results, got max: {len(max_results)}, min: {len(min_results)}")
                        
                        # Create lists to store results in original order
                        ordered_max_results = [None] * num_params
                        ordered_min_results = [None] * num_params
                        
                        for param_id in range(num_params):
                            if param_id not in max_results or param_id not in min_results:
                                raise RuntimeError(f"Missing results for parameter ID {param_id}")
                            ordered_max_results[param_id] = max_results[param_id]
                            ordered_min_results[param_id] = min_results[param_id]
                        
                        # Vectorized matrix construction following project patterns
                        if len(ordered_max_results) == 1:
                            max_pos_matrix = ordered_max_results[0]
                            min_pos_matrix = ordered_min_results[0]
                        else:
                            max_pos_matrix = torch.stack(ordered_max_results, dim=0)
                            min_pos_matrix = torch.stack(ordered_min_results, dim=0)
                        
                        logging.info(f"Successfully built both extreme position matrices with shape: {max_pos_matrix.shape}")
                            
                except Exception as e:
                    logging.error(f"Parallel processing failed: {str(e)}")
                    logging.info("Falling back to sequential processing")
                    parallel = False
            
            if not parallel:
                # Sequential processing (fallback or by choice) following project patterns
                logging.info("Using sequential processing")
                
                # Pre-allocate results lists for vectorized construction
                max_pos_results = []
                min_pos_results = []
                converged_count = 0
                non_converged_count = 0
                
                for parameter_id, reach_param in enumerate(reach_parameters):
                    try:
                        # Reset field state following project patterns
                        self.field.pos_matrix = 0
                        self.field.agents_pos = og_pos.clone()
                        self.agents_pos = og_pos.clone()
                        
                        # Set parameters based on domain type following project patterns
                        if self.domain_type in ['1d'] and learning_rate==None:
                            self.field.learning_rate = [
                                10**(-1*(max(3, 5*(parameter_id+1)/len(reach_parameters)))), 
                                1/10000, 
                                500
                            ]
                            self.field.parameters = np.array(reach_param)
                        elif  self.domain_type in ['1d']:
                            self.field.learning_rate = [
                                10**(-1*(max(learning_rate[1], 5*(parameter_id+1)/len(reach_parameters)))), 
                                learning_rate[0], 
                                500
                            ]
                        elif self.domain_type in ['2d', 'simplex']:
                            self.field.parameters = torch.tensor(reach_param).clone()
                        
                        # Run gradient ascent
                        self.field.gradient_ascent(show_out=False)
                        
                        # Check if gradient ascent converged
                        converged = len(self.field.pos_matrix) < time_steps if time_steps else False
                        
                        if converged:
                            # Use final position for converged cases (both max and min are the same)
                            if self.domain_type == 'simplex':
                                final_pos_row = simplex_utils.ba2xy_vectorized(
                                    barycentric_coords=self.field.pos_matrix[-1].clone(),
                                    corners=self.corners
                                )
                            else:
                                final_pos_row = self.field.pos_matrix[-1].clone()
                            
                            max_pos_results.append(final_pos_row)
                            min_pos_results.append(final_pos_row)
                            converged_count += 1
                            
                        else:
                            # Use extreme positions from specified percentage of iterations for non-converged cases
                            trajectory_length = len(self.field.pos_matrix)
                            if percentage == 1.0:
                                # Use entire trajectory
                                percentage_subset = self.field.pos_matrix
                            else:
                                # Use last percentage of trajectory
                                start_idx = max(0, int(trajectory_length * (1.0 - percentage)))
                                percentage_subset = self.field.pos_matrix[start_idx:]
                            
                            if self.domain_type == 'simplex':
                                # Convert percentage subset to xy coordinates first
                                trajectory_xy = simplex_utils.ba2xy_vectorized(
                                    barycentric_coords=percentage_subset,
                                    corners=self.corners
                                )
                                
                                max_pos_row = torch.max(trajectory_xy, dim=0)[0]
                                min_pos_row = torch.min(trajectory_xy, dim=0)[0]
                            else:
                                # For 1d and 2d domains
                                max_pos_row = torch.max(percentage_subset, dim=0)[0]
                                min_pos_row = torch.min(percentage_subset, dim=0)[0]
                            
                            max_pos_results.append(max_pos_row)
                            min_pos_results.append(min_pos_row)
                            non_converged_count += 1
                        
                        # Reset agents position following project patterns
                        self.field.agents_pos = og_pos.clone()
                        
                        # Progress logging
                        if (parameter_id + 1) % max(1, len(reach_parameters) // 10) == 0:
                            logging.info(f"Completed {parameter_id + 1}/{len(reach_parameters)} parameters")
                            
                    except Exception as e:
                        logging.error(f"Error processing parameter {parameter_id}: {str(e)}")
                        raise RuntimeError(f"Failed to process parameter {parameter_id}: {str(e)}")
                
                # Vectorized matrix construction for sequential processing following project patterns
                if len(max_pos_results) == 1:
                    max_pos_matrix = max_pos_results[0]
                    min_pos_matrix = min_pos_results[0]
                else:
                    max_pos_matrix = torch.stack(max_pos_results, dim=0)
                    min_pos_matrix = torch.stack(min_pos_results, dim=0)
                
                logging.info(f"Successfully built both position matrices with shape: {max_pos_matrix.shape}")
                logging.info(f"Converged cases: {converged_count}, Non-converged cases: {non_converged_count}")
            
            logging.info(f"Successfully computed both extreme positions for all {len(reach_parameters)} parameters")
            
        except Exception as e:
            logging.error(f"Critical error in extreme_pos_over_reach_both: {str(e)}")
            raise
            
        finally:
            # Restore original state following project patterns
            try:
                self.field.tolerated_agents = og_tolerated_agents
                self.field.tolerance = og_tolerance 
                self.field.agents_pos = og_pos.clone()
                self.field.learning_rate = og_lr.clone()
                self.agents_pos = og_pos.clone()
                self.field.parameters = og_parameters
            except Exception as e:
                logging.warning(f"Error restoring original state: {str(e)}")
        
        return {
            'max': max_pos_matrix,
            'min': min_pos_matrix
        }
    
    def _compute_single_parameter_both_extremes_percentage(self, parameter_data: Dict) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Helper function to compute both max and min positions for a single parameter from the specified percentage of iterations,
        but only if convergence was not achieved. For converged cases, returns the final position for both.
        Designed to be used with multiprocessing for the extreme_pos_over_reach_both function.
        
        :param parameter_data: Dictionary containing parameter data and configuration
        :type parameter_data: Dict
        
        :return: Tuple of parameter_id, max_position_row, and min_position_row
        :rtype: Tuple[int, torch.Tensor, torch.Tensor]
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
            percentage = parameter_data['percentage']
            
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
            
            # Set parameters based on domain type following project patterns
            if domain_type in ['1d']:
                temp_field.learning_rate = [10**(-1*(max(3,5*(total_params-parameter_id)/total_params))), 1/10000, 500]
                temp_field.parameters = np.array(reach_param)
            elif domain_type in ['2d', 'simplex']:
                temp_field.parameters = torch.tensor(reach_param).clone()
            
            # Run gradient ascent
            temp_field.gradient_ascent(show_out=False)
            
            # Check if gradient ascent converged
            converged = len(temp_field.pos_matrix) < time_steps if time_steps else False
            
            if converged:
                # Use final position for converged cases (both max and min are the same)
                if domain_type == 'simplex':
                    final_pos_row = simplex_utils.ba2xy_vectorized(
                        barycentric_coords=temp_field.pos_matrix[-1].clone(),
                        corners=self.corners
                    )
                else:
                    final_pos_row = temp_field.pos_matrix[-1].clone()
                
                max_pos_row = final_pos_row
                min_pos_row = final_pos_row
            else:
                # Use extreme positions from specified percentage of iterations for non-converged cases
                trajectory_length = len(temp_field.pos_matrix)
                if percentage == 1.0:
                    # Use entire trajectory
                    percentage_subset = temp_field.pos_matrix
                else:
                    # Use last percentage of trajectory
                    start_idx = max(0, int(trajectory_length * (1.0 - percentage)))
                    percentage_subset = temp_field.pos_matrix[start_idx:]
                
                if domain_type == 'simplex':
                    # Convert percentage subset to xy coordinates first
                    trajectory_xy = simplex_utils.ba2xy_vectorized(
                        barycentric_coords=percentage_subset,
                        corners=self.corners
                    )
                    
                    max_pos_row = torch.max(trajectory_xy, dim=0)[0]
                    min_pos_row = torch.min(trajectory_xy, dim=0)[0]
                else:
                    # For 1d and 2d domains
                    max_pos_row = torch.max(percentage_subset, dim=0)[0]
                    min_pos_row = torch.min(percentage_subset, dim=0)[0]
                
            return parameter_id, max_pos_row, min_pos_row
            
        except Exception as e:
            logging.error(f"Error computing positions for parameter {parameter_id}: {str(e)}")
            raise RuntimeError(f"Failed to compute positions for parameter {parameter_id}: {str(e)}")

    def equilibrium_bifurcation_complete(self,
                                        reach_start: float = .03,
                                        reach_end: float = .3,
                                        reach_num_points: int = 30,
                                        time_steps: int = 100,
                                        initial_pos: Union[List[float], torch.Tensor] = None,
                                        tolerance: Optional[float] = None,
                                        tolerated_agents: Optional[int] = None,
                                        parallel_configs: Dict[str, Union[bool, int]] = None,
                                        envelope: bool = False,
                                        verbose: bool = True
                                        ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Optimized equilibrium bifurcation plot computation following project patterns.
        
        Key optimizations:
        - Proper state management with restoration
        - Memory efficient matrix clearing
        - Better parameter validation
        - Following project's torch tensor conventions
        - Optimized position generation logic
        """
        
        # Preserve original state following project patterns
        og_time_steps = self.time_steps
        og_pos = self.agents_pos.clone()
        og_field_pos = self.field.agents_pos.clone()
        
        # Parameter validation and defaults
        if initial_pos is None:
            initial_pos = self.agents_pos.clone()
        elif not isinstance(initial_pos, torch.Tensor):
            initial_pos = torch.tensor(initial_pos, dtype=torch.float32)
        else:
            initial_pos = initial_pos.clone()
        
        if parallel_configs is None:
            parallel_configs = {'parallel': True, 'max_workers': 4, 'batch_size': 2}
        
        if tolerance is None:
            tolerance = self.tolerance
        if tolerated_agents is None:
            tolerated_agents = self.tolerated_agents
        
        # Set time steps for computation
        self.field.time_steps = time_steps
        
        try:
            matrix_list = []
            
            # Domain-specific optimization
            if self.domain_type == "1d":
                # Generate reach parameters once
                reach_parameters = general.agent_parameter_setup(
                    num_agents=self.num_agents,
                    infl_type=self.infl_type,
                    setup_type="parameter_space",
                    reach_start=reach_start,
                    reach_end=reach_end,
                    reach_num_points=reach_num_points
                )
                
                # Optimized position generation - create all variations at once
                player_positions = self._generate_position_variants(initial_pos, self.num_agents)
                
                if verbose:
                    print(f"Processing {len(player_positions)} position variants over {reach_num_points} reach parameters...")
                
                start_time = time.time()
                
                # Process each position variant
                for pos_id, position_variant in enumerate(player_positions):
                    try:
                        # Set agent positions for this variant
                        self.agents_pos = position_variant.clone()
                        self.field.agents_pos = position_variant.clone()
                        
                        # Choose processing method based on envelope flag
                        if envelope and pos_id == 1:  # Only use envelope for middle variant
                            final_positions = self.final_pos_over_reach_envelope(
                                reach_parameters, 
                                tolerance=tolerance, 
                                tolerated_agents=tolerated_agents, 
                                parallel=parallel_configs.get('parallel', True),
                                max_workers=parallel_configs.get('max_workers', 4),
                                batch_size=parallel_configs.get('batch_size', 2),
                                time_steps=time_steps,
                                percentage=0.5
                            )
                        else:
                            final_positions = self.final_pos_over_reach(
                                reach_parameters, 
                                tolerance=tolerance, 
                                tolerated_agents=tolerated_agents, 
                                parallel=parallel_configs.get('parallel', True),
                                max_workers=parallel_configs.get('max_workers', 4),
                                batch_size=parallel_configs.get('batch_size', 2),
                                time_steps=time_steps
                            )
                            
                        matrix_list.append(final_positions)
                        
                        # Clear matrices to save memory following project patterns
                        if hasattr(self.field, 'pos_matrix'):
                            self.field.pos_matrix = 0
                        if hasattr(self.field, 'grad_matrix'):
                            self.field.grad_matrix = 0
                        
                        if verbose:
                            print(f"Completed position variant {pos_id + 1}/{len(player_positions)}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Error processing position variant {pos_id}: {str(e)}")
                        # Append empty tensor as fallback
                        matrix_list.append(torch.empty(0))
                
                if verbose:
                    total_time = time.time() - start_time
                    print(f"Total processing time: {total_time:.2f} seconds")
            
            else:
                # Handle other domain types if needed
                if verbose:
                    print(f"Domain type {self.domain_type} not optimized yet")
                matrix_list = []
        
        finally:
            # Restore original state following project patterns
            self.time_steps = og_time_steps
            self.agents_pos = og_pos
            self.field.agents_pos = og_field_pos
        
        return matrix_list 

    def _generate_position_variants(self,
                                    initial_pos: torch.Tensor,
                                    num_agents: int) -> List[torch.Tensor]:
        """
        Optimized position variant generation following project patterns.
        
        Creates strategic initial position variants for bifurcation analysis:
        1. Clustered variant (agents 1 to n-1 at last agent's position)
        2. Original position
        3. Alternative clustered variant (agents 1 to n-1 at first agent's position)
        """
        player_positions = []
        
        # Variant 1: Cluster agents 1 to n-1 at the last agent's position
        if num_agents > 2:
            clustered_pos_last = initial_pos.clone()
            for i in range(1, num_agents - 1):
                clustered_pos_last[i] = clustered_pos_last[-1]
            player_positions.append(clustered_pos_last)
        
        # Variant 2: Original position
        player_positions.append(initial_pos.clone())
        
        # Variant 3: Cluster agents 1 to n-1 at the first agent's position
        if num_agents > 2:
            clustered_pos_first = initial_pos.clone()
            for i in range(1, num_agents - 1):
                clustered_pos_first[i] = clustered_pos_first[0]
            player_positions.append(clustered_pos_first)
        
        return player_positions

    def find_convergence_intersections(matrix_list: List[torch.Tensor], 
                                    reach_parameters: torch.Tensor,
                                    tolerance: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        Find parameter values where agent positions converge across different variants.
        
        Following project patterns:
        - Use torch tensors for compatibility
        - Proper error handling
        - Return structured results
        """
        if len(matrix_list) < 2:
            return {'convergence_points': torch.empty(0), 'parameter_indices': torch.empty(0)}
        
        try:
            # Find where agents converge to same position within each matrix
            convergence_masks = []
            
            for matrix in matrix_list:
                if matrix.numel() == 0:
                    continue
                    
                # Check if all agents have same final position (within tolerance)
                agent_diffs = torch.abs(matrix[:, 1:] - matrix[:, :-1])  # Differences between adjacent agents
                converged_points = torch.all(agent_diffs < tolerance, dim=1)
                convergence_masks.append(converged_points)
            
            if len(convergence_masks) < 2:
                return {'convergence_points': torch.empty(0), 'parameter_indices': torch.empty(0)}
            
            # Find intersection of convergence points across variants
            common_convergence = convergence_masks[0]
            for mask in convergence_masks[1:]:
                if len(mask) == len(common_convergence):
                    common_convergence = common_convergence & mask
            
            # Get parameter values where convergence occurs
            convergence_indices = torch.where(common_convergence)[0]
            convergence_parameters = reach_parameters[convergence_indices] if len(convergence_indices) > 0 else torch.empty(0)
            
            return {
                'convergence_points': convergence_parameters,
                'parameter_indices': convergence_indices,
                'num_convergent': len(convergence_indices)
            }
            
        except Exception as e:
            print(f"Error finding convergence intersections: {str(e)}")
            return {'convergence_points': torch.empty(0), 'parameter_indices': torch.empty(0)}

    def learning_rate_large_end(self,resource_parameter,second_run=False,high_end=False)-> float:
        """
        Determine a large learning rate based on the resource parameter.
        
        :param resource_parameter: Resource parameter value
        :type resource_parameter: float
        
        :return: Calculated large learning rate
        :rtype: float
        """
        if second_run:
            if resource_parameter <= .2:
                return .001
            elif .2<resource_parameter < .7:
                return .000001
            elif .7<=resource_parameter <= .9:
                return .0001
            else:
                return .00001
        elif high_end:
            if resource_parameter <= .75:
                return [.0001,.0001,100],
        
            else:
                return [.001,.01,1000]
        else:
            if resource_parameter <= .2:
                return .01
            elif .2<resource_parameter < .7:
                return .00001
            elif .7<=resource_parameter <= .9:
                return .001
            else:
                return .0001
    def find_second_order_bifs(self,
                        bin_points: Union[List[float], np.ndarray],
                        fixed_parameters_lst: List[List[float]],
                        agents_pos: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
                        resource_distribution_type: str = "multi_modal_gaussian_distribution_1D",
                        alpha_st: float = 0,
                        alpha_end: float = 1,
                        varying_parameter_type: str = 'mean',
                        learning_rate_p: List[float] = [.0001, .01, 100],
                        parallel: bool = True,
                        max_workers: Optional[int] = None,
                        batch_size: Optional[int] = None,
                        time_steps: int = 10000,
                        second_run: bool = False,
                        data: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
                        num_points: int = 100,
                        direct_method: bool = True) -> Dict[str, List]:
        """
        Find second-order bifurcation points using parallel processing. Note that this function is only applicable to 1-1-1 equalbiria for 3 players.
        
        :param num_agents: Number of agents in the system
        :type num_agents: int
        :param bin_points: Discretization points for the domain
        :type bin_points: Union[List[float], np.ndarray]
        :param fixed_parameters_lst: Fixed parameters for resource distribution
        :type fixed_parameters_lst: List[List[float]]
        :param resource_distribution_type: Type of resource distribution
        :type resource_distribution_type: str
        :param alpha_st: Starting alpha value
        :type alpha_st: float
        :param alpha_end: Ending alpha value
        :type alpha_end: float
        :param varying_parameter_type: Type of parameter variation
        :type varying_parameter_type: str
        :param learning_rate_p: Learning rate parameters [start, end, decay]
        :type learning_rate_p: List[float]
        :param parallel: Whether to use parallel processing
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers
        :type max_workers: Optional[int]
        :param batch_size: Batch size for parallel processing
        :type batch_size: Optional[int]
        :param time_steps: Maximum time steps for gradient ascent
        :type time_steps: int
        :param direct_method: If True, uses direct gradient=0 solving with symmetric split at 0.5. If False, uses gradient ascent method.
        :type direct_method: bool
        :return: Dictionary containing sigma_star and final_parameters lists
        :rtype: Dict[str, List]
        """
        
        
        # Store original state
        og_pos = self.agents_pos.clone()
        og_field_pos = self.field.agents_pos.clone()
        og_resource_dist = self.resource_distribution.clone()
        og_learning_rate = self.learning_rate.clone()
        num_agents = self.num_agents
        try:
            # Setup initial position
            if agents_pos is None:
                agents_pos = self.agents_pos.clone()
            if agents_pos is not None and not torch.is_tensor(agents_pos):
                agents_pos = torch.tensor(agents_pos, dtype=torch.float32)
            else:
                #check dtype
                agents_pos = agents_pos.to(torch.float32)

            pos = agents_pos
            self.agents_pos = pos.clone()
            
            # Generate resource parameters
            resource_parameters, alphas = general.resource_parameter_setup(
                resource_distribution_type=resource_distribution_type,
                varying_parameter_type=varying_parameter_type,
                alpha_st=alpha_st,
                alpha_end=alpha_end,
                fixed_parameters_lst=fixed_parameters_lst,
                alpha_num_points=num_points
            )
            
            # Setup learning rates and subtract values
            t = np.linspace(0, 1, len(resource_parameters))
            if self.infl_type=='beta':
                #learning_rates = .1 - (.1 - .0001) * (t ** 2)
                subtract = .005 + (.01 - .005) * (t ** 2)
            else:
                subtract = [.02] * len(resource_parameters)
                learning_rates = learning_rate_p['min'] + (learning_rate_p['max'] - learning_rate_p['min']) * (t ** 2)
                #learning_rates = learning_rates[::-1]  # Reverse for correct order
            # Determine optimal batch size and workers
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(resource_parameters))
            if batch_size is None:
                batch_size = max(1, len(resource_parameters) // max_workers)
            
            # Prepare parameter data for parallel processing
            parameter_data_list = []
            for resource_parameter_id, resource_param in enumerate(resource_parameters):
                if self.infl_type=='beta':
                    learning_rate_p = [learning_rate_p[0],self.learning_rate_large_end(resource_parameter=alphas[resource_parameter_id],second_run=second_run),learning_rate_p[2]]
                if self.learning_rate_type=='gradient_magnitude':
                   lr= [learning_rate_p['min_simulated'],learning_rates[resource_parameter_id],learning_rate_p['period']]
                elif self.learning_rate_type=='cosine_annealing':
                   lr= [learning_rate_p['min_simulated'],learning_rates[resource_parameter_id],learning_rate_p['period']]
                else:
                   lr=[learning_rates[resource_parameter_id],learning_rates[resource_parameter_id],learning_rate_p['period']]


                if second_run:
                    parameter_data = {
                        'resource_parameter_id': resource_parameter_id,
                        'resource_param': resource_param,
                        'bin_points': bin_points,
                        'resource_distribution_type': resource_distribution_type,
                        'num_agents': num_agents,
                        'learning_rate': lr,
                        'subtract_val': subtract[resource_parameter_id],
                        'time_steps': time_steps,
                        'tolerance': self.tolerance,
                        'tolerated_agents': self.tolerated_agents,
                        'total_params': len(resource_parameters),
                        'pos': pos.clone(),
                        'sigma_star': data['final_parameters'][resource_parameter_id]['unstable_flip'][0],
                        'sigma_star_bool': second_run,
                        'direct_method': direct_method
                    }
                else:
                    parameter_data = {
                        'resource_parameter_id': resource_parameter_id,
                        'resource_param': resource_param,
                        'bin_points': bin_points,
                        'resource_distribution_type': resource_distribution_type,
                        'num_agents': num_agents,
                        'learning_rate': lr,
                        'subtract_val': subtract[resource_parameter_id],
                        'time_steps': time_steps,
                        'tolerance': self.tolerance,
                        'tolerated_agents': self.tolerated_agents,
                        'total_params': len(resource_parameters),
                        'pos': pos.clone(),
                        'sigma_star': None,
                        'sigma_star_bool': second_run,
                        'direct_method': direct_method
                    }
                parameter_data_list.append(parameter_data)
            
            # Initialize result storage
            
            final_parameters = [0] * len(resource_parameters)
            sigma_star_list = [0] * len(resource_parameters)
            
            if parallel and len(resource_parameters) > 1:
                # Parallel processing
                logging.info(f"Using parallel processing with {max_workers} workers for {len(resource_parameters)} parameters")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_data = {
                            executor.submit(self._compute_single_bifurcation_parameter, param_data): param_data
                            for param_data in parameter_data_list
                        }
                        
                        # Collect results as they complete
                        completed_count = 0
                        for future in as_completed(future_to_data):
                            param_data = future_to_data[future]
                            try:
                                if direct_method or second_run:
                                    param_id, critical_parameters, sigma_star = future.result()
                                    final_parameters[param_id] = critical_parameters
                                    sigma_star_list[param_id] = sigma_star
                                else: 
                                    param_id, critical_param, sigma_star, final_pos_row = future.result()
                                    final_parameters[param_id] = critical_param
                                    sigma_star_list[param_id] = sigma_star
                                    print(final_pos_row)
                                completed_count += 1
                                if completed_count % max(1, len(resource_parameters) // 10) == 0:
                                    logging.info(f"Completed {completed_count}/{len(resource_parameters)} bifurcation parameters")
                                    
                            except Exception as e:
                                logging.error(f"Error processing parameter {param_data['resource_parameter_id']}: {str(e)}")
                                # Set default values for failed computation
                                param_id = param_data['resource_parameter_id']
                                if direct_method:
                                    final_parameters[param_id] = {'unstable_flip': [], 'stable_flip': []}
                                else:
                                    final_parameters[param_id] = 0.0
                                sigma_star_list[param_id] = 0.0
                        
                except Exception as e:
                    logging.error(f"Parallel processing failed: {str(e)}")
                    logging.info("Falling back to sequential processing")
                    parallel = False
            
            if not parallel:
                # Sequential processing (fallback or by choice)
                logging.info("Using sequential processing")
                
                for resource_parameter_id, param_data in enumerate(parameter_data_list):
                    try:
                        if direct_method or second_run:
                            param_id, critical_parameters, sigma_star = self._compute_single_bifurcation_parameter(param_data)
                            final_parameters[param_id] = critical_parameters
                            sigma_star_list[param_id] = sigma_star
                        else:
                            print(alphas[resource_parameter_id])
                            param_id, critical_param, sigma_star, final_pos_row = self._compute_single_bifurcation_parameter(param_data)
                            final_parameters[param_id] = critical_param
                            sigma_star_list[param_id] = sigma_star
                            print(final_pos_row)
                        print(f'{resource_parameter_id} complete, learning rate: {learning_rate_p}')
                        
                        # Progress reporting
                        if (resource_parameter_id + 1) % max(1, len(resource_parameters) // 10) == 0:
                            logging.info(f"Completed {resource_parameter_id + 1}/{len(resource_parameters)} parameters")
                            
                    except Exception as e:
                        logging.error(f"Error processing parameter {resource_parameter_id}: {str(e)}")
                        if direct_method:
                            final_parameters[resource_parameter_id] = {'unstable_flip': [], 'stable_flip': []}
                        else:
                            final_parameters[resource_parameter_id] = 0.0
                        sigma_star_list[resource_parameter_id] = 0.0
            
            logging.info(f"Successfully computed bifurcation parameters for all {len(resource_parameters)} parameters")
            
        except Exception as e:
            logging.error(f"Critical error in find_second_order_bifs: {str(e)}")
            raise
            
        finally:
            # Restore original state
            try:
                self.agents_pos = og_pos
                self.field.agents_pos = og_field_pos
                self.resource_distribution = og_resource_dist
                self.learning_rate = og_learning_rate
            except Exception as e:
                logging.warning(f"Error restoring original state: {str(e)}")
        if second_run:
            data['second_run'] = np.array(final_parameters)
            return data
        else:
            return {'sigma_star': sigma_star_list, 'final_parameters': final_parameters}

    def _compute_single_bifurcation_parameter(self, parameter_data: Dict) -> Tuple[int, float, float]:
        """
        Helper function to compute bifurcation parameters for a single resource parameter.
        Designed to be used with multiprocessing.
        
        :param parameter_data: Dictionary containing parameter data and configuration
        :type parameter_data: Dict
        
        :return: Tuple of parameter_id, critical_parameter, and sigma_star
        :rtype: Tuple[int, float, float]
        """
        sigma_indicator=0
        try:
            # Extract parameters
            resource_parameter_id = parameter_data['resource_parameter_id']
            resource_param = parameter_data['resource_param']
            bin_points = parameter_data['bin_points']
            resource_distribution_type = parameter_data['resource_distribution_type']
            num_agents = parameter_data['num_agents']
            learning_rate = parameter_data['learning_rate']
            subtract_val = parameter_data['subtract_val']
            time_steps = parameter_data['time_steps']
            tolerance = parameter_data['tolerance']
            tolerated_agents = parameter_data['tolerated_agents']
            total_params = parameter_data['total_params']
            pos = parameter_data['pos']
            sigma_star_i=parameter_data['sigma_star']
            sigma_star_bool=parameter_data.get('sigma_star_bool',False)
            direct_method=parameter_data.get('direct_method', True)  # Default to True for backward compatibility

            # Generate resource distribution
            resource_distribution = torch.tensor(rd.resource_distribution_choice(
                bin_points=bin_points,
                resource_type=resource_distribution_type,
                resource_parameters=resource_param
            ))
            
            # Calculate sigma_star
            if sigma_star_bool == False:
                if self.infl_type== 'gaussian':
                    sigma_star = np.sqrt((num_agents-2)/(num_agents-1)) * torch.sqrt(
                        general.discrete_variance(
                            bin_points=torch.tensor(bin_points),
                            resource_distribution=resource_distribution,
                            mean=general.discrete_mean(
                                bin_points=torch.tensor(bin_points),
                                resource_distribution=resource_distribution
                            )
                        )
                    )
                elif self.infl_type=='beta':

                    # Define the equation to solve: left_side(x) - right_side = 0
                    # Note: scipy.optimize and scipy.special.digamma expect float inputs/outputs
                    sigma_star=beta.sigma_star(num_agents=num_agents,bin_points=self.bin_points,resource_distribution=resource_distribution,parameter_instance=None,nash=0.5)
            else:
                sigma_indicator=1
                # copy or clone based on dtype
                if not torch.is_tensor(sigma_star_i):
                    sigma_star = torch.tensor(sigma_star_i, dtype=torch.float32)
                else:
                    sigma_star = sigma_star_i.clone()


            # Create temporary field environment for computation
            if self.domain_type == 'simplex':
                temp_field = AdaptiveEnv(
                    num_agents=num_agents,
                    agents_pos=pos.clone(),
                    parameters=torch.tensor([0.1]*num_agents),  # Will be overwritten
                    bin_points=torch.tensor(bin_points),
                    resource_distribution=resource_distribution,
                    infl_configs={'infl_type': self.infl_type},
                    learning_rate_type=self.learning_rate_type,
                    infl_fshift=self.infl_fshift,
                    Q=self.Q,
                    tolerance=tolerance,
                    tolerated_agents=tolerated_agents,
                    time_steps=time_steps,
                    domain_type=self.domain_type,
                    learning_rate=learning_rate
                )
                # Set corners for simplex domain
                temp_field.corners = self.corners
            else:
                temp_field = AdaptiveEnv(
                    num_agents=num_agents,
                    agents_pos=pos.clone(),
                    parameters=torch.tensor([3]*num_agents),  # Will be overwritten
                    bin_points=torch.tensor(bin_points),
                    resource_distribution=resource_distribution,
                    infl_configs={'infl_type': self.infl_type},
                    learning_rate_type=self.learning_rate_type,
                    infl_fshift=self.infl_fshift,
                    Q=self.Q,
                    tolerance=tolerance,
                    tolerated_agents=tolerated_agents,
                    time_steps=time_steps,
                    domain_type=self.domain_type,
                    learning_rate=learning_rate,
                    ignore_zero_infl=True
                )
            
            # Generate reach parameters
            if self.infl_type=='beta':
                reach_parameters = general.agent_parameter_setup(
                    num_agents=num_agents,
                    infl_type=self.infl_type,
                    setup_type="parameter_space",
                    reach_start=sigma_star - subtract_val,
                    reach_end=max(sigma_star - .30, 0.001),
                    reach_num_points=100
                )
            else:
                reach_parameters = general.agent_parameter_setup(
                    num_agents=num_agents,
                    infl_type=self.infl_type,
                    setup_type="parameter_space",
                    reach_start=sigma_star - subtract_val,
                    reach_end=max(sigma_star - .3, 0.01),
                    reach_num_points=200 
                )
            _stability_mask=[]
            if direct_method:
                
                def _stability_function(alpha, parameter_instance):
                    agent_pos=torch.tensor([.5-alpha]+[.5]*(num_agents-2)+[alpha+.5],dtype=torch.float32)
                    grad_temp= temp_field.gradient_function(agents_pos=agent_pos, parameter_instance=parameter_instance,two_a=True)
                    return grad_temp[0]
                # for each parameter in reach_parameters, find root of the gradient using brentq method
                from scipy.optimize import brentq
                crticial_alpha=[]
                sym_positions=[]
                for param in reach_parameters:
                    root_finder_function = lambda x: _stability_function(x,parameter_instance=param)
                    try:
                        root=brentq(root_finder_function,0.00001,.49998)
                        crticial_alpha.append(root) 
                        sym_positions.append(torch.tensor([.5-root]+[.5]*(num_agents-2)+[root+.5],dtype=torch.float32))
                    except Exception as e:
                        print('error in brentq:', str(e))
                        root=.5
                        crticial_alpha.append(root)
                        sym_positions.append(torch.tensor([.5-root]+[.5]*(num_agents-2)+[root+.5],dtype=torch.float32))
                        continue
            # Find critical parameter using direct stability stop logic
            import InflGame.adaptive.jacobian as jc
            
            # Initialize variables for stability checking
            stopped_early = False
            stop_index = -1
            instability_position = None
            critical_parameter = [reach_parameters[-1]]  # Default to last parameter
            final_pos_row = pos.clone()  # Initialize with starting position as default
            # Loop through reach parameters to find instability
            for parameter_id, reach_param in enumerate(reach_parameters):
                try:
                    # Set parameters based on domain type
                    if self.domain_type in ['1d']:
                        current_params = torch.tensor(reach_param, dtype=torch.float32)
                        temp_field.parameters = np.array(reach_param)
                    elif self.domain_type in ['2d', 'simplex']:
                        temp_field.parameters = torch.tensor(reach_param).clone()
                        current_params = torch.tensor(reach_param).clone()
                    
                    if direct_method:
                        final_pos_row=sym_positions[parameter_id]
                    else:
                        #use gradient ascent method
                        # Reset temporary field state
                        temp_field.pos_matrix = 0
                        temp_field.agents_pos = pos.clone()
                        
                        # Run gradient ascent
                        temp_field.gradient_ascent(show_out=False)
                        
                        # Extract final position
                        if self.domain_type == 'simplex':
                            final_pos_row = simplex_utils.ba2xy_vectorized(
                                barycentric_coords=temp_field.pos_matrix[-1].clone(),
                                corners=self.corners
                            )
                        else:
                            final_pos_row = temp_field.pos_matrix[-1].clone()
                    
                    # Set agents position for jacobian calculation (using temporary field)
                    temp_field.agents_pos = final_pos_row.clone()
                    
                        
                    try:
                        jacobian_matrix = jc.jacobian_matrix(
                            num_agents=num_agents,
                            parameters=current_params,
                            agents_pos=final_pos_row,
                            bin_points=torch.tensor(bin_points),
                            resource_distribution=resource_distribution,
                            infl_type=self.infl_type,
                            infl_fshift=self.infl_fshift,
                            Q=self.Q,
                            infl_matrix=temp_field.influence_matrix(parameter_instance=current_params),
                            prob_matrix=temp_field.prob_matrix(parameter_instance=current_params),
                            d_lnf_matrix=temp_field.d_lnf_matrix(parameter_instance=current_params),
                            x=final_pos_row,
                        )
                        
                        eigenvalues = torch.linalg.eigvals(jacobian_matrix)
                        real_parts = torch.real(eigenvalues)
                        
                        # Check for positive real parts (instability)
                        has_positive = torch.any(real_parts > 0).item()
                        # Check stopping condition - instability detected
                        if direct_method or sigma_indicator==1:
                            
                            
                            if num_agents==3:
                                # append list false (0) and true (1)
                                _stability_mask.append(int(has_positive))
                            
                            if sigma_indicator==1 and len(torch.unique(final_pos_row))==2:
                                print('skipping jacobian calculation at sigma star:', sigma_star.item())
                                _stability_mask.append(1)
                                break
                            else:
                                _stability_mask.append(int(torch.sum(real_parts > 0).item()>1))
                                if sigma_indicator==1 and torch.sum(real_parts > 0).item()>1:
                                    break
                                 
                        else:    
                            if has_positive:
                                stopped_early = True
                                stop_index = parameter_id
                                instability_position = final_pos_row
                                critical_parameter = [reach_param]
                                break
                        
                    except Exception as e:
                        # Continue to next parameter if jacobian calculation fails
                        continue
                    
                        
                except Exception as e:
                    # Continue to next parameter if any other error occurs
                    continue
            
            # No need to restore original state since we used a temporary field
            if direct_method or sigma_indicator==1:
                # Find where the parameters switch from stable to unstable and vice versa
                stability_changes = np.diff(_stability_mask)
                
                # Unstable flip: 0 -> 1 (stable to unstable), change = +1
                unstable_flip_indices = np.where(stability_changes == 1)[0]
                # Stable flip: 1 -> 0 (unstable to stable), change = -1
                stable_flip_indices = np.where(stability_changes == -1)[0]
                
                # Get the parameter values at these flip points
                # Note: diff gives us the index of the value BEFORE the change
                # So the flip happens between index i and i+1
                unstable_flip_params = [float(reach_parameters[i, 0]) for i in unstable_flip_indices] if len(unstable_flip_indices) > 0 else []
                stable_flip_params = [float(reach_parameters[i, 0]) for i in stable_flip_indices] if len(stable_flip_indices) > 0 else []
                
                if sigma_indicator==1 and len(unstable_flip_params)==0:
                    unstable_flip_params=[sigma_star.item()]
                critical_parameters = {
                    'unstable_flip': unstable_flip_params,
                    'stable_flip': stable_flip_params
                }
                if sigma_indicator==1:
                    return resource_parameter_id, [critical_parameters['unstable_flip'][i] for i in range(len(critical_parameters['unstable_flip']))], sigma_star.item()
                else:
                    return resource_parameter_id, critical_parameters, sigma_star.item()
            else:
                return resource_parameter_id, critical_parameter[0], sigma_star.item(), final_pos_row.clone()

        except Exception as e:
            logging.error(f"Error computing bifurcation parameter {resource_parameter_id}: {str(e)}")
            raise RuntimeError(f"Failed to compute bifurcation parameter {resource_parameter_id}: {str(e)}")

    def find_third_order_bifurcations_refined(self,
                                              int_position,
                                              second_order_bif,
                                              guess_distance,
                                              sig_edge,
                                              num_refinements,
                                              learning_rate_p,
                                              resource_distribution_type,
                                              varying_parameter_type,
                                              alpha_st,
                                              alpha_end,
                                              alpha_num_points,
                                              fixed_parameters_lst,
                                              learning_rate_type: str = None,
                                              method_type: str = 'bottom_up',
                                              parallel: bool = True,
                                              max_workers: Optional[int] = None,
                                              batch_size: Optional[int] = None,
                                              time_steps: int = 5000,
                                              verbose: bool = True) -> List[torch.Tensor]:
        """
        Find third-order bifurcation parameters using optimized parallel processing.
        
        This method processes multiple resource parameters in parallel to find cycle ends
        for third-order bifurcation analysis. Optimized following project patterns with:
        - Parallel processing support using ProcessPoolExecutor
        - Proper state management and error handling
        - Memory efficient processing with batch support
        - Comprehensive logging and progress tracking
        
        :param int_position: Initial position for agents
        :type int_position: torch.Tensor
        :param second_order_bif: Second-order bifurcation parameters for each resource parameter
        :type second_order_bif: List[torch.Tensor]
        :param guess_distance: Distance for initial guess estimation
        :type guess_distance: torch.Tensor
        :param sig_edge: Minimum sigma value constraint
        :type sig_edge: float
        :param num_refinements: Number of refinement iterations
        :type num_refinements: int
        :param learning_rate_p: Learning rate parameters [start, end, decay]
        :type learning_rate_p: List[float]
        :param resource_distribution_type: Type of resource distribution
        :type resource_distribution_type: str
        :param varying_parameter_type: Type of parameter variation
        :type varying_parameter_type: str
        :param alpha_st: Starting alpha value
        :type alpha_st: float
        :param alpha_end: Ending alpha value
        :type alpha_end: float
        :param fixed_parameters_lst: Fixed parameters for resource distribution
        :type fixed_parameters_lst: List[List[float]]
        :param bin_points: Discretization points for the domain
        :type bin_points: Union[List[float], np.ndarray]
        :param parallel: Whether to use parallel processing
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers
        :type max_workers: Optional[int]
        :param batch_size: Batch size for parallel processing
        :type batch_size: Optional[int]
        :param time_steps: Maximum time steps for gradient ascent
        :type time_steps: int
        :param verbose: Whether to show progress information
        :type verbose: bool
        
        :return: List of cycle end parameters for each resource parameter
        :rtype: List[torch.Tensor]
        
        :raises ValueError: If input parameters are invalid
        :raises RuntimeError: If computation fails
        """
        bin_points = self.bin_points
        if method_type not in ['bottom_up','top_down','top_down_n1','bottom_up_n1']:
            raise ValueError('Method type must be either "bottom_up", "top_down", "top_down_n1", or "bottom_up_n1"')
        if len(second_order_bif) == 0:
            raise ValueError("second_order_bif cannot be empty")

        if learning_rate_type == None:
            learning_rate_type = self.learning_rate_type

         # Validate input parameters    
        
        if num_refinements < 0:
            raise ValueError(f"num_refinements must be non-negative, got {num_refinements}")
            
        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, got {time_steps}")
        
        # Convert inputs to tensors with consistent float32 data type following project patterns
        # Convert int_position to tensor
        if not isinstance(int_position, torch.Tensor):
            int_position = torch.tensor(int_position, dtype=torch.float32)
        else:
            int_position = int_position.to(torch.float32)
        
        # Convert second_order_bif to list of tensors
        converted_second_order_bif = []
        for i, bif in enumerate(second_order_bif):
            if not isinstance(bif, torch.Tensor):
                converted_second_order_bif.append(torch.tensor(bif, dtype=torch.float32))
            else:
                converted_second_order_bif.append(bif.to(torch.float32)) 
        second_order_bif = converted_second_order_bif
        
        # Convert guess_distance to tensor
        if not isinstance(guess_distance, torch.Tensor):
            guess_distance = torch.tensor(guess_distance, dtype=torch.float32)
        else:
            guess_distance = guess_distance.to(torch.float32)
        
        # Convert fixed_parameters_lst to tensor format
        converted_fixed_parameters = []
        for param_list in fixed_parameters_lst:
            if not isinstance(param_list, torch.Tensor):
                converted_fixed_parameters.append(torch.tensor(param_list, dtype=torch.float32))
            else:
                converted_fixed_parameters.append(param_list.to(torch.float32))
        fixed_parameters_lst = converted_fixed_parameters
        
        # Generate resource parameters
        resource_parameters, _ = general.resource_parameter_setup(
            resource_distribution_type=resource_distribution_type,
            varying_parameter_type=varying_parameter_type,
            alpha_st=alpha_st, 
            alpha_end=alpha_end, 
            fixed_parameters_lst=fixed_parameters_lst,
            alpha_num_points=alpha_num_points
        )
        
        if len(resource_parameters) != len(second_order_bif):
            raise ValueError(f"Length mismatch: resource_parameters ({len(resource_parameters)}) != second_order_bif ({len(second_order_bif)})")
        
        # Generate learning rate schedule
        if self.infl_type == 'beta':
            t = np.linspace(0, 1, len(resource_parameters))
            # reverse learning rate schedule for beta influence
            learning_rates = learning_rate_p[1] + (learning_rate_p[0] - learning_rate_p[1]) * (-4 * t ** 2 + 4 * t)
        else:
            t = np.linspace(0, 1, len(resource_parameters))  
            learning_rates = learning_rate_p['min'] + (learning_rate_p['max'] - learning_rate_p['min']) * (t ** 2)
            #flip the order 
            #learning_rates=learning_rates[::-1]
            if 1==0:
                guess_distance= guess_distance+(.2-guess_distance)*(t**2)
            else:
                guess_distance= [guess_distance]*len(resource_parameters)
        if learning_rate_type=='inside_decay':
            learning_rates=learning_rate_p['outside_min'] + (learning_rate_p['outside_max'] - learning_rate_p['outside_min']) * (t ** 2)

         # Logging setup

        if verbose:
            logging.info(f"Finding third-order bifurcations for {len(resource_parameters)} parameters using domain '{self.domain_type}'")
        
        # Determine optimal batch size and workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(resource_parameters))
        
        if batch_size is None:
            batch_size = max(1, len(resource_parameters) // max_workers)
        
        # Store original state following project patterns
        og_pos = self.agents_pos.clone()
        og_field_pos = self.field.agents_pos.clone()
        og_tolerance = self.tolerance
        og_tolerated_agents = self.tolerated_agents
        
        try:
            # Prepare parameter data for parallel processing
            parameter_data_list = []
            for i in range(len(resource_parameters)):
                resource_distribution_tensor = torch.tensor(rd.resource_distribution_choice(
                    bin_points=bin_points,
                    resource_type=resource_distribution_type,
                    resource_parameters=resource_parameters[i]
                ))
                if learning_rate_type=='gradient_magnitude':
                   lr= [learning_rate_p['min_simulated'],learning_rates[i],learning_rate_p['period']]
                elif learning_rate_type=='cosine_annealing':
                   lr= [learning_rate_p['min_simulated'],learning_rates[i],learning_rate_p['period']]
                elif learning_rate_type=='exponential_growth':
                   lr=[learning_rates[i],learning_rates[i],learning_rate_p['period']]
                elif learning_rate_type=='inside_decay':
                   lr={'min':learning_rate_p['min'],'max':learning_rates[i],'period':learning_rate_p['period'],'min_simulated':learning_rate_p['min_simulated']}
                parameter_data = {
                    'parameter_id': i,
                    'int_position': int_position.clone(),
                    'bif_1': second_order_bif[i],
                    'guess_distance': guess_distance[i],
                    'min_sig': sig_edge,
                    'resource_distribution': resource_distribution_tensor,
                    'time_steps': time_steps,
                    'num_refinements': num_refinements,
                    'learning_rate': lr,
                    'num_agents': self.num_agents,
                    'bin_points': bin_points,
                    'infl_configs': self.infl_configs,
                    'learning_rate_type': 'cosine_annealing' if learning_rate_type == 'exponential_growth' else learning_rate_type,
                    'fp': self.fp,
                    'infl_cshift': self.infl_cshift,
                    'cshift': self.cshift,
                    'infl_fshift': self.infl_fshift,
                    'Q': self.Q,
                    'domain_type': self.domain_type,
                    'domain_bounds': self.domain_bounds,
                    'tolerance': self.tolerance,
                    'tolerated_agents': self.tolerated_agents,
                    'ignore_zero_infl': self.ignore_zero_infl,
                    'total_params': len(resource_parameters),
                    'method_type': method_type
                }
                parameter_data_list.append(parameter_data)
            
            # Initialize result storage
            parm_list = [None] * len(resource_parameters)
            
            if parallel and len(resource_parameters) > 1:
                # Parallel processing
                if verbose:
                    logging.info(f"Using parallel processing with {max_workers} workers for {len(resource_parameters)} parameters")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_data = {
                            executor.submit(self._compute_single_third_order_parameter, param_data): param_data
                            for param_data in parameter_data_list
                        }
                        
                        # Collect results as they complete
                        completed_count = 0
                        for future in as_completed(future_to_data):
                            try:
                                param_id, cycle_end_result = future.result()
                                parm_list[param_id] = cycle_end_result
                                completed_count += 1
                                
                                if verbose and completed_count % max(1, len(resource_parameters) // 10) == 0:
                                    logging.info(f"Completed {completed_count}/{len(resource_parameters)} third-order bifurcation parameters")
                                    
                            except Exception as e:
                                param_data = future_to_data[future]
                                param_id = param_data['parameter_id']
                                logging.error(f"Error processing parameter {param_id}: {str(e)}")
                                # Set default result for failed computation
                                parm_list[param_id] = torch.tensor([sig_edge] * self.num_agents)
                        
                        if verbose:
                            logging.info(f"Parallel processing completed: {completed_count}/{len(resource_parameters)} successful")
                        
                except Exception as e:
                    logging.error(f"Parallel processing failed: {str(e)}")
                    logging.info("Falling back to sequential processing")
                    parallel = False
            
            if not parallel:
                # Sequential processing (fallback or by choice)
                if verbose:
                    logging.info("Using sequential processing")
                
                for i, param_data in enumerate(parameter_data_list):
                    try:
                        print(param_data['learning_rate'])
                        param_id, cycle_end_result = self._compute_single_third_order_parameter(param_data)
                        parm_list[param_id] = cycle_end_result
                        
                        if verbose and (i + 1) % max(1, len(resource_parameters) // 10) == 0:
                            logging.info(f"Completed {i + 1}/{len(resource_parameters)} third-order bifurcation parameters")
                            
                    except Exception as e:
                        logging.error(f"Error processing parameter {i}: {str(e)}")
                        # Set default result for failed computation
                        parm_list[i] = torch.tensor([sig_edge] * self.num_agents)
            
            if verbose:
                logging.info(f"Successfully computed third-order bifurcations for all {len(resource_parameters)} parameters")
            
        except Exception as e:
            logging.error(f"Critical error in find_third_order_bifurcations_refined: {str(e)}")
            raise
            
        finally:
            # Restore original state following project patterns
            try:
                self.agents_pos = og_pos
                self.field.agents_pos = og_field_pos
                self.tolerance = og_tolerance
                self.tolerated_agents = og_tolerated_agents
            except Exception as e:
                logging.warning(f"Error restoring original state: {str(e)}")
        
        return parm_list

    def _internal_learning_rate_schedule(self, learning_rate_p: Dict, estimate) -> List[float]:
        # Increases learning rate as estimate increases
        rate=10**(estimate*10-5)
        rate = max(rate, learning_rate_p['min'])
        rate = min(rate, learning_rate_p['max'])
        return [rate,rate,learning_rate_p['period']]
    
    def _compute_single_third_order_parameter(self, parameter_data: Dict) -> Tuple[int, torch.Tensor]:
        """
        Helper function to compute cycle ends for a single third-order bifurcation parameter.
        Designed to be used with multiprocessing for the find_third_order_bifurcations_refined function.
        
        This method creates a temporary AdaptiveEnv instance and calls find_cycle_ends
        to compute the third-order bifurcation parameter.
        
        :param parameter_data: Dictionary containing parameter data and configuration
        :type parameter_data: Dict
        
        :return: Tuple of parameter_id and cycle_end_result
        :rtype: Tuple[int, torch.Tensor]
        
        :raises RuntimeError: If computation fails
        """
        try:
            # Extract parameters from the data dictionary
            parameter_id = parameter_data['parameter_id']
            int_position = parameter_data['int_position']
            bif_1 = parameter_data['bif_1']
            guess_distance = parameter_data['guess_distance']
            min_sig = parameter_data['min_sig']
            resource_distribution = parameter_data['resource_distribution']
            time_steps = parameter_data['time_steps']
            num_refinements = parameter_data['num_refinements']
            learning_rate = parameter_data['learning_rate']
            
            # Ensure tensor inputs have consistent float32 data type following project patterns
            # Convert int_position to proper tensor format
            if not isinstance(int_position, torch.Tensor):
                int_position = torch.tensor(int_position, dtype=torch.float32)
            else:
                int_position = int_position.to(torch.float32)
                
            # Convert bif_1 to proper tensor format
            if not isinstance(bif_1, torch.Tensor):
                bif_1 = torch.tensor(bif_1, dtype=torch.float32)
            else:
                bif_1 = bif_1.to(torch.float32)
                
            # Convert guess_distance to proper tensor format
            if not isinstance(guess_distance, torch.Tensor):
                guess_distance = torch.tensor(guess_distance, dtype=torch.float32)
            else:
                guess_distance = guess_distance.to(torch.float32)
                
            # Convert resource_distribution to proper tensor format
            if not isinstance(resource_distribution, torch.Tensor):
                resource_distribution = torch.tensor(resource_distribution, dtype=torch.float32)
            else:
                resource_distribution = resource_distribution.to(torch.float32)
            
            # Environment configuration parameters
            num_agents = parameter_data['num_agents']
            bin_points = parameter_data['bin_points']
            infl_configs = parameter_data['infl_configs']
            learning_rate_type = parameter_data['learning_rate_type']
            if learning_rate_type=='inside_decay':
                learning_rate_type='cosine_annealing'
                inside_decay=True
                learning_rate_p=parameter_data['learning_rate']
                learning_rate=[learning_rate_p['min_simulated'],learning_rate_p['min_simulated'],learning_rate_p['period']]
            else:
                inside_decay=False
            fp = parameter_data['fp']
            infl_cshift = parameter_data['infl_cshift']
            cshift = parameter_data['cshift']
            infl_fshift = parameter_data['infl_fshift']
            Q = parameter_data['Q']
            domain_type = parameter_data['domain_type']
            domain_bounds = parameter_data['domain_bounds']
            tolerance = parameter_data['tolerance']
            tolerated_agents = parameter_data['tolerated_agents']
            ignore_zero_infl = parameter_data['ignore_zero_infl']
            total_params = parameter_data['total_params']
            method_type = parameter_data.get('method_type', 'bottom_up')
            
            # Create a temporary AdaptiveEnv field for this computation
            # Following project patterns for temporary environment creation
            if domain_type == 'simplex':
                temp_field = AdaptiveEnv(
                    num_agents=num_agents,
                    agents_pos=int_position.clone(),
                    parameters=torch.tensor([0.1] * num_agents, dtype=torch.float32),  # Will be overwritten
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
                    tolerance=tolerance,
                    tolerated_agents=tolerated_agents,
                    ignore_zero_infl=ignore_zero_infl
                )
            else:
                temp_field = AdaptiveEnv(
                    num_agents=num_agents,
                    agents_pos=int_position.clone(),
                    parameters=torch.tensor([0.1] * num_agents, dtype=torch.float32),  # Will be overwritten
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
                    tolerated_agents=tolerated_agents,
                    ignore_zero_infl=ignore_zero_infl
                )
            
            # Implement find_cycle_ends logic directly in the helper function
            # This avoids creating an entire bif environment and is more efficient
            estimate = torch.max(bif_1 - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
            if inside_decay:
                if estimate[0] < .1:
                    temp_field.learning_rate = [learning_rate_p['min_simulated'],learning_rate_p['min_simulated'],learning_rate_p['period']]
                if estimate[0] >= .1:
                    # linearly increase the learning rate from learning_rate['min']
                    temp_field.learning_rate = self._internal_learning_rate_schedule(learning_rate_p=learning_rate_p, estimate=estimate[0])
            converged = False
            tracker = 0
            if method_type in ['top_down', 'top_down_n1']:
                converged = True
                while converged:
                    tracker += 1
                    temp_field.parameters = estimate
                    temp_field.agents_pos = int_position.clone()
                    temp_field.gradient_ascent(show_out=False)
                    unique_vals = torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))
                    print(f'Parameter ID {parameter_id}, Tracker {tracker}, Unique Positions: {unique_vals},len_pos: {len(temp_field.pos_matrix)}')
                    if len(temp_field.pos_matrix) < time_steps:
                        if method_type=='top_down_n1':
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                                estimate = torch.max(estimate - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
                                if estimate[0].item()<=min_sig:
                                    converged=False
                            else:
                                converged = False
                        else:
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                                converged = False
                            else:
                                estimate = torch.max(estimate - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
                                if estimate[0].item()<=min_sig:
                                    converged=False
                    else:
                        
                        converged=False
                    
                    if inside_decay:
                        if estimate[0] < .1:
                            temp_field.learning_rate = [learning_rate_p['min_simulated'],learning_rate_p['min_simulated'],learning_rate_p['period']]
                        if estimate[0] >= .1:
                            # linearly increase the learning rate from learning_rate['min']
                            temp_field.learning_rate = self._internal_learning_rate_schedule(learning_rate_p=learning_rate_p, estimate=estimate[0])
                    # Reset for next iteration
                    temp_field.pos_matrix = 0
                for refinement in range(num_refinements):
                    if method_type == 'top_down_n1':
                        estimate_new = bif_1 - 1 * (bif_1 - estimate) / 3
                    else:
                        estimate_new = bif_1 - 1 * (bif_1 - estimate) / 3
                    
                    if inside_decay:
                        if estimate_new[0] < .1:
                            temp_field.learning_rate = [learning_rate_p['min_simulated'],learning_rate_p['min_simulated'],learning_rate_p['period']]
                        if estimate_new[0] >= .1:
                            # linearly increase the learning rate from learning_rate['min']
                            temp_field.learning_rate = self._internal_learning_rate_schedule(learning_rate_p=learning_rate_p, estimate=estimate_new[0])

                    temp_field.parameters = estimate_new
                    temp_field.agents_pos = int_position.clone()
                    temp_field.pos_matrix = 0  # Reset matrix
                    temp_field.gradient_ascent(show_out=False)
                    print(f'Refinement {refinement}, Parameter ID {parameter_id}, bif1: {bif_1}, estimate: {estimate}, Unique Positions: {torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=2))}, len_pos: {len(temp_field.pos_matrix)}')
                    if method_type=='top_down_n1':
                        if len(temp_field.pos_matrix) == time_steps:
                            # If estimate_new is approximately equal to estimate, break
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=3))) != 2:
                                if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                    estimate = estimate_new
                                    break
                                else:
                                    estimate = estimate_new
                            else:
                                if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                    bif_1 = estimate_new
                                    break
                                else:
                                    bif_1 = estimate_new
                        elif len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=3))) != 2:
                            # If estimate_new is approximately equal to estimate, break
                            if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                estimate = estimate_new
                                break
                            else:
                                estimate = estimate_new
                        else:
                            if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                bif_1 = estimate_new
                                break
                            else:
                                bif_1 = estimate_new
                    else:
                        if len(temp_field.pos_matrix) == time_steps:
                            # If estimate_new is approximately equal to estimate, break
                            if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                estimate = estimate_new
                                break
                            else:
                                estimate = estimate_new
                        elif len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                            # If estimate_new is approximately equal to estimate, break
                            if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                estimate = estimate_new
                                break
                            else:
                                estimate = estimate_new
                        else:
                            if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                bif_1 = estimate_new
                                break
                            else:
                                bif_1 = estimate_new
                cycle_end_result = estimate
                
                return parameter_id, cycle_end_result
            else:
                # Initial convergence loop
                while not converged:
                    tracker += 1
                    temp_field.parameters = estimate
                    temp_field.agents_pos = int_position.clone()
                    temp_field.gradient_ascent(show_out=False)
                    unique_vals = torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))
                    print(f'Parameter ID {parameter_id}, Tracker {tracker}, Unique Positions: {unique_vals}, len_pos: {len(temp_field.pos_matrix)}')
                    if len(temp_field.pos_matrix) < time_steps:
                        if method_type=='bottom_up_n1':
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                                converged = False
                                estimate = torch.max(estimate - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
                                if estimate[0].item()<=min_sig:
                                    converged=True
                            else:
                                converged = True
                        else:
                            converged = True
                    else:
                        if method_type=='bottom_up_n1':
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 3:
                                converged = True
                            else:
                                converged = False
                                estimate = torch.max(estimate - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
                                if estimate[0].item()<=min_sig:
                                    converged=True
                        else:
                            estimate = torch.max(estimate - guess_distance, torch.tensor([min_sig] * num_agents, dtype=torch.float32))
                            if estimate[0].item()<=min_sig:
                                converged=True
                    # Reset for next iteration
                    temp_field.pos_matrix = 0
                print(f'{parameter_id} complete')
                # Refinement loop
                for refinement in range(num_refinements):
                    if method_type == 'bottom_up_n1':
                        estimate_new = bif_1 - 1 * (bif_1 - estimate) / 2
                    else:
                        estimate_new = bif_1 - 3 * (bif_1 - estimate) / 4
                        
                    
                    temp_field.parameters = estimate_new
                    temp_field.agents_pos = int_position.clone()
                    temp_field.pos_matrix = 0  # Reset matrix
                    temp_field.gradient_ascent(show_out=False)
                    print(f'Refinement {refinement}, Parameter ID {parameter_id}, bif1: {bif_1}, estimate: {estimate}, Unique Positions: {torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))}, len_pos: {len(temp_field.pos_matrix)}')
                    if len(temp_field.pos_matrix) < time_steps:
                        # If estimate_new is approximately equal to estimate, break
                        if method_type=='bottom_up_n1':
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                                if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                    bif_1 = estimate_new
                                    break
                                else:
                                    bif_1 = estimate_new
                            else:
                                if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                    estimate = estimate_new
                                    break
                                else:
                                    estimate = estimate_new
                        else:
                            if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                estimate = estimate_new
                                break
                            else:
                                estimate = estimate_new
                    else:
                        if method_type=='bottom_up_n1':
                            if len(torch.unique(torch.round(temp_field.pos_matrix[-1], decimals=4))) == 2:
                                # If estimate_new is approximately equal to estimate, break
                                if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                    bif_1 = estimate_new
                                    break
                                else:
                                    bif_1 = estimate_new
                            else:
                                if torch.all(torch.abs(estimate_new - estimate) < 1e-5):
                                    estimate = estimate_new
                                    break
                                else:
                                    estimate = estimate_new
                        else:
                            if torch.all(torch.abs(bif_1 - estimate) < 1e-5):
                                bif_1 = estimate_new
                                break
                            else:
                                bif_1 = estimate_new
                
                cycle_end_result = estimate
                
                return parameter_id, cycle_end_result
            
        except Exception as e:
            parameter_id = parameter_data.get('parameter_id', -1)
            logging.error(f"Error computing third-order parameter {parameter_id}: {str(e)}")
            raise RuntimeError(f"Failed to compute third-order parameter {parameter_id}: {str(e)}")


    def first_order_bifurcation_plot(self,
                                     infl_type: str = 'gaussian',
                                     alpha_st: float = 0,
                                     alpha_end: float = 1,
                                     processed_data = None,
                                     alpha_values = None,
                                     cutoff_index = None,
                                     title_ads: List[str] = [],
                                     save: bool = False,
                                     name_ads: List[str] = [],
                                     font ={'default_size': 24, 'cbar_size': 16, 'title_size': 34, 'legend_size': 12, 'table_size':15,'label_size':10,'font_family': 'sans-serif',},
                                     save_types: List[str] = ['.png', '.svg'],
                                     paper_figure: dict= {'paper':False,'section':'3_2_6','figure_id':'bif_diag'}
                                     ) -> matplotlib.figure.Figure:
        r"""
            Plots the first-order bifurcation for agents over a range of alpha values (resource parameters) via func:`jacobian_stability_fast`. 
            
            Now supports both original format (Gaussian) and new beta format with processed stability flips.

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
            :param processed_data: Processed test results with 'unstable_flip', 'stable_flip', and optionally 'cycles_end'
            :type processed_data: Optional[dict]
            :param alpha_values: Alpha values array
            :type alpha_values: Optional[np.ndarray]
            :param cutoff_index: Index to cutoff the data
            :type cutoff_index: Optional[int]

            :return: The generated plot figure.
            :rtype: matplotlib.figure.Figure
            """


                                
        
        font['font.family'] = font.get('font_family', 'sans-serif')
        default_font_size = font.get('default_size', 12)
        title_font_size = font.get('title_size', 14)
        legend_font_size = font.get('legend_size', 12)
        mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
        mpl.rcParams['legend.fontsize'] = legend_font_size
        
        fig,ax=plt.subplots(figsize=(24, 16))
        ax.set_box_aspect(1)
        
        # Apply font settings
        plt.rcParams.update({'font.size': font['default_size'], 'font.family': font['font_family']})
        
        # Determine which format we're using
        # Original Gaussian case: using test['final_parameters'] as arrays
        if alpha_values is None:
                raise ValueError("alpha_values must be provided when using processed_data")
            
        if cutoff_index is None:
                cutoff_index = len(alpha_values)
        
        alpha = alpha_values[:cutoff_index]
        has_cycles = 'cycles_end' in processed_data and processed_data['cycles_end'] is not None
        if self.num_agents>=5:
            y = processed_data['sigma_star'][:cutoff_index]
            ax.plot(alpha, y, label=r'$\sigma^{*}_0$ exact', color='black')
            ax.fill_between(alpha, y, max(max(y), .3), where=(alpha <= alpha_end), color='#87CEEB', alpha=0.5)  # Sky blue for symmetric Nash
            ax.fill_between(alpha, 0, y,where=(alpha >= alpha_st), color="#388001", alpha=0.5)
        else:
            y = processed_data['sigma_star'][:cutoff_index]
            z = processed_data['unstable_flip'][:cutoff_index, 0]
            
            
            
            # Plot sigma_star and unstable flip
            ax.plot(alpha, y, label=r'$\sigma^{*}_0$ exact', color='black')
            ax.fill_between(alpha, y, max(max(y), .3), where=(alpha <= alpha_end), color='#87CEEB', alpha=0.5)  # Sky blue for symmetric Nash
            
            if infl_type == 'beta':
                # If stable flip exists, add it
                if processed_data['max_stable_len'] > 0:
                    w = processed_data['stable_flip'][:cutoff_index, 0]
                    ax.plot(alpha, w, label=r'$\sigma^{*}_1$ est.', color='black', linestyle='dashdot')
                    ax.fill_between(alpha, w, y, where=(alpha >= alpha_st), color='#FF6B6B', alpha=0.5)  # Coral red for 2-1
                    ax.fill_between(alpha, z, w, where=(alpha >= alpha_st), color='#9370DB', alpha=0.5)  # Medium purple for 1-1-1
                else:
                    ax.fill_between(alpha, 0, z, where=(alpha >= alpha_st), color='#FF6B6B', alpha=0.5)  # Coral red for 2-1
                ax.plot(alpha, z, label=r'$\sigma^{*}_2$ est.', color='black', linestyle='--')
                
                # Fill regions with improved colors
                if has_cycles:
                    # Handle both 1D and 2D arrays for cycles_end
                    cycles_data = processed_data['cycles_end']
                    if isinstance(cycles_data, np.ndarray) and cycles_data.ndim > 1:
                        w = cycles_data[:cutoff_index].flatten() if cycles_data.shape[1] == 1 else cycles_data[:cutoff_index, 0]
                    else:
                        w = np.array(cycles_data)[:cutoff_index]
                    ax.plot(alpha, w, label=r'$\sigma^{*}_3$ est.', color='black', linestyle='dotted')
                    ax.fill_between(alpha, w, z, where=(alpha >= alpha_st), color='#FFD700', alpha=0.5)  # Gold for cycle
                    ax.fill_between(alpha, 0, w, where=(alpha >= alpha_st), color='#FF6B6B', alpha=0.5)  # Coral red for 2-1
                
            
            else:
                if self.num_agents==4:
                    # Gaussian case: Fill regions with improved colors
                    ax.fill_between(alpha, y, z, where=(alpha >= alpha_st), color='#9370DB', alpha=0.5)
                    ax.plot(alpha, z, label=r'$\sigma^{*}_1$ est.', color='black', linestyle='--')
                    ax.fill_between(alpha, 0, z,where=(alpha >= alpha_st), color="#388001", alpha=0.5)
                    ax.fill_between(alpha, 0, y,where=(alpha >= alpha_st), facecolor='none', edgecolor="#090909", hatch='\\\\\\', linewidth=0)
                        
                else:
                    # Gaussian case: Fill regions with improved colors
                    ax.fill_between(alpha, y, z, where=(alpha >= alpha_st), color='#9370DB', alpha=0.5)
                    ax.plot(alpha, z, label=r'$\sigma^{*}_1$ est.', color='black', linestyle='--')
                    
                    if has_cycles:
                        # Handle both 1D and 2D arrays for cycles_end
                        cycles_data = processed_data['cycles_end']
                        if isinstance(cycles_data, np.ndarray) and cycles_data.ndim > 1:
                            w = cycles_data[:cutoff_index].flatten() if cycles_data.shape[1] == 1 else cycles_data[:cutoff_index, 0]
                        else:
                            w = np.array(cycles_data)[:cutoff_index]
                        ax.plot(alpha, w, label=r'$\sigma^{*}_2$ est.', color='black', linestyle='dotted')
                        ax.fill_between(alpha, w, z, where=(alpha >= alpha_st), color='#FFD700', alpha=0.5)  # Gold for cycle
                        ax.fill_between(alpha, 0, w, where=(alpha >= alpha_st), color='#FF6B6B', alpha=0.5)  # Coral red for 2-1
            
        
        plt.ylim(0, max(max(y), .3))
        
        # Legend patches with improved colors
        red_p = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=f'({self.num_agents-1},1) or (1,{self.num_agents-1})')
        yellow_p = mpatches.Patch(color='#FFD700', alpha=0.5, label='cycle')
        purple_p = mpatches.Patch(color='#9370DB', alpha=0.5, label=f'(1,{self.num_agents-2},1)')
        blue_p = mpatches.Patch(color='#87CEEB', alpha=0.5, label=r'sym nash')
        green_p = mpatches.Patch(color='#388001', alpha=0.5, label=r'Mixed results')
        black_p = mpatches.Patch(facecolor='white', edgecolor='#090909', hatch='\\\\\\', label='(2,2) dom')
        old_handles, labels = ax.get_legend_handles_labels()
        
        # Conditionally add legend patches based on what's plotted
        legend_handles = [blue_p]
        if infl_type == 'beta':
            legend_handles.append(red_p)
            legend_handles.append(purple_p)
            if has_cycles:
                legend_handles.append(yellow_p)
        else:
            if self.num_agents==4:
                legend_handles.append(black_p)
                legend_handles.append(purple_p)
                legend_handles.append(green_p)
            elif self.num_agents>=5:
                legend_handles.append(green_p)
            else:
                legend_handles.append(purple_p)
            if has_cycles:
                legend_handles.append(yellow_p)
                legend_handles.append(red_p)
            
            

        
        plt.legend(handles=legend_handles + old_handles)
        plt.ylabel("$\sigma$ (reach)", fontsize=font['default_size'])
        plt.xlabel(r"$\alpha$ (mode distance)", fontsize=font['default_size'])
        
        title = r"$(\alpha,\sigma)$ $x^{*}$ stability bifurication for " + str(self.num_agents) + " players"
        if len(title_ads) > 0:
            for title_additon in title_ads:
                title = title + " " + title_additon
        plt.title(title, fontsize=font['title_size'])
        plt.xlim(alpha_st, alpha[cutoff_index-1]) 
        plt.close()
        if save==True:
            file_names=data_management.data_final_name({'data_type':'plot',"plot_type":'bif_diagram','domain_type':self.domain_type,'num_agents':self.num_agents,'section':paper_figure['section'],'figure_id':paper_figure.get('figure_id','bif_diagram')},name_ads=name_ads,save_types=save_types,paper_figure=paper_figure['paper'])
            for file_name in file_names:
                fig.savefig(file_name,bbox_inches='tight')
            
        return fig ,ax
