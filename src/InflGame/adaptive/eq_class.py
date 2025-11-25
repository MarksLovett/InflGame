""".. module:: selfualization
   :synopsis: Provides selfualization tools for analyzing and understanding the dynamics of adaptive environments and agent interactions for influencer games.


Visualization Module
====================



This module provides selfualization tools for analyzing and understanding the dynamics o             raise RuntimeError("Infinite values detected in computed gradient")
                
            except Exception as e:
                if isinstance(e, (ValueError, RuntimeError)):
                    raise
                else:
                    raise RuntimeError(f"Gradient computation failed: {str(e)}") from e
            
            finally:
                # Always restore original environment state
                self.agents_pos = og_pos
                if self.infl_type == 'dirichlet':
                    self.alpha_matrix = og_alpha
            
            return grad
            
        except Exception as e:
            # Final catch-all with state restoration
            try:
                self.agents_pos = og_pos
                if self.infl_type == 'dirichlet':
                    self.alpha_matrix = og_alpha
            except:
                pass  # Don't override original error if restoration fails
            
            if isinstance(e, (ValueError, RuntimeError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in gradient_function: {str(e)}") from e

f adaptive environments and agent interactions for influencer games.
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

import matplotlib.figure
from typing import Union, List, Dict, Optional, Tuple
from collections import Counter


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



class search_env:
    """
    The Shell class provides a framework for simulating and selfualizing adaptive dynamics
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
        self.results_dict=None
        self.stats=None
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
   
    def monte_carlo_unit_hypercube(self, number_samples: int, seed: int) -> torch.Tensor:
        # Generate m random (x1, x2,...,xn) coordinates in [0, 1]^n
        torch.manual_seed(seed)
        samples = torch.rand(number_samples, self.num_agents)
        return samples

    def unit_cube_3d(self,resolution=10):
            """Generates a 3D unit cube as a torch tensor grid of points."""
            lin = torch.linspace(0, 1, resolution + 2)[1:-1]
            grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
            
            # Stack the coordinates to create 3D points
            points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
            
            # Return all points without filtering
            return points

    ## Non-monte carlo specific methods for equilibrium classification
    def detect_approximate_cycles(self,sequence, tolerance=0.05, min_period=10, max_period=None):
        """
        Detect approximate cycles by looking for similar patterns with some tolerance.
        This handles cases where cycles exist but aren't exactly identical.
        """
        if max_period is None:
            max_period = len(sequence) // 3
        
        data = np.array(sequence)
        best_period = None
        best_score = 0
        
        for period in range(min_period, min(max_period, len(data) // 2)):
            scores = []
            
            # Compare segments of this period length
            for start in range(len(data) - 2 * period):
                seg1 = data[start:start + period]
                seg2 = data[start + period:start + 2 * period]
                
                # Calculate normalized correlation
                if np.std(seg1) > 1e-6 and np.std(seg2) > 1e-6:
                    correlation = np.corrcoef(seg1, seg2)[0, 1]
                    if not np.isnan(correlation):
                        scores.append(abs(correlation))
            
            if scores:
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                
                # Combined score favoring consistency and strength
                combined_score = avg_score * max_score
                
                if combined_score > best_score and avg_score > 0.3:
                    best_score = combined_score
                    best_period = period
        
        return {
            'has_approximate_cycle': best_period is not None,
            'period': best_period,
            'correlation_score': best_score,
            'confidence': min(1.0, best_score) if best_period else 0
        }

    def _process_grid_point_worker(self, args):
        """Worker function for processing grid search equilibrium points."""
        point, shell_params, time_steps = args
        try:
            # Create a temporary AdaptiveEnv field directly
            temp_field = grad_func_env.AdaptiveEnv(
                num_agents=shell_params['num_agents'],
                agents_pos=shell_params['agents_pos'], 
                parameters=shell_params['parameters'],
                resource_distribution=shell_params['resource_distribution'],
                bin_points=shell_params['bin_points'],
                infl_configs=shell_params['infl_configs'],
                learning_rate_type=shell_params['learning_rate_type'],
                learning_rate=shell_params['learning_rate'],
                time_steps=shell_params['time_steps'],
                fp=shell_params['fp'],
                infl_cshift=shell_params['infl_cshift'],
                cshift=shell_params['cshift'],
                infl_fshift=shell_params['infl_fshift'],
                Q=shell_params['Q'],
                domain_type=shell_params['domain_type'],
                domain_bounds=shell_params['domain_bounds'],
                tolerance=shell_params['tolerance'],
                tolerated_agents=shell_params['tolerated_agents'],
                ignore_zero_infl=shell_params['ignore_zero_infl']
            )
            
            # Reset the field's agents positions for this run
            #if temp_field.num_agents != 3:
                # we need to add number of players -3 values equal to player 2 position
            #    extra_agents = temp_field.num_agents - 3
            #    point_extended = point.tolist() + [point[1].item()] * extra_agents
            #    temp_field.agents_pos = torch.tensor(point_extended)
            #else:
            temp_field.agents_pos = point.clone()
            
            temp_field.gradient_ascent()
            
            # Get the path data
            #if shell_params['num_agents'] != 3:
            #    pos_matrix = temp_field.pos_matrix[:, :3].numpy()  # Only take first 3 dimensions for plotting
            #else:
            pos_matrix = temp_field.pos_matrix.numpy()  # Convert to numpy for plotly
            
            converged = len(temp_field.pos_matrix) < time_steps
            
            result = {'converged': converged}
            if converged:
                converged_pos = pos_matrix[-1]
                result['pos_converged'] = converged_pos
            else:
                # Fast inline cycle detection using optimized approach
                # Use smaller, more recent sequence for faster processing
                sequence_len = min(500, len(pos_matrix) - 1000)  # Use last 500 points if available
                start_idx = max(1000, len(pos_matrix) - sequence_len)
                sequence = pos_matrix[start_idx:start_idx + sequence_len, 0]
                
                # Fast cycle detection using variance-based approach
                cycle_detected = False
                if len(sequence) > 50:  # Need minimum points for meaningful analysis
                    data = np.array(sequence)
                    
                    # Quick variance check - if variance is very low, likely converged (not cycling)
                    if np.var(data) > 1e-8:
                        # Use autocorrelation for fast period detection
                        # Only check reasonable period ranges
                        min_period = 5
                        max_period = min(50, len(data) // 4)  # Much smaller range
                        
                        best_score = 0
                        # Use step size to speed up search
                        for period in range(min_period, max_period, 2):  # Step by 2 for speed
                            if len(data) >= 3 * period:
                                # Fast correlation using just 2-3 segments instead of all overlapping segments
                                seg1 = data[:period]
                                seg2 = data[period:2*period] 
                                seg3 = data[2*period:3*period] if len(data) >= 3*period else None
                                
                                # Quick std check before expensive correlation
                                if np.std(seg1) > 1e-6 and np.std(seg2) > 1e-6:
                                    # Fast correlation calculation
                                    corr12 = np.corrcoef(seg1, seg2)[0, 1]
                                    if not np.isnan(corr12) and abs(corr12) > 0.5:  # Higher threshold for speed
                                        score = abs(corr12)
                                        
                                        # Check third segment if available for confirmation
                                        if seg3 is not None and np.std(seg3) > 1e-6:
                                            corr13 = np.corrcoef(seg1, seg3)[0, 1]
                                            if not np.isnan(corr13):
                                                score = (abs(corr12) + abs(corr13)) / 2
                                        
                                        if score > best_score and score > 0.6:  # Higher threshold
                                            best_score = score
                                            cycle_detected = True
                                            break  # Early exit on first good match
                
                if cycle_detected:
                    result['pos_converged'] = 'cycle'
            
            return result
                    
        except Exception as e:
            print(f"Error processing point {point}: {str(e)}")
            return None

    def search_Eq(self,resolution=None,number_samples=None,time_steps=10000,use_parallel=True,num_workers=None,use_monte_carlo=False,seed=42): 
        """
        Search for equilibrium positions from multiple starting points.
        Analyzes convergence behavior and detects cycles in the gradient ascent dynamics.
        
        Args:
            resolution (int): Resolution of the cube grid (points per dimension) or number of Monte Carlo samples
            time_steps (int): Maximum number of steps for gradient ascent
            use_parallel (bool): Whether to use parallel processing
            num_workers (int): Number of parallel workers (defaults to CPU count)
            use_monte_carlo (bool): Whether to use Monte Carlo sampling instead of regular grid
            seed (int): Random seed for Monte Carlo sampling (only used if use_monte_carlo=True)
            
        Returns:
            dict: Dictionary containing convergence results for each starting point
        """
    
        # Generate cube points
        if use_monte_carlo:
            if number_samples is None:
                number_samples = 1000  # Default number of samples
            cube_points = self.monte_carlo_unit_hypercube(number_samples=number_samples, seed=seed)
            print(f"Using Monte Carlo sampling with {number_samples} random points...")
            resolution=number_samples
        else:
            cube_points = self.unit_cube_3d(resolution=resolution)
            print(f"Using regular grid with {len(cube_points)} grid points...")
        
        print(f"Processing {len(cube_points)} starting points...")
        
        # Start timing
        start_time = time.time()
        results_dict = {}
        results_list = []

        if use_parallel and len(cube_points) > 1:
            # Parallel processing
            if num_workers is None:
                num_workers = min(mp.cpu_count(), len(cube_points))
            
            print(f"Using parallel processing with {num_workers} workers...")
            
            # Create shell parameters dictionary for workers
            def create_worker_args():
                worker_args = []
                shell_params = {
                    'num_agents': self.num_agents,
                    'agents_pos': self.agents_pos,
                    'parameters': self.parameters,
                    'resource_distribution': self.resource_distribution,
                    'bin_points': self.bin_points,
                    'infl_configs': self.infl_configs,
                    'learning_rate_type': self.learning_rate_type,
                    'learning_rate': self.learning_rate,
                    'time_steps': time_steps,
                    'fp': self.fp,
                    'infl_cshift': self.infl_cshift,
                    'cshift': self.cshift,
                    'infl_fshift': self.infl_fshift,
                    'Q': self.Q,
                    'domain_type': self.domain_type,
                    'domain_bounds': self.domain_bounds,
                    'tolerance': self.tolerance,
                    'tolerated_agents': self.tolerated_agents,
                    'ignore_zero_infl': self.ignore_zero_infl
                }
                for point in cube_points:
                    worker_args.append((point, shell_params, time_steps))
                return worker_args

            # Prepare worker arguments
            worker_args = create_worker_args()
            
            try:
                # Use ProcessPoolExecutor for better resource management
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    future_to_args = {
                        executor.submit(self._process_grid_point_worker, args): i 
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
                            if completed_count % max(1, len(cube_points) // 10) == 0:
                                print(f"Parallel progress: {completed_count}/{len(cube_points)} points ({completed_count/len(cube_points)*100:.1f}%)")
                                
                        except Exception as e:
                            arg_index = future_to_args[future]
                            print(f"Worker {arg_index} failed: {str(e)}")
                    
                    # Convert results back to ordered list
                    for i in range(len(cube_points)):
                        if i in completed_results:
                            results_list.append(completed_results[i])
                            results_dict[f'{i}'] = completed_results[i]
                            
            except Exception as e:
                print(f"Parallel processing failed: {str(e)}")
                print("Falling back to sequential processing...")
                use_parallel = False
        
        if not use_parallel or len(results_list) == 0:
            # Sequential processing (fallback or by choice)
            print("Using sequential processing...")

            temp_field = grad_func_env.AdaptiveEnv(num_agents=self.num_agents,agents_pos=self.agents_pos,parameters=self.parameters,
                                                 resource_distribution=self.resource_distribution,bin_points=self.bin_points,
                                                 infl_configs=self.infl_configs,learning_rate_type=self.learning_rate_type,learning_rate=self.learning_rate,time_steps=self.time_steps,fp=self.fp,infl_cshift=self.infl_cshift,cshift=self.cshift,
                                                 infl_fshift=self.infl_fshift,Q=self.Q,domain_type=self.domain_type,domain_bounds=self.domain_bounds,tolerance=self.tolerance,tolerated_agents=self.tolerated_agents,ignore_zero_infl=self.ignore_zero_infl) 
            # Process points sequentially
            temp_field.time_steps = time_steps 
            for i, point in enumerate(cube_points):
                try:            
                    # Reset the field's agents positions for this run
                    #if temp_field.num_agents != 3:
                    #    # we need to add number of players -3 values equal to player 2 position
                    #    extra_agents = temp_field.num_agents - 3
                    #    point_extended = point.tolist() + [point[1].item()] * extra_agents
                    #    temp_field.agents_pos = torch.tensor(point_extended)
                    #else:
                    temp_field.agents_pos = point.clone()
                    temp_field.gradient_ascent()
                    
                    # Get the path data
                    #if self.num_agents != 3:
                    #   pos_matrix= temp_field.pos_matrix[:, :3].numpy()  # Only take first 3 dimensions for plotting
                    #else:
                    pos_matrix = temp_field.pos_matrix.numpy()  # Convert to numpy for plotly
                    converged = len(temp_field.pos_matrix) < time_steps
                    if converged:
                        converged_pos = pos_matrix[-1]
                        results_dict[f'{i}'] = {'converged':converged,'pos_converged':converged_pos}
                    else:
                        cycle_test=self.detect_approximate_cycles(pos_matrix[2000:3000,0])
                        if cycle_test['has_approximate_cycle']:
                            results_dict[f'{i}'] = {'converged':converged,'pos_converged':'cycle'}

                    # Update progress periodically
                    if (i + 1) % 5 == 0 or (i + 1) == len(cube_points):
                        print(f"Sequential progress: {i+1}/{len(cube_points)} points ({(i+1)/len(cube_points)*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error processing point {i}: {str(e)}")
        
        # End timing and report
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        self.results_dict = results_dict
        
        return results_dict

    ## Non-class methods for equilibrium analysis
    def analyze_grid_search_results(self, results_dict=None, position_tolerance=1e-3):
        """
        Analyze grid search results to classify equilibrium types and provide statistics.
        
        Args:
            results_dict: Dictionary from grid_search_Eq containing convergence results
            position_tolerance: Tolerance for considering positions as "equal"
        
        Returns:
            Dictionary with statistics and classifications
        """
        if results_dict==None:
            results_dict = self.results_dict
            if results_dict is None:
                raise ValueError("No results_dict provided and no stored results found.")
        else:
            self.results_dict = results_dict
        total_points = len(results_dict)
        cycles = 0
        converged = 0
        no_result = 0
        
        equilibrium_types = []
        convergence_positions = []
        
        for key, result in results_dict.items():
            if result is None:
                no_result += 1
                continue
                
            if not result.get('converged', False):
                if result.get('pos_converged') == 'cycle':
                    cycles += 1
                else:
                    no_result += 1  # Didn't converge and no cycle detected
            else:
                converged += 1
                pos = result.get('pos_converged')
                if pos is not None:
                    convergence_positions.append(pos)
                    
                    # Classify equilibrium type based on position clustering
                    eq_type = self.classify_equilibrium_type(pos, position_tolerance)
                    equilibrium_types.append(eq_type)
        
        # Calculate percentages
        cycle_percentage = (cycles / total_points) * 100
        converged_percentage = (converged / total_points) * 100
        no_result_percentage = (no_result / total_points) * 100
        
        # Count equilibrium types
        eq_type_counts = Counter(equilibrium_types)
        eq_type_percentages = {eq_type: (count / converged) * 100 if converged > 0 else 0 
                            for eq_type, count in eq_type_counts.items()}
        
        self.stats={
            'total_points': total_points,
            'cycles': cycles,
            'converged': converged,
            'no_result': no_result,
            'cycle_percentage': cycle_percentage,
            'converged_percentage': converged_percentage,
            'no_result_percentage': no_result_percentage,
            'equilibrium_types': dict(eq_type_counts),
            'equilibrium_type_percentages': eq_type_percentages,
            'convergence_positions': convergence_positions
        }

        return self.stats

    def classify_equilibrium_type(self, positions, tolerance=1e-3):
        """
        Classify equilibrium type based on how many agents share similar positions,
        distinguishing between different spatial arrangements.
        
        Args:
            positions: Array of agent positions
            tolerance: Tolerance for considering positions as equal
        
        Returns:
            String describing equilibrium type with position ordering:
            - '(n)' : All agents together
            - '(n-1,1)' : n-1 agents grouped at lower position, 1 isolated higher
            - '(1,n-1)' : 1 agent isolated at lower position, n-1 grouped higher
            - 'c(1,n-2,1)' : Middle group centered around 0.5
            - 'l(1,n-2,1)' : Middle group below 0.5
            - 'u(1,n-2,1)' : Middle group above 0.5
            - etc.
        """
        positions = np.array(positions)
        n_agents = self.num_agents
        
        # Sort positions for easier comparison
        sorted_pos = np.sort(positions)
        
        # Group positions that are within tolerance
        groups = []
        group_positions = []  # Store mean position of each group
        current_group = [sorted_pos[0]]
        
        for i in range(1, len(sorted_pos)):
            if abs(sorted_pos[i] - current_group[-1]) <= tolerance:
                current_group.append(sorted_pos[i])
            else:
                groups.append(len(current_group))
                group_positions.append(np.mean(current_group))
                current_group = [sorted_pos[i]]
        
        # Don't forget the last group
        groups.append(len(current_group))
        group_positions.append(np.mean(current_group))
        
        # Now classify based on group structure and positions
        num_groups = len(groups)
        
        # Case 1: All agents at same position
        if num_groups == 1:
            return f'({n_agents})'
        
        # Case 2: Two groups - distinguish by which is larger and positions
        elif num_groups == 2:
            larger_group = max(groups)
            smaller_group = min(groups)
            
            # Find which group is at lower position
            if groups[0] > groups[1]:  # First (lower position) group is larger
                # (n-1, 1) pattern - most agents at lower position
                if larger_group == n_agents - 1:
                    return f'({n_agents-1},1)'
                else:
                    return f'({larger_group},{smaller_group})'
            else:  # Second (higher position) group is larger
                # (1, n-1) pattern - most agents at higher position
                if larger_group == n_agents - 1:
                    return f'(1,{n_agents-1})'
                else:
                    return f'({smaller_group},{larger_group})'
        
        # Case 3: Three groups - special case (1, n-2, 1) with position checks
        elif num_groups == 3:
            # Check if it's a (1, n-2, 1) pattern
            if groups[0] == 1 and groups[2] == 1:
                middle_group_size = groups[1]
                middle_position = group_positions[1]
                
                # Check position of middle group relative to 0.5
                if abs(middle_position - 0.5) < tolerance * 2:
                    # Middle group centered around 0.5
                    if middle_group_size == n_agents - 2:
                        return f'c(1,{n_agents-2},1)'
                    else:
                        return f'c(1,{middle_group_size},1)'
                elif middle_position < 0.5:
                    # Middle group below 0.5
                    if middle_group_size == n_agents - 2:
                        return f'l(1,{n_agents-2},1)'
                    else:
                        return f'l(1,{middle_group_size},1)'
                else:
                    # Middle group above 0.5
                    if middle_group_size == n_agents - 2:
                        return f'u(1,{n_agents-2},1)'
                    else:
                        return f'u(1,{middle_group_size},1)'
            else:
                # General three-group pattern, sorted by size
                sorted_groups = sorted(groups, reverse=True)
                return f'({",".join(map(str, sorted_groups))})'
        
        # Case 4: More than three groups - just list sizes
        else:
            sorted_groups = sorted(groups, reverse=True)
            return f'({",".join(map(str, sorted_groups))})'

    def print_results_summary(self):
        """
        Print a formatted summary of the analysis results with enhanced descriptions
        for position-aware equilibrium types.
        """
        if not hasattr(self, 'stats'):
            print("No analysis statistics available. Please run analyze_grid_search_results() first.")
            return
        print("="*70)
        print("GRID SEARCH EQUILIBRIUM ANALYSIS RESULTS")
        print("="*70)
        print(f"Total starting points analyzed: {self.stats['total_points']}")
        print()
        
        print("CONVERGENCE BEHAVIOR:")
        print(f"  Converged to equilibrium: {self.stats['converged']:4d} ({self.stats['converged_percentage']:5.1f}%)")
        print(f"  Cycling behavior:        {self.stats['cycles']:4d} ({self.stats['cycle_percentage']:5.1f}%)")
        print(f"  No clear result:         {self.stats['no_result']:4d} ({self.stats['no_result_percentage']:5.1f}%)")
        print()
        
        if self.stats['converged'] > 0:
            print("EQUILIBRIUM TYPES:")
            print("  Type                Count   Percentage   Description")
            print("  ----                -----   ----------   -----------")
            
            def get_description(eq_type_str):
                """Generate human-readable description from equilibrium type string."""
                
                # Handle special position-aware types with short prefixes
                if eq_type_str.startswith('l(') and eq_type_str.count(',') == 2:
                    # e.g., 'l(1,n-2,1)' or 'l(1,1,1)'
                    parts = eq_type_str[2:-1].split(',')  # Remove 'l(' and ')'
                    return f"{parts[0]} low, {parts[1]} grouped below 0.5, {parts[2]} high"
                
                elif eq_type_str.startswith('c(') and eq_type_str.count(',') == 2:
                    # e.g., 'c(1,n-2,1)' or 'c(1,1,1)'
                    parts = eq_type_str[2:-1].split(',')  # Remove 'c(' and ')'
                    return f"{parts[0]} low, {parts[1]} at center (~0.5), {parts[2]} high"
                
                elif eq_type_str.startswith('u(') and eq_type_str.count(',') == 2:
                    # e.g., 'u(1,n-2,1)' or 'u(1,1,1)'
                    parts = eq_type_str[2:-1].split(',')  # Remove 'u(' and ')'
                    return f"{parts[0]} low, {parts[1]} grouped above 0.5, {parts[2]} high"
                
                # Handle standard tuple notation
                elif eq_type_str.startswith('(') and eq_type_str.endswith(')'):
                    groups_str = eq_type_str.strip('()').split(',')
                    
                    if len(groups_str) == 1:
                        return f"All {groups_str[0]} agents at same position"
                    
                    elif len(groups_str) == 2:
                        # Two groups - determine spatial arrangement
                        size1, size2 = groups_str[0].strip(), groups_str[1].strip()
                        # Assume larger group determines position
                        try:
                            n1, n2 = int(size1) if size1.isdigit() else None, int(size2) if size2.isdigit() else None
                            if n1 and n2:
                                if n1 > n2:
                                    return f"{size1} agents at lower position, {size2} isolated higher"
                                else:
                                    return f"{size1} isolated at lower position, {size2} agents grouped higher"
                        except:
                            pass
                        return f"{size1} and {size2} agents in separate groups"
                    
                    # Build description for multiple groups
                    group_desc = []
                    for size in groups_str:
                        size = size.strip()
                        if size == '1':
                            group_desc.append('1 agent')
                        else:
                            group_desc.append(f'{size} agents together')
                    
                    return ', '.join(group_desc)
                
                # Fallback
                return eq_type_str
            
            # Sort by count (descending) for better readability
            sorted_types = sorted(self.stats['equilibrium_types'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for eq_type, count in sorted_types:
                percentage = self.stats['equilibrium_type_percentages'][eq_type]
                description = get_description(eq_type)
                # Adjust spacing for longer type names
                print(f"  {eq_type:18s}  {count:5d}   {percentage:8.1f}%   {description}")
        
        print("="*70)

    ## Plotting functions  

    def plot_equilibrium_analysis(self,
                                  results_dict: Optional[Dict] = None,
                                  plot_types: List[str] = ['convergence', 'distribution', 'percentage'],
                                  save: bool = False,
                                  name_ads: List[str] = [],
                                  save_types: List[str] = ['.png', '.svg'],
                                  paper_figure: Dict = {'paper': False, 'section': 'equilibrium_analysis', 'figure_id': 'equilibrium_analysis'},
                                  font: Dict = {
                                    'default_size': 12,
                                    'title_size': 14,
                                    'label_size': 11,
                                    'tick_size': 10,
                                    'font_family': 'sans-serif'
                                  },
                                  colors: Dict = {
                                    'converged': '#2E8B57',
                                    'cycling': '#FF6B6B',
                                    'no_result': '#95A5A6',
                                    'bar_distribution': '#3498DB',
                                    'bar_percentage': '#E74C3C'
                                  },
                                  text_configs: Dict = {
                                    'convergence_title': 'Convergence Behavior Distribution',
                                    'distribution_title': 'Equilibrium Types Distribution',
                                    'percentage_title': 'Equilibrium Types (% of Converged)',
                                    'xlabel_distribution': 'Equilibrium Type',
                                    'ylabel_distribution': 'Count',
                                    'xlabel_percentage': 'Equilibrium Type',
                                    'ylabel_percentage': 'Percentage (%)'
                                  },
                                  figsize: Optional[tuple] = None
                                                                ) -> matplotlib.figure.Figure:
        """
        Create comprehensive visualizations of equilibrium analysis results.
        
        Parameters
        ----------
        stats : Dict
            Statistics dictionary containing convergence results and equilibrium types.
        plot_types : List[str], optional
            List of plot types to include: 'convergence', 'distribution', 'percentage'.
            Default is all three.
        save : bool, optional
            Whether to save the figure. Default is False.
        name_ads : List[str], optional
            List of additional name components for file naming. Default is [].
        save_types : List[str], optional
            File extensions for saving. Default is ['.png', '.svg'].
        paper_figure : Dict, optional
            Dictionary with keys 'paper' (bool), 'section' (str), 'figure_id' (str).
            Controls paper-style naming and organization. Default is 
            {'paper': False, 'section': 'equilibrium_analysis', 'figure_id': 'equilibrium_analysis'}.
        font : Dict, optional
            Font configuration dictionary with keys: 'default_size', 'title_size',
            'label_size', 'tick_size', 'font_family'.
        colors : Dict, optional
            Color configuration for different plot elements.
        text_configs : Dict, optional
            Text configuration for titles and labels.
        figsize : Optional[tuple], optional
            Figure size. If None, automatically determined based on number of plots.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.
        
        Examples
        --------
        >>> # Show only convergence pie chart
        >>> fig = plot_equilibrium_analysis(stats, plot_types=['convergence'])
        >>> 
        >>> # Show distribution and percentage, save with custom naming
        >>> fig = plot_equilibrium_analysis(
        ...     stats,
        ...     plot_types=['distribution', 'percentage'],
        ...     save=True,
        ...     name_ads=['gaussian', 'n3'],
        ...     paper_figure={'paper': True, 'section': 'results', 'figure_id': 'eq_analysis'}
        ... )
        """

        font['font.family'] = font.get('font_family', 'sans-serif')
        default_font_size = font.get('default_size', 12)
        legend_font_size = font.get('legend_size', 12)
        mpl.rcParams.update({'font.size': default_font_size, 'font.family': font['font.family']})
        mpl.rcParams['legend.fontsize'] = legend_font_size
        if results_dict is None:
            results_dict = self.results_dict
            if results_dict is None:
                raise ValueError("No results_dict provided and no stored results found.")
        else:
            self.results_dict = results_dict

        
        self.analyze_grid_search_results(results_dict=results_dict)
        stats = self.stats
        # Validate plot_types
        valid_types = ['convergence', 'distribution', 'percentage']
        plot_types = [pt for pt in plot_types if pt in valid_types]
        
        if not plot_types:
            raise ValueError(f"No valid plot types specified. Valid types: {valid_types}")
        
        num_plots = len(plot_types)
        
        # Determine figure size - fixed subplot dimensions
        if figsize is None:
            # Each subplot gets equal size: 5x5 inches
            subplot_width = 5
            subplot_height = 5
            # Add spacing between subplots
            total_width = num_plots * subplot_width + (num_plots - 1) * 0.5
            figsize = (total_width, subplot_height)
        
        # Create figure with GridSpec for precise control
        fig = plt.figure(figsize=figsize)
        
        # Calculate subplot positions to ensure equal sizes
        subplot_width_ratio = 1.0 / num_plots
        axes = []
        for i in range(num_plots):
            # Each subplot gets equal width fraction of the figure
            left = i * subplot_width_ratio + 0.05 / num_plots
            bottom = 0.15
            width = subplot_width_ratio * 0.85
            height = 0.7
            ax = fig.add_axes([left, bottom, width, height])
            axes.append(ax)
        
        # Set font
        plt.rcParams['font.family'] = font.get('font_family', 'sans-serif')
        plt.rcParams['font.size'] = font.get('default_size', 12)
        
        plot_idx = 0
        
        # Plot 1: Convergence behavior (pie chart)
        if 'convergence' in plot_types:
            ax = axes[plot_idx]
            plot_idx += 1

            label_names = ['Converged', 'Cycling', 'No Result']
            sizes = [stats['converged'], stats['cycles'], stats['no_result']]
            pie_colors = [colors['converged'], colors['cycling'], colors['no_result']]
            
            wedges, texts = ax.pie(
                sizes, startangle=90,
                colors=pie_colors,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
            )
            
            # Custom label positioning with percentages
            self._add_custom_pie_labels(ax, wedges, label_names, sizes, font)
            
            # Fix axis limits to prevent resizing when labels change
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_aspect('equal')
            
            ax.set_title(
                text_configs.get('convergence_title', 'Convergence Behavior Distribution'),
                fontsize=font.get('title_size', 14),
                fontweight='bold'
            )
        
        # Plot 2: Equilibrium types distribution (bar chart)
        if 'distribution' in plot_types:
            ax = axes[plot_idx]
            plot_idx += 1
            
            if stats['equilibrium_types']:
                # Sort equilibrium types by count (descending) for better readability
                sorted_items = sorted(stats['equilibrium_types'].items(), 
                                    key=lambda x: x[1], reverse=True)
                eq_types = [item[0] for item in sorted_items]
                eq_counts = [item[1] for item in sorted_items]
                
                bars = ax.bar(range(len(eq_types)), eq_counts, color=colors['bar_distribution'], alpha=0.7)
                ax.set_title(
                    text_configs.get('distribution_title', 'Equilibrium Types Distribution'),
                    fontsize=font.get('title_size', 14),
                    fontweight='bold'
                )
                ax.set_xlabel(
                    text_configs.get('xlabel_distribution', 'Equilibrium Type'),
                    fontsize=font.get('label_size', 11)
                )
                ax.set_ylabel(
                    text_configs.get('ylabel_distribution', 'Count'),
                    fontsize=font.get('label_size', 11)
                )
                
                # Set x-axis labels with proper positioning
                ax.set_xticks(range(len(eq_types)))
                ax.set_xticklabels(eq_types, rotation=45, ha='right', 
                                fontsize=font.get('tick_size', 10))
                ax.tick_params(axis='y', labelsize=font.get('tick_size', 10))
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=font.get('tick_size', 10))
            else:
                ax.set_visible(False)
        
        # Plot 3: Equilibrium type percentages (bar chart)
        if 'percentage' in plot_types:
            ax = axes[plot_idx]
            plot_idx += 1
            
            if stats['equilibrium_type_percentages']:
                # Sort equilibrium types by percentage (descending) for better readability
                sorted_items = sorted(stats['equilibrium_type_percentages'].items(), 
                                    key=lambda x: x[1], reverse=True)
                eq_types = [item[0] for item in sorted_items]
                eq_percentages = [item[1] for item in sorted_items]
                
                bars = ax.bar(range(len(eq_types)), eq_percentages, color=colors['bar_percentage'], alpha=0.7)
                ax.set_title(
                    text_configs.get('percentage_title', 'Equilibrium Types (% of Converged)'),
                    fontsize=font.get('title_size', 14),
                    fontweight='bold'
                )
                ax.set_xlabel(
                    text_configs.get('xlabel_percentage', 'Equilibrium Type'),
                    fontsize=font.get('label_size', 11)
                )
                ax.set_ylabel(
                    text_configs.get('ylabel_percentage', 'Percentage (%)'),
                    fontsize=font.get('label_size', 11)
                )
                
                # Set x-axis labels with proper positioning
                ax.set_xticks(range(len(eq_types)))
                ax.set_xticklabels(eq_types, rotation=45, ha='right',
                                fontsize=font.get('tick_size', 10))
                ax.tick_params(axis='y', labelsize=font.get('tick_size', 10))
                
                # Add percentage labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom',
                        fontsize=font.get('tick_size', 10))
            else:
                ax.set_visible(False)
        
        # Don't use tight_layout since we want fixed subplot sizes
        # plt.tight_layout() removed
        
        # Save figure if requested
        if save:
            # Get num_agents from stats if available, otherwise use a default
            num_agents = stats.get('num_agents', 3)
            domain_type = stats.get('domain_type', '1d')
            
            # Build data_parameters dictionary for data_management
            data_parameters = {
                'data_type': 'plot',
                'plot_type': 'equilibrium_analysis',
                'domain_type': domain_type,
                'num_agents': num_agents,
                'section': paper_figure.get('section', 'equilibrium_analysis'),
                'figure_id': paper_figure.get('figure_id', 'equilibrium_analysis')
            }
            
            # Generate file names using data_management
            file_names = data_management.data_final_name(
                data_parameters=data_parameters,
                name_ads=name_ads,
                save_types=save_types,
                paper_figure=paper_figure.get('paper', False)
            )
            
            # Save figure with all specified extensions
            for file_name in file_names:
                fig.savefig(file_name, bbox_inches='tight', dpi=300)
                print(f"Figure saved: {file_name}")
        plt.close()
        return fig, axes

    def _add_custom_pie_labels(self, ax, wedges, labels, sizes, font):
        """Helper function to add custom positioned labels with connecting lines to pie chart."""
        label_positions = []
        total = sum(sizes)
        
        # Calculate angular positions for each wedge
        for wedge, size, label in zip(wedges, sizes, labels):
            if size == 0:  # Skip zero slices
                continue
                
            # Get the angle at the middle of the wedge
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            
            # Calculate position on the pie edge
            x_pie = np.cos(np.radians(angle))
            y_pie = np.sin(np.radians(angle))
            
            # Calculate initial label position (radial distance from center)
            label_distance = 1.15
            x_label = label_distance * x_pie
            y_label = label_distance * y_pie
            
            # Determine horizontal alignment based on position
            ha = 'left' if x_label > 0 else 'right'
            
            # Calculate percentage
            percentage = (size / total) * 100 if total > 0 else 0
            label_with_pct = f'{label}\n{percentage:.1f}%'
            
            label_positions.append({
                'label': label_with_pct,
                'x_pie': x_pie * 1.0,
                'y_pie': y_pie * 1.0,
                'x_label': x_label,
                'y_label': y_label,
                'ha': ha,
                'angle': angle
            })
        
        # Adjust overlapping labels
        label_positions = self._adjust_label_positions(label_positions)
        
        # Draw labels and connecting lines
        for pos in label_positions:
            # Draw line from pie edge to label
            ax.plot([pos['x_pie'], pos['x_label']], 
                [pos['y_pie'], pos['y_label']], 
                color='gray', linewidth=0.8, linestyle='-', zorder=0)
            
            # Draw label text
            ax.text(pos['x_label'], pos['y_label'], pos['label'],
                ha=pos['ha'], va='center', 
                fontsize=font.get('default_size', 12),
                fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))

    def _adjust_label_positions(self, positions, min_separation=0.15):
        """Helper function to adjust label positions to avoid overlaps."""
        if len(positions) <= 1:
            return positions
        
        # Sort by y position
        positions = sorted(positions, key=lambda p: p['y_label'])
        
        # Iteratively adjust overlapping labels
        max_iterations = 10
        for _ in range(max_iterations):
            adjusted = False
            for i in range(len(positions) - 1):
                curr = positions[i]
                next_pos = positions[i + 1]
                
                # Check if labels overlap vertically
                y_diff = abs(next_pos['y_label'] - curr['y_label'])
                if y_diff < min_separation:
                    # Push labels apart
                    adjustment = (min_separation - y_diff) / 2
                    curr['y_label'] -= adjustment
                    next_pos['y_label'] += adjustment
                    adjusted = True
            
            if not adjusted:
                break
        
        # Adjust horizontal positions based on which side they're on
        for pos in positions:
            # Push labels further out horizontally to reduce crowding
            if pos['x_label'] > 0:
                pos['x_label'] = max(pos['x_label'], 1.15)
            else:
                pos['x_label'] = min(pos['x_label'], -1.15)
        
        return positions


        ## outdated- Monte Carlo specific methods for equilibrium classification

    
    
    
    
    
    
    
    
    ## Outddated - Monte Carlo specific methods for equilibrium classification
    def gradient_ascent_monte_carlo(self,
                                    number_samples: int,
                                    seed: int,
                                    tolerance: float = 10**-5,
                                    tolerated_agents: Optional[int] = None,
                                    parallel: bool = True,
                                    max_workers: Optional[int] = None,
                                    batch_size: Optional[int] = None,
                                    time_steps: Optional[int] = None) -> torch.Tensor:
        """
        Run gradient ascent using Monte Carlo generated starting positions with parallelization.
        
        This function generates random starting positions using the monte_carlo_unit_hypercube method
        and then runs gradient ascent for each position in parallel, similar to final_pos_over_reach
        but using random initial positions instead of varying parameters.

        :param number_samples: Number of Monte Carlo samples to generate
        :type number_samples: int
        :param seed: Random seed for reproducibility
        :type seed: int
        :param tolerance: Tolerance for convergence
        :type tolerance: float
        :param tolerated_agents: Number of agents allowed to tolerate deviations
        :type tolerated_agents: Optional[int]
        :param parallel: Whether to use parallel processing
        :type parallel: bool
        :param max_workers: Maximum number of parallel workers (defaults to CPU count)
        :type max_workers: Optional[int]
        :param batch_size: Batch size for processing (auto-calculated if None)
        :type batch_size: Optional[int]
        :param time_steps: Maximum time steps for gradient ascent
        :type time_steps: Optional[int]

        :return: The final positions of agents for each Monte Carlo sample
        :rtype: torch.Tensor
        
        :raises ValueError: If input parameters are invalid
        :raises RuntimeError: If computation fails
        """
        # Input validation
        if number_samples <= 0:
            raise ValueError(f"number_samples must be positive, got {number_samples}")
            
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
            
        if tolerated_agents is None:
            tolerated_agents = self.num_agents
        elif tolerated_agents < 0:
            raise ValueError(f"tolerated_agents must be non-negative, got {tolerated_agents}")
        
        # Validate domain type
        if self.domain_type not in ['1d', '2d', 'simplex']:
            raise ValueError(f"Unsupported domain_type: {self.domain_type}")
            
        logging.info(f"Running gradient ascent for {number_samples} Monte Carlo samples using domain '{self.domain_type}'")
        
        # Generate Monte Carlo starting positions
        monte_carlo_positions = self.monte_carlo_unit_hypercube(number_samples, seed)
        
        # Convert positions to appropriate domain bounds if needed
        if self.domain_type in ['1d', '2d'] and hasattr(self, 'domain_bounds'):
            if isinstance(self.domain_bounds, (list, tuple)) and len(self.domain_bounds) == 2:
                # Scale from [0,1] to domain bounds
                lower_bound, upper_bound = self.domain_bounds
                monte_carlo_positions = monte_carlo_positions * (upper_bound - lower_bound) + lower_bound
        
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
                max_workers = min(mp.cpu_count(), number_samples)
            
            if batch_size is None:
                batch_size = max(1, number_samples // max_workers)
            
            # Prepare sample data for parallel processing
            sample_data_list = []
            for sample_id, start_pos in enumerate(monte_carlo_positions):
                sample_data = {
                    'sample_id': sample_id,
                    'start_pos': start_pos,
                    'tolerance': tolerance,
                    'tolerated_agents': tolerated_agents,
                    'domain_type': self.domain_type,
                    'total_samples': number_samples,
                    'time_steps': time_steps
                }
                sample_data_list.append(sample_data)
            
            # Initialize result storage
            final_pos_matrix = 0
            
            if parallel and number_samples > 1:
                # Parallel processing
                logging.info(f"Using parallel processing with {max_workers} workers")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_sample = {
                            executor.submit(self._compute_single_monte_carlo_sample, sample_data): sample_data['sample_id']
                            for sample_data in sample_data_list
                        }
                        
                        # Collect results as they complete
                        results = {}
                        completed_count = 0
                        
                        for future in as_completed(future_to_sample):
                            try:
                                sample_id, final_pos_row = future.result()
                                results[sample_id] = final_pos_row
                                completed_count += 1
                                
                                if completed_count % max(1, number_samples // 10) == 0:
                                    logging.info(f"Completed {completed_count}/{number_samples} Monte Carlo samples")
                                    
                            except Exception as e:
                                sample_id = future_to_sample[future]
                                logging.error(f"Monte Carlo sample {sample_id} failed: {str(e)}")
                                raise RuntimeError(f"Failed to compute Monte Carlo sample {sample_id}: {str(e)}")
                        
                        # Build final matrix from results - vectorized approach
                        # Ensure results are stored in original sample order
                        if len(results) != number_samples:
                            raise RuntimeError(f"Expected {number_samples} results, got {len(results)}")
                        
                        # Create a list to store results in original order
                        ordered_results = [None] * number_samples
                        for sample_id, result in results.items():
                            if sample_id >= number_samples:
                                raise RuntimeError(f"Invalid sample ID {sample_id}, expected 0-{number_samples-1}")
                            ordered_results[sample_id] = result
                        
                        # Check that we have all results
                        if None in ordered_results:
                            missing_ids = [i for i, res in enumerate(ordered_results) if res is None]
                            raise RuntimeError(f"Missing results for sample IDs: {missing_ids}")
                        
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
                
                for sample_id, start_pos in enumerate(monte_carlo_positions):
                    try:
                        # Reset field state
                        self.field.pos_matrix = 0
                        self.field.agents_pos = start_pos.clone()
                        self.agents_pos = start_pos.clone()
                        
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
                        
                        # Store result in order
                        final_pos_results.append(final_pos_row)
                        
                        # Progress logging
                        if (sample_id + 1) % max(1, number_samples // 10) == 0:
                            logging.info(f"Completed {sample_id + 1}/{number_samples} Monte Carlo samples")
                            
                    except Exception as e:
                        logging.error(f"Error processing Monte Carlo sample {sample_id}: {str(e)}")
                        raise RuntimeError(f"Failed to process Monte Carlo sample {sample_id}: {str(e)}")
                
                # Vectorized matrix construction for sequential processing
                if len(final_pos_results) == 1:
                    final_pos_matrix = final_pos_results[0]
                else:
                    final_pos_matrix = torch.stack(final_pos_results, dim=0)
                
                logging.info(f"Successfully built final position matrix with shape: {final_pos_matrix.shape}")
            
            logging.info(f"Successfully completed gradient ascent for all {number_samples} Monte Carlo samples")
            
        except Exception as e:
            logging.error(f"Critical error in gradient_ascent_monte_carlo: {str(e)}")
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
                logging.warning(f"Could not fully restore original state: {str(e)}")
        
        return final_pos_matrix

    def _compute_single_monte_carlo_sample(self,
                                           sample_data: Dict
                                           ) -> Tuple[int, torch.Tensor]:
        """
        Helper function to compute final position for a single Monte Carlo sample.
        Designed to be used with multiprocessing.
        
        :param sample_data: Dictionary containing sample data and configuration
        :type sample_data: Dict
        
        :return: Tuple of sample_id and final position row
        :rtype: Tuple[int, torch.Tensor]
        """
        try:
            sample_id = sample_data['sample_id']
            start_pos = sample_data['start_pos']
            tolerance = sample_data['tolerance']
            tolerated_agents = sample_data['tolerated_agents']
            domain_type = sample_data['domain_type']
            total_samples = sample_data['total_samples']
            time_steps = sample_data['time_steps']
            
            # Create a temporary field environment for this computation
            if domain_type == 'simplex':
                temp_field = grad_func_env.AdaptiveEnv(
                    num_agents=self.num_agents,
                    agents_pos=start_pos.clone(),
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
                    agents_pos=start_pos.clone(),
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
            
            # Run gradient ascent from the Monte Carlo starting position
            temp_field.gradient_ascent(show_out=False)
            
            # Extract final position
            final_pos_row = temp_field.pos_matrix[-1].clone()
                
            return sample_id, final_pos_row
            
        except Exception as e:
            logging.error(f"Error computing Monte Carlo sample {sample_id}: {str(e)}")
            raise RuntimeError(f"Failed to compute Monte Carlo sample {sample_id}: {str(e)}") from e

    

# outdated
def analyze_unique_equilibria(result_tensor, tolerance=1e-4):
    """
    Analyze Monte Carlo results to find unique equilibria and their frequencies.
    
    :param result_tensor: Tensor of final positions from Monte Carlo runs
    :type result_tensor: torch.Tensor
    :param tolerance: Tolerance for considering positions as "near-unique"
    :type tolerance: float
    
    :return: Dictionary with unique equilibria, frequencies, and statistics
    :rtype: dict
    """
    
    if result_tensor.dim() == 1:
        # Single sample case - reshape to 2D
        result_tensor = result_tensor.unsqueeze(0)
    
    num_samples, num_agents = result_tensor.shape
    print(f"Analyzing {num_samples} samples with {num_agents} agents each")
    
    # Convert to numpy for easier manipulation
    positions = result_tensor.cpu().numpy()
    
    # Find unique equilibria using clustering approach
    unique_equilibria = []
    frequencies = []
    sample_assignments = []  # Which unique equilibrium each sample belongs to
    
    for i, pos in enumerate(positions):
        # Check if this position is close to any existing unique equilibrium
        found_match = False
        
        for j, unique_pos in enumerate(unique_equilibria):
            # Calculate Euclidean distance between positions
            distance = np.linalg.norm(pos - unique_pos)
            
            if distance < tolerance:
                # This position is close enough to an existing unique equilibrium
                frequencies[j] += 1
                sample_assignments.append(j)
                found_match = True
                break
        
        if not found_match:
            # This is a new unique equilibrium
            unique_equilibria.append(pos.copy())
            frequencies.append(1)
            sample_assignments.append(len(unique_equilibria) - 1)
    
    # Convert to numpy arrays first
    unique_equilibria_array = np.array(unique_equilibria)
    frequencies_array = np.array(frequencies)
    
    # Sort by frequency (most common first) - fix for negative stride issue
    sorted_indices = np.argsort(frequencies_array)[::-1].copy()  # Add .copy() to avoid negative stride
    unique_equilibria_sorted = unique_equilibria_array[sorted_indices].copy()  # Make copy for tensor conversion
    frequencies_sorted = frequencies_array[sorted_indices].copy()  # Make copy
    
    # Convert to tensor after sorting
    unique_equilibria_tensor = torch.tensor(unique_equilibria_sorted)
    
    # Calculate statistics
    num_unique = len(unique_equilibria)
    total_samples = num_samples
    
    # Calculate diversity metrics
    shannon_entropy = -np.sum((frequencies_sorted / total_samples) * 
                             np.log(frequencies_sorted / total_samples))
    
    # Most common equilibrium statistics
    most_common_freq = frequencies_sorted[0] if num_unique > 0 else 0
    most_common_prob = most_common_freq / total_samples if total_samples > 0 else 0
    
    results = {
        'unique_equilibria': unique_equilibria_tensor,
        'frequencies': frequencies_sorted,
        'sample_assignments': np.array(sample_assignments),
        'num_unique': num_unique,
        'total_samples': total_samples,
        'tolerance_used': tolerance,
        'shannon_entropy': shannon_entropy,
        'most_common_frequency': most_common_freq,
        'most_common_probability': most_common_prob,
        'diversity_ratio': num_unique / total_samples if total_samples > 0 else 0
    }
    
    return results

def print_equilibrium_analysis(analysis_results):
    """
    Print a detailed analysis of the equilibrium results.
    
    :param analysis_results: Results from analyze_unique_equilibria
    :type analysis_results: dict
    """
    
    print("="*60)
    print("MONTE CARLO EQUILIBRIUM ANALYSIS")
    print("="*60)
    
    print(f"Total samples analyzed: {analysis_results['total_samples']}")
    print(f"Number of unique equilibria found: {analysis_results['num_unique']}")
    print(f"Tolerance used for clustering: {analysis_results['tolerance_used']}")
    print(f"Diversity ratio: {analysis_results['diversity_ratio']:.3f}")
    print(f"Shannon entropy: {analysis_results['shannon_entropy']:.3f}")
    print()
    
    print("UNIQUE EQUILIBRIA (sorted by frequency):")
    print("-" * 60)
    
    unique_equilibria = analysis_results['unique_equilibria']
    frequencies = analysis_results['frequencies']
    total_samples = analysis_results['total_samples']
    
    for i, (equilibrium, freq) in enumerate(zip(unique_equilibria, frequencies)):
        probability = freq / total_samples
        print(f"Equilibrium #{i+1}: (frequency: {freq}, probability: {probability:.3f})")
        
        # Print agent positions
        for j, pos in enumerate(equilibrium):
            print(f"  Agent {j+1}: {pos:.6f}")
        
        # Add spacing between equilibria
        if i < len(unique_equilibria) - 1:
            print()
    
    print("\n" + "="*60)
    print(f"Most common equilibrium appears in {analysis_results['most_common_probability']:.1%} of samples")
    print("="*60)

def visualize_equilibrium_clustering(analysis_results, original_results):
    """
    Create visualizations for the equilibrium clustering analysis.
    
    :param analysis_results: Results from analyze_unique_equilibria
    :param original_results: Original Monte Carlo results tensor
    """
    
    # Create figure with 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])
    
    # Top row - original plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Bottom row - table spanning both columns
    ax_table = fig.add_subplot(gs[1, :])
    
    # 1. Frequency bar chart
    frequencies = analysis_results['frequencies']
    equilibrium_labels = [f"Eq {i+1}" for i in range(len(frequencies))]
    
    bars = ax1.bar(equilibrium_labels, frequencies, 
                   color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Equilibrium')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Frequency of Each Unique Equilibrium')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add frequency labels on bars
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{freq}', ha='center', va='bottom')
    
    # 2. Probability pie chart (top 10 equilibria only)
    top_n = min(10, len(frequencies))
    top_frequencies = frequencies[:top_n]
    remaining_freq = np.sum(frequencies[top_n:]) if len(frequencies) > top_n else 0
    
    if remaining_freq > 0:
        pie_frequencies = np.append(top_frequencies, remaining_freq)
        pie_labels = equilibrium_labels[:top_n] + [f"Others ({len(frequencies)-top_n})"]
    else:
        pie_frequencies = top_frequencies
        pie_labels = equilibrium_labels[:top_n]
    
    ax2.pie(pie_frequencies, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of Equilibria (Probabilities)')
    
    # 3. Create equilibrium positions table
    unique_equilibria = analysis_results['unique_equilibria']
    num_agents = unique_equilibria.shape[1]
    
    # Prepare table data
    table_data = []
    column_labels = ['Equilibrium', 'Frequency', 'Probability'] + [f'Agent {i+1}' for i in range(num_agents)]
    
    for i, (equilibrium, freq) in enumerate(zip(unique_equilibria, frequencies)):
        probability = freq / analysis_results['total_samples']
        row = [f"Eq {i+1}", str(freq), f"{probability:.3f}"]
        # Add agent positions
        for pos in equilibrium:
            row.append(f"{pos:.4f}")
        table_data.append(row)
    
    # Create table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create the table
    table = ax_table.table(cellText=table_data,
                          colLabels=column_labels,
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header row
    for i in range(len(column_labels)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors for better readability
    for i in range(1, len(table_data) + 1):
        for j in range(len(column_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Color frequency column based on values
    max_freq = max(frequencies) if frequencies.size > 0 else 1
    for i, freq in enumerate(frequencies):
        # Color intensity based on frequency
        intensity = freq / max_freq
        color_intensity = 0.3 + 0.7 * intensity  # Range from 0.3 to 1.0
        table[(i+1, 1)].set_facecolor(plt.cm.Blues(color_intensity))
    
    ax_table.set_title('Equilibrium Positions Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Also create a pandas DataFrame for easy data manipulation
    import pandas as pd
    
    # Create DataFrame
    df_data = {
        'Equilibrium': [f"Eq {i+1}" for i in range(len(frequencies))],
        'Frequency': frequencies,
        'Probability': frequencies / analysis_results['total_samples']
    }
    
    # Add agent position columns
    for i in range(num_agents):
        df_data[f'Agent_{i+1}_Position'] = unique_equilibria[:, i]
    
    equilibrium_df = pd.DataFrame(df_data)
    
    print("\nEquilibrium Data as DataFrame:")
    print("="*60)
    print(equilibrium_df.round(4))
    
    return equilibrium_df