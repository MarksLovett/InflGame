"""
.. module:: one_utils
   :synopsis: Provides utility functions for setting up and managing 1D domains in influencer games.

1D Utility Module
=================

This module provides utility functions for setting up and managing 1D domains in influencer games. 
It includes tools for creating color schemes, plotting critical values, and handling domain-specific configurations.

The module is designed to work with the `InflGame` package and supports creating structured 1D environments 
for simulations involving agent dynamics and resource distributions.

Dependencies:
-------------
- InflGame.utils

Usage:
------
The `critical_values_plot` function can be used to plot critical values for a given resource distribution and number of agents, while the `color_list_maker` function generates random colors for agents.


"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from random import randint
from typing import Union, List, Tuple
import InflGame.utils.general as general
from InflGame.utils.general import generate_color_palette
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


import InflGame.utils.general as general


def critical_values_plot(num_agents: int,
                         bin_points: np.ndarray,
                         resource_distribution: torch.Tensor,
                         axis: plt.Axes,
                         reach_start: float = 0.3,
                         reach_end: float = 0,
                         refinements: int = 2,
                         crit_cs: str = 'Greys') -> tuple:
    """
    Plot critical values given a resource distribution and number of agents t_* (assume symmetric splitting).

    :param num_agents: Number of agents.
    :type num_agents: int
    :param bin_points: Points representing bins for resources.
    :type bin_points: np.ndarray
    :param resource_distribution: Distribution of resources.
    :type resource_distribution: torch.Tensor
    :param axis: Matplotlib axis to plot on.
    :type axis: plt.Axes
    :param reach_start: Starting reach value, defaults to 0.3.
    :type reach_start: float, optional
    :param reach_end: Ending reach value, defaults to 0.
    :type reach_end: float, optional
    :param refinements: Number of refinements for splitting, defaults to 2.
    :type refinements: int, optional
    :param crit_cs: Color scheme for the plot, defaults to 'Greys'.
    :type crit_cs: str, optional
    :return: Updated axis, list of means for axis, and standard deviations.
    :rtype: tuple
    """
    # Convert bin_points to tensor for consistent operations
    bin_points_tensor = torch.tensor(bin_points) if not isinstance(bin_points, torch.Tensor) else bin_points
    
    # Pre-calculate values
    num_sub_divisions = int(np.ceil(np.log2(num_agents)))
    colors_lst = generate_color_palette(num_sub_divisions, crit_cs)
    
    # Initialize storage
    mean_divisions = []
    mean_for_axis = []
    std_divisions = []
    
    for sub_division in range(num_sub_divisions):
        if sub_division == 0:
            # Initial calculation for the root level
            mean_star = general.discrete_mean(bin_points_tensor, resource_distribution)
            variance_star = general.discrete_variance(bin_points_tensor, resource_distribution, mean_star)
            std_star = torch.sqrt((num_agents - 2) / (num_agents - 1) * variance_star)
            
            mean_divisions.append([mean_star])
            std_divisions.append([std_star])
            mean_for_axis.append(mean_star)
            
            # Plot initial lines
            axis.axhline(mean_star.item(), ls='--', color='#404040', linewidth=1)
            if std_star.item() < reach_start or std_star.item() > reach_end:
                # Remove the std line if it is outside the reach range
                pass   
            else:
                axis.axvline(std_star.item(), color='#404040',
                             label=f'$t_*={std_star.item():.3f}$', linewidth=1)
        else:
            # Calculate splits for subsequent levels
            mean_division = mean_divisions[sub_division - 1]
            group_agent_counts = general.split_favor_bottom(num_agents=num_agents, division=sub_division)
            
            # Refine symmetric splits
            for _ in range(refinements):
                symmetric_splits, _ = symmetric_splitting(
                    bin_points=bin_points_tensor, 
                    resource_distribution=resource_distribution,
                    bifurcation_count=sub_division, 
                    means=mean_division
                )
                symmetric_splits = sorted(symmetric_splits)
                # Calculate midpoints for next iteration
                if len(symmetric_splits) > 1:
                    symmetric_splits_tensor = torch.stack(symmetric_splits)
                    mid_point = (symmetric_splits_tensor[1:] + symmetric_splits_tensor[:-1]) / 2
                    mean_division = mid_point.tolist()
            
            # Process each split
            split_stds = []
            split_new_means = []
            num_splits = 2 ** sub_division
            
            for split_id in range(num_splits):
                # Determine support region for this split
                support_mask = _get_support_mask(split_id, num_splits, mid_point, bin_points_tensor)
                
                # Create local resource distribution
                values_supported = torch.zeros_like(resource_distribution)
                values_supported[support_mask] = resource_distribution[support_mask]
                
                # Calculate local mean
                if sub_division != num_sub_divisions - 1:
                    mean_local = general.discrete_mean(bin_points_tensor, values_supported)
                    split_new_means.append(mean_local)
                    mean_for_axis.append(mean_local)
                
                # Calculate and plot standard deviation if needed
                if sub_division != num_sub_divisions - 1:
                    group_agent_count = group_agent_counts[split_id]
                    if group_agent_count not in [1, 2]:
                        variance_local = general.discrete_variance(bin_points_tensor, values_supported, mean_local)
                        std_local = torch.sqrt((group_agent_count - 2) / (group_agent_count - 1) * variance_local)
                        split_stds.append(std_local)
                        
                        # Plot lines
                        axis.axvline(std_local.item(), 
                                   color=colors_lst[sub_division])
                        axis.hlines(mean_local.item(), xmin=reach_start, xmax=std_local.item(),
                                  ls='--', color=colors_lst[sub_division])
            
            # Clean up standard deviations (remove duplicates)
            if split_stds:
                split_stds_array = torch.stack(split_stds)
                unique_stds = torch.unique(torch.round(split_stds_array, decimals=4))
                
                # Check if all values are approximately the same
                if len(unique_stds) > 1:
                    avg_std = torch.mean(split_stds_array)
                    if torch.allclose(split_stds_array, avg_std, atol=1e-1):
                        unique_stds = torch.tensor([avg_std])
                
                std_divisions.append(unique_stds.tolist())
            else:
                std_divisions.append([])
            
            mean_divisions.append(sorted(split_new_means, key=lambda x: x.item()))
    
    return axis, mean_for_axis, std_divisions


def _get_support_mask(split_id: int, num_splits: int, mid_point: torch.Tensor, 
                     bin_points: torch.Tensor) -> torch.Tensor:
    """
    Helper function to get support mask for a given split.
    
    :param split_id: ID of the current split
    :param num_splits: Total number of splits
    :param mid_point: Midpoint values for splits
    :param bin_points: Bin points tensor
    :return: Boolean mask for the support region
    """
    if split_id == 0:
        return mid_point[split_id] > bin_points
    elif split_id == num_splits - 1:
        return bin_points > mid_point[split_id - 1]
    else:
        return (mid_point[split_id] > bin_points) & (bin_points > mid_point[split_id - 1])

def symmetric_splitting(bin_points: Union[np.ndarray, torch.Tensor],
                        resource_distribution: Union[torch.Tensor, np.ndarray],
                        bifurcation_count: int,
                        means: List[float]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Perform symmetric splitting of resources based on means.

    :param bin_points: Points representing bins for resources.
    :type bin_points: Union[np.ndarray, torch.Tensor]
    :param resource_distribution: Distribution of resources.
    :type resource_distribution: Union[torch.Tensor, np.ndarray]
    :param bifurcation_count: Number of bifurcations.
    :type bifurcation_count: int
    :param means: List of means for splitting.
    :type means: List[float]
    :return: Symmetric splits and final array of means.
    :rtype: Tuple[List[torch.Tensor], List[torch.Tensor]]
    """
    # Convert inputs to tensors for consistent operations
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points)
    if not isinstance(resource_distribution, torch.Tensor):
        resource_distribution = torch.tensor(resource_distribution)
    
    symmetric_splits = []
    final_array = []
    num_splits = 2 ** bifurcation_count
    
    for split_id in range(num_splits):
        # Get support mask for this split
        support_mask = _get_support_mask_symmetric(split_id, num_splits, means, bin_points)
        # Add means to final array for non-edge cases
        if split_id == 0:
            if split_id < len(means):
                if not torch.is_tensor(means[split_id]):
                    means[split_id] = torch.tensor(means[split_id])
                final_array.append(means[split_id].clone())
        elif split_id != num_splits - 1:
            if split_id < len(means):
                if not torch.is_tensor(means[split_id]):
                    means[split_id] = torch.tensor(means[split_id])
                final_array.append(means[split_id].clone())
        
        
        # Create supported values
        values_supported = torch.zeros_like(resource_distribution)
        values_supported[support_mask] = resource_distribution[support_mask]
        
        # Calculate local mean
        mean_local = general.discrete_mean(bin_points, values_supported)
        symmetric_splits.append(mean_local)
        final_array.append(mean_local)
    # remove nan values from symmetric_splits
    symmetric_splits = [s for s in symmetric_splits if not torch.isnan(s)]
    
    return symmetric_splits, final_array


def _get_support_mask_symmetric(split_id: int, num_splits: int, means: List[float], 
                               bin_points: torch.Tensor) -> torch.Tensor:
    """
    Helper function to get support mask for symmetric splitting.
    
    :param split_id: ID of the current split
    :param num_splits: Total number of splits
    :param means: List of mean values
    :param bin_points: Bin points tensor
    :return: Boolean mask for the support region
    """
    if split_id == 0:
        # For the first split, check if we have any means
        if len(means) > 0:
            return means[split_id] > bin_points
        else:
            return torch.ones_like(bin_points, dtype=torch.bool)
    elif split_id == num_splits - 1:
        # For the last split, use the previous mean if it exists
        if split_id - 1 < len(means):
            return bin_points > means[split_id - 1]
        else:
            return torch.ones_like(bin_points, dtype=torch.bool)
    else:
        # For middle splits, check bounds on both means
        if split_id < len(means) and split_id - 1 < len(means):
            return (means[split_id] > bin_points) & (bin_points > means[split_id - 1])
        else:
            return torch.ones_like(bin_points, dtype=torch.bool)

def direction_strength_1d(gradient_function,
                          two_a: bool,
                          parameter_instance: Union[list, np.ndarray, torch.Tensor] = 0,
                          ids: List[int] = [0, 1],
                          pos: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the gradient strength in a 1D direction using PyTorch operations.

    :param gradient_function: Function to compute gradients.
    :type gradient_function: callable
    :param two_a: Flag indicating whether to use two arguments.
    :type two_a: bool
    :param parameter_instance: Parameters for the gradient function, defaults to 0.
    :type parameter_instance: Union[list, np.ndarray, torch.Tensor], optional
    :param ids: Indices for the gradient computation, defaults to [0, 1].
    :type ids: List[int], optional
    :param pos: Position tensor, defaults to None.
    :type pos: torch.Tensor, optional
    :return: Computed gradients.
    :rtype: torch.Tensor
    """
    # Create coordinate grid using torch to match the OLD version's np.mgrid behavior
    # np.mgrid[0:1:100j, 0:1:100j] creates Y, X order
    y_coords = torch.linspace(0, 1, 100)
    x_coords = torch.linspace(0, 1, 100)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')  # This matches np.mgrid order
    
    # Flatten to match the OLD version's a1, a2 assignment
    a1 = X.flatten()  # x coordinates
    a2 = Y.flatten()  # y coordinates
    
    if two_a == False:
        grads = []
        for x, y in zip(a1, a2):
            pos[ids[0]] = x.item()   
            pos[ids[1]] = y.item()
            grad_result = gradient_function(pos, parameter_instance, ids=ids, two_a=two_a)
            
            # Handle tensor conversion
            if torch.is_tensor(grad_result):
                if grad_result.numel() == 1:
                    grads.append(grad_result.item())
                else:
                    grads.append(grad_result.detach().numpy())
            else:
                grads.append(float(grad_result))
        
        # Convert to torch tensor
        if isinstance(grads[0], (int, float)):
            grads = torch.tensor(grads, dtype=torch.float32)
        else:
            grads = torch.tensor(np.array(grads), dtype=torch.float32)
    else:
        grads_list = []
        for x, y in zip(a1, a2):
            coord_array = torch.tensor([x.item(), y.item()], dtype=torch.float32)
            grad_result = gradient_function(coord_array, parameter_instance, ids=ids)
            
            # Handle tensor conversion
            if torch.is_tensor(grad_result):
                if grad_result.numel() == 1:
                    grads_list.append(grad_result.item())
                else:
                    grads_list.append(grad_result.detach().numpy())
            else:
                grads_list.append(float(grad_result))
        
        # Convert to torch tensor
        if isinstance(grads_list[0], (int, float)):
            grads = torch.tensor(grads_list, dtype=torch.float32)
        else:
            grads = torch.tensor(np.array(grads_list), dtype=torch.float32)
    
    return grads


def direction_strength_1d_OLD(gradient_function,
                          two_a: bool,
                          parameter_instance: list | np.ndarray | torch.Tensor = 0,
                          ids: list = [0, 1],
                          pos: torch.Tensor = None):
    """
    Compute the gradient strength in a 1D direction.

    :param gradient_function: Function to compute gradients.
    :type gradient_function: callable
    :param two_a: Flag indicating whether to use two arguments.
    :type two_a: bool
    :param parameter_instance: Parameters for the gradient function, defaults to 0.
    :type parameter_instance: list | np.ndarray | torch.Tensor, optional
    :param ids: Indices for the gradient computation, defaults to [0, 1].
    :type ids: list, optional
    :param pos: Position tensor, defaults to None.
    :type pos: torch.Tensor, optional
    :return: Computed gradients.
    :rtype: np.ndarray
    """
    Y, X = np.mgrid[0:1:100j, 0:1:100j]
    a1=X.flatten()
    a2=Y.flatten()
    if two_a==False:
        grads=[]
        for x,y in zip(a1,a2):
            pos[ids[0]]=x   
            pos[ids[1]]=y 
            grads.append(gradient_function(pos,parameter_instance,ids=ids,two_a=two_a).numpy())
        grads=np.array(grads)
    else:
        grads=np.array([gradient_function(np.array([x,y]),parameter_instance,ids=ids).numpy() for x,y in zip(a1,a2)])
    return grads