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
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import re
from matplotlib.patches import Patch
import matplotlib.patches as patches

import InflGame.utils.general as general
from InflGame.utils.general import generate_color_palette
import  InflGame.adaptive.jacobian as jc

def critical_values_plot(num_agents: int,
                         bin_points: np.ndarray,
                         resource_distribution: torch.Tensor,
                         axis: plt.Axes,
                         reach_start: float = 0.3,
                         reach_end: float = 0,
                         refinements: int = 2,
                         crit_cs: str = 'Greys') -> tuple:
    """
    Plot critical values given a resource distribution and number of agents :math:`t_*` (assuming symmetric splitting).
    
    This function calculates and visualizes critical threshold values for agent positioning in a 1D domain,
    using recursive symmetric splitting to determine equilibrium configurations.

    Parameters
    ----------
    num_agents : int
        Number of agents in the system.
    bin_points : np.ndarray
        Points representing bins for resource discretization.
    resource_distribution : torch.Tensor
        Distribution of resources across the domain.
    axis : plt.Axes
        Matplotlib axis object to plot on.
    reach_start : float, optional
        Starting reach value for critical threshold visualization, by default 0.3.
    reach_end : float, optional
        Ending reach value for critical threshold visualization, by default 0.
    refinements : int, optional
        Number of refinement iterations for symmetric splitting, by default 2.
    crit_cs : str, optional
        Colormap scheme for the critical value plot, by default 'Greys'.

    Returns
    -------
    tuple
        A tuple containing:
        - axis (plt.Axes): Updated matplotlib axis with plotted critical values
        - mean_for_axis (List[torch.Tensor]): List of mean values for each split level
        - std_divisions (List[List[float]]): List of standard deviation values at each bifurcation level
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
    Helper function to compute the support mask for a given split region.
    
    Determines which bin points fall within the support region of a particular split,
    based on the split ID and midpoint boundaries.

    Parameters
    ----------
    split_id : int
        ID of the current split region.
    num_splits : int
        Total number of splits in the current bifurcation level.
    mid_point : torch.Tensor
        Tensor containing midpoint values that define split boundaries.
    bin_points : torch.Tensor
        Tensor of bin points for resource discretization.

    Returns
    -------
    torch.Tensor
        Boolean mask tensor indicating which bin points fall within the support region.
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
    Perform symmetric splitting of resource distribution based on means.
    
    Recursively divides the resource distribution into symmetric regions based on bifurcation count,
    calculating local means for each split region.

    Parameters
    ----------
    bin_points : Union[np.ndarray, torch.Tensor]
        Points representing bins for resource discretization.
    resource_distribution : Union[torch.Tensor, np.ndarray]
        Distribution of resources across the domain.
    bifurcation_count : int
        Number of bifurcation levels, determines :math:`2^{bifurcation\_count}` splits.
    means : List[float]
        List of mean values from previous bifurcation level used as boundaries.

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor]]
        A tuple containing:
        - symmetric_splits (List[torch.Tensor]): List of locally computed means for each split region
        - final_array (List[torch.Tensor]): Combined list of boundary means and local means
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
    Helper function to compute support mask for symmetric splitting operations.
    
    Determines which bin points belong to each symmetric split region based on mean boundaries.

    Parameters
    ----------
    split_id : int
        ID of the current split region.
    num_splits : int
        Total number of splits at the current bifurcation level.
    means : List[float]
        List of mean values defining split boundaries.
    bin_points : torch.Tensor
        Tensor of bin points for resource discretization.

    Returns
    -------
    torch.Tensor
        Boolean mask tensor indicating which bin points fall within the symmetric split region.
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
    Compute gradient strength in a 1D direction using PyTorch operations.
    
    Evaluates the gradient function across a 2D grid to compute directional gradient strengths,
    useful for vector field visualization in 1D domains.

    Parameters
    ----------
    gradient_function : callable
        Function to compute gradients, should accept position and parameters.
    two_a : bool
        Flag indicating whether the gradient function uses a two-argument coordinate array format.
    parameter_instance : Union[list, np.ndarray, torch.Tensor], optional
        Parameters for the influence function, by default 0.
    ids : List[int], optional
        Indices of agents to compute gradients for, by default [0, 1].
    pos : torch.Tensor, optional
        Position tensor for agents, by default None.

    Returns
    -------
    torch.Tensor
        Computed gradient values as a flattened tensor of shape :math:`(10000,)` for a 100x100 grid.
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
    Compute gradient strength in a 1D direction (legacy NumPy implementation).
    
    .. deprecated::
        This is the original NumPy-based implementation. Use :func:`direction_strength_1d` for 
        PyTorch-based operations with autograd support.

    Parameters
    ----------
    gradient_function : callable
        Function to compute gradients.
    two_a : bool
        Flag indicating whether to use two-argument coordinate array format.
    parameter_instance : list | np.ndarray | torch.Tensor, optional
        Parameters for the gradient function, by default 0.
    ids : list, optional
        Indices for the gradient computation, by default [0, 1].
    pos : torch.Tensor, optional
        Position tensor, by default None.

    Returns
    -------
    np.ndarray
        Computed gradients as a NumPy array.
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

def projection_to_plane_coordinates(matrix: torch.Tensor) -> torch.Tensor:
    """
    Project 3D simplex coordinates to 2D plane coordinates using orthogonal projection.
    
    Uses an orthogonal projection matrix to map 3D barycentric coordinates onto a 2D plane
    for visualization purposes. The projection preserves relative distances and orientations.
    
    Following project patterns:
    
    - Use vectorized torch operations for performance
    - Handle dtype compatibility automatically
    - Maintain tensor device consistency
    - Handle single vector and batch inputs
    - Ensure consistent dtype=torch.float32 output

    Parameters
    ----------
    matrix : torch.Tensor
        Input tensor of shape :math:`(3,)` or :math:`(N, 3)` containing 3D coordinates.

    Returns
    -------
    torch.Tensor
        Projected 2D coordinates of shape :math:`(2,)` or :math:`(N, 2)` with dtype=torch.float32.
    """
    # Ensure input is 2D for batch processing - following project patterns
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Ensure input matrix is float32 - following project patterns
    if matrix.dtype != torch.float32:
        matrix = matrix.to(torch.float32)
    
    # Create projection matrix with float32 dtype - FIXED DIMENSIONS
    axis_project = torch.tensor([
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [0, -1/np.sqrt(2), 1/np.sqrt(2)]
    ], dtype=torch.float32, device=matrix.device)
    
    # FIXED: Correct matrix multiplication order
    # matrix: (N, 3), axis_project: (2, 3) -> result: (N, 2)
    coordinates = torch.matmul(matrix, axis_project.T)
    
    # Ensure output is float32 - following project patterns
    coordinates = coordinates.to(torch.float32)
    
    # Return single vector if input was single vector
    if squeeze_output:
        coordinates = coordinates.squeeze(0)
    
    return coordinates

def projection_to_3d_auto_constrained(matrix: torch.Tensor, target_bounds: tuple = (0.0, 1.0), 
                                     tolerance: float = 1e-8, max_iterations: int = 100) -> torch.Tensor:
    """
    Project 2D plane coordinates back to 3D simplex coordinates with automatic constraint satisfaction.
    
    Uses the Moore-Penrose pseudo-inverse to compute a base 3D projection from 2D coordinates,
    then applies an offset along the :math:`[1,1,1]` direction to ensure all coordinates satisfy
    simplex constraints (all elements within target_bounds).
    
    Following Influencer Games patterns:
    
    - Use torch tensor operations for autograd compatibility
    - State management with .clone() for torch tensors
    - Adaptive optimization to find optimal offset parameter
    - Handle single vector and batch inputs
    - Ensure consistent dtype=torch.float32 output
    - Convergence checking with project tolerance patterns

    Parameters
    ----------
    matrix : torch.Tensor
        Input tensor of shape :math:`(2,)` or :math:`(N, 2)` containing 2D plane coordinates.
    target_bounds : tuple, optional
        Tuple :math:`(min\_val, max\_val)` for coordinate constraints, by default (0.0, 1.0).
    tolerance : float, optional
        Convergence tolerance for constraint satisfaction, by default 1e-8.
    max_iterations : int, optional
        Maximum iterations for offset optimization (currently unused), by default 100.

    Returns
    -------
    torch.Tensor
        Projected 3D coordinates of shape :math:`(3,)` or :math:`(N, 3)` with dtype=torch.float32,
        all elements guaranteed to be within target_bounds.
    """
    # Handle single vector input - following project patterns
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Ensure input matrix is float32 - following project patterns
    if matrix.dtype != torch.float32:
        matrix = matrix.to(torch.float32)
    
    # The forward projection matrix P from 3D to 2D - ENSURE FLOAT32
    P = torch.tensor([
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [0, -1/np.sqrt(2), 1/np.sqrt(2)]
    ], dtype=torch.float32, device=matrix.device)
    
    # Calculate the TRUE Moore-Penrose pseudo-inverse
    P_pinv = torch.linalg.pinv(P)  # Shape: (3, 2)
    
    # Apply pseudo-inverse projection to get base 3D coordinates (without normal offset)
    base_coordinates = torch.matmul(matrix, P_pinv.T)  # Shape: (N, 3)
    
    # Normalized [1,1,1] direction for offset - FLOAT32
    normal = torch.tensor([1, 1, 1], dtype=torch.float32, device=matrix.device)
      # Normalize
    
    def apply_c_offset(c_val):
        """Apply c offset and return coordinates"""
        return base_coordinates + c_val * normal.unsqueeze(0)
    min_c=torch.min(base_coordinates)
    max_c=torch.max(base_coordinates)
    if max_c > target_bounds[1]:
        best_c=-min_c
    elif min_c < target_bounds[0]:
        best_c=-min_c
    else:
        best_c=0.0

    # Apply the best c value found
    coordinates = apply_c_offset(best_c)
    
    # Final clipping as safety measure - following project robustness patterns
    coordinates = torch.clamp(coordinates, 0, 1)
    
    # Ensure output is float32 - following project patterns
    coordinates = coordinates.to(torch.float32)
    
    if squeeze_output:
        coordinates = coordinates.squeeze(0)
    
    return coordinates

def generate_constrained_2d_points(num_points=50, method='analytical_transform'):
    """
    Generate 2D points satisfying simplex projection constraints.
    
    Generates 2D points :math:`(x,y)` that satisfy the constraints:
    
    1. :math:`x \\in [-1/\\sqrt{2}, 1/\\sqrt{2}]`
    2. :math:`y \\in [-1/\\sqrt{2}, 1/\\sqrt{2}]`
    3. :math:`(x-y) \\in [-1/\\sqrt{2}, 1/\\sqrt{2}]`
    
    These constraints ensure the points can be validly projected back to 3D simplex coordinates.
    
    Following Influencer Games patterns:
    
    - Use torch tensor operations for autograd compatibility
    - State management with .clone() for torch tensors
    - Handle domain bounds properly for 1D domain type
    - Return torch.float32 tensors
    - Memory management with matrix clearing

    Parameters
    ----------
    num_points : int, optional
        Number of valid 2D points to generate, by default 50.
    method : str, optional
        Generation method, by default 'analytical_transform'.
        
        - ``'analytical_transform'``: Transform from :math:`(u,v)` coordinates with analytic bounds
        - ``'rejection_sampling'``: Random sampling with constraint rejection
        - ``'grid_filtering'``: Grid-based approach with constraint filtering

    Returns
    -------
    dict
        Dictionary containing:
        
        - ``'points_2d'`` (torch.Tensor): Generated 2D points of shape :math:`(N, 2)`
        - ``'constraint_values'`` (torch.Tensor): Constraint check values :math:`[x, y, x-y]`
        - ``'method'`` (str): Method used for generation
        - Additional metadata depending on method (success_rate, acceptance_rate, etc.)
    """
    sqrt2_inv = 1.0 / np.sqrt(2)
    
    if method == 'analytical_transform':
        # Method 1: Transform from unconstrained variables using change of variables
        # Use (u, v) coordinates where u = x+y, v = x-y
        # Then x = (u+v)/2, y = (u-v)/2
        
        # Constraints become:
        # |x| ≤ 1/√2 → |(u+v)/2| ≤ 1/√2 → |u+v| ≤ 2/√2 = √2
        # |y| ≤ 1/√2 → |(u-v)/2| ≤ 1/√2 → |u-v| ≤ 2/√2 = √2  
        # |v| ≤ 1/√2 (given constraint on x-y)
        
        # The feasible region in (u,v) space is the intersection of:
        # |u+v| ≤ √2, |u-v| ≤ √2, |v| ≤ 1/√2
        
        print(f"Generating {num_points} points using analytical transformation...")
        
        valid_points = []
        constraint_values = []
        
        # Generate points uniformly in the feasible (u,v) region
        for _ in range(num_points * 2):  # Oversample for efficiency
            # Sample v first (constrained to [-1/√2, 1/√2])
            v = torch.rand(1, dtype=torch.float32).item() * (2 * sqrt2_inv) - sqrt2_inv
            
            # For this v, find valid u range
            # Need: |u+v| ≤ √2 AND |u-v| ≤ √2
            # This gives: max(-√2-v, -√2+v) ≤ u ≤ min(√2-v, √2+v)
            
            sqrt2 = np.sqrt(2)
            u_min = max(-sqrt2 - v, -sqrt2 + v)
            u_max = min(sqrt2 - v, sqrt2 + v)
            
            if u_min <= u_max:  # Feasible region exists
                u = torch.rand(1, dtype=torch.float32).item() * (u_max - u_min) + u_min
                
                # Transform back to (x,y)
                x = (u + v) / 2
                y = (u - v) / 2
                
                # Verify constraints (safety check)
                if (abs(x) <= sqrt2_inv + 1e-10 and 
                    abs(y) <= sqrt2_inv + 1e-10 and 
                    abs(x - y) <= sqrt2_inv + 1e-10):
                    
                    valid_points.append(torch.tensor([x, y], dtype=torch.float32))
                    constraint_values.append(torch.tensor([x, y, x-y], dtype=torch.float32))
                    
                    if len(valid_points) >= num_points:
                        break
        
        return {
            'points_2d': torch.stack(valid_points[:num_points]) if valid_points else torch.empty(0, 2),
            'constraint_values': torch.stack(constraint_values[:num_points]) if constraint_values else torch.empty(0, 3),
            'method': method,
            'success_rate': len(valid_points) / (num_points * 2) if num_points > 0 else 0
        }
    
    elif method == 'rejection_sampling':
        # Method 2: Simple rejection sampling
        print(f"Generating {num_points} points using rejection sampling...")
        
        valid_points = []
        constraint_values = []
        attempts = 0
        max_attempts = num_points * 100
        
        while len(valid_points) < num_points and attempts < max_attempts:
            # Sample uniformly in the square [-1/√2, 1/√2]²
            x = torch.rand(1, dtype=torch.float32).item() * (2 * sqrt2_inv) - sqrt2_inv
            y = torch.rand(1, dtype=torch.float32).item() * (2 * sqrt2_inv) - sqrt2_inv
            
            # Check constraint: |x-y| ≤ 1/√2
            if abs(x - y) <= sqrt2_inv:
                valid_points.append(torch.tensor([x, y], dtype=torch.float32))
                constraint_values.append(torch.tensor([x, y, x-y], dtype=torch.float32))
            
            attempts += 1
            
            # Progress reporting following project patterns
            if attempts % (max_attempts // 20) == 0:
                acceptance_rate = len(valid_points) / attempts * 100
                print(f"Attempts: {attempts} - Found: {len(valid_points)} - Acceptance: {acceptance_rate:.2f}%")
        
        return {
            'points_2d': torch.stack(valid_points) if valid_points else torch.empty(0, 2),
            'constraint_values': torch.stack(constraint_values) if constraint_values else torch.empty(0, 3),
            'method': method,
            'attempts': attempts,
            'acceptance_rate': len(valid_points) / attempts if attempts > 0 else 0
        }
    
    elif method == 'grid_filtering':
        # Method 3: Grid-based approach with filtering
        print(f"Generating {num_points} points using grid filtering...")
        
        # Create dense grid in [-1/√2, 1/√2]²
        grid_size = int(np.sqrt(num_points * 4))  # Oversample for filtering
        x_grid = torch.linspace(-sqrt2_inv, sqrt2_inv, grid_size, dtype=torch.float32)
        y_grid = torch.linspace(-sqrt2_inv, sqrt2_inv, grid_size, dtype=torch.float32)
        
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # Filter points satisfying |x-y| ≤ 1/√2
        x_coords = grid_points[:, 0]
        y_coords = grid_points[:, 1]
        diff_constraint = torch.abs(x_coords - y_coords) <= sqrt2_inv
        
        valid_grid_points = grid_points[diff_constraint]
        
        # Randomly sample from valid points
        if len(valid_grid_points) >= num_points:
            indices = torch.randperm(len(valid_grid_points))[:num_points]
            selected_points = valid_grid_points[indices]
        else:
            selected_points = valid_grid_points
            print(f"Warning: Only found {len(valid_grid_points)} valid points out of {num_points} requested")
        
        # Compute constraint values
        constraint_vals = []
        for point in selected_points:
            x, y = point
            constraint_vals.append(torch.tensor([x.item(), y.item(), (x-y).item()], dtype=torch.float32))
        
        return {
            'points_2d': selected_points,
            'constraint_values': torch.stack(constraint_vals) if constraint_vals else torch.empty(0, 3),
            'method': method,
            'total_grid_points': len(grid_points),
            'valid_grid_points': len(valid_grid_points),
            'filtering_efficiency': len(valid_grid_points) / len(grid_points) if len(grid_points) > 0 else 0
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")

def classify_equilibrium_type(positions, tolerance=1e-3):
    """
    Classify equilibrium type based on spatial grouping of agents.
    
    Groups agents whose positions are within tolerance and returns a classification string
    describing the group sizes in spatial order (left to right).

    Parameters
    ----------
    positions : array_like
        Array of agent positions (supports both NumPy arrays and torch tensors).
    tolerance : float, optional
        Tolerance for considering positions as equal, by default 1e-3.

    Returns
    -------
    str
        String describing equilibrium type with groups in spatial order:
        
        - ``'(n)'``: All :math:`n` agents at the same position
        - ``'(n-1,1)'``: :math:`n-1` agents grouped at lower position, 1 isolated higher
        - ``'(1,n-1)'``: 1 agent isolated at lower position, :math:`n-1` grouped higher
        - ``'(2,1,1,2)'``: Groups listed left to right by position
        
    Examples
    --------
    >>> positions = np.array([0.2, 0.2, 0.5, 0.8, 0.8])
    >>> classify_equilibrium_type(positions)
    '(2,1,2)'
    """
    # Convert to numpy array, handling torch tensors
    if hasattr(positions, 'numpy'):
        positions = positions.numpy()
    else:
        positions = np.asarray(positions)
    
    n_agents = len(positions)
    
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
    
    # Return groups in spatial order (already ordered since we sorted positions)
    num_groups = len(groups)
    
    # Case 1: All agents at same position
    if num_groups == 1:
        return f'({n_agents})'
    
    # All other cases: return groups in order they appear spatially
    else:
        return f'({",".join(map(str, groups))})'

def _type_dict_helper(matrix, reach_parameters, tolerance):
    """
    Helper function to create classification dictionary from equilibrium matrix.
    
    Processes the output from :func:`classify_equilibrium_type` for each equilibrium configuration
    in the matrix and associates it with the corresponding reach parameter.

    Parameters
    ----------
    matrix : array_like
        Matrix of equilibrium positions for different parameter values.
    reach_parameters : array_like
        Array of reach parameter values corresponding to each row in the matrix.
    tolerance : float
        Tolerance for position grouping in equilibrium classification.

    Returns
    -------
    dict
        Dictionary mapping item IDs to classification information:
        
        - ``'classification'`` (str): Equilibrium type string
        - ``'reach_parameter'`` (float): Associated reach parameter value
    """
    classification_dict={}
    for item_id in range(len(matrix)):
        item_classification = classify_equilibrium_type(matrix[item_id],tolerance=tolerance)
        classification_dict[str(item_id)] = {'classification': item_classification, 'reach_parameter': reach_parameters[item_id][0].item()}
    return classification_dict

def _find_bifurcation_split(classification_dict):
    """
    Identify bifurcation points where equilibrium classification changes.
    
    Scans through the classification dictionary in reverse order to detect transitions
    in equilibrium structure as the reach parameter varies.

    Parameters
    ----------
    classification_dict : dict
        Dictionary mapping item IDs to classification and reach parameter information.

    Returns
    -------
    dict
        Dictionary of bifurcation points, where each key maps to:
        
        - ``'classification_old'`` (str): Equilibrium type before bifurcation
        - ``'classification_new'`` (str): Equilibrium type after bifurcation
        - ``'reach_parameter'`` (float): Reach parameter value at bifurcation
    """
    #we find where the classification changes as reach varies
    classification=classification_dict[str(len(classification_dict)-1)]['classification']
    bifurcations={}
    for item in list(classification_dict.keys())[::-1]:
        if classification != classification_dict[item]['classification']:
            classification_new=classification_dict[item]['classification']
            bifurcations[item]= {'classification_old': classification, 'classification_new': classification_new, 'reach_parameter': classification_dict[item]['reach_parameter']}
            classification=classification_new
    return bifurcations

def bifurcation_type_helper(matrix, reach_parameters, tolerance=1e-2):
    """
    Classify bifurcation types based on equilibrium structure transitions.
    
    Identifies two types of bifurcations:
    
    1. **Type 1 (Symmetry-breaking)**: Local symmetric equilibrium becomes unstable and splits 
       into asymmetric equilibria (number of groups increases)
    2. **Type 2 (Basin shift)**: Basins of attraction shift local grouping without increasing 
       group count (number of groups stays same or decreases)

    Parameters
    ----------
    matrix : dict
        Dictionary containing equilibrium position matrix under key ``'max'``.
    reach_parameters : array_like
        Array of reach parameter values corresponding to equilibrium configurations.
    tolerance : float, optional
        Tolerance for position grouping in equilibrium classification, by default 1e-2.

    Returns
    -------
    dict
        Dictionary of classified bifurcations, where each key maps to:
        
        - ``'reach_parameter'`` (float): Reach parameter value at bifurcation
        - ``'type'`` (str): Bifurcation type (``'1'`` or ``'2'``)
        - ``'classification_new'`` (str): New equilibrium structure after bifurcation
        
    Notes
    -----
    The ``classification_new`` field shows the equilibrium structure after bifurcation occurs.
    Type 1 bifurcations indicate structural instability, while Type 2 indicate basin reorganization.
    """
    bifurcations=_find_bifurcation_split(_type_dict_helper(matrix['max'],reach_parameters=reach_parameters,tolerance=tolerance))
    # A type 1 bifurcation can only occur if there are less groups in the old classification than in the new classification. While a type 2 bifurcation occurs when the number of groups remains the same or decreases.
    bifurcation_types = {}
    for key, value in bifurcations.items():
        old_nums = [int(x) for x in value['classification_old'].strip('()').split(',')]
        new_nums = [int(x) for x in value['classification_new'].strip('()').split(',')]
        
        old_groups = len(old_nums)
        new_groups = len(new_nums)
        
        # Count groups of size 2 in old and new classifications
        old_groups_of_2 = old_nums.count(2)
        new_groups_of_2 = new_nums.count(2)
        
        # Check if there was a group of 2 in old but fewer groups of 2 in new
        
        if old_groups < new_groups:
            if old_groups_of_2 > 0 and new_groups_of_2 < old_groups_of_2:
                bifurcation_types[key] = {'reach_parameter': value['reach_parameter'], 'type': '2', 'classification_new': value['classification_new']}
            else:
                bifurcation_types[key] = {'reach_parameter': value['reach_parameter'], 'type': '1', 'classification_new': value['classification_new']}
        else:
            bifurcation_types[key] = {'reach_parameter': value['reach_parameter'], 'type': '2', 'classification_new': value['classification_new']}
    return bifurcation_types

def find_label_position(x_pos, y_pos, existing_positions, base_offset=0.03, min_distance=0.06):
        """Find a good label position offset from the data point that doesn't overlap with existing labels."""
        
        # Try positions around the data point in order of preference
        offset_positions = [
            (x_pos + base_offset, y_pos + base_offset),      # top-right
            (x_pos - base_offset, y_pos + base_offset),      # top-left  
            (x_pos + base_offset, y_pos - base_offset),      # bottom-right
            (x_pos - base_offset, y_pos - base_offset),      # bottom-left
            (x_pos, y_pos + base_offset * 1.5),             # directly above
            (x_pos, y_pos - base_offset * 1.5),             # directly below
            (x_pos + base_offset * 1.5, y_pos),             # directly right
            (x_pos - base_offset * 1.5, y_pos),             # directly left
        ]
        
        # Check each position for conflicts
        for candidate_x, candidate_y in offset_positions:
            # Skip if outside plot bounds
            if not (0 <= candidate_x <= 1 and 0 <= candidate_y <= 1):
                continue
                
            # Check distance from existing labels
            conflict = False
            for ex_x, ex_y in existing_positions:
                distance = np.sqrt((candidate_x - ex_x)**2 + (candidate_y - ex_y)**2)
                if distance < min_distance:
                    conflict = True
                    break
            
            if not conflict:
                return candidate_x, candidate_y
        
        # If no good position found, use spiral search with larger offsets
        for radius in [base_offset * 2, base_offset * 3]:
            angles = np.linspace(0, 2*np.pi, 12)
            for angle in angles:
                candidate_x = x_pos + radius * np.cos(angle)
                candidate_y = y_pos + radius * np.sin(angle)
                
                if not (0 <= candidate_x <= 1 and 0 <= candidate_y <= 1):
                    continue
                    
                conflict = False
                for ex_x, ex_y in existing_positions:
                    distance = np.sqrt((candidate_x - ex_x)**2 + (candidate_y - ex_y)**2)
                    if distance < min_distance:
                        conflict = True
                        break
                
                if not conflict:
                    return candidate_x, candidate_y
        
        # Fallback: use offset even if there's minor overlap
        return x_pos + base_offset, y_pos + base_offset
    
def movement_direction(num_agents,evals, evec, tolerance=1e-6):
    movement_vec = torch.zeros(num_agents, dtype=torch.float32)
    
    for i in range(num_agents):
        if evals[i].real < 0:
            movement_vec += evec[:, i].real  # Use real part of eigenvector
        elif evals[i].real == 0:
            print('0 eval detected')
            pass
    
    # Count how many elements are approximately the same
    unique_values = []
    counts = []
    
    for i, val in enumerate(movement_vec):
        found_match = False
        for j, unique_val in enumerate(unique_values):
            if torch.isclose(val, unique_val, atol=tolerance):
                counts[j] += 1
                found_match = True
                break
        
        if not found_match:
            unique_values.append(val)
            counts.append(1)
    
   
    
    return movement_vec, unique_values, counts

def stability_analysis(num_agents,field,results):
    # Initialize dictionary outside the loop
    stability_result = {}

    # Check if results exist and have equilibria
    
    for i in range(len(results['unique_final_positions'])):
        # Preserve original state before modification
        original_pos = field.agents_pos.clone()
        
        # Set position for current equilibrium
        field.agents_pos =results['unique_final_positions'][i].clone()
        
        # Compute Jacobian and eigenvalues
        jac = jc.compute_jacobian_optimized(field, position=field.agents_pos, device='cpu')
        evals, evec = torch.linalg.eig(jac)
        
        # Analyze movement direction
        _, unique_vals, counts = movement_direction(num_agents=num_agents, evals=evals, evec=evec)
        
        # Classify stability based on number of unique movement directions
        if torch.all(evals.real < 0):
            stability_type = 'stable'
        elif len(unique_vals) == 1:
            stability_type = 'line-stable'
        elif len(unique_vals) == 2:
            stability_type = f'({num_agents-1},1) stable'
        elif len(unique_vals) == 3:
            if num_agents==3:
                stability_type='unstable'
            else:    
                stability_type = f'(1,{num_agents-2},1) stable'
        elif len(unique_vals) == num_agents:
            stability_type = 'unstable'
        # Store result
        stability_result[f'E{i+1}'] = {
            'stability_type': stability_type,
            'unique_values': len(unique_vals),
            'counts': counts,
            'position': results['unique_final_positions'][i]
        }
        
        # Restore original state
        field.agents_pos = original_pos
    
    return stability_result

# Helper function to find a node by display label, searching in priority order
def find_node_by_label(label, G,preferred_branches=None):
    """Find a node with the given display label, preferring certain branch types."""
    if preferred_branches is None:
        preferred_branches = ['main', 'left', 'right']
    
    for branch_type in preferred_branches:
        matching = [n for n in G.nodes() if G.nodes[n].get('display_label') == label 
                    and G.nodes[n].get('branch_type') == branch_type]
        if matching:
            return matching[-1]  # Return the last one (most recent)
    return None

def fig_to_array(fig):
        """Convert a matplotlib figure to a numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)

def fig_to_svg_bytes(fig):
        """Convert a matplotlib figure to SVG bytes."""
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()

# Matches a full <image .../> element that carries embedded PNG data.
# Captured groups: (1) all attributes string, (2) the full match.
_PNG_IMAGE_ELEM_RE = re.compile(
    r'<image\b([^>]*?data:image/png;base64[^>]*?)/>',
    re.DOTALL,
)

# Matches a <g id="node_subfig_N"> ... <image .../> ... </g> block produced by
# matplotlib when artist.set_gid() is used on an AnnotationBbox.
# Group 1 = gid value, Group 2 = image attributes string.
_GID_IMAGE_BLOCK_RE = re.compile(
    r'<g\s+id="(node_subfig_\d+)"[^>]*>'
    r'(?:(?!</g>).)*?'
    r'<image\b([^>]*?data:image/png;base64[^>]*?)/>'
    r'(?:(?!</g>).)*?'
    r'</g>',
    re.DOTALL,
)

# Helpers for extracting individual attributes from an attribute string
def _svg_attr(attrs, name):
    """Return the value of a named attribute from an SVG attribute string."""
    m = re.search(rf'\b{re.escape(name)}=["\']([^"\']*)["\']', attrs)
    return m.group(1) if m else None

# For extracting sub-SVG dimensions
_SVG_VIEWBOX_RE = re.compile(
    r'\bviewBox=["\']([0-9.e+\-]+)\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)["\']'
)
_SVG_WH_PT_RE = re.compile(
    r'<svg\b[^>]*\bwidth=["\']([0-9.]+)pt["\'][^>]*\bheight=["\']([0-9.]+)pt["\']'
    r'|<svg\b[^>]*\bheight=["\']([0-9.]+)pt["\'][^>]*\bwidth=["\']([0-9.]+)pt["\']',
    re.DOTALL,
)
_SVG_OPEN_END_RE = re.compile(r'<svg\b[^>]*>')

def _subfig_viewbox(subfig_str):
    """Return (x0, y0, w, h) viewBox for a sub-figure SVG string."""
    vb = _SVG_VIEWBOX_RE.search(subfig_str)
    if vb:
        return (float(vb.group(1)), float(vb.group(2)),
                float(vb.group(3)), float(vb.group(4)))
    wh = _SVG_WH_PT_RE.search(subfig_str)
    if wh:
        w = float(wh.group(1) or wh.group(4))
        h = float(wh.group(2) or wh.group(3))
        return (0.0, 0.0, w, h)
    return (0.0, 0.0, 500.0, 500.0)

def _subfig_body(subfig_str):
    """Return the inner content of an SVG (everything between opening and closing tag)."""
    open_m = _SVG_OPEN_END_RE.search(subfig_str)
    if not open_m:
        return subfig_str
    close_pos = subfig_str.rfind('</svg>')
    if close_pos == -1:
        return subfig_str[open_m.end():]
    return subfig_str[open_m.end():close_pos]

def compose_svg_with_subfigures(main_svg_bytes, subfig_bytes_list):
        """
        Replace rasterized PNG ``<image/>`` elements in an SVG with inline ``<svg>``
        elements whose content is the corresponding sub-figure, so that the result is
        fully vector and renderable in Adobe Illustrator / Acrobat.

        Matplotlib's SVG backend places AnnotationBbox images inside a coordinate
        group that applies a vertical flip.  To compensate, the sub-figure body is
        wrapped in ``<g transform="translate(0,H) scale(1,-1)">`` so it appears
        right-side-up after the parent flip is applied.

        Parameters
        ----------
        main_svg_bytes : bytes
            The main (tree) SVG as raw bytes.
        subfig_bytes_list : list of bytes
            Sub-figure SVGs in the same order as the AnnotationBbox artists
            (i.e. the order in ``placed_figures``).

        Returns
        -------
        bytes
            Composed SVG with all PNG images replaced by inline vector content.
        """
        svg_str = main_svg_bytes.decode('utf-8')

        # ------------------------------------------------------------------
        # GID-based matching (preferred): subfig_bytes_list is a dict
        # { gid_str: bytes } built in visualization.py.
        # We locate each <g id="node_subfig_N"> block, grab the <image/>
        # inside it, and replace with an inline <svg> for vector fidelity.
        # ------------------------------------------------------------------
        if isinstance(subfig_bytes_list, dict):
            gid_map = subfig_bytes_list  # { gid: svg_bytes }

            gid_blocks = list(_GID_IMAGE_BLOCK_RE.finditer(svg_str))
            if not gid_blocks:
                # Fallback: GID groups not present in SVG — return as-is
                return main_svg_bytes

            def _make_nested_svg(img_attrs, subfig_bytes, outer_transform=None):
                x         = _svg_attr(img_attrs, 'x')         or '0'
                y         = _svg_attr(img_attrs, 'y')         or '0'
                width     = _svg_attr(img_attrs, 'width')     or '100'
                height    = _svg_attr(img_attrs, 'height')    or '100'
                transform = _svg_attr(img_attrs, 'transform') or outer_transform
                subfig_str = subfig_bytes.decode('utf-8')
                x0, y0, vb_w, vb_h = _subfig_viewbox(subfig_str)
                body = _subfig_body(subfig_str)
                inner_body = (
                    f'<g transform="translate(0,{vb_h}) scale(1,-1)">'
                    + body + '</g>'
                )
                transform_attr = f' transform="{transform}"' if transform else ''
                return (
                    f'<svg x="{x}" y="{y}" width="{width}" height="{height}"'
                    f'{transform_attr} '
                    f'viewBox="{x0} {y0} {vb_w} {vb_h}" '
                    f'preserveAspectRatio="xMidYMid meet" '
                    f'xmlns="http://www.w3.org/2000/svg" '
                    f'xmlns:xlink="http://www.w3.org/1999/xlink">'
                    + inner_body + '</svg>'
                )

            parts = []
            offset = 0

            # Build a lookup: gid -> block match, image-attrs string
            # Also extract the outer-group transform (if any) to pass into
            # the nested svg so placement is preserved even when the <image>
            # has no transform attribute itself.
            _OUTER_TRANSFORM_RE = re.compile(
                r'<g\s+id="[^"]*"[^>]*\btransform="([^"]*)"', re.DOTALL)

            for block_match in gid_blocks:
                gid = block_match.group(1)
                img_attrs = block_match.group(2)
                if gid not in gid_map:
                    continue

                outer_t_m = _OUTER_TRANSFORM_RE.match(block_match.group(0))
                outer_transform = outer_t_m.group(1) if outer_t_m else None

                # Replace just the <image .../> inside this block keeping
                # the surrounding <g id="..."> wrapper intact.
                block_str = block_match.group(0)
                image_in_block_m = _PNG_IMAGE_ELEM_RE.search(block_str)
                if image_in_block_m:
                    nested = _make_nested_svg(img_attrs, gid_map[gid], outer_transform)
                    new_block = (
                        block_str[:image_in_block_m.start()]
                        + nested
                        + block_str[image_in_block_m.end():]
                    )
                else:
                    new_block = block_str

                parts.append(svg_str[offset:block_match.start()])
                parts.append(new_block)
                offset = block_match.end()

            parts.append(svg_str[offset:])
            return ''.join(parts).encode('utf-8')

        # ------------------------------------------------------------------
        # Legacy sequential matching (fallback for old callers that pass a list).
        # ------------------------------------------------------------------
        matches = list(_PNG_IMAGE_ELEM_RE.finditer(svg_str))

        if len(matches) == 0:
            return main_svg_bytes

        if len(matches) != len(subfig_bytes_list):
            print(
                f"SVG compositing: found {len(matches)} rasterized <image> element(s) but "
                f"{len(subfig_bytes_list)} sub-figure(s). Skipping compositing."
            )
            return main_svg_bytes

        parts = []
        offset = 0
        for match, subfig_bytes in zip(matches, subfig_bytes_list):
            attrs = match.group(1)
            x         = _svg_attr(attrs, 'x')         or '0'
            y         = _svg_attr(attrs, 'y')         or '0'
            width     = _svg_attr(attrs, 'width')     or '100'
            height    = _svg_attr(attrs, 'height')    or '100'
            # Preserve the transform that matplotlib uses for placement + Y-flip
            transform = _svg_attr(attrs, 'transform')

            subfig_str = subfig_bytes.decode('utf-8')
            x0, y0, vb_w, vb_h = _subfig_viewbox(subfig_str)
            body = _subfig_body(subfig_str)

            # Matplotlib's image transform is typically matrix(w, 0, 0, -h, tx, ty)
            # — the negative h-scale is a Y-flip applied to the PNG data so it
            # displays correctly.  When we embed a proper SVG (Y-axis already correct)
            # inside that same flipped coordinate space we must compensate with an
            # inner flip within the sub-figure's own viewBox coordinate space.
            inner_body = (
                f'<g transform="translate(0,{vb_h}) scale(1,-1)">'
                + body
                + '</g>'
            )

            transform_attr = f' transform="{transform}"' if transform else ''

            nested_svg = (
                f'<svg x="{x}" y="{y}" width="{width}" height="{height}"'
                f'{transform_attr} '
                f'viewBox="{x0} {y0} {vb_w} {vb_h}" '
                f'preserveAspectRatio="xMidYMid meet" '
                f'xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink">'
                + inner_body
                + '</svg>'
            )

            parts.append(svg_str[offset:match.start()])
            parts.append(nested_svg)
            offset = match.end()

        parts.append(svg_str[offset:])
        return ''.join(parts).encode('utf-8')

def load_image(img_source):
        """Load image from various sources."""
        if isinstance(img_source, str):
            # It's a file path
            return plt.imread(img_source)
        elif isinstance(img_source, plt.Figure):
            # It's a matplotlib figure
            return fig_to_array(img_source)
        elif isinstance(img_source, np.ndarray):
            # It's already an array
            return img_source
        elif hasattr(img_source, 'convert'):
            # It's a PIL Image
            return np.array(img_source)
        else:
            raise ValueError(f"Unsupported image type: {type(img_source)}")

def process_matrix_tree(matrix,num_agents, reach_parameters,reach_start,reach_end,key_tolerance=2):
        """Process a matrix and extract segment info for tree visualization.
        
        Parameters
        ----------
        matrix : dict
            Bifurcation matrix containing 'max', 'min', etc.
        key_tolerance : int
            Minimum distance between keys to include. If next key is closer than this, skip current key.
        """
        # Get bifurcation types for this matrix
        bifurcation_types = bifurcation_type_helper(
            matrix=matrix, 
            reach_parameters=reach_parameters
        )
        
        # Find cycle locations
        locs = torch.where(
            torch.round(matrix['min'][:,0], decimals=2) != 
            torch.round(matrix['max'][:,0], decimals=2)
        )
        excluded_indices = set(locs[0].tolist())
        
        # Get cycle reach parameters if they exist
        cycle_reach_params = []
        if len(excluded_indices) > 0:
            for idx in excluded_indices:
                cycle_reach_params.append(reach_parameters[idx][0].item())
        
        # Get sorted keys to check distances
        all_keys_sorted = sorted([int(k) for k in bifurcation_types.keys()])
        
        # Collect bifurcation info, skipping keys that are too close to next key
        bifurcation_info = []
        for key, value in bifurcation_types.items():
            if int(key) in excluded_indices:
                continue
            
            # Check if next key is too close
            key_int = int(key)
            if key_int in all_keys_sorted:
                key_idx = all_keys_sorted.index(key_int)
                if key_idx < len(all_keys_sorted) - 1:
                    next_key = all_keys_sorted[key_idx + 1]
                    if next_key - key_int < key_tolerance:
                        continue  # Skip this key, too close to next
            
            bifurcation_info.append({
                'reach': value['reach_parameter'],
                'type': value['type'],
                'classification': value['classification_new'],
                'key': key
            })
        
        # Sort by reach parameter
        bifurcation_info.sort(key=lambda x: x['reach'])
                
        # Create boundaries
        boundaries = [reach_start] + \
                     [info['reach'] for info in bifurcation_info] + \
                     [reach_end]
        
        # Build final segments with labels
        final_boundaries = []
        final_labels = []
        
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            
            # Check for cycles in this region
            cycles_in_region = []
            cycle_indices = []
            for idx, cycle_reach in enumerate(cycle_reach_params):
                if x_start < cycle_reach < x_end:
                    cycles_in_region.append(cycle_reach)
                    cycle_indices.append(idx)
            
            # Get original label
            if i == len(boundaries) - 2:
                original_label = f'({num_agents})'
            elif i < len(bifurcation_info):
                original_label = bifurcation_info[i]['classification']
            else:
                original_label = ''
            
            if cycles_in_region:
                sorted_pairs = sorted(zip(cycles_in_region, cycle_indices))
                cycles_in_region = [val for val, idx in sorted_pairs]
                cycle_indices = [idx for val, idx in sorted_pairs]
                
                min_idx = cycle_indices[0]
                max_idx = cycle_indices[-1]
                
                cycle_start = x_start
                cycle_end = cycle_reach_params[min(len(cycle_reach_params) - 1, max_idx + 2)]
                
                final_boundaries.append((cycle_start, cycle_end))
                final_labels.append('Cycles')
                
                if cycle_end < x_end:
                    final_boundaries.append((cycle_end, x_end))
                    final_labels.append(original_label)
            else:
                final_boundaries.append((x_start, x_end))
                final_labels.append(original_label)
        
        # Remove consecutive duplicate labels (keep first occurrence)
        dedupe_boundaries = []
        dedupe_labels = []
        for i, (boundary, label) in enumerate(zip(final_boundaries, final_labels)):
            if i == 0 or label != dedupe_labels[-1]:
                dedupe_boundaries.append(boundary)
                dedupe_labels.append(label)
            else:
                # Extend the previous boundary to include this one
                prev_start, _ = dedupe_boundaries[-1]
                _, curr_end = boundary
                dedupe_boundaries[-1] = (prev_start, curr_end)
        
        return {
            'boundaries': dedupe_boundaries,
            'labels': dedupe_labels
        }
    
def get_new_labels(parent_labels, current_labels):
        """Find where lists diverge and return remaining labels from that point plus the divergence index."""
        # Find the first index where they differ
        for i in range(min(len(parent_labels), len(current_labels))):
            if parent_labels[i] != current_labels[i]:
                return current_labels[i:], i
        
        # If parent is shorter, return the remaining current labels
        if len(current_labels) > len(parent_labels):
            return current_labels[len(parent_labels):], len(parent_labels)
        
        # Lists are identical or current is shorter
        return [], -1

def make_unique_node_name(label, suffix, label_counts):
    """
    Create a unique node name by tracking label counts per branch.
    If a label has been seen before in this branch, append a counter.
    
    Args:
        label: The base label (e.g., '$(6)$' or 'Cycles')
        suffix: The branch suffix (e.g., '_m', '_l0', '_r1')
        label_counts: A dict tracking how many times each label has been used for this branch suffix
    
    Returns
    -------
        A unique node name
    """
    # Create a key for tracking this label on this branch type
    key = (label, suffix)
    
    if key not in label_counts:
        label_counts[key] = 0
    
    count = label_counts[key]
    label_counts[key] += 1
    
    if count == 0:
        # First occurrence - no counter needed
        return f'{label}{suffix}'
    else:
        # Subsequent occurrences - add counter
        return f'{label}_{count}{suffix}'
    
def process_matrix(matrix,num_agents,reach_parameters,reach_start,reach_end):
        # Get bifurcation types for this matrix
        bifurcation_types = bifurcation_type_helper(
            matrix=matrix, 
            reach_parameters=reach_parameters
        )
        
        # Find cycle locations
        locs = torch.where(
            torch.round(matrix['min'][:,0], decimals=2) != 
            torch.round(matrix['max'][:,0], decimals=2)
        )
        excluded_indices = set(locs[0].tolist())
        
        # Get cycle reach parameters if they exist
        cycle_reach_params = []
        if len(excluded_indices) > 0:
            for idx in excluded_indices:
                cycle_reach_params.append(reach_parameters[idx][0].item())
        
        # Collect bifurcation info
        bifurcation_info = []
        for key, value in bifurcation_types.items():
            if int(key) in excluded_indices:
                continue
            
            bifurcation_info.append({
                'reach': value['reach_parameter'],
                'type': value['type'],
                'classification': value['classification_new'],
                'key': key
            })
        
        # Sort by reach parameter
        bifurcation_info.sort(key=lambda x: x['reach'])
        
        # Create boundaries
        boundaries = [reach_start] + \
                     [info['reach'] for info in bifurcation_info] + \
                     [reach_end]
        
        # Build final segments with labels
        final_boundaries = []
        final_labels = []
        
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            
            # Check for cycles in this region
            cycles_in_region = []
            cycle_indices = []
            for idx, cycle_reach in enumerate(cycle_reach_params):
                if x_start < cycle_reach < x_end:
                    cycles_in_region.append(cycle_reach)
                    cycle_indices.append(idx)
            
            # Get original label
            if i == len(boundaries) - 2:
                original_label = f'$({num_agents})$'
            elif i < len(bifurcation_info):
                original_label = bifurcation_info[i]['classification']
            else:
                original_label = ''
            
            if cycles_in_region:
                sorted_pairs = sorted(zip(cycles_in_region, cycle_indices))
                cycles_in_region = [val for val, idx in sorted_pairs]
                cycle_indices = [idx for val, idx in sorted_pairs]
                
                min_idx = cycle_indices[0]
                max_idx = cycle_indices[-1]
                
                cycle_start = x_start
                cycle_end = cycle_reach_params[min(len(cycle_reach_params) - 1, max_idx + 2)]
                
                final_boundaries.append((cycle_start, cycle_end))
                final_labels.append('Cycles')
                
                if cycle_end < x_end:
                    final_boundaries.append((cycle_end, x_end))
                    final_labels.append(original_label)
            else:
                final_boundaries.append((x_start, x_end))
                final_labels.append(original_label)
        
        return {
            'boundaries': final_boundaries,
            'labels': final_labels
        }
    
def sigma_to_y(sigma,box_height=10, rect_y_start=1, sigma_min=0.0, sigma_max=1.0):
        return rect_y_start + (sigma - sigma_min) / (sigma_max - sigma_min) * box_height

# Draw main rectangle
def draw_rectangle(segments, x_pos, label_to_color,rect_width, edge_color='black', edge_width=2, start_from_sigma=None):
        """
        Draw rectangles, optionally starting only from a specific sigma value.
        
        Parameters
        ----------
        segments : dict
            Segment boundaries and labels
        x_pos : float
            X position for rectangle
        edge_color : str
            Color for rectangle edges
        edge_width : float
            Width of rectangle edges
        start_from_sigma : float, optional
            Only draw segments BELOW this sigma value (i.e., from reach_start up to start_from_sigma)
        """
        for i, (y_start_sigma, y_end_sigma) in enumerate(segments['boundaries']):
            # If threshold is set, only draw segments that are below it
            if start_from_sigma is not None:
                # Skip segments entirely above the threshold
                if y_start_sigma >= start_from_sigma:
                    continue
                
                # Trim segment if it extends above the threshold
                if y_end_sigma > start_from_sigma:
                    y_end_sigma = start_from_sigma
            
            segment_y_start = sigma_to_y(y_start_sigma)
            segment_y_end = sigma_to_y(y_end_sigma)
            segment_height = segment_y_end - segment_y_start
            
            if segment_height <= 0:
                continue
            
            label = segments['labels'][i]
            color = label_to_color.get(label, '#CCCCCC')
            
            # Create vertical rectangle
            rectangle = patches.Rectangle(
                (x_pos, segment_y_start),
                rect_width,
                segment_height,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.7
            )
            ax.add_patch(rectangle)
    
# Function to find first difference scanning top to bottom and draw branches
def find_first_difference_and_draw(prev_segments, curr_segments, prev_x, curr_x, rect_width, branch_color='blue'):
    """
    Find where classifications first differ (scanning high to low sigma) and draw branches.
    Returns the sigma value where differences start.
    """
    # Create sorted list of all boundary points (from high to low)
    all_points = set()
    for (start, end) in prev_segments['boundaries']:
        all_points.add(np.round(start,decimals=3))
        all_points.add(np.round(end,decimals=3))
    for (start, end) in curr_segments['boundaries']:
        all_points.add(np.round(start,decimals=3))
        all_points.add(np.round(end,decimals=3))
    
    sorted_points = sorted(all_points, reverse=True)  # High to low
    
    # Build lookup dictionaries
    prev_dict = {}
    for (start, end), label in zip(prev_segments['boundaries'], prev_segments['labels']):
        prev_dict[(start, end)] = label
    
    curr_dict = {}
    for (start, end), label in zip(curr_segments['boundaries'], curr_segments['labels']):
        curr_dict[(start, end)] = label
    
    # Scan from high to low to find first difference
    first_diff_sigma = None
    
    for i in range(len(sorted_points) - 1):
        high_sigma = sorted_points[i]
        low_sigma = sorted_points[i + 1]
        mid_sigma = (high_sigma + low_sigma) / 2
        
        # Find which segment contains this midpoint in each matrix
        prev_label = None
        curr_label = None
        
        for (start, end), label in prev_dict.items():
            if start <= mid_sigma <= end:
                prev_label = label
                break
        
        for (start, end), label in curr_dict.items():
            if start <= mid_sigma <= end:
                curr_label = label
                break
        
        # Check if they differ
        if prev_label != curr_label and prev_label is not None and curr_label is not None:
            first_diff_sigma = high_sigma
            
            # Draw branch at the divergence point (mid_sigma)
            y_pos = sigma_to_y(mid_sigma)
            
            
            break
    
    # Draw connecting line based on display_type
    if first_diff_sigma is not None:
        # Rectangle mode: horizontal line at top
        top_y = sigma_to_y(first_diff_sigma)
        
        if curr_x < prev_x:  # Left branch
            ax.plot(
                [prev_x, curr_x + rect_width],
                [top_y, top_y],
                color=branch_color,
                linewidth=10,
                alpha=0.6,
                linestyle='--'
            )
        else:  # Right branch
            ax.plot(
                [prev_x + rect_width, curr_x],
                [top_y, top_y],
                color=branch_color,
                linewidth=10,
                alpha=0.6,
                linestyle='--'
            )
    
    return first_diff_sigma

# Helper function to get ordered labels for a segment list
def get_ordered_labels(segments, start_from_sigma=None):
    """
    Get ordered labels from segments, optionally filtering by start_from_sigma.
    
    Parameters
    ----------
    segments : dict
        Segment boundaries and labels
    start_from_sigma : float, optional
        Only include segments below this sigma value
    
    Returns
    -------
    list : Ordered labels from high sigma to low sigma (top to bottom)
    """
    ordered_labels = []
    seen = set()
    # Iterate in reverse (high sigma to low sigma, which is top to bottom in plot)
    for i in range(len(segments['boundaries']) - 1, -1, -1):
        y_start_sigma, y_end_sigma = segments['boundaries'][i]
        label = segments['labels'][i]
        
        # Skip segments entirely above the start_from_sigma threshold
        if start_from_sigma is not None and y_start_sigma >= start_from_sigma:
            continue
        
        # Include segment if it has any portion below the threshold
        if label and label not in seen:
            ordered_labels.append(label)
            seen.add(label)
    return ordered_labels
