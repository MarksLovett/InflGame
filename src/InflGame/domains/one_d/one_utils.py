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
