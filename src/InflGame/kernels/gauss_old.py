r"""
.. module:: gauss
   :synopsis: Implements the Gaussian influence kernel for modeling agent interactions with resources in 1D domains.

Gaussian Influence Kernel Module
================================

This module implements the Gaussian influence kernel and its associated computations. The Gaussian kernel models the 
influence of agents based on a Gaussian distribution centered at their positions.

Mathematical Definitions:
-------------------------
The Gaussian influence kernel is defined as:

.. math::
    f_i(x_i, b) = \exp\left(-\frac{(b - x_i)^2}{2\sigma_i^2}\right)

where:
  - :math:`x_i` is the position of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`\sigma_i` is the parameter for agent :math:`i`

Usage:
------
The `influence` function computes the influence of an agent at specific bin points,
 while the `d_ln_f` function calculates the gradient of the logarithm of the Gaussian influence kernel. 
The `symmetric_nash_stability` function computes the parameter needed for the symmetric Nash to be stable.

New vectorized functions are available for improved performance:
- `influence_vectorized` : Compute influence for all agents simultaneously
- `d_ln_f_vectorized` : Compute gradients for all agents simultaneously
- `symmetric_nash_stability_vectorized` : Vectorized stability computation

Example:
--------

.. code-block:: python

  import numpy as np
  import torch
  from InflGame.kernels.gauss import influence, d_ln_f, symmetric_nash_stability
  from InflGame.kernels.gauss import influence_vectorized, d_ln_f_vectorized

  # Define parameters
  num_agents = 3
  parameter_instance = [0.5, 0.5, 0.5]
  agents_pos = np.array([0.2, 0.5, 0.8])
  bin_points = np.linspace(0, 1, 100)
  resource_distribution = np.random.rand(100)

  # Single agent computation (backward compatible)
  influence_values = influence(agent_id=0, parameter_instance=parameter_instance, 
                              agents_pos=agents_pos, bin_points=bin_points)
  print("Influence values:", influence_values)

  # Vectorized computation (all agents at once)
  all_influences = influence_vectorized(parameter_instance=parameter_instance, 
                                       agents_pos=agents_pos, bin_points=bin_points)
  print("All influences shape:", all_influences.shape)  # (num_agents, num_bins)

  # Single agent gradient
  gradient = d_ln_f(agent_id=0, parameter_instance=parameter_instance, 
                   agents_pos=agents_pos, bin_points=bin_points)
  print("Gradient values:", gradient)

  # Vectorized gradients (all agents at once)
  all_gradients = d_ln_f_vectorized(parameter_instance=parameter_instance,
                                   agents_pos=agents_pos, bin_points=bin_points)
  print("All gradients shape:", all_gradients.shape)  # (num_agents, num_bins)
"""

import numpy as np
import torch
from typing import Union, List


# ========================= VECTORIZED FUNCTIONS =========================

def influence_vectorized(parameter_instance: Union[list, np.ndarray, torch.Tensor],
                         agents_pos: Union[np.ndarray, torch.Tensor],
                         bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Compute the Gaussian influence for all agents simultaneously using vectorized operations.
    
    This function calculates the influence matrix where each row represents an agent's
    influence across all bin points, providing significant performance improvements over
    single-agent computations.
    
    The influence is computed as:

    .. math::
        f_i(x_i,b) = \exp\left(-\frac{(b-x_i)^2}{2\sigma_i^2}\right)

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameters (:math:`\sigma_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`), shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the influence of agent i at bin point j.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.2, 0.5, 0.8])
    >>> parameters = np.array([0.1, 0.15, 0.1])
    >>> bins = np.linspace(0, 1, 50)
    >>> influences = influence_vectorized(parameters, agents_pos, bins)
    >>> print(influences.shape)
    torch.Size([3, 50])
    """
    # Convert to tensors with consistent dtype
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float32)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not isinstance(parameter_instance, torch.Tensor):
        parameter_instance = torch.tensor(parameter_instance, dtype=torch.float32)
    
    # Reshape for broadcasting: 
    # agents_pos: (N,) -> (N, 1)
    # bin_points: (K,) -> (1, K)  
    # parameters: (N,) -> (N, 1)
    agents_expanded = agents_pos.unsqueeze(1)  # Shape: (N, 1)
    bins_expanded = bin_points.unsqueeze(0)    # Shape: (1, K)
    params_expanded = parameter_instance.unsqueeze(1)  # Shape: (N, 1)
    
    # Vectorized computation: (N, 1) - (1, K) = (N, K)
    diff_squared = (bins_expanded - agents_expanded) ** 2
    variance = 2 * params_expanded ** 2
    
    # Compute influence matrix: (N, K)
    influence_matrix = torch.exp(-diff_squared / variance)
    
    return influence_matrix


def d_ln_f_vectorized(parameter_instance: Union[list, np.ndarray, torch.Tensor],
                      agents_pos: Union[np.ndarray, torch.Tensor],
                      bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Compute the gradient of the logarithm of the Gaussian influence kernel for all agents.
    
    This vectorized function calculates gradients for all agents simultaneously,
    providing significant performance improvements over single-agent computations.

    The gradient is calculated as:

    .. math::
        d_{i}(x_i,b) = -\frac{(b - x_i)}{\sigma_i^2}

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameters (:math:`\sigma_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`), shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the gradient of agent i at bin point j.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.2, 0.5, 0.8])
    >>> parameters = np.array([0.1, 0.15, 0.1])
    >>> bins = np.linspace(0, 1, 50)
    >>> gradients = d_ln_f_vectorized(parameters, agents_pos, bins)
    >>> print(gradients.shape)
    torch.Size([3, 50])
    """
    # Convert to tensors with consistent dtype
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float32)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not isinstance(parameter_instance, torch.Tensor):
        parameter_instance = torch.tensor(parameter_instance, dtype=torch.float32)
    
    # Reshape for broadcasting
    agents_expanded = agents_pos.unsqueeze(1)      # Shape: (N, 1)
    bins_expanded = bin_points.unsqueeze(0)        # Shape: (1, K)
    params_squared = parameter_instance.unsqueeze(1) ** 2  # Shape: (N, 1)
    
    # Vectorized gradient computation: (N, K)
    gradient_matrix = (bins_expanded - agents_expanded) / params_squared
    
    return gradient_matrix


def symmetric_nash_stability_vectorized(num_agents: int,
                                       d_values: torch.Tensor,
                                       resource_distribution: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Calculate the symmetric stability parameter using vectorized operations.
    
    This function provides improved performance for symmetric Nash equilibrium
    stability calculations through vectorized tensor operations.

    The symmetric stability parameter is given by:

    .. math::
        \sigma^* = \sqrt{\frac{(N-2)}{(N-1)} \cdot \frac{\sum_{b\in \mathbb{B}} (b-x)^2 \cdot B(b)}{\sum_{b\in \mathbb{B}} B(b)}}

    Parameters
    ----------
    num_agents : int
        Number of agents (:math:`N`).
    d_values : torch.Tensor
        Gradient values (:math:`d_{(i,k)}`) at equilibrium, shape (num_agents, num_bins) or (num_bins,).
    resource_distribution : list | np.ndarray | torch.Tensor
        Distribution of resources (:math:`r_k`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Symmetric stability parameter (:math:`\sigma^*`).
        
    Examples
    --------
    >>> import torch
    >>> num_agents = 5
    >>> d_vals = torch.randn(100)  # Gradient values
    >>> resources = torch.rand(100)  # Resource distribution
    >>> sigma_star = symmetric_nash_stability_vectorized(num_agents, d_vals, resources)
    >>> print(f"Sigma*: {sigma_star.item():.4f}")
    """
    # Convert to tensors
    if not isinstance(d_values, torch.Tensor):
        d_values = torch.tensor(d_values, dtype=torch.float32)
    if not isinstance(resource_distribution, torch.Tensor):
        resource_distribution = torch.tensor(resource_distribution, dtype=torch.float32)
    
    # Handle both single agent (1D) and multi-agent (2D) cases
    if d_values.dim() == 2:
        # Multi-agent case: average across agents or use first agent for symmetric case
        d_values = d_values[0]  # Symmetric case - all agents identical
    
    # Vectorized computation
    weighted_sum = torch.sum(d_values ** 2 * resource_distribution)
    total_resources = torch.sum(resource_distribution)
    
    # Stability parameter calculation
    factor = (num_agents - 2) / (num_agents - 1)
    sigma_star = torch.sqrt(factor * weighted_sum / total_resources)
    
    return sigma_star


# ================= BACKWARD COMPATIBLE FUNCTIONS =================

def influence(agent_id: int,
              parameter_instance: Union[list, np.ndarray, torch.Tensor],
              agents_pos: Union[np.ndarray, torch.Tensor],
              bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Calculates the influence of a single agent using the Gaussian influence kernel.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations when beneficial.

    The influence is computed as:

    .. math::
        f_i(x_i,b) = \exp\left(-\frac{(b-x_i)^2}{2\sigma_i^2}\right)

    where:
      - :math:`x_i` is the position of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`\sigma_i` is the parameter for agent :math:`i`

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID.
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameter(s) unique to the agent's influence distribution (:math:`\sigma_i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
        
    Returns
    -------
    torch.Tensor
        The agent's influence calculated using the Gaussian method.
        
    Notes
    -----
    For improved performance when computing influence for multiple agents,
    consider using :func:`influence_vectorized` instead.
    """
    # Use vectorized computation and extract single agent result
    influence_matrix = influence_vectorized(parameter_instance, agents_pos, bin_points)
    return influence_matrix[agent_id]


def d_ln_f(agent_id: int,
           parameter_instance: Union[list, np.ndarray, torch.Tensor],
           agents_pos: Union[np.ndarray, torch.Tensor],
           bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Computes the gradient of the logarithm of the Gaussian influence kernel for a single agent.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations.

    The gradient is calculated as:

    .. math::
        d_{i}(x_i,b) = -\frac{(b - x_i)}{\sigma_i^2}

    where:
      - :math:`x_i` is the position of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`\sigma_i` is the parameter for agent :math:`i`

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID (:math:`i`).
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameter(s) (:math:`\sigma_i`) for the Gaussian distribution.
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`).
        
    Returns
    -------
    torch.Tensor
        Gradient values :math:`d_{(i,k)}` for all :math:`k` values.
        
    Notes
    -----
    For improved performance when computing gradients for multiple agents,
    consider using :func:`d_ln_f_vectorized` instead.
    """
    # Use vectorized computation and extract single agent result
    d_matrix = d_ln_f_vectorized(parameter_instance, agents_pos, bin_points)
    return d_matrix[agent_id]


def symmetric_nash_stability(num_agents: int,
                            d_values: torch.Tensor,
                            resource_distribution: Union[list, np.ndarray]) -> torch.Tensor:
    r"""
    Calculates the symmetric stability parameter (:math:`\sigma^*`) for symmetric Gaussian influence kernels.
    
    This function provides backward compatibility while using optimized internal computations.

    The symmetric stability parameter is given by:

    .. math::
        \sigma^* = \sqrt{\frac{(N-2)}{(N-1)} \cdot \frac{\sum_{b\in \mathbb{B}} (b-x)^2 \cdot B(b)}{\sum_{b\in \mathbb{B}} B(b)}}

    where:
      - :math:`N` is the number of agents
      - :math:`b\in \mathbb{B}` is the set of bin points
      - :math:`B(b)` is the resource distribution at bin point :math:`b` 

    Parameters
    ----------
    num_agents : int
        Number of agents (:math:`N`).
    d_values : torch.Tensor
        Gradient values (:math:`d_{(i,k)}`) at equilibrium.
    resource_distribution : list | np.ndarray
        Distribution of resources (:math:`r_k`).
        
    Returns
    -------
    torch.Tensor
        Symmetric stability parameter (:math:`\sigma^*`).
    """
    return symmetric_nash_stability_vectorized(num_agents, d_values, resource_distribution)