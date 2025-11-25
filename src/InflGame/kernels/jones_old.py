r"""
.. module:: jones
   :synopsis: Implements the Jones influence kernel for modeling agent interactions in 1D domains.


Jones Influence Kernel Module
=============================

This module implements the Jones influence kernel and its associated computations. The Jones kernel models the influence 
of agents based on a power-law relationship between their positions and resource/bin points.

Mathematical Definitions:
-------------------------
The Jones influence kernel is defined as:

.. math::
    f_i(x_i, b) = \frac{1}{|x_i - b|^{P_i}}

where:
  - :math:`x_i` is the position of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`P_i` is the parameter for agent :math:`i`

Usage:
------
The `influence` function computes the influence of an agent at specific bin points, while the `d_ln_f` function calculates the gradient of the logarithm of the Jones influence kernel.

New vectorized functions are available for improved performance:
- `influence_vectorized` : Compute influence for all agents simultaneously
- `d_ln_f_vectorized` : Compute gradients for all agents simultaneously

Example:
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.kernels.jones import influence, d_ln_f
    from InflGame.kernels.jones import influence_vectorized, d_ln_f_vectorized

    # Define parameters
    num_agents = 3
    parameter_instance = [2.0, 3.0, 4.0]
    agents_pos = np.array([0.2, 0.5, 0.8])
    bin_points = np.linspace(0, 1, 100)

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
    Compute the Jones influence for all agents simultaneously using vectorized operations.
    
    This function calculates the influence matrix where each row represents an agent's
    influence across all bin points, providing significant performance improvements over
    single-agent computations.
    
    The influence is computed as:

    .. math::
        f_i(x_i, b) = \frac{1}{|x_i - b|^{P_i}}

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameters (:math:`P_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`), shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the influence of agent i at bin point j.
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.2, 0.5, 0.8])
    >>> parameters = np.array([2.0, 3.0, 4.0])
    >>> bins = np.linspace(0, 1, 50)
    >>> influences = influence_vectorized(parameters, agents_pos, bins)
    >>> print(influences.shape)
    torch.Size([3, 50])
    """
    
    try:
        # Input validation and conversion
        if isinstance(parameter_instance, list):
            parameter_tensor = torch.tensor(parameter_instance, dtype=torch.float32)
        elif isinstance(parameter_instance, np.ndarray):
            parameter_tensor = torch.from_numpy(parameter_instance.astype(np.float32))
        elif isinstance(parameter_instance, torch.Tensor):
            parameter_tensor = parameter_instance.to(torch.float32)
        else:
            raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
        
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
        
        if isinstance(bin_points, np.ndarray):
            bin_points_tensor = torch.from_numpy(bin_points.astype(np.float32))
        elif isinstance(bin_points, torch.Tensor):
            bin_points_tensor = bin_points.to(torch.float32)
        else:
            raise TypeError(f"bin_points must be np.ndarray or torch.Tensor, got {type(bin_points)}")
        
        # Validate dimensions
        if len(parameter_tensor) != len(agents_pos_tensor):
            raise ValueError(f"parameter_instance length ({len(parameter_tensor)}) must match agents_pos length ({len(agents_pos_tensor)})")
        
        if len(bin_points_tensor) == 0:
            raise ValueError("bin_points cannot be empty")
        
        # Validate parameter values
        if torch.any(parameter_tensor <= 0):
            raise ValueError("All parameters must be positive")
        
        if torch.any(torch.isnan(parameter_tensor)) or torch.any(torch.isinf(parameter_tensor)):
            raise ValueError("parameter_instance contains NaN or infinite values")
        
        if torch.any(torch.isnan(agents_pos_tensor)) or torch.any(torch.isinf(agents_pos_tensor)):
            raise ValueError("agents_pos contains NaN or infinite values")
        
        if torch.any(torch.isnan(bin_points_tensor)) or torch.any(torch.isinf(bin_points_tensor)):
            raise ValueError("bin_points contains NaN or infinite values")
        
        # Vectorized computation using broadcasting
        # Reshape for broadcasting: 
        # agents_pos: (N,) -> (N, 1)
        # bin_points: (K,) -> (1, K)  
        # parameters: (N,) -> (N, 1)
        agents_expanded = agents_pos_tensor.unsqueeze(1)  # Shape: (N, 1)
        bins_expanded = bin_points_tensor.unsqueeze(0)    # Shape: (1, K)
        params_expanded = parameter_tensor.unsqueeze(1)   # Shape: (N, 1)
        
        # Compute absolute differences: (N, 1) - (1, K) = (N, K)
        abs_diff = torch.abs(agents_expanded - bins_expanded)
        
        # Apply small epsilon to avoid division by zero
        abs_diff = torch.maximum(abs_diff, torch.tensor(1e-10, dtype=torch.float32))
        
        # Compute influence matrix using vectorized power operation
        # f_i(x_i, b) = 1 / |x_i - b|^{P_i}
        influence_matrix = 1.0 / torch.pow(abs_diff, params_expanded)  # Shape: (N, K)
        
        # Apply numerical stability bounds
        influence_matrix = torch.clamp(influence_matrix, min=1e-10, max=1e10)
        
        # Final validation
        if torch.any(torch.isnan(influence_matrix)):
            raise RuntimeError("NaN values detected in computed influence matrix")
        
        if torch.any(torch.isinf(influence_matrix)):
            raise RuntimeError("Infinite values detected in computed influence matrix")
        
        return influence_matrix
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Jones vectorized influence computation: {str(e)}") from e


def d_ln_f_vectorized(parameter_instance: Union[list, np.ndarray, torch.Tensor],
                      agents_pos: Union[np.ndarray, torch.Tensor],
                      bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Compute the gradient of the logarithm of the Jones influence kernel for all agents.
    
    This vectorized function calculates gradients for all agents simultaneously,
    providing significant performance improvements over single-agent computations.

    The gradient is calculated as:

    .. math::
        d_{i}(x_i, b) = P_i \cdot \frac{\text{sgn}(x_i - b)}{|x_i - b|} = \frac{P_i}{x_i - b}

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameters (:math:`P_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`), shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the gradient of agent i at bin point j.
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.2, 0.5, 0.8])
    >>> parameters = np.array([2.0, 3.0, 4.0])
    >>> bins = np.linspace(0, 1, 50)
    >>> gradients = d_ln_f_vectorized(parameters, agents_pos, bins)
    >>> print(gradients.shape)
    torch.Size([3, 50])
    """
    
    try:
        # Input validation and conversion
        if isinstance(parameter_instance, list):
            parameter_tensor = torch.tensor(parameter_instance, dtype=torch.float32)
        elif isinstance(parameter_instance, np.ndarray):
            parameter_tensor = torch.from_numpy(parameter_instance.astype(np.float32))
        elif isinstance(parameter_instance, torch.Tensor):
            parameter_tensor = parameter_instance.to(torch.float32)
        else:
            raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
        
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
        
        if isinstance(bin_points, np.ndarray):
            bin_points_tensor = torch.from_numpy(bin_points.astype(np.float32))
        elif isinstance(bin_points, torch.Tensor):
            bin_points_tensor = bin_points.to(torch.float32)
        else:
            raise TypeError(f"bin_points must be np.ndarray or torch.Tensor, got {type(bin_points)}")
        
        # Validate dimensions
        if len(parameter_tensor) != len(agents_pos_tensor):
            raise ValueError(f"parameter_instance length ({len(parameter_tensor)}) must match agents_pos length ({len(agents_pos_tensor)})")
        
        if len(bin_points_tensor) == 0:
            raise ValueError("bin_points cannot be empty")
        
        # Validate parameter values
        if torch.any(parameter_tensor <= 0):
            raise ValueError("All parameters must be positive")
        
        if torch.any(torch.isnan(parameter_tensor)) or torch.any(torch.isinf(parameter_tensor)):
            raise ValueError("parameter_instance contains NaN or infinite values")
        
        if torch.any(torch.isnan(agents_pos_tensor)) or torch.any(torch.isinf(agents_pos_tensor)):
            raise ValueError("agents_pos contains NaN or infinite values")
        
        if torch.any(torch.isnan(bin_points_tensor)) or torch.any(torch.isinf(bin_points_tensor)):
            raise ValueError("bin_points contains NaN or infinite values")
        
        # Vectorized gradient computation using broadcasting
        # Reshape for broadcasting
        agents_expanded = agents_pos_tensor.unsqueeze(1)      # Shape: (N, 1)
        bins_expanded = bin_points_tensor.unsqueeze(0)        # Shape: (1, K)
        params_expanded = parameter_tensor.unsqueeze(1)       # Shape: (N, 1)
        
        # Compute position differences (vectorized)
        diff = agents_expanded - bins_expanded  # Shape: (N, K)
        
        # Apply numerical stability for small differences
        # Use a small epsilon to avoid division by zero
        diff_stable = torch.where(
            torch.abs(diff) < 1e-10,
            torch.sign(diff) * 1e-10,  # Preserve sign but avoid zero
            diff
        )
        
        # Compute gradient: d_ln_f = P_i / (x_i - b)
        gradient_matrix = params_expanded / diff_stable  # Shape: (N, K)
        
        # Apply numerical bounds to prevent extreme values
        gradient_matrix = torch.clamp(gradient_matrix, min=-1000.0, max=1000.0)
        
        # Final validation
        if torch.any(torch.isnan(gradient_matrix)):
            raise RuntimeError("NaN values detected in computed gradient matrix")
        
        if torch.any(torch.isinf(gradient_matrix)):
            raise RuntimeError("Infinite values detected in computed gradient matrix")
        
        return gradient_matrix
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Jones vectorized gradient computation: {str(e)}") from e


# ================= BACKWARD COMPATIBLE FUNCTIONS =================

def influence(agent_id: int,
              parameter_instance: Union[list, np.ndarray, torch.Tensor],
              agents_pos: Union[np.ndarray, torch.Tensor],
              bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Calculates the influence of a single agent using the Jones influence kernel.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations when beneficial.

    The influence is computed as:

    .. math::
        f_i(x_i, b) = \frac{1}{|x_i - b|^{P_i}}

    where:
      - :math:`x_i` is the position of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`P_i` is the parameter for agent :math:`i`

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID (:math:`i`).
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameter(s) unique to the agent's influence distribution (:math:`P_i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
        
    Returns
    -------
    torch.Tensor
        The agent's influence calculated using the Jones method.
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
    IndexError
        If agent_id is out of bounds.
        
    Notes
    -----
    For improved performance when computing influence for multiple agents,
    consider using :func:`influence_vectorized` instead.
    """
    
    try:
        # Input validation
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")
        
        # Convert inputs to proper format for vectorized function
        if isinstance(parameter_instance, list):
            parameter_tensor = torch.tensor(parameter_instance, dtype=torch.float32)
        elif isinstance(parameter_instance, np.ndarray):
            parameter_tensor = torch.from_numpy(parameter_instance.astype(np.float32))
        elif isinstance(parameter_instance, torch.Tensor):
            parameter_tensor = parameter_instance.to(torch.float32)
        else:
            raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
        
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= len(parameter_tensor):
            raise IndexError(f"agent_id {agent_id} is out of bounds for parameter_instance with length {len(parameter_tensor)}")
        
        if agent_id < 0 or agent_id >= len(agents_pos_tensor):
            raise IndexError(f"agent_id {agent_id} is out of bounds for agents_pos with length {len(agents_pos_tensor)}")
        
        # Use vectorized computation and extract single agent result
        influence_matrix = influence_vectorized(parameter_tensor, agents_pos_tensor, bin_points)
        return influence_matrix[agent_id]
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Jones influence computation: {str(e)}") from e


def d_ln_f(agent_id: int,
           parameter_instance: Union[list, np.ndarray, torch.Tensor],
           agents_pos: Union[np.ndarray, torch.Tensor],
           bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Computes the gradient of the logarithm of the Jones influence kernel for a single agent.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations.

    The gradient is calculated as:

    .. math::
        d_{i}(x_i, b) = P_i \cdot \frac{\text{sgn}(x_i - b)}{|x_i - b|} = \frac{P_i}{x_i - b}

    where:
      - :math:`x_i` is the position of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`P_i` is the parameter for agent :math:`i`
      - :math:`\text{sgn}` is the sign function

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID (:math:`i`).
    parameter_instance : list | np.ndarray | torch.Tensor
        Parameter(s) (:math:`P_i`) for the Jones distribution.
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`).
        
    Returns
    -------
    torch.Tensor
        Gradient values :math:`d_{(i,k)}` for all :math:`k` values.
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
    IndexError
        If agent_id is out of bounds.
        
    Notes
    -----
    For improved performance when computing gradients for multiple agents,
    consider using :func:`d_ln_f_vectorized` instead.
    """
    
    try:
        # Input validation
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")
        
        # Convert inputs to proper format for vectorized function
        if isinstance(parameter_instance, list):
            parameter_tensor = torch.tensor(parameter_instance, dtype=torch.float32)
        elif isinstance(parameter_instance, np.ndarray):
            parameter_tensor = torch.from_numpy(parameter_instance.astype(np.float32))
        elif isinstance(parameter_instance, torch.Tensor):
            parameter_tensor = parameter_instance.to(torch.float32)
        else:
            raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
        
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= len(parameter_tensor):
            raise IndexError(f"agent_id {agent_id} is out of bounds for parameter_instance with length {len(parameter_tensor)}")
        
        if agent_id < 0 or agent_id >= len(agents_pos_tensor):
            raise IndexError(f"agent_id {agent_id} is out of bounds for agents_pos with length {len(agents_pos_tensor)}")
        
        # Use vectorized computation and extract single agent result
        d_matrix = d_ln_f_vectorized(parameter_tensor, agents_pos_tensor, bin_points)
        return d_matrix[agent_id]
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Jones gradient computation: {str(e)}") from e