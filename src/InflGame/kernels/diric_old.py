r"""
.. module:: diric
   :synopsis: Implements the Dirichlet influence kernel for modeling agent interactions with resources in simplex domains.

Dirichlet Influence Kernel Module
=================================

This module implements the Dirichlet influence kernel and its associated computations. The Dirichlet kernel models the 
influence of agents based on a multivariate probability distribution centered on their position and defined over a simplex.

Mathematical Definitions:
-------------------------
The Dirichlet influence kernel is defined as:

.. math::
    f_i(\alpha, b) = \frac{1}{\beta(\alpha)} \prod_{l=1}^{L} b_{l}^{\alpha_{l} - 1}

where:
  - :math:`\alpha` is the vector of parameters for the Dirichlet distribution, defined by the `param` function.
  - :math:`b` is the bin point.
  - :math:`\beta(\alpha)` is the beta function.

Dependencies:
-------------
- InflGame.utils

Usage:
------
The `param` function generates the alpha parameters for the Dirichlet kernel from agents positions,
while the `influence` function computes the influence of agents at specific bin points.
The `d_ln_f` function calculates the gradient of the logarithm of the Dirichlet PDF with respect to agent positions.

New vectorized functions are available for improved performance:
- `param_vectorized` : Generate alpha parameters for all agents simultaneously
- `influence_vectorized` : Compute influence for all agents simultaneously
- `d_ln_f_vectorized` : Compute gradients for all agents simultaneously

Example:
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.kernels.diric import param, influence, d_ln_f
    from InflGame.kernels.diric import param_vectorized, influence_vectorized, d_ln_f_vectorized

    # Define parameters
    num_agents = 3
    parameter_instance = [2.0, 3.0, 4.0]
    agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
    fixed_pa = 2
    bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

    # Single agent computation (backward compatible)
    alpha_matrix = param(num_agents, parameter_instance, agents_pos, fixed_pa)
    influence_values = influence(agent_id=0, bin_points=bin_points, alpha_matrix=alpha_matrix)
    print("Influence values:", influence_values)

    # Vectorized computation (all agents at once)
    alpha_matrix_vec = param_vectorized(num_agents, parameter_instance, agents_pos, fixed_pa)
    all_influences = influence_vectorized(bin_points=bin_points, alpha_matrix=alpha_matrix_vec)
    print("All influences shape:", all_influences.shape)  # (num_agents, num_bins)

    # Single agent gradient
    gradient = d_ln_f(agent_id=0, agents_pos=agents_pos, bin_points=bin_points, 
                     alpha_matrix=alpha_matrix, fixed_pa=fixed_pa)
    print("Gradient values:", gradient)

    # Vectorized gradients (all agents at once)
    all_gradients = d_ln_f_vectorized(agents_pos=agents_pos, bin_points=bin_points,
                                     alpha_matrix=alpha_matrix_vec, fixed_pa=fixed_pa)
    print("All gradients shape:", all_gradients.shape)  # (num_agents, num_dims, num_bins)
    
"""

import numpy as np
import torch
from scipy.stats import dirichlet
from scipy.special import psi
import InflGame.utils.general as general
from typing import Union, List


# ========================= VECTORIZED FUNCTIONS =========================

def param_vectorized(num_agents: int,
                     parameter_instance: Union[list, np.ndarray, torch.Tensor],
                     agents_pos: Union[list, np.ndarray, torch.Tensor],
                     fixed_pa: int) -> torch.Tensor:
    r"""
    Generate alpha parameters matrix for all agents using vectorized operations.
    
    This function provides significant performance improvements over single-agent
    parameter generation through vectorized tensor operations.

    The alpha matrix is defined as:

    .. math::
        A = \begin{bmatrix}
          \alpha_{1,1} & \alpha_{1,2} & \cdots & \alpha_{1,L} \\ 
          \alpha_{2,1} & \alpha_{2,2} & \cdots & \alpha_{2,L} \\ 
          \vdots & \vdots & \ddots & \vdots \\ 
          \alpha_{N,1} & \alpha_{N,2} & \cdots & \alpha_{N,L}
        \end{bmatrix}

    Parameters
    ----------
    num_agents : int
        Number of agents (:math:`N`).
    parameter_instance : list | np.ndarray | torch.Tensor
        Fixed alpha values for each agent (:math:`\alpha_i`), shape (num_agents,).
    agents_pos : list | np.ndarray | torch.Tensor
        Positions of agents in the space (:math:`x_{i,l}`), shape (num_agents, num_dims).
    fixed_pa : int
        Index of the fixed coordinate direction (:math:`\varphi`).
        
    Returns
    -------
    torch.Tensor
        Alpha matrix (:math:`A`) of shape (num_agents, num_dims), where each row 
        corresponds to the alpha parameters of an agent.
        
    Raises
    -----
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
    
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    >>> parameters = [2.0, 3.0]
    >>> alpha_matrix = param_vectorized(2, parameters, agents_pos, fixed_pa=2)
    >>> print(alpha_matrix.shape)
    torch.Size([2, 3])
    """
    
    try:
        
        if isinstance(parameter_instance, list):
            parameter_tensor = torch.tensor(parameter_instance, dtype=torch.float32)
        elif isinstance(parameter_instance, np.ndarray):
            parameter_tensor = torch.from_numpy(parameter_instance.astype(np.float32))
        elif isinstance(parameter_instance, torch.Tensor):
            parameter_tensor = parameter_instance.to(torch.float32)
        else:
            raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
        
        if isinstance(agents_pos, list):
            agents_pos_tensor = torch.tensor(agents_pos, dtype=torch.float32)
        elif isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}")
        
        # Vectorized alpha computation
        try:
            # Extract fixed parameter positions for all agents: shape (num_agents,)
            fixed_positions = agents_pos_tensor[:, fixed_pa]  # Shape: (N,)
            
            # Expand parameter_tensor and fixed_positions for broadcasting
            params_expanded = parameter_tensor.unsqueeze(1)  # Shape: (N, 1)
            fixed_expanded = fixed_positions.unsqueeze(1)    # Shape: (N, 1)
            
            # Compute alpha matrix using vectorized operations
            # For non-fixed dimensions: alpha_i,l = (alpha_i / x_i,fixed) * x_i,l
            # For fixed dimension: alpha_i,fixed = alpha_i
            alpha_matrix = (params_expanded / fixed_expanded) * agents_pos_tensor  # Shape: (N, L)
            
            # Set fixed parameter column to original parameter values
            alpha_matrix[:, fixed_pa] = parameter_tensor  # Shape: (N,)
            
            # Apply minimum value constraint to prevent numerical issues
            alpha_matrix = torch.maximum(alpha_matrix, torch.tensor(1e-7, dtype=torch.float32))
            
            # Final validation
            if torch.any(torch.isnan(alpha_matrix)):
                raise RuntimeError("NaN values detected in computed alpha matrix")
            
            if torch.any(torch.isinf(alpha_matrix)):
                raise RuntimeError("Infinite values detected in computed alpha matrix")
            
            if torch.any(alpha_matrix <= 0):
                raise RuntimeError("Alpha matrix contains non-positive values")
            
            return alpha_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Alpha matrix computation failed: {str(e)}") from e
                
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet parameter computation: {str(e)}") from e


def influence_vectorized(bin_points: Union[np.ndarray, torch.Tensor],
                         alpha_matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the Dirichlet influence for all agents simultaneously using vectorized operations.
    
    This function calculates the influence matrix where each row represents an agent's
    influence across all bin points, providing significant performance improvements over
    single-agent computations.

    The influence is calculated as:

    .. math::
        f_i(\alpha, b) = \frac{1}{\beta(\alpha)} \prod_{l=1}^{L} b_{l}^{\alpha_{l} - 1}

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins, num_dims).
    alpha_matrix : torch.Tensor
        Alpha parameters for the Dirichlet influence (:math:`\alpha`), shape (num_agents, num_dims).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the influence of agent i at bin point j.
        
    Raises
    ------
    RuntimeError
        If computation fails due to numerical issues.
    
        
    Examples
    --------
    >>> import numpy as np
    >>> bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    >>> alpha_matrix = torch.tensor([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]])
    >>> influences = influence_vectorized(bin_points, alpha_matrix)
    >>> print(influences.shape)
    torch.Size([2, 2])
    """
    
    try:
        if torch.is_tensor(bin_points):
            bin_points_tensor = bin_points.to(torch.float32)
        elif isinstance(bin_points, np.ndarray):
            bin_points_tensor = torch.from_numpy(bin_points.astype(np.float32))
        else:
            raise TypeError(f"bin_points must be np.ndarray or torch.Tensor, got {type(bin_points)}")
        
        alpha_matrix = alpha_matrix.to(torch.float32)
        num_bins, num_dims = bin_points_tensor.shape
        num_agents, alpha_dims = alpha_matrix.shape
        # Validate simplex constraints for bin_points
        if torch.any(bin_points_tensor <= 0):
            # Allow small negative values due to floating point precision
            bin_points_tensor = torch.clamp(bin_points_tensor, min=1e-10)
        
        # Vectorized influence computation
        try:
            # Pre-allocate result matrix
            influence_matrix = torch.zeros((num_agents, num_bins), dtype=torch.float32)
            
            # Compute influence for each agent using vectorized operations
            for agent_id in range(num_agents):
                agent_alpha = alpha_matrix[agent_id]  # Shape: (num_dims,)
                
                # Vectorized computation across all bin points for this agent
                # Check for zero or negative bin points
                valid_mask = torch.all(bin_points_tensor > 1e-10, dim=1)  # Shape: (num_bins,)
                
                if torch.any(valid_mask):
                    valid_bins = bin_points_tensor[valid_mask]  # Shape: (valid_bins, num_dims)
                    
                    # Compute Dirichlet PDF for valid bin points
                    # Using log-space computation for numerical stability
                    log_influence = torch.zeros(valid_bins.shape[0], dtype=torch.float32)
                    
                    for dim in range(num_dims):
                        log_influence += (agent_alpha[dim] - 1) * torch.log(valid_bins[:, dim] + 1e-10)
                    
                    # Add normalization constant (log beta function)
                    log_beta = torch.sum(torch.lgamma(agent_alpha)) - torch.lgamma(torch.sum(agent_alpha))
                    log_influence -= log_beta
                    
                    # Convert back from log space
                    valid_influence = torch.exp(log_influence)
                    
                    # Fill result matrix
                    influence_matrix[agent_id, valid_mask] = valid_influence
                    # Invalid bin points remain 0
            
            # Final validation
            if torch.any(torch.isnan(influence_matrix)):
                raise RuntimeError("NaN values detected in computed influence matrix")
            
            if torch.any(torch.isinf(influence_matrix)):
                raise RuntimeError("Infinite values detected in computed influence matrix")
            
            return influence_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Influence computation failed: {str(e)}") from e
                
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet vectorized influence computation: {str(e)}") from e


def d_ln_f_vectorized(agents_pos: Union[np.ndarray, torch.Tensor],
                      bin_points: Union[np.ndarray, torch.Tensor],
                      alpha_matrix: torch.Tensor,
                      fixed_pa: int) -> torch.Tensor:
    r"""
    Compute the gradient of the logarithm of the Dirichlet influence kernel for all agents.
    
    This vectorized function calculates gradients for all agents simultaneously,
    providing significant performance improvements over single-agent computations.

    The gradient is calculated as described in the single-agent function.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_{i,l}`), shape (num_agents, num_dims).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins, num_dims).
    alpha_matrix : torch.Tensor
        Alpha parameters for the Dirichlet influence (:math:`\alpha`), shape (num_agents, num_dims).
    fixed_pa : int
        Index of the fixed coordinate direction (:math:`\varphi`).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_dims, num_bins) where element [i,l,j] 
        represents the gradient of agent i in dimension l at bin point j.
        
    Raises
    ------
    RuntimeError
        If computation fails due to numerical issues.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    >>> bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    >>> alpha_matrix = torch.tensor([[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]])
    >>> gradients = d_ln_f_vectorized(agents_pos, bin_points, alpha_matrix, fixed_pa=2)
    >>> print(gradients.shape)
    torch.Size([2, 3, 2])
    """
    
    try:
        # Input validation and conversion
        if isinstance(bin_points, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
        
        if torch.is_tensor(bin_points):
            bin_points_tensor = bin_points.to(torch.float32)
        elif isinstance(bin_points, np.ndarray):
            bin_points_tensor = torch.from_numpy(bin_points.astype(np.float32))
        else:
            raise TypeError(f"bin_points must be np.ndarray or torch.Tensor, got {type(bin_points)}")
        
        alpha_matrix = alpha_matrix.to(torch.float32)
        num_agents, num_dims = agents_pos_tensor.shape
        num_bins, bin_dims = bin_points_tensor.shape
        
        if torch.any(bin_points_tensor <= 0):
            # Allow small values, clamp to avoid log(0)
            bin_points_tensor = torch.clamp(bin_points_tensor, min=1e-10)
        
        # Vectorized gradient computation
        try:
            # Pre-allocate result tensor: (num_agents, num_dims, num_bins)
            gradient_matrix = torch.zeros((num_agents, num_dims, num_bins), dtype=torch.float32)
            
            # Compute gradients for all agents using vectorized operations
            for agent_id in range(num_agents):
                agent_pos = agents_pos_tensor[agent_id]  # Shape: (num_dims,)
                agent_alpha = alpha_matrix[agent_id]     # Shape: (num_dims,)
                
                # Common factor for non-fixed dimensions
                c_f = agent_alpha[fixed_pa] / agent_pos[fixed_pa]
                
                # Compute psi terms (digamma function)
                psi_alpha = torch.zeros(num_dims, dtype=torch.float32)
                for dim in range(num_dims):
                    psi_alpha[dim] = psi(agent_alpha[dim].item())
                
                psi_sum = psi(torch.sum(agent_alpha).item())
                
                # Vectorized computation for each dimension
                for dim in range(num_dims):
                    if dim == fixed_pa:
                        # Fixed dimension: compute as sum of other dimensions
                        gradient_matrix[agent_id, dim] = torch.zeros(num_bins)
                    else:
                        # Non-fixed dimension: vectorized computation across all bin points
                        log_bins = torch.log(bin_points_tensor[:, dim] + 1e-10)  # Shape: (num_bins,)
                        gradient_matrix[agent_id, dim] = c_f * (log_bins - psi_alpha[dim] + psi_sum)
                
                # Compute fixed dimension as weighted sum of other dimensions
                fixed_gradient = torch.zeros(num_bins, dtype=torch.float32)
                for dim in range(num_dims):
                    if dim != fixed_pa:
                        weight = -agent_pos[dim] / agent_pos[fixed_pa]
                        fixed_gradient += weight * gradient_matrix[agent_id, dim]
                
                gradient_matrix[agent_id, fixed_pa] = fixed_gradient
            
            # Final validation
            if torch.any(torch.isnan(gradient_matrix)):
                raise RuntimeError("NaN values detected in computed gradient matrix")
            
            if torch.any(torch.isinf(gradient_matrix)):
                raise RuntimeError("Infinite values detected in computed gradient matrix")
            
            return gradient_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Gradient computation failed: {str(e)}") from e
                
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet vectorized gradient computation: {str(e)}") from e


# ================= BACKWARD COMPATIBLE FUNCTIONS =================


def param(num_agents: int,
          parameter_instance: Union[list, np.ndarray, torch.Tensor],
          agents_pos: Union[list, np.ndarray, torch.Tensor],
          fixed_pa: int) -> torch.Tensor:
    r"""
    Generates a matrix of alpha parameters for all agents based on their positions and a fixed parameter.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations when beneficial.

    The alpha matrix is defined as:

    .. math::
        A = \begin{bmatrix}
          \alpha_{1,1} & \alpha_{1,2} & \cdots & \alpha_{1,L} \\ 
          \alpha_{2,1} & \alpha_{2,2} & \cdots & \alpha_{2,L} \\ 
          \vdots & \vdots & \ddots & \vdots \\ 
          \alpha_{N,1} & \alpha_{N,2} & \cdots & \alpha_{N,L}
        \end{bmatrix}

    The alpha parameters are calculated as follows:

    - If :math:`l \neq \varphi`:

      .. math::
        \alpha_{i,l} = \frac{\alpha_{i}}{x_{i,\varphi}} \cdot x_{i,l}

    - If :math:`l = \varphi`:

      .. math::
        \alpha_{i,l} = \alpha_{i}

    Where:
        - :math:`N` is the number of agents
        - :math:`L` is the number of dimensions
        - :math:`x_{i,l}` is the position of agent :math:`i` in dimension :math:`l`
        - :math:`\varphi` is the fixed parameter index
        - :math:`\alpha_{i}` is the parameter for agent :math:`i` and is a fixed parameter.

    Parameters
    ----------
    num_agents : int
        Number of agents (:math:`N`).
    parameter_instance : list | np.ndarray | torch.Tensor
        Fixed alpha values for each agent (:math:`\alpha_i`).
    agents_pos : list | np.ndarray | torch.Tensor
        Positions of agents in the space (:math:`x_{i,l}`).
    fixed_pa : int
        Index of the fixed coordinate direction (:math:`\varphi`).
        
    Returns
    -------
    torch.Tensor
        Alpha matrix (:math:`A`), where each row corresponds to the alpha parameters of an agent.
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
    IndexError
        If fixed_pa is out of bounds.
        
    Notes
    -----
    For improved performance with large agent populations,
    consider using :func:`param_vectorized` directly.
    """
    
    try:
        # Use vectorized computation for efficiency
        return param_vectorized(num_agents, parameter_instance, agents_pos, fixed_pa)
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet parameter computation: {str(e)}") from e


def influence(agent_id: int,
              bin_points: Union[np.ndarray, torch.Tensor],
              alpha_matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the influence of a single agent using the Dirichlet kernel.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations when beneficial.

    The influence is calculated as the Dirichlet PDF evaluated at the bin points:

    .. math::
        f_i(\alpha, b) = \frac{1}{\beta(\alpha)} \prod_{l=1}^{L} b_{l}^{\alpha_{l} - 1}

    where:
      - :math:`L` is the number of dimensions.
      - :math:`b` is the bin point.
      - :math:`\alpha` is the vector of parameters for the Dirichlet distribution calculated by the `param` function.
      - :math:`\beta(\alpha)` is the beta function.

    Parameters
    ----------
    agent_id : int
        The ID of the agent (:math:`i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
    alpha_matrix : torch.Tensor
        Alpha parameters for the Dirichlet influence (:math:`\alpha`).
        
    Returns
    -------
    torch.Tensor
        Influence values for the agent at each bin point.
        
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
        
        if not isinstance(alpha_matrix, torch.Tensor):
            raise TypeError(f"alpha_matrix must be torch.Tensor, got {type(alpha_matrix)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= alpha_matrix.shape[0]:
            raise IndexError(f"agent_id {agent_id} is out of bounds for alpha_matrix with {alpha_matrix.shape[0]} agents")
        
        # Use vectorized computation and extract single agent result
        influence_matrix = influence_vectorized(bin_points, alpha_matrix)
        return influence_matrix[agent_id]
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet influence computation: {str(e)}") from e


def d_ln_f(agent_id: int,
           agents_pos: Union[np.ndarray, torch.Tensor],
           bin_points: Union[np.ndarray, torch.Tensor],
           alpha_matrix: torch.Tensor,
           fixed_pa: int) -> torch.Tensor:
    r"""
    Computes the gradient of the logarithm of the Dirichlet PDF with respect to agent positions for a single agent.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations.

    The gradient is calculated as follows:

    - For :math:`l \neq \varphi`:

      .. math::
        d_{(i,l)} = \frac{\alpha_{i,\varphi}}{x_{i,\varphi}} \left(\ln(b_{l}) - \psi(\alpha_{i,l}) + \psi\left(\sum_{l=1}^{L} \alpha_{i,l}\right)\right)

    - For :math:`l = \varphi`:

      .. math::
        d_{(i,\varphi)} = - \sum_{\substack{j=1 \\ j \neq \varphi}}^{L} \frac{x_{i,j}}{x_{i,\varphi}} \cdot d_{(i,j)}

    where:
      - :math:`L` is the number of dimensions.
      - :math:`b_{l}` is the bin point in dimension :math:`l`.
      - :math:`x_{i,l}` is the position of agent :math:`i` in dimension :math:`l`.
      - :math:`\varphi` is the fixed parameter index.
      - :math:`\alpha_{i,l}` is the parameter for agent :math:`i` in dimension :math:`l`.
      - :math:`\psi` is the digamma function.

    Parameters
    ----------
    agent_id : int
        The ID of the agent (:math:`i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_{i,l}`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
    alpha_matrix : torch.Tensor
        Alpha parameters for the Dirichlet influence (:math:`\alpha`).
    fixed_pa : int
        Index of the fixed coordinate direction (:math:`\varphi`).
        
    Returns
    -------
    torch.Tensor
        Gradient values (:math:`d_{(i,l)}`) for all :math:`l` values.
        Shape: (num_dims, num_bins)
        
    Raises
    ------
    ValueError
        If input parameters are invalid or incompatible.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
    IndexError
        If agent_id or fixed_pa is out of bounds.
        
    Notes
    -----
    For improved performance when computing gradients for multiple agents,
    consider using :func:`d_ln_f_vectorized` instead.
    """
    
    try:
        # Input validation
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")
        
        if not isinstance(alpha_matrix, torch.Tensor):
            raise TypeError(f"alpha_matrix must be torch.Tensor, got {type(alpha_matrix)}")
        
        if not isinstance(fixed_pa, int):
            raise TypeError(f"fixed_pa must be an integer, got {type(fixed_pa)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= alpha_matrix.shape[0]:
            raise IndexError(f"agent_id {agent_id} is out of bounds for alpha_matrix with {alpha_matrix.shape[0]} agents")
        
        # Use vectorized computation and extract single agent result
        d_matrix = d_ln_f_vectorized(agents_pos, bin_points, alpha_matrix, fixed_pa)
        return d_matrix[agent_id]  # Shape: (num_dims, num_bins)
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in Dirichlet gradient computation: {str(e)}") from e


