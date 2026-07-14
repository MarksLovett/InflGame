r"""
.. module:: diric_mode
   :synopsis: Implements the mode-parameterized Dirichlet influence kernel for modeling agent interactions with resources in simplex domains.

Mode-Parameterized Dirichlet Influence Kernel Module
=====================================================

This module implements the mode-parameterized Dirichlet influence kernel and its associated computations. 
Unlike the standard Dirichlet kernel which parameterizes the distribution by a fixed alpha, 
this kernel parameterizes the Dirichlet distribution so that the mode of the distribution 
equals the agent's position on the simplex.

Mathematical Definitions:
-------------------------
The mode-parameterized Dirichlet influence kernel uses the inverse parameterization:

.. math::
    \alpha_{(i,l)} = 1 + \frac{x_{(i,l)}}{\sigma}

where:
  - :math:`\sigma > 0` is the spread parameter (higher values = more spread out)
  - :math:`x_{(i,l)}` is the position of agent :math:`i` in dimension :math:`l`
  - The sum :math:`\alpha_0 = \sum_l \alpha_{(i,l)} = L + 1/\sigma` where :math:`L` is the dimension
  - Higher :math:`\sigma` means more spread out (less concentrated) distribution

The log-density of the Dirichlet distribution is:

.. math::
    \ln f = \ln \Gamma(L + 1/\sigma) - \sum_{l=1}^{L} \ln \Gamma(1 + x_{(i,l)}/\sigma) + \frac{1}{\sigma} \sum_{l=1}^{L} x_{(i,l)} \ln b_l

The gradient with respect to agent position is:

.. math::
    d_{(i,l)} = \frac{1}{\sigma} \left( \ln(b_l) - \psi_0(1 + x_{(i,l)}/\sigma) \right)

where :math:`\psi_0` is the digamma function.

The Hessian is a diagonal matrix:

.. math::
    H = -\frac{1}{\sigma^2} \, \text{diag}(\psi_1(1 + x_{(i,1)}/\sigma), \ldots, \psi_1(1 + x_{(i,L)}/\sigma))

where :math:`\psi_1` is the trigamma function.

Dependencies:
-------------
- InflGame.utils
- scipy.special (psi, polygamma)
- torch
- numpy

Usage:
------
The `param_vectorized` function generates the alpha parameters for the mode-parameterized Dirichlet kernel,
while the `influence_vectorized` function computes the influence of agents at specific bin points.
The `d_ln_f_vectorized` function calculates the gradient of the log-density with respect to agent positions.
The `hessian_vectorized` function calculates the Hessian matrix for stability analysis.

Examples
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.kernels.diric_mode import param_vectorized, influence_vectorized, d_ln_f_vectorized, hessian_vectorized

    # Define parameters
    num_agents = 3
    sigma = 0.2  # Spread parameter (higher = more spread out)
    agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
    bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

    # Compute alpha matrix from positions
    alpha_matrix = param_vectorized(agents_pos, sigma)
    
    # Compute influence for all agents
    all_influences = influence_vectorized(bin_points, alpha_matrix)
    print("All influences shape:", all_influences.shape)  # (num_agents, num_bins)

    # Compute gradients for all agents
    all_gradients = d_ln_f_vectorized(agents_pos, bin_points, sigma)
    print("All gradients shape:", all_gradients.shape)  # (num_agents, num_dims, num_bins)
    
    # Compute Hessian for all agents at each bin point
    hessians = hessian_vectorized(agents_pos, sigma)
    print("Hessians shape:", hessians.shape)  # (num_agents, num_dims, num_dims)
    
"""

import numpy as np
import torch
import InflGame.utils.general as general
from typing import Union, List, Optional


# ========================= JIT-COMPILED HELPER FUNCTIONS =========================

@torch.jit.script
def _param_vectorized_core(
    agents_pos_tensor: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    JIT-compiled core computation for mode-parameterized alpha parameter generation.
    
    The mode parameterization uses inverse concentration:
        alpha_{(i,l)} = 1 + x_{(i,l)} / sigma
    
    Args:
        agents_pos_tensor: Agent positions (N, L) on the simplex
        sigma: Spread parameter (sigma > 0), higher values = more spread out
    
    Returns
    -------
        torch.Tensor: Alpha matrix (N, L)
    """
    # Mode parameterization with inverse concentration: alpha = 1 + x / sigma
    alpha_matrix = 1.0 + agents_pos_tensor / sigma
    
    # Apply minimum value constraint for numerical stability
    alpha_matrix = torch.maximum(alpha_matrix, torch.tensor(1e-10, dtype=torch.float32))
    
    return alpha_matrix


@torch.jit.script
def _influence_log_computation_core(
    bin_points_tensor: torch.Tensor,
    agent_alpha: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core for Dirichlet influence computation in log space.
    
    Args:
        bin_points_tensor: Valid bin points (K, L)
        agent_alpha: Alpha parameters for single agent (L,)
    
    Returns
    -------
        torch.Tensor: Log influence values (K,)
    """
    num_dims = agent_alpha.shape[0]
    num_bins = bin_points_tensor.shape[0]
    
    # Compute log influence using vectorized operations
    log_influence = torch.zeros(num_bins, dtype=torch.float32)
    
    for dim in range(num_dims):
        log_bins = torch.log(bin_points_tensor[:, dim] + 1e-10)
        log_influence += (agent_alpha[dim] - 1.0) * log_bins
    
    # Add normalization constant (log beta function)
    # log(beta(alpha)) = sum(lgamma(alpha_l)) - lgamma(sum(alpha))
    log_beta = torch.sum(torch.lgamma(agent_alpha)) - torch.lgamma(torch.sum(agent_alpha))
    log_influence -= log_beta
    
    return log_influence


@torch.jit.script
def _gradient_computation_core(
    bin_points_tensor: torch.Tensor,
    agent_alpha: torch.Tensor,
    sigma: float,
    psi_alpha: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core for mode-parameterized Dirichlet gradient computation.
    
    The gradient with inverse parameterization is:
        d_{(i,l)} = (1/sigma) * (ln(b_l) - psi_0(1 + x_{(i,l)}/sigma))
                  = (1/sigma) * (ln(b_l) - psi_0(alpha_{(i,l)}))
    
    Args:
        bin_points_tensor: Bin points (K, L)
        agent_alpha: Single agent alpha parameters (L,)
        sigma: Spread parameter
        psi_alpha: Digamma values for alpha parameters (L,)
    
    Returns
    -------
        torch.Tensor: Gradient matrix (L, K)
    """
    num_dims = agent_alpha.shape[0]
    num_bins = bin_points_tensor.shape[0]
    
    # Pre-allocate gradient matrix
    gradient_matrix = torch.zeros((num_dims, num_bins), dtype=torch.float32)
    
    # Compute 1/sigma for gradient scaling
    inv_sigma = 1.0 / sigma
    
    # Vectorized computation for each dimension
    # d_{(i,l)} = (1/sigma) * (ln(b_l) - psi(alpha_l))
    for dim in range(num_dims):
        log_bins = torch.log(bin_points_tensor[:, dim] + 1e-10)  # Shape: (num_bins,)
        gradient_matrix[dim] = inv_sigma * (log_bins - psi_alpha[dim])
    
    return gradient_matrix


@torch.jit.script
def _hessian_computation_core(
    agent_alpha: torch.Tensor,
    sigma: float,
    psi1_alpha: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core for mode-parameterized Dirichlet Hessian computation.
    
    The Hessian with inverse parameterization is a diagonal matrix:
        H_{l,l} = -(1/sigma^2) * psi_1(alpha_{(i,l)})
    
    where psi_1 is the trigamma function.
    
    Args:
        agent_alpha: Single agent alpha parameters (L,)
        sigma: Spread parameter
        psi1_alpha: Trigamma values for alpha parameters (L,)
    
    Returns
    -------
        torch.Tensor: Hessian matrix (L, L) - diagonal
    """
    num_dims = agent_alpha.shape[0]
    
    # Create diagonal Hessian matrix
    # H_{l,l} = -(1/sigma^2) * psi_1(alpha_l)
    inv_sigma_sq = 1.0 / (sigma * sigma)
    hessian_diag = -inv_sigma_sq * psi1_alpha
    hessian_matrix = torch.diag(hessian_diag)
    
    return hessian_matrix


# ========================= VECTORIZED FUNCTIONS =========================

def param_vectorized(agents_pos: Union[list, np.ndarray, torch.Tensor],
                     sigma: float) -> torch.Tensor:
    r"""
    Generate alpha parameters matrix for all agents using mode parameterization.
    
    This function uses the inverse parameterization:
    
    .. math::
        \alpha_{(i,l)} = 1 + \frac{x_{(i,l)}}{\sigma}
    
    This ensures that the mode of the Dirichlet distribution is exactly at the agent's position.
    Higher sigma means more spread out (less concentrated) distribution.

    Parameters
    ----------
    agents_pos : list | np.ndarray | torch.Tensor
        Positions of agents on the simplex (:math:`x_{i,l}`), shape (num_agents, num_dims).
        Each row should sum to 1.
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        Higher values create more spread out (less peaked) distributions.
        
    Returns
    -------
    torch.Tensor
        Alpha matrix of shape (num_agents, num_dims), where each row 
        corresponds to the alpha parameters of an agent.
        
    Raises
    ------
    ValueError
        If sigma is not positive.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    >>> sigma = 0.2  # Spread parameter
    >>> alpha_matrix = param_vectorized(agents_pos, sigma)
    >>> print(alpha_matrix.shape)
    torch.Size([2, 3])
    >>> print(alpha_matrix)  # 1 + pos/sigma for each element
    tensor([[2.0000, 2.5000, 3.5000],
            [3.0000, 3.0000, 2.0000]])
    """
    
    try:
        # Validate sigma
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        
        # Convert inputs to tensors
        if isinstance(agents_pos, list):
            agents_pos_tensor = torch.tensor(agents_pos, dtype=torch.float32)
        elif isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}")
        
        # Use JIT-compiled core for optimal performance
        try:
            alpha_matrix = _param_vectorized_core(agents_pos_tensor, float(sigma))
            
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
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet parameter computation: {str(e)}") from e


def influence_vectorized(bin_points: Union[np.ndarray, torch.Tensor],
                         alpha_matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the mode-parameterized Dirichlet influence for all agents simultaneously.
    
    This function calculates the influence matrix where each row represents an agent's
    influence across all bin points.

    The influence is calculated as the Dirichlet PDF:

    .. math::
        f_i(\alpha, b) = \frac{1}{\text{B}(\alpha)} \prod_{l=1}^{L} b_{l}^{\alpha_{l} - 1}

    where :math:`\alpha_{(i,l)} = 1 + \sigma x_{(i,l)}`.

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins, num_dims).
    alpha_matrix : torch.Tensor
        Alpha parameters from `param_vectorized`, shape (num_agents, num_dims).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the influence of agent i at bin point j.
        
    Raises
    ------
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    >>> alpha_matrix = torch.tensor([[2.0, 2.5, 3.5], [3.0, 3.0, 2.0]])
    >>> influences = influence_vectorized(bin_points, alpha_matrix)
    >>> print(influences.shape)
    torch.Size([2, 2])
    """
    
    try:
        # Convert bin_points to tensor
        if torch.is_tensor(bin_points):
            bin_points_tensor = bin_points.to(torch.float32)
        elif isinstance(bin_points, np.ndarray):
            bin_points_tensor = torch.from_numpy(bin_points.astype(np.float32))
        else:
            raise TypeError(f"bin_points must be np.ndarray or torch.Tensor, got {type(bin_points)}")
        
        alpha_matrix = alpha_matrix.to(torch.float32)

        # Clamp bin points once to avoid log(0); avoids repeated per-agent masking
        bin_points_tensor = bin_points_tensor.clamp(min=1e-10)

        try:
            # ---- FULLY VECTORIZED computation (no Python loops over agents) ----
            # log_bins: (K, L)
            log_bins = torch.log(bin_points_tensor)

            # log_numerator[n, k] = sum_l (alpha[n,l] - 1) * log_bins[k,l]
            # (N, L) @ (L, K) = (N, K)
            log_numerator = (alpha_matrix - 1.0) @ log_bins.T

            # Log-beta normalisation per agent: (N,)
            # log B(alpha) = sum_l lgamma(alpha_l) - lgamma(sum_l alpha_l)
            log_beta = (torch.lgamma(alpha_matrix).sum(dim=1)
                        - torch.lgamma(alpha_matrix.sum(dim=1)))

            # Influence matrix: (N, K)
            influence_matrix = torch.exp(log_numerator - log_beta.unsqueeze(1))

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
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet vectorized influence computation: {str(e)}") from e


def d_ln_f_vectorized(agents_pos: Union[np.ndarray, torch.Tensor],
                      bin_points: Union[np.ndarray, torch.Tensor],
                      sigma: float,
                      alpha_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute the gradient of the log-density for all agents using mode parameterization.
    
    The gradient for the mode-parameterized Dirichlet with inverse parameterization is:

    .. math::
        d_{(i,l)} = \frac{1}{\sigma} \left( \ln(b_l) - \psi_0(1 + x_{(i,l)}/\sigma) \right)

    where :math:`\psi_0` is the digamma function.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_{i,l}`), shape (num_agents, num_dims).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins, num_dims).
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_dims, num_bins) where element [i,l,j] 
        represents the gradient of agent i in dimension l at bin point j.
        
    Raises
    ------
    ValueError
        If sigma is not positive.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    >>> bin_points = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    >>> sigma = 0.2  # Spread parameter
    >>> gradients = d_ln_f_vectorized(agents_pos, bin_points, sigma)
    >>> print(gradients.shape)
    torch.Size([2, 3, 2])
    """

    try:
        # Validate sigma
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Input validation and conversion
        if isinstance(agents_pos, np.ndarray):
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

        # Reuse pre-computed alpha_matrix when available (avoids redundant param_vectorized call)
        if alpha_matrix is None:
            alpha_matrix = param_vectorized(agents_pos_tensor, sigma)
        else:
            alpha_matrix = alpha_matrix.to(torch.float32)

        # Clamp bin points once to avoid log(0)
        bin_points_tensor = bin_points_tensor.clamp(min=1e-10)

        try:
            # ---- FULLY VECTORIZED computation (no Python loops over agents or dims) ----

            # 1. Digamma for all agents × dims at once: (N, L)
            psi_alpha_matrix = torch.special.digamma(alpha_matrix)

            # 2. Log bin points: (K, L) → transposed to (L, K)
            log_bins = torch.log(bin_points_tensor).T  # (L, K)

            # 3. Gradient: (1/sigma) * (log_bins[l,k] - psi_alpha[n,l])
            #    Broadcasting: (1, L, K) - (N, L, 1) → (N, L, K)
            inv_sigma = 1.0 / sigma
            gradient_matrix = inv_sigma * (log_bins.unsqueeze(0) - psi_alpha_matrix.unsqueeze(2))
            # Shape: (N, L, K)

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
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet vectorized gradient computation: {str(e)}") from e


def hessian_vectorized(agents_pos: Union[np.ndarray, torch.Tensor],
                       sigma: float,
                       alpha_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute the Hessian of the log-density for all agents using mode parameterization.
    
    The Hessian for the mode-parameterized Dirichlet is a diagonal matrix:

    .. math::
        H_{l,l} = -\frac{1}{\sigma^2} \psi_1(1 + x_{(i,l)}/\sigma)

    where :math:`\psi_1` is the trigamma function.
    
    Note: The Hessian is independent of the bin points since the second derivative
    with respect to position only involves the normalization constant terms.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents (:math:`x_{i,l}`), shape (num_agents, num_dims).
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        
    Returns
    -------
    torch.Tensor
        Hessian tensor of shape (num_agents, num_dims, num_dims) where element [i,:,:] 
        is the Hessian matrix for agent i (diagonal).
        
    Raises
    ------
    ValueError
        If sigma is not positive.
    RuntimeError
        If computation fails due to numerical issues.
    TypeError
        If input types are not supported.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    >>> sigma = 0.2  # Spread parameter
    >>> hessians = hessian_vectorized(agents_pos, sigma)
    >>> print(hessians.shape)
    torch.Size([2, 3, 3])
    """

    try:
        # Validate sigma
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Input validation and conversion
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")

        # Reuse pre-computed alpha_matrix when available
        if alpha_matrix is None:
            alpha_matrix = param_vectorized(agents_pos_tensor, sigma)
        else:
            alpha_matrix = alpha_matrix.to(torch.float32)

        try:
            # ---- FULLY VECTORIZED computation (no Python loops over agents or dims) ----

            # Trigamma for all agents × dims at once: (N, L)
            psi1_alpha_matrix = torch.special.polygamma(1, alpha_matrix)

            inv_sigma_sq = 1.0 / (sigma * sigma)
            hessian_diags = -inv_sigma_sq * psi1_alpha_matrix  # (N, L)

            # Build a batch of diagonal matrices: (N, L, L)
            hessian_tensor = torch.diag_embed(hessian_diags)

            # Final validation
            if torch.any(torch.isnan(hessian_tensor)):
                raise RuntimeError("NaN values detected in computed Hessian tensor")
            if torch.any(torch.isinf(hessian_tensor)):
                raise RuntimeError("Infinite values detected in computed Hessian tensor")

            return hessian_tensor

        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Hessian computation failed: {str(e)}") from e

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet Hessian computation: {str(e)}") from e


# ================= BACKWARD COMPATIBLE FUNCTIONS =================


def param(agents_pos: Union[list, np.ndarray, torch.Tensor],
          sigma: float) -> torch.Tensor:
    r"""
    Generates a matrix of alpha parameters for all agents based on mode parameterization.
    
    Backward-compatible wrapper for :func:`param_vectorized`.

    Parameters
    ----------
    agents_pos : list | np.ndarray | torch.Tensor
        Positions of agents on the simplex, shape (num_agents, num_dims).
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        
    Returns
    -------
    torch.Tensor
        Alpha matrix of shape (num_agents, num_dims).
    """
    return param_vectorized(agents_pos, sigma)


def influence(agent_id: int,
              bin_points: Union[np.ndarray, torch.Tensor],
              alpha_matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the influence of a single agent using the mode-parameterized Dirichlet kernel.
    
    Backward-compatible wrapper that extracts a single agent's influence from the vectorized computation.

    Parameters
    ----------
    agent_id : int
        The ID of the agent (:math:`i`).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
    alpha_matrix : torch.Tensor
        Alpha parameters from `param_vectorized`.
        
    Returns
    -------
    torch.Tensor
        Influence values for the agent at each bin point.
        
    Raises
    ------
    TypeError
        If input types are not supported.
    IndexError
        If agent_id is out of bounds.
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
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet influence computation: {str(e)}") from e


def d_ln_f(agent_id: int,
           agents_pos: Union[np.ndarray, torch.Tensor],
           bin_points: Union[np.ndarray, torch.Tensor],
           sigma: float) -> torch.Tensor:
    r"""
    Computes the gradient of the log-density for a single agent using mode parameterization.
    
    Backward-compatible wrapper that extracts a single agent's gradient from the vectorized computation.

    Parameters
    ----------
    agent_id : int
        The ID of the agent (:math:`i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents.
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points.
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        
    Returns
    -------
    torch.Tensor
        Gradient values of shape (num_dims, num_bins).
        
    Raises
    ------
    TypeError
        If input types are not supported.
    IndexError
        If agent_id is out of bounds.
    """
    
    try:
        # Input validation
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")
        
        # Get agents_pos shape for validation
        if isinstance(agents_pos, np.ndarray):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, torch.Tensor):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, list):
            num_agents = len(agents_pos)
        else:
            raise TypeError(f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= num_agents:
            raise IndexError(f"agent_id {agent_id} is out of bounds for {num_agents} agents")
        
        # Use vectorized computation and extract single agent result
        d_matrix = d_ln_f_vectorized(agents_pos, bin_points, sigma)
        return d_matrix[agent_id]  # Shape: (num_dims, num_bins)
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet gradient computation: {str(e)}") from e


def hessian(agent_id: int,
            agents_pos: Union[np.ndarray, torch.Tensor],
            sigma: float) -> torch.Tensor:
    r"""
    Computes the Hessian of the log-density for a single agent using mode parameterization.
    
    Backward-compatible wrapper that extracts a single agent's Hessian from the vectorized computation.

    Parameters
    ----------
    agent_id : int
        The ID of the agent (:math:`i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions of all agents.
    sigma : float
        Spread parameter (:math:`\sigma > 0`).
        
    Returns
    -------
    torch.Tensor
        Hessian matrix of shape (num_dims, num_dims) - diagonal.
        
    Raises
    ------
    TypeError
        If input types are not supported.
    IndexError
        If agent_id is out of bounds.
    """
    
    try:
        # Input validation
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")
        
        # Get agents_pos shape for validation
        if isinstance(agents_pos, np.ndarray):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, torch.Tensor):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, list):
            num_agents = len(agents_pos)
        else:
            raise TypeError(f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}")
        
        # Validate agent_id bounds
        if agent_id < 0 or agent_id >= num_agents:
            raise IndexError(f"agent_id {agent_id} is out of bounds for {num_agents} agents")
        
        # Use vectorized computation and extract single agent result
        h_tensor = hessian_vectorized(agents_pos, sigma, alpha_matrix=None)
        return h_tensor[agent_id]  # Shape: (num_dims, num_dims)
        
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in mode-parameterized Dirichlet Hessian computation: {str(e)}") from e


# ================= UTILITY FUNCTIONS =================


def get_alpha_sum(agents_pos: Union[np.ndarray, torch.Tensor],
                  sigma: float) -> torch.Tensor:
    r"""
    Compute the sum of alpha parameters (alpha_0) for each agent.
    
    For mode parameterization with 1/sigma:
    
    .. math::
        \alpha_0 = \sum_l \alpha_{(i,l)} = \sum_l (1 + x_{(i,l)}/\sigma) = L + 1/\sigma
    
    Note that alpha_0 is constant for all agents when positions sum to 1.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Agent positions, shape (num_agents, num_dims).
    sigma : float
        Spread parameter.
        
    Returns
    -------
    torch.Tensor
        Alpha sums of shape (num_agents,). Should all equal L + 1/sigma.
    """
    alpha_matrix = param_vectorized(agents_pos, sigma)
    return torch.sum(alpha_matrix, dim=1)


def validate_simplex_positions(agents_pos: Union[np.ndarray, torch.Tensor],
                               tolerance: float = 1e-5) -> bool:
    r"""
    Validate that agent positions lie on the simplex (non-negative, sum to 1).

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Agent positions, shape (num_agents, num_dims).
    tolerance : float
        Tolerance for checking sum-to-one constraint.
        
    Returns
    -------
    bool
        True if all positions are valid simplex points.
    """
    if isinstance(agents_pos, np.ndarray):
        pos = agents_pos
        sums = np.sum(pos, axis=1)
        return np.all(pos >= 0) and np.all(np.abs(sums - 1.0) < tolerance)
    elif isinstance(agents_pos, torch.Tensor):
        pos = agents_pos
        sums = torch.sum(pos, dim=1)
        return bool(torch.all(pos >= 0) and torch.all(torch.abs(sums - 1.0) < tolerance))
    else:
        raise TypeError(f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}")
