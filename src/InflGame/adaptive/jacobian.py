import numpy as np
import torch
import InflGame.utils.general as general
from typing import Union, List
from scipy.special import polygamma as psi

"""
.. module:: jacobian
   :synopsis: Optimized Jacobian matrix computation for adaptive dynamics in influencer games.


Jacobian Matrix Computation Module
===================================

This module provides highly optimized functions for computing Jacobian matrices in adaptive dynamics
using vectorized PyTorch operations and JIT compilation. The Jacobian matrix is essential for stability
analysis of Nash equilibria in influencer games.

**Key Features:**

- JIT-compiled core computations for maximum performance
- Vectorized operations using PyTorch broadcasting instead of nested loops
- Pre-allocated tensors to avoid dynamic memory allocation
- Consistent tensor dtypes (float32) for optimal GPU/CPU performance
- Elimination of unnecessary tensor conversions

**Mathematical Background:**

The Jacobian matrix :math:`J` captures the first-order sensitivity of agent payoffs to position changes.
For agent :math:`i`, the utility function is:

.. math::
    u_i(x) = \sum_{k=1}^{K} G_{i,k}(x, b_k) B(b_k)

where :math:`G_{i,k}` is the probability of agent :math:`i` influencing bin :math:`k`, and :math:`B(b_k)`
is the resource value at that bin.

The Jacobian elements are:

.. math::
    J_{ij} = \frac{\partial^2 u_i}{\partial x_i \partial x_j}

Diagonal elements (:math:`i=j`) represent how an agent's gradient changes with its own position,
while off-diagonal elements capture strategic interactions between agents.

**Performance Optimizations:**

1. **JIT Compilation**: Critical functions are decorated with ``@torch.jit.script`` for compilation
2. **Vectorization**: Broadcasting eliminates explicit loops over bin points
3. **Memory Efficiency**: Pre-allocated output tensors reduce garbage collection overhead
4. **Type Consistency**: All tensors use float32 to avoid implicit conversions

Dependencies:
-------------
- torch (PyTorch for tensor operations and JIT compilation)
- numpy (for array conversions)
- scipy.special (for polygamma function in beta kernel derivatives)
- InflGame.utils.general (for utility functions)

Usage:
------
The primary entry point is :func:`jacobian_matrix`, which computes the full Jacobian.
For convenience, :func:`compute_jacobian_optimized` provides a high-level interface.

Examples
--------

.. code-block:: python

    import torch
    from InflGame.adaptive.jacobian import jacobian_matrix
    
    # Assume we have an adaptive environment set up
    jacobian = jacobian_matrix(
        num_agents=3,
        parameters=torch.tensor([0.2, 0.2, 0.2]),
        agents_pos=torch.tensor([0.3, 0.5, 0.7]),
        bin_points=torch.linspace(0, 1, 100),
        resource_distribution=torch.ones(100),
        infl_type='gaussian',
        infl_fshift=False,
        Q=0.0,
        infl_matrix=infl_mat,
        prob_matrix=prob_mat,
        d_lnf_matrix=d_lnf_mat
    )
    
    # Check stability
    eigenvalues = torch.linalg.eigvals(jacobian)
    is_stable = torch.all(eigenvalues.real < 0)

See Also
--------
- :mod:`InflGame.adaptive.grad_func_env` : Gradient ascent dynamics
- :mod:`InflGame.adaptive.bifurcation_analysis` : Bifurcation analysis tools
- :mod:`InflGame.adaptive.visualization` : Visualization of stability regions
"""

import numpy as np
import torch
import InflGame.utils.general as general
from typing import Union, List
from scipy.special import polygamma as psi

# JIT-compiled helper functions for optimal performance
@torch.jit.script
def _shift_matrix_jacobian_core(
    agents_pos: torch.Tensor,
    bin_points: torch.Tensor,
    num_agents: int,
    Q: float,
    denom: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for shift matrix Jacobian.
    
    This internal function computes the first-order partial derivatives of the shift function
    with respect to agent positions. Used for computing off-diagonal Jacobian elements when
    influence function shifts are enabled.
    
    Parameters
    ----------
    agents_pos : torch.Tensor
        Agent positions tensor of shape (N,).
    bin_points : torch.Tensor
        Bin points tensor of shape (K,).
    num_agents : int
        Number of agents in the system.
    Q : float
        Scaling factor for the shift function.
    denom : torch.Tensor
        Denominator tensor from influence matrix sum, shape (K,).
    
    Returns
    -------
    torch.Tensor
        Shift matrix of shape (N, K).
    
    .. note::
        This function is JIT-compiled and should not be called directly.
        Use :func:`shift_matrix_jacobian` instead.
    """
    diff = bin_points.unsqueeze(0) - agents_pos.unsqueeze(1)
    shift_matrix = -2.0 * Q * torch.pow(diff, 2 * num_agents - 1)
    return shift_matrix / denom

@torch.jit.script
def _shift_matrix_jacobian_ii_core(
    agents_pos: torch.Tensor,
    bin_points: torch.Tensor,
    num_agents: int,
    Q: float,
    denom: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for second-order shift matrix Jacobian.
    
    This internal function computes the second-order partial derivatives of the shift function
    with respect to an agent's own position. Used for computing diagonal Jacobian elements when
    influence function shifts are enabled.
    
    Parameters
    ----------
    agents_pos : torch.Tensor
        Agent positions tensor of shape (N,).
    bin_points : torch.Tensor
        Bin points tensor of shape (K,).
    num_agents : int
        Number of agents in the system.
    Q : float
        Scaling factor for the shift function.
    denom : torch.Tensor
        Denominator tensor from influence matrix sum, shape (K,).
    
    Returns
    -------
    torch.Tensor
        Second-order shift matrix of shape (N, K).
    
    .. note::
        This function is JIT-compiled and should not be called directly.
        Use :func:`shift_matrix_jacobian_ii` instead.
    """
    diff = bin_points.unsqueeze(0) - agents_pos.unsqueeze(1)
    shift_matrix = 2.0 * Q * torch.pow(diff, 2 * num_agents - 2)
    return shift_matrix / denom

@torch.jit.script
def _shift_matrix_jacobian_ij_core(
    agents_pos: torch.Tensor,
    bin_points: torch.Tensor,
    num_agents: int,
    Q: float,
    denom: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for mixed partial derivative shift matrix Jacobian.
    
    This internal function computes the mixed second-order partial derivatives of the shift
    function with respect to positions of two different agents. Used for computing
    off-diagonal Jacobian elements when influence function shifts are enabled.
    
    Parameters
    ----------
    agents_pos : torch.Tensor
        Agent positions tensor of shape (N,).
    bin_points : torch.Tensor
        Bin points tensor of shape (K,).
    num_agents : int
        Number of agents in the system.
    Q : float
        Scaling factor for the shift function.
    denom : torch.Tensor
        Denominator tensor from influence matrix sum, shape (K,).
    
    Returns
    -------
    torch.Tensor
        Mixed partial derivative shift matrix of shape (N, K).
    
    .. note::
        This function is JIT-compiled and should not be called directly.
        Use :func:`shift_matrix_jacobian_ij` instead.
    """
    diff = bin_points.unsqueeze(0) - agents_pos.unsqueeze(1)
    shift_matrix = 4.0 * Q * torch.pow(diff, 2 * num_agents - 2)
    return shift_matrix / denom

@torch.jit.script
def _jacobian_off_diag_core(
    resource_distribution: torch.Tensor,
    infl_fshift: bool,
    di: torch.Tensor,
    pi: torch.Tensor,
    dj: torch.Tensor,
    pj: torch.Tensor,
    shift_i: torch.Tensor,
    shift_j: torch.Tensor,
    shift_ij: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for off-diagonal Jacobian elements.
    
    Computes :math:`J_{ij}` where :math:`i \neq j`, representing how agent :math:`i`'s
    utility gradient changes with respect to agent :math:`j`'s position. This captures
    strategic interactions between agents.
    
    Parameters
    ----------
    resource_distribution : torch.Tensor
        Resource distribution tensor of shape (K,).
    infl_fshift : bool
        Whether to include influence function shifts in computation.
    di : torch.Tensor
        First derivative of log-influence for agent i, shape (K,).
    pi : torch.Tensor
        Probability distribution for agent i, shape (K,).
    dj : torch.Tensor
        First derivative of log-influence for agent j, shape (K,).
    pj : torch.Tensor
        Probability distribution for agent j, shape (K,).
    shift_i : torch.Tensor
        Shift function values for agent i, shape (K,).
    shift_j : torch.Tensor
        Shift function values for agent j, shape (K,).
    shift_ij : torch.Tensor
        Mixed shift function values for agents i and j, shape (K,).
    
    Returns
    -------
    torch.Tensor
        Single scalar value representing the off-diagonal Jacobian element.
    
    .. note::
        This function is JIT-compiled and should not be called directly.
        Use :func:`jacobian_off_diag` instead.
    """
    # Base computation
    j_elm = di * dj * (-pi * pj * (1.0 - pi) + pi * pi * pj) * resource_distribution
    
    if infl_fshift:
        shift_term = (-shift_ij + 2.0 * dj * pj * shift_i + di * (1.0 + 2.0 * pi) * shift_j + 2.0 * shift_j * shift_i) * pi * resource_distribution
        j_elm = j_elm + shift_term
    
    return torch.sum(j_elm)

@torch.jit.script
def _jacobian_diag_core(
    resource_distribution: torch.Tensor,
    infl_fshift: bool,
    dd_i: torch.Tensor,
    di: torch.Tensor,
    pi: torch.Tensor,
    shift_i: torch.Tensor,
    shift_ii: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for diagonal Jacobian elements.
    
    Computes :math:`J_{ii}`, representing how agent :math:`i`'s utility gradient changes
    with respect to its own position. This determines the agent's local stability and
    convergence behavior.
    
    Parameters
    ----------
    resource_distribution : torch.Tensor
        Resource distribution tensor of shape (K,).
    infl_fshift : bool
        Whether to include influence function shifts in computation.
    dd_i : torch.Tensor
        Second derivative of log-influence for agent i (scalar).
    di : torch.Tensor
        First derivative of log-influence for agent i, shape (K,).
    pi : torch.Tensor
        Probability distribution for agent i, shape (K,).
    shift_i : torch.Tensor
        Shift function values for agent i, shape (K,).
    shift_ii : torch.Tensor
        Second-order shift function values for agent i, shape (K,).
    
    Returns
    -------
    torch.Tensor
        Single scalar value representing the diagonal Jacobian element.
    
    .. note::
        This function is JIT-compiled and should not be called directly.
        Use :func:`jacobian_diag` instead.
    """
    # Base computation
    pi_comp = 1.0 - pi  # Complementary probability
    j_elm = (dd_i * pi * pi_comp + di * di * pi * pi_comp * pi_comp - di * di * pi * pi * pi_comp) * resource_distribution
    
    if infl_fshift:
        shift_term = (((di * (3.0 * pi - 1.0) + 2.0 * shift_i) * shift_i - shift_ii) * pi) * resource_distribution
        j_elm = j_elm + shift_term
    
    return torch.sum(j_elm)

"""
    ..automodule:: influencer_games.adaptive_dynamics.jacobian
    :optimized:
    :ignore-module-all:

This module contains optimized functions for computing Jacobian matrices in adaptive dynamics using vectorized PyTorch operations. 
Performance improvements include:
- Vectorized operations using torch broadcasting instead of nested loops
- Pre-allocated tensors instead of iterative matrix building
- Consistent tensor dtypes (float32) for optimal GPU/CPU performance
- Elimination of unnecessary tensor conversions
"""

def shift_matrix_jacobian(num_agents: int,
                          agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                          bin_points: Union[List[float], np.ndarray, torch.Tensor],
                          Q: float,
                          infl_matrix: torch.Tensor,
                          ) -> torch.Tensor:
    r"""
    Compute the shift matrix Jacobian using JIT-optimized vectorized operations.
    
    This function calculates the first-order partial derivatives of the shift function
    :math:`S_i(x_i, b_k)` with respect to agent positions. The shift function modifies
    the influence kernel to account for boundary effects or strategic considerations.
    
    The shift Jacobian is used in computing off-diagonal elements of the full Jacobian matrix.

    Parameters
    ----------
    num_agents : int
        Number of agents in the system.
    agents_pos : Union[List[float], np.ndarray, torch.Tensor]
        Positions of the agents (can be list, numpy array, or tensor).
    bin_points : Union[List[float], np.ndarray, torch.Tensor]
        Discretized domain points (can be list, numpy array, or tensor).
    Q : float
        Scaling factor for the shift function magnitude.
    infl_matrix : torch.Tensor
        Pre-computed influence matrix of shape (N, K).
    
    Returns
    -------
    torch.Tensor
        Shift matrix Jacobian of shape (N, K) where N is number of agents and K is number of bins.
    
    .. note::
        Input types are automatically converted to torch.float32 tensors for consistency.
        This function uses JIT compilation for optimal performance.
    """
    # Convert inputs to tensors with consistent dtype
    agents_pos = torch.as_tensor(agents_pos, dtype=torch.float32)
    bin_points = torch.as_tensor(bin_points, dtype=torch.float32)
    
    denom = torch.sum(infl_matrix, 0)
    
    # Use JIT-compiled core for optimal performance
    return _shift_matrix_jacobian_core(agents_pos, bin_points, num_agents, Q, denom)

def shift_matrix_jacobian_ii(num_agents: int,
                             agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                             bin_points: Union[List[float], np.ndarray, torch.Tensor],
                             Q: float,
                             infl_matrix: torch.Tensor,
                             ) -> torch.Tensor:
    r"""
    Compute second-order shift matrix Jacobian using JIT-optimized vectorized operations.
    
    This function calculates the second-order partial derivatives
    :math:`\frac{\partial^2 S_i}{\partial x_i^2}` of the shift function with respect to
    an agent's own position. These derivatives contribute to the diagonal elements of the
    full Jacobian matrix.

    Parameters
    ----------
    num_agents : int
        Number of agents in the system.
    agents_pos : Union[List[float], np.ndarray, torch.Tensor]
        Positions of the agents (can be list, numpy array, or tensor).
    bin_points : Union[List[float], np.ndarray, torch.Tensor]
        Discretized domain points (can be list, numpy array, or tensor).
    Q : float
        Scaling factor for the shift function magnitude.
    infl_matrix : torch.Tensor
        Pre-computed influence matrix of shape (N, K).
    
    Returns
    -------
    torch.Tensor
        Second-order shift matrix Jacobian of shape (N, K).
    
    .. note::
        Input types are automatically converted to torch.float32 tensors for consistency.
    """
    # Convert inputs to tensors with consistent dtype
    agents_pos = torch.as_tensor(agents_pos, dtype=torch.float32)
    bin_points = torch.as_tensor(bin_points, dtype=torch.float32)
    
    denom = torch.sum(infl_matrix, 0)
    
    # Use JIT-compiled core for optimal performance
    return _shift_matrix_jacobian_ii_core(agents_pos, bin_points, num_agents, Q, denom)

def shift_matrix_jacobian_ij(num_agents: int,
                             agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                             bin_points: Union[List[float], np.ndarray, torch.Tensor],
                             Q: float,
                             infl_matrix: torch.Tensor,
                             ) -> torch.Tensor:
    r"""
    Compute mixed partial derivative shift matrix Jacobian using JIT-optimized vectorized operations.
    
    This function calculates the mixed second-order partial derivatives
    :math:`\frac{\partial^2 S_i}{\partial x_i \partial x_j}` of the shift function with
    respect to positions of two different agents. These derivatives contribute to the
    off-diagonal elements of the full Jacobian matrix.

    Parameters
    ----------
    num_agents : int
        Number of agents in the system.
    agents_pos : Union[List[float], np.ndarray, torch.Tensor]
        Positions of the agents (can be list, numpy array, or tensor).
    bin_points : Union[List[float], np.ndarray, torch.Tensor]
        Discretized domain points (can be list, numpy array, or tensor).
    Q : float
        Scaling factor for the shift function magnitude.
    infl_matrix : torch.Tensor
        Pre-computed influence matrix of shape (N, K).
    
    Returns
    -------
    torch.Tensor
        Mixed partial derivative shift matrix Jacobian of shape (N, K).
    
    .. note::
        Input types are automatically converted to torch.float32 tensors for consistency.
    """
    # Convert inputs to tensors with consistent dtype
    agents_pos = torch.as_tensor(agents_pos, dtype=torch.float32)
    bin_points = torch.as_tensor(bin_points, dtype=torch.float32)
    
    denom = torch.sum(infl_matrix, 0)
    
    # Use JIT-compiled core for optimal performance
    return _shift_matrix_jacobian_ij_core(agents_pos, bin_points, num_agents, Q, denom)

def dd_lnf_matrix(agent_id: int,
                  parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                  infl_type: str,
                  x: Union[float, torch.Tensor]= None,
                  ) -> torch.Tensor:
    r"""
    Calculate the second derivative of the log-influence function.
    
    Computes :math:`\frac{\partial^2}{\partial x_i^2} \ln(f_i(x_i, b_k))` where :math:`f_i`
    is the influence kernel for agent :math:`i`. This second derivative is needed for
    computing diagonal elements of the Jacobian matrix.
    
    **Supported Influence Types:**
    
    - **'gaussian'**: Returns :math:`-1/\sigma_i^2` where :math:`\sigma_i` is the reach parameter
    - **'beta'**: Uses polygamma function to compute beta distribution second derivatives
    
    Parameters
    ----------
    agent_id : int
        Index of the agent (0 to N-1).
    parameter_instance : Union[List[float], np.ndarray, torch.Tensor]
        Parameters for the influence function (e.g., reach/sigma values).
    infl_type : str
        Type of influence kernel ('gaussian', 'beta', etc.).
    x : Union[float, torch.Tensor], optional
        Agent positions (required for beta kernel, optional otherwise).
    
    Returns
    -------
    torch.Tensor
        Second derivative value as a scalar tensor.
    
    Raises
    ------
    ValueError
        If influence type is not recognized.
    
    .. note::
        For beta kernels, the computation uses scipy's polygamma function which requires
        scalar inputs, so tensor values are converted to float.
    """
    # Convert to tensor with consistent dtype
    parameter_instance = torch.as_tensor(parameter_instance, dtype=torch.float32)
    
    if infl_type == 'gaussian':
        dd_i = -1 / (parameter_instance[agent_id]**2)
    elif infl_type == 'beta':
        x=x[agent_id]
        # Extract float value from nash_equilibrium_x
        if torch.is_tensor(x):
            x_float = x.item() if x.dim() == 0 else float(x)
        else:
            x_float = float(x)
        
        if torch.is_tensor(parameter_instance[agent_id]):
            # Convert to float for scipy.special.polygamma
            sig_float = 1/float(parameter_instance[agent_id])
            result = psi(1, (1 - x_float) * (sig_float - 2) + 1) + psi(1, x_float * (sig_float - 2) + 1)
            return -(sig_float - 2)**2*result
        else:
            # sig is already a float
            return  -(1/parameter_instance[agent_id] - 2)**2*psi(1, (1 - x_float) * (1/parameter_instance[agent_id] - 2) + 1) + psi(1, x_float * (1/parameter_instance[agent_id] - 2) + 1)
    else:
        raise ValueError(f"Unknown influence function type: {infl_type}.")
    
    return torch.as_tensor(dd_i, dtype=torch.float32)


def jacobian_off_diag(resource_distribution: Union[List[float], np.ndarray, torch.Tensor],
                      infl_fshift: bool,
                      di: torch.Tensor,
                      pi: torch.Tensor,
                      dj: torch.Tensor,
                      pj: torch.Tensor,
                      shift_i: Union[float, torch.Tensor] = 0,
                      shift_j: Union[float, torch.Tensor] = 0,
                      shift_ij: Union[float, torch.Tensor] = 0,
                      ) -> torch.Tensor:
    """
    Compute off-diagonal Jacobian matrix elements using JIT-optimized vectorized operations.
    
    Calculates :math:`J_{ij}` for :math:`i \neq j`, representing the cross-derivative of
    agent :math:`i`'s utility with respect to agent :math:`j`'s position. This captures
    how agents strategically respond to each other's positions.

    Parameters
    ----------
    resource_distribution : Union[List[float], np.ndarray, torch.Tensor]
        Resource values at each bin point.
    infl_fshift : bool
        Whether to include influence function shift corrections.
    di : torch.Tensor
        First derivative of log-influence :math:`\partial \ln(f_i)/\partial x_i` for agent i.
    pi : torch.Tensor
        Probability distribution for agent i across bins.
    dj : torch.Tensor
        First derivative of log-influence :math:`\partial \ln(f_j)/\partial x_j` for agent j.
    pj : torch.Tensor
        Probability distribution for agent j across bins.
    shift_i : Union[float, torch.Tensor]
        Shift function correction for agent i (used if infl_fshift=True).
    shift_j : Union[float, torch.Tensor]
        Shift function correction for agent j (used if infl_fshift=True).
    shift_ij : Union[float, torch.Tensor]
        Mixed shift function correction for agents i and j (used if infl_fshift=True).
    
    Returns
    -------
    torch.Tensor
        Scalar Jacobian element :math:`J_{ij}`.
    
    .. note::
        All tensor inputs are automatically converted to float32 for consistency.
    """
    # Convert inputs to tensors with consistent dtype
    resource_distribution = torch.as_tensor(resource_distribution, dtype=torch.float32)
    shift_i = torch.as_tensor(shift_i, dtype=torch.float32)
    shift_j = torch.as_tensor(shift_j, dtype=torch.float32)
    shift_ij = torch.as_tensor(shift_ij, dtype=torch.float32)
    
    # Use JIT-compiled core for optimal performance
    return _jacobian_off_diag_core(
        resource_distribution, infl_fshift, di, pi, dj, pj,
        shift_i, shift_j, shift_ij
    )

def jacobian_diag(resource_distribution: Union[List[float], np.ndarray, torch.Tensor],
                  infl_fshift: bool,
                  dd_i: Union[float, torch.Tensor],
                  di: torch.Tensor,
                  pi: torch.Tensor,
                  shift_i: Union[float, torch.Tensor] = 0,
                  shift_ii: Union[float, torch.Tensor] = 0,
                  ) -> torch.Tensor:
    """
    Compute diagonal Jacobian matrix elements using JIT-optimized vectorized operations.
    
    Calculates :math:`J_{ii}`, representing the second derivative of agent :math:`i`'s
    utility with respect to its own position. This determines local stability and
    convergence rate for the agent.

    Parameters
    ----------
    resource_distribution : Union[List[float], np.ndarray, torch.Tensor]
        Resource values at each bin point.
    infl_fshift : bool
        Whether to include influence function shift corrections.
    dd_i : Union[float, torch.Tensor]
        Second derivative of log-influence :math:`\partial^2 \ln(f_i)/\partial x_i^2` for agent i.
    di : torch.Tensor
        First derivative of log-influence :math:`\partial \ln(f_i)/\partial x_i` for agent i.
    pi : torch.Tensor
        Probability distribution for agent i across bins.
    shift_i : Union[float, torch.Tensor]
        Shift function correction for agent i (used if infl_fshift=True).
    shift_ii : Union[float, torch.Tensor]
        Second-order shift function correction for agent i (used if infl_fshift=True).
    
    Returns
    -------
    torch.Tensor
        Scalar Jacobian element :math:`J_{ii}`.
    
    .. note::
        All tensor inputs are automatically converted to float32 for consistency.
    """
    # Convert inputs to tensors with consistent dtype
    resource_distribution = torch.as_tensor(resource_distribution, dtype=torch.float32)
    dd_i = torch.as_tensor(dd_i, dtype=torch.float32)
    shift_i = torch.as_tensor(shift_i, dtype=torch.float32)
    shift_ii = torch.as_tensor(shift_ii, dtype=torch.float32)
    
    # Use JIT-compiled core for optimal performance
    return _jacobian_diag_core(
        resource_distribution, infl_fshift, dd_i, di, pi,
        shift_i, shift_ii
    )


def jacobian_matrix(num_agents: int,
                    parameters: Union[List[float], np.ndarray, torch.Tensor],
                    agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                    bin_points: Union[List[float], np.ndarray, torch.Tensor],
                    resource_distribution: Union[List[float], np.ndarray, torch.Tensor],
                    infl_type: str,
                    infl_fshift: bool,
                    Q: float,
                    infl_matrix: torch.Tensor,
                    prob_matrix: torch.Tensor,
                    d_lnf_matrix: torch.Tensor,
                    x: Union[float, torch.Tensor]= None,
                    ) -> torch.Tensor:
    r"""
    Compute the full Jacobian matrix for multi-agent adaptive dynamics using optimized vectorized operations.
    
    The Jacobian matrix :math:`J` is an :math:`N \times N` matrix where element :math:`J_{ij}` represents
    the partial derivative of agent :math:`i`'s utility gradient with respect to agent :math:`j`'s position.
    
    **Mathematical Formulation:**
    
    For agent :math:`i` with utility:
    
    .. math::
        u_i(x) = \sum_{k=1}^{K} G_{i,k}(x, b_k) B(b_k)
    
    The Jacobian elements are:
    
    .. math::
        J_{ij} = \frac{\partial^2 u_i}{\partial x_i \partial x_j}
    
    **Stability Analysis:**
    
    The eigenvalues of :math:`J` determine equilibrium stability:
    - All eigenvalues with negative real parts → stable equilibrium
    - Any eigenvalue with positive real part → unstable equilibrium
    - Zero real parts → marginal stability (further analysis needed)

    Parameters
    ----------
    num_agents : int
        Number of agents in the system.
    parameters : Union[List[float], np.ndarray, torch.Tensor]
        Influence function parameters (e.g., reach/sigma values for each agent).
    agents_pos : Union[List[float], np.ndarray, torch.Tensor]
        Current positions of all agents.
    bin_points : Union[List[float], np.ndarray, torch.Tensor]
        Discretized domain points for resource distribution.
    resource_distribution : Union[List[float], np.ndarray, torch.Tensor]
        Resource values at each bin point.
    infl_type : str
        Type of influence kernel ('gaussian', 'beta', 'multi_gaussian', etc.).
    infl_fshift : bool
        Whether to include influence function shift corrections.
    Q : float
        Scaling factor for shift functions (relevant if infl_fshift=True).
    infl_matrix : torch.Tensor
        Pre-computed influence matrix of shape (N, K).
    prob_matrix : torch.Tensor
        Pre-computed probability matrix of shape (N, K).
    d_lnf_matrix : torch.Tensor
        Pre-computed first derivatives of log-influence, shape (N, K).
    x : Union[float, torch.Tensor], optional
        Agent positions (used for beta kernel second derivatives, optional).
    
    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (N, N).
    
    .. note::
        This is the primary function for Jacobian computation. It pre-computes shift matrices
        and second derivatives before assembling the full Jacobian using vectorized operations.
        
    .. seealso::
        :func:`compute_jacobian_optimized` - Convenience wrapper for use with AdaptiveEnv objects.
    """
    # Convert inputs to tensors with consistent dtype
    parameters = torch.as_tensor(parameters, dtype=torch.float32)
    agents_pos = torch.as_tensor(agents_pos, dtype=torch.float32)
    bin_points = torch.as_tensor(bin_points, dtype=torch.float32)
    resource_distribution = torch.as_tensor(resource_distribution, dtype=torch.float32)
    
    # Pre-allocate Jacobian matrix
    j_matrix = torch.zeros((num_agents, num_agents), dtype=torch.float32)
    
    # Compute shift matrices if needed
    if infl_fshift:
        shift_i = shift_matrix_jacobian(num_agents, agents_pos, bin_points, Q, infl_matrix)
        shift_ii = shift_matrix_jacobian_ii(num_agents, agents_pos, bin_points, Q, infl_matrix)
        shift_ij = shift_matrix_jacobian_ij(num_agents, agents_pos, bin_points, Q, infl_matrix)
    else:
        shift_i = torch.zeros((num_agents, len(bin_points)), dtype=torch.float32)
        shift_ii = torch.zeros((num_agents, len(bin_points)), dtype=torch.float32)
        shift_ij = torch.zeros((num_agents, len(bin_points)), dtype=torch.float32)

    # Vectorized computation of second derivatives for diagonal elements
    dd_params = torch.zeros(num_agents, dtype=torch.float32)
    for agent_id in range(num_agents):
        dd_params[agent_id] = dd_lnf_matrix(agent_id=agent_id, parameter_instance=parameters, infl_type=infl_type,x=x)

    # Compute Jacobian matrix elements
    for agent_id in range(num_agents):
        pi = prob_matrix[agent_id]
        di = d_lnf_matrix[agent_id]
        dd_i = dd_params[agent_id]
        
        for a_id2 in range(num_agents):
            if agent_id == a_id2:
                # Diagonal element
                j_matrix[agent_id, a_id2] = jacobian_diag(
                    resource_distribution, infl_fshift, dd_i, di, pi,
                    shift_i=shift_i[agent_id] if infl_fshift else 0,
                    shift_ii=shift_ii[agent_id] if infl_fshift else 0
                )
            else:
                # Off-diagonal element
                dj = d_lnf_matrix[a_id2]
                pj = prob_matrix[a_id2]
                j_matrix[agent_id, a_id2] = jacobian_off_diag(
                    resource_distribution, infl_fshift, di, pi, dj, pj,
                    shift_i=shift_i[agent_id] if infl_fshift else 0,
                    shift_j=shift_i[a_id2] if infl_fshift else 0,
                    shift_ij=shift_ij[agent_id] if infl_fshift else 0
                )
    
    return j_matrix


def compute_jacobian_optimized(adaptive_env,
                              position: torch.Tensor,
                              infl_fshift: bool = False,
                              device: str = 'cpu',
                              ) -> torch.Tensor:
    """
    Convenience function to compute the Jacobian matrix from an AdaptiveEnv object.
    
    This high-level wrapper extracts all necessary data from an adaptive environment instance
    and computes the Jacobian matrix. It handles device placement and automatically computes
    required intermediate matrices (influence, probability, derivatives).
    
    **Usage Example:**
    
    .. code-block:: python
    
        from InflGame.adaptive.grad_func_env import AdaptiveEnv
        from InflGame.adaptive.jacobian import compute_jacobian_optimized
        
        # Create and set up environment
        env = AdaptiveEnv(...)
        env.gradient_ascent()
        
        # Compute Jacobian at final positions
        jacobian = compute_jacobian_optimized(env, env.agents_pos)
        
        # Check stability
        eigenvalues = torch.linalg.eigvals(jacobian)
        is_stable = torch.all(eigenvalues.real < 0)
    
    Parameters
    ----------
    adaptive_env : AdaptiveEnv
        The adaptive environment containing agent configuration and dynamics.
    position : torch.Tensor
        Agent positions at which to evaluate the Jacobian.
    infl_fshift : bool
        Whether to include influence function shift corrections.
    device : str
        Device for tensor computations ('cpu' or 'cuda').
    
    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (N, N) on the specified device.
    
    .. note::
        This function automatically computes influence_matrix, prob_matrix, and d_lnf_matrix
        from the environment. For repeated Jacobian evaluations at the same position, consider
        caching these intermediate matrices.
    """
    # Extract necessary data from adaptive environment
    num_agents = adaptive_env.num_agents
    parameters = adaptive_env.parameters
    bin_points = adaptive_env.bin_points
    resource_distribution = adaptive_env.resource_distribution
    infl_type = adaptive_env.infl_type
    Q = getattr(adaptive_env, 'Q', 1.0)  # Default Q value if not present
    
    adaptive_env.agents_pos = position
    # Compute necessary matrices
    infl_matrix = adaptive_env.influence_matrix()
    prob_matrix = adaptive_env.prob_matrix()
    d_lnf_matrix = adaptive_env.d_lnf_matrix()
    
    # Move tensors to specified device for optimal performance
    if isinstance(infl_matrix, torch.Tensor):
        infl_matrix = infl_matrix.to(device)
    if isinstance(prob_matrix, torch.Tensor):
        prob_matrix = prob_matrix.to(device)
    if isinstance(d_lnf_matrix, torch.Tensor):
        d_lnf_matrix = d_lnf_matrix.to(device)
    
    # Compute Jacobian matrix
    jacobian = jacobian_matrix(
        num_agents=num_agents,
        parameters=parameters,
        agents_pos=position,
        bin_points=bin_points,
        resource_distribution=resource_distribution,
        infl_type=infl_type,
        infl_fshift=infl_fshift,
        Q=Q,
        infl_matrix=infl_matrix,
        prob_matrix=prob_matrix,
        d_lnf_matrix=d_lnf_matrix,
        x=position
    )
    
    return jacobian.to(device)