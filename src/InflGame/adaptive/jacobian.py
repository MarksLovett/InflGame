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
    
    Args:
        agents_pos: Agent positions tensor (N,)
        bin_points: Bin points tensor (K,)
        num_agents: Number of agents
        Q: Scaling factor
        denom: Denominator tensor from influence matrix sum (K,)
    
    Returns:
        torch.Tensor: Shift matrix (N, K)
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
    
    Args:
        agents_pos: Agent positions tensor (N,)
        bin_points: Bin points tensor (K,)
        num_agents: Number of agents
        Q: Scaling factor
        denom: Denominator tensor from influence matrix sum (K,)
    
    Returns:
        torch.Tensor: Second-order shift matrix (N, K)
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
    
    Args:
        agents_pos: Agent positions tensor (N,)
        bin_points: Bin points tensor (K,)
        num_agents: Number of agents
        Q: Scaling factor
        denom: Denominator tensor from influence matrix sum (K,)
    
    Returns:
        torch.Tensor: Mixed partial derivative shift matrix (N, K)
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
    
    Args:
        resource_distribution: Resource distribution tensor (K,)
        infl_fshift: Whether to include functional shifts
        di: First derivative for agent i (K,)
        pi: Probability for agent i (K,)
        dj: First derivative for agent j (K,)
        pj: Probability for agent j (K,)
        shift_i: Shift for agent i (K,)
        shift_j: Shift for agent j (K,)
        shift_ij: Mixed shift for agents i and j (K,)
    
    Returns:
        torch.Tensor: Off-diagonal Jacobian element (scalar)
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
    
    Args:
        resource_distribution: Resource distribution tensor (K,)
        infl_fshift: Whether to include functional shifts
        dd_i: Second derivative for agent i (scalar)
        di: First derivative for agent i (K,)
        pi: Probability for agent i (K,)
        shift_i: Shift for agent i (K,)
        shift_ii: Second-order shift for agent i (K,)
    
    Returns:
        torch.Tensor: Diagonal Jacobian element (scalar)
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
    JIT-optimized function. Computes the shift matrix Jacobian for the given agents and bin points using vectorized operations.


    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: Positions of the agents.
    :type agents_pos: Union[List[float], np.ndarray, torch.Tensor]
    :param bin_points: Bin points.
    :type bin_points: Union[List[float], np.ndarray, torch.Tensor]
    :param Q: Scaling factor.
    :type Q: float
    :param infl_matrix: Influence matrix.
    :type infl_matrix: torch.Tensor
    :return: The computed shift matrix Jacobian.
    :rtype: torch.Tensor
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
    JIT-optimized function. Computes the diagonal elements of the shift matrix Jacobian (second order derivatives) for the given agents and bin points using vectorized operations.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: Positions of the agents.
    :type agents_pos: Union[List[float], np.ndarray, torch.Tensor]
    :param bin_points: Bin points.
    :type bin_points: Union[List[float], np.ndarray, torch.Tensor]
    :param Q: Scaling factor.
    :type Q: float
    :param infl_matrix: Influence matrix.
    :type infl_matrix: torch.Tensor
    :return: The computed second-order shift matrix Jacobian.
    :rtype: torch.Tensor
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
    JIT-optimized function. Computes the mixed partial derivative shift matrix Jacobian (off diagonal) for the given agents and bin points using vectorized operations.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: Positions of the agents.
    :type agents_pos: Union[List[float], np.ndarray, torch.Tensor]
    :param bin_points: Bin points.
    :type bin_points: Union[List[float], np.ndarray, torch.Tensor]
    :param Q: Scaling factor.
    :type Q: float
    :param infl_matrix: Influence matrix.
    :type infl_matrix: torch.Tensor
    :return: The computed mixed partial derivative shift matrix Jacobian.
    :rtype: torch.Tensor
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
    Optimized function. Calculates the second derivative of the natural logarithm of the influence function :math:`\frac{\partial^2}{\partial x_i^2} \ln(f_{i,k})`.

    :param agent_id: ID of the agent.
    :type agent_id: int
    :param parameter_instance: Parameters unique to the influence function.
    :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
    :param infl_type: Type of influence function (e.g., 'gaussian').
    :type infl_type: str
    :return: The second derivative of the natural logarithm of the influence function.
    :rtype: torch.Tensor
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
    JIT-optimized function. Computes the off-diagonal elements of the Jacobian matrix using vectorized operations.

    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[List[float], np.ndarray, torch.Tensor]
    :param infl_fshift: Whether to include influence function shifts.
    :type infl_fshift: bool
    :param di: First derivative of the influence function for agent i.
    :type di: torch.Tensor
    :param pi: Probability for agent i.
    :type pi: torch.Tensor
    :param dj: First derivative of the influence function for agent j.
    :type dj: torch.Tensor
    :param pj: Probability for agent j.
    :type pj: torch.Tensor
    :param shift_i: Shift for agent i. Defaults to 0.
    :type shift_i: Union[float, torch.Tensor]
    :param shift_j: Shift for agent j. Defaults to 0.
    :type shift_j: Union[float, torch.Tensor]
    :param shift_ij: Mixed shift for agents i and j. Defaults to 0.
    :type shift_ij: Union[float, torch.Tensor]
    :return: The computed off-diagonal element of the Jacobian matrix.
    :rtype: torch.Tensor
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
    JIT-optimized function. Computes the diagonal elements of the Jacobian matrix using vectorized operations.

    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[List[float], np.ndarray, torch.Tensor]
    :param infl_fshift: Whether to include influence function shifts.
    :type infl_fshift: bool
    :param dd_i: Second derivative of the influence function for agent i.
    :type dd_i: Union[float, torch.Tensor]
    :param di: First derivative of the influence function for agent i.
    :type di: torch.Tensor
    :param pi: Probability for agent i.
    :type pi: torch.Tensor
    :param shift_i: Shift for agent i. Defaults to 0.
    :type shift_i: Union[float, torch.Tensor]
    :param shift_ii: Second-order shift for agent i. Defaults to 0.
    :type shift_ii: Union[float, torch.Tensor]
    :return: The computed diagonal element of the Jacobian matrix.
    :rtype: torch.Tensor
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
    """
    Optimized function. Computes the Jacobian matrix for the given agents and parameters using vectorized operations.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param parameters: Parameters unique to the influence function.
    :type parameters: Union[List[float], np.ndarray, torch.Tensor]
    :param agents_pos: Positions of the agents.
    :type agents_pos: Union[List[float], np.ndarray, torch.Tensor]
    :param bin_points: Bin points.
    :type bin_points: Union[List[float], np.ndarray, torch.Tensor]
    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[List[float], np.ndarray, torch.Tensor]
    :param infl_type: Type of influence function (e.g., 'gaussian').
    :type infl_type: str
    :param infl_fshift: Whether to include influence function shifts.
    :type infl_fshift: bool
    :param Q: Scaling factor.
    :type Q: float
    :param infl_matrix: Influence matrix.
    :type infl_matrix: torch.Tensor
    :param prob_matrix: Probability matrix.
    :type prob_matrix: torch.Tensor
    :param d_lnf_matrix: First derivatives of the natural logarithm of the influence function.
    :type d_lnf_matrix: torch.Tensor
    :return: The computed Jacobian matrix.
    :rtype: torch.Tensor
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
    Convenience function to compute the Jacobian matrix with optimized tensor operations.
    
    :param adaptive_env: The adaptive environment containing all necessary data.
    :type adaptive_env: AdaptiveEnv
    :param infl_fshift: Whether to include influence function shifts.
    :type infl_fshift: bool
    :param device: Device to perform computations on ('cpu' or 'cuda').
    :type device: str
    :return: The computed Jacobian matrix.
    :rtype: torch.Tensor
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