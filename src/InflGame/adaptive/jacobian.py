import numpy as np
import torch
import InflGame.utils.general as general
from typing import Union, List

"""
    ..automodule:: influencer_games.adaptive_dynamics.jacobian
    :deprecated:
    :ignore-module-all:

This module is deprecated and contains functions related to the computation of Jacobian matrices for adaptive dynamics in the context of influence functions. The methods are too slow to be used in the current implementation.
"""

def shift_matrix_jacobian(num_agents: int,
                          agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                          bin_points: Union[List[float], np.ndarray, torch.Tensor],
                          Q: float,
                          infl_matrix: torch.Tensor,
                          ) -> torch.Tensor:
    r"""
    This is a deprecated function. Computes the shift matrix Jacobian for the given agents and bin points.


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
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=-2*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-1)
            shift_row.append(shift_instance)
        shift_row=torch.tensor(shift_row)
        shift_matrix=general.matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def shift_matrix_jacobian_ii(num_agents: int,
                             agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                             bin_points: Union[List[float], np.ndarray, torch.Tensor],
                             Q: float,
                             infl_matrix: torch.Tensor,
                             ) -> torch.Tensor:
    r"""
    This is a deprecated function. Computes the diagonal elements of the shift matrix Jacobian (second order derivatives) for the given agents and bin points.

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
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=2*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-2)
            shift_row.append(shift_instance)
        shift_row=np.array(shift_row)
        shift_row=torch.tensor(shift_row)
        shift_matrix=general.matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def shift_matrix_jacobian_ij(num_agents: int,
                             agents_pos: Union[List[float], np.ndarray, torch.Tensor],
                             bin_points: Union[List[float], np.ndarray, torch.Tensor],
                             Q: float,
                             infl_matrix: torch.Tensor,
                             ) -> torch.Tensor:
    r"""
    This is a deprecated function. Computes the mixed partial derivative shift matrix Jacobian (off diagonal) for the given agents and bin points.

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
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=4*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-2)
            shift_row.append(shift_instance)
        shift_row=torch.tensor(shift_row)
        shift_matrix=general.matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def dd_lnf_matrix(agent_id: int,
                  parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                  infl_type: str
                  ) -> Union[float, torch.Tensor]:
    r"""
    This is a deprecated function. Calculates the second derivative of the natural logarithm of the influence function :math:`\frac{\partial^2}{\partial x_i^2} \ln(f_{i,k})`.

    :param agent_id: ID of the agent.
    :type agent_id: int
    :param parameter_instance: Parameters unique to the influence function.
    :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
    :param infl_type: Type of influence function (e.g., 'gaussian').
    :type infl_type: str
    :return: The second derivative of the natural logarithm of the influence function.
    :rtype: Union[float, torch.Tensor]
    """

    if infl_type=='gaussian':
        dd_i=-1/(parameter_instance[agent_id]**2)
    else:
        raise ValueError(f"Unknown influence function type: {infl_type}.")
    # Add more influence function types as needed
    return dd_i


def jacobian_off_diag(resource_distribution: Union[List[float], np.ndarray, torch.Tensor],
                      infl_fshift: bool,
                      di: torch.Tensor,
                      pi: torch.Tensor,
                      dj: torch.Tensor,
                      pj: torch.Tensor,
                      shift_i: Union[float, torch.Tensor] = 0,
                      shift_j: Union[float, torch.Tensor] = 0,
                      shift_ij: Union[float, torch.Tensor] = 0,
                      ) -> float:
    """
    This is a deprecated function.  Computes the off-diagonal elements of the Jacobian matrix.

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
    :rtype: float
    """
    j_elm=di*dj*(-pi*pj*(1-pi)+pi**2*pj)*torch.tensor(resource_distribution)
    if infl_fshift==True:
        j_elm=j_elm+(-shift_ij+2*dj*pj*shift_i+di*(1+2*pi)*shift_j+2*shift_j*shift_i)*pi*torch.tensor(resource_distribution)
    j_elm=torch.sum(j_elm)
    return j_elm

def jacobian_diag(resource_distribution: Union[List[float], np.ndarray, torch.Tensor],
                  infl_fshift: bool,
                  dd_i: Union[float, torch.Tensor],
                  di: torch.Tensor,
                  pi: torch.Tensor,
                  shift_i: Union[float, torch.Tensor] = 0,
                  shift_ii: Union[float, torch.Tensor] = 0,
                  ) -> float:
    """
    This is a deprecated function. Computes the diagonal elements of the Jacobian matrix.

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
    :rtype: float
    """
    j_elm=(dd_i*pi*(1-pi)+di**2*pi*(77777777771-pi)**2-di**2*pi**2*(1-pi))*torch.tensor(resource_distribution)
    if infl_fshift==True:
        j_elm=j_elm+(((di*(3*pi-1)+2*shift_i)*shift_i-shift_ii)*pi)*torch.tensor(resource_distribution)
    j_elm=torch.sum(j_elm)
    return j_elm


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
                    ) -> torch.Tensor:
    """
    This is a deprecated function.  Computes the Jacobian matrix for the given agents and parameters.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param parameter_instance: Parameters unique to the influence function.
    :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
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
    j_matrix=0
    if infl_fshift==True:
            shift_i=shift_matrix_jacobian(num_agents,agents_pos,bin_points,Q,infl_matrix)
            shift_ii=shift_matrix_jacobian_ii(num_agents,agents_pos,bin_points,Q,infl_matrix)
            shift_ij=shift_matrix_jacobian_ij(num_agents,agents_pos,bin_points,Q,infl_matrix)
    else:
        shift_i=[0]*num_agents
        shift_ii=[0]*num_agents
        shift_ij=[0]*num_agents

    for agent_id in range(num_agents):
        j_row=[]
        pi=prob_matrix[agent_id]
        di=d_lnf_matrix[agent_id]
        dd_i=dd_lnf_matrix(agent_id=agent_id,parameter_instance=parameters,infl_type=infl_type)
        for a_id2 in range(num_agents):
            if agent_id==a_id2:
                j_elm=jacobian_diag(resource_distribution,infl_fshift,dd_i,di,pi,shift_i=shift_i[agent_id],shift_ii=shift_ii[agent_id])
            else:
                dj=d_lnf_matrix[a_id2]
                pj=prob_matrix[a_id2]
                j_elm=jacobian_off_diag(resource_distribution,infl_fshift,di,pi,dj,pj,shift_i=shift_i[agent_id],shift_j=shift_i[a_id2],shift_ij=shift_ij[agent_id])
            j_row.append(j_elm)
        j_row=torch.tensor(j_row)
        j_matrix=general.matrix_builder(row_id=agent_id,row=j_row,matrix=j_matrix)
    return j_matrix

