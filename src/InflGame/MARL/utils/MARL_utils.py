r"""
.. module:: MARL_utils
   :synopsis: Provides utility functions for multi-agent reinforcement learning (MARL) in influencer games.

Multi-Agent Reinforcement Learning Utilities
============================================

This module provides utility functions for multi-agent reinforcement learning (MARL) in influencer games. 
It includes functions to compute influence matrices, probability matrices, and influence kernels for agents 
interacting in a shared environment. The module supports various influence kernels, including Gaussian, Jones, 
Dirichlet, and Multi-variate Gaussian kernels.

Mathematical Definitions:
-------------------------
1. **Probability Matrix**:
   The probability matrix :math:`G` is defined as:

   .. math::
       G_{i,k} = \frac{f_i(x_i, b_k)}{\sum_{j=1}^{N} f_j(x_j, b_k)}

   where:
     - :math:`f_i(x_i, b_k)` is the influence of agent :math:`i` on resource point :math:`b_k`
     - :math:`N` is the total number of agents
     - :math:`b_k` is the `k`th resource point

2. **Influence Matrix**:
   The influence matrix :math:`I` is defined as having the components:

   .. math::
       \iota_{i,k} = f_i(x_i, b_k)

   where:
     - :math:`f_i(x_i, b_k)` is the influence of agent :math:`i` on resource point :math:`b_k`

3. **Influence Kernels**:
   Various influence kernels are supported, including Gaussian, Jones, Dirichlet, and Multi-variate Gaussian kernels.

Dependencies:
-------------
- InflGame.utils
- InflGame.kernels


Usage:
------
The `prob_matrix` function computes the probability matrix for agents influencing resource points, while the 
`influence_matrix` function calculates the influence matrix for all agents. The `influence` function computes 
the influence of a specific agent's kernel over resource points.

"""



import torch
import numpy as np
import itertools
from functools import lru_cache
from typing import Union, Dict, List, Tuple

# from the package
import InflGame.utils.general as general

# for influence kernel specific functions
import InflGame.kernels.gauss as gauss
import InflGame.kernels.jones as jones
import InflGame.kernels.diric as diric
import InflGame.kernels.MV_gauss as MV_gauss


@lru_cache(maxsize=1024)
def _cached_influence_computation(agent_id: int, 
                                 agents_pos_tuple: tuple, 
                                 bin_points_tuple: tuple,
                                 infl_type: str,
                                 parameters_tuple: tuple) -> torch.Tensor:
    """Cached influence computation for repeated calculations."""
    agents_pos = torch.tensor(agents_pos_tuple)
    bin_points = torch.tensor(bin_points_tuple)
    parameters = torch.tensor(parameters_tuple)
    
    if infl_type == 'gaussian':
        return gauss.influence(agent_id=agent_id, parameter_instance=parameters, 
                              agents_pos=agents_pos, bin_points=bin_points)
    elif infl_type == 'gaussian_og':
        return torch.exp(-(agents_pos[agent_id] - bin_points) ** 2 / (2 * parameters[agent_id] ** 2))
    elif infl_type == 'Jones_M':
        return jones.influence(agent_id=agent_id, parameter_instance=parameters, 
                              agents_pos=agents_pos, bin_points=bin_points)
    else:
        # Fall back to uncached for other types
        return None


def prob_matrix_optimized(num_agents: int,
                         agents_pos: Union[torch.Tensor, np.ndarray],
                         bin_points: Union[torch.Tensor, np.ndarray],
                         infl_configs: dict,
                         parameters: Union[list, np.ndarray, torch.Tensor],
                         fixed_pa: int,
                         infl_cshift: bool = False,
                         infl_fshift: bool = False,
                         cshift: torch.Tensor = None,
                         Q: float = 0.0) -> torch.Tensor:
    """
    Optimized probability matrix computation using vectorization and caching.
    """
    # Convert to tensors if needed and ensure proper dtypes
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float32)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not isinstance(parameters, torch.Tensor):
        parameters = torch.tensor(parameters, dtype=torch.float32)
    
    # Get influence matrix
    infl_matrix = influence_matrix_optimized(
        num_agents=num_agents, 
        agents_pos=agents_pos, 
        bin_points=bin_points,
        infl_configs=infl_configs, 
        parameters=parameters, 
        fixed_pa=fixed_pa,
        infl_cshift=infl_cshift, 
        infl_fshift=infl_fshift, 
        cshift=cshift, 
        Q=Q
    )
    
    # Vectorized probability computation with numerical stability
    denom = torch.sum(infl_matrix, dim=0, keepdim=True)
    # Add small epsilon to prevent division by zero
    denom = torch.clamp(denom, min=1e-10)
    agent_prob_matrix = infl_matrix / denom
    
    return agent_prob_matrix


def influence_matrix_optimized(num_agents: int,
                              agents_pos: Union[torch.Tensor, np.ndarray],
                              bin_points: Union[torch.Tensor, np.ndarray],
                              infl_configs: dict,
                              parameters: Union[list, np.ndarray, torch.Tensor],
                              fixed_pa: int,
                              infl_cshift: bool = False,
                              infl_fshift: bool = False,
                              cshift: torch.Tensor = None,
                              Q: float = 0.0) -> torch.Tensor:
    r"""
    Compute the influence of a specific agent's influence kernel over the bin points.
    
    i.e.

    .. math::
        f_{i}(x_i,b)

    Where :math:`x_i` is the position of the :math:`i` th agent and :math:`b \in \mathbb{B}` is the resource/bin points in the environment.

    There are several types of preset influence kernels, including:

    - **Gaussian influence kernel**

        .. math::
            f_i(x_i,b,\sigma) = e^{-\frac {(x_i-b)^2}{2\sigma^2}}

    - **Jones influence kernel**

        .. math::
            f_i(x_i,b,p) = |x-b|^p

    - **Dirichlet influence kernel**

        .. math::
            f_i(\mathbb{\alpha},b)=\frac{1}{\beta(\alpha)}\prod_{l=1}^{L} b_l^{(\alpha_l-1)}

    where :math:`L` is the number of dimensions and :math:`b_l` is the :math:`l` th component of the bin point :math:`b`.
    
    Here :math:`\mathbf{\alpha}` is the parameter vector for the Dirichlet influence kernel, but :math:`\alpha_\phi` is the fixed parameter such that

        .. math::
            \alpha_l=\frac{\alpha_\phi}{x_{(i,\phi)}}*x_{(i,l)}

    where :math:`x_{(i,\phi)}` is the :math:`\phi` th component of the position of the :math:`i` th agent and :math:`x_{(i,l)}` is the the :math:`l` th component of the position of the :math:`i` th agent.

    - **Multi-variate Gaussian influence kernel**

        .. math::
            f_i(\mathbf{x}_i,\mathbf{b},\Sigma) = e^{-\frac{(\mathbf{x}_i-\mathbf{b})^T \Sigma^{-1} (\mathbf{x}_i-\mathbf{b})}{2}}

    where :math:`\Sigma` is the covariance matrix of the multi-variate Gaussian influence kernel.

    - **Custom influence kernel (user-defined)**

    This influence kernel is defined by the user and can be any function that takes in the agent's position, bin points, and parameters.
    Examples of custom influence kernels are provided in the demos.

    :param agent_id: The ID of the agent for which influence is being calculated.
    :type agent_id: int
    :param agents_pos: Positions of the agents.
    :type agents_pos: torch.Tensor | numpy.ndarray
    :param bin_points: Positions of the resource points.
    :type bin_points: torch.Tensor | numpy.ndarray
    :param infl_configs: Configuration for the influence type.
    :type infl_configs: dict
    :param parameters: Parameters for the influence function.
    :type parameters: list | numpy.ndarray | torch.Tensor
    :param alpha_matrix: Alpha parameters for Dirichlet influence. Defaults to 0.
    :type alpha_matrix: torch.Tensor, optional
    :return: A vector representing the influence of the agent over all resource points.
    :rtype: torch.Tensor
    """

    # Convert to tensors
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float32)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not isinstance(parameters, torch.Tensor):
        parameters = torch.tensor(parameters, dtype=torch.float32)
    
    infl_type = infl_configs['infl_type']
    
    # Handle special influence types that can be vectorized
    if infl_type in ['gaussian']:
        return gauss.influence_vectorized(agents_pos=agents_pos, bin_points=bin_points, parameter_instance=parameters)
    elif infl_type == 'Jones_M':
        return _vectorized_jones_influence(agents_pos, bin_points, parameters)
    elif infl_type == 'custom':
        custom_influence_vectorized = infl_configs['custom_influence']
        #x_torch = torch.tensor(agents_pos[agent_id])?
        #p = np.array([parameters[agent_id]])?
        return custom_influence_vectorized(agents_pos, bin_points=bin_points, parameter_instance=parameters)


def _vectorized_jones_influence(agents_pos: torch.Tensor, 
                               bin_points: torch.Tensor, 
                               parameters: torch.Tensor) -> torch.Tensor:
    """Vectorized computation for Jones influence."""
    agents_expanded = agents_pos.unsqueeze(1)  # Shape: (N, 1)
    bins_expanded = bin_points.unsqueeze(0)    # Shape: (1, K)
    params_expanded = parameters.unsqueeze(1)  # Shape: (N, 1)
    
    # Jones influence: |x - b|^p
    abs_diff = torch.abs(agents_expanded - bins_expanded)
    infl_matrix = torch.pow(abs_diff, params_expanded)
    
    return infl_matrix


def reward_obs_optimized(observations: Dict[str, int],
                        possible_agents: List[str],
                        possible_positions: List[List[int]],
                        num_agents: int,
                        bin_points: Union[List, np.ndarray],
                        infl_configs: dict,
                        parameters: Union[List, np.ndarray, torch.Tensor],
                        fixed_pa: int,
                        infl_fshift: bool,
                        infl_cshift: bool,
                        cshift: torch.Tensor,
                        Q: float,
                        resource_distribution: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Optimized reward computation using vectorized operations.
    """
    # Convert position indices to actual positions
    position = np.array([possible_positions[observations[agent]] for agent in possible_agents])
    
    # Convert to tensors for faster computation
    if not isinstance(resource_distribution, torch.Tensor):
        resource_distribution = torch.tensor(resource_distribution, dtype=torch.float32)
    
    # Get probability matrix
    pr_matrix = prob_matrix_optimized(
        num_agents=num_agents,
        agents_pos=position,
        bin_points=bin_points,
        infl_configs=infl_configs,
        parameters=parameters,
        fixed_pa=fixed_pa,
        infl_cshift=infl_cshift,
        infl_fshift=infl_fshift,
        cshift=cshift,
        Q=Q
    )
    
    # Vectorized reward computation
    reward_vec = torch.sum(pr_matrix * resource_distribution.unsqueeze(0), dim=1)
    
    # Convert back to dictionary
    reward = {possible_agents[agent_id]: reward_vec[agent_id].item() for agent_id in range(num_agents)}
    return reward


def reward_dict_optimized(possible_agents: List[str],
                         possible_positions: List[List[int]],
                         num_observations: int,
                         num_agents: int,
                         bin_points: Union[List, np.ndarray],
                         infl_configs: dict,
                         parameters: Union[List, np.ndarray, torch.Tensor],
                         fixed_pa: int,
                         infl_fshift: bool,
                         infl_cshift: bool,
                         cshift: torch.Tensor,
                         Q: float,
                         resource_distribution: Union[List, np.ndarray],
                         normalize: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Highly optimized reward dictionary computation using batch processing.
    """
    # Convert to tensors once
    if not isinstance(resource_distribution, torch.Tensor):
        resource_distribution = torch.tensor(resource_distribution, dtype=torch.float32)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float32)
    if not isinstance(parameters, torch.Tensor):
        parameters = torch.tensor(parameters, dtype=torch.float32)
    
    # Generate all possible observations
    possible_observation = possible_observations(possible_agents, num_observations, num_agents)
    
    # Batch process all observations
    all_positions = []
    obs_keys = []
    
    for observation in possible_observation:
        position = np.array([possible_positions[observation[agent]] for agent in possible_agents])
        all_positions.append(position)
        obs_keys.append(str(observation))
    
    # Convert to tensor for batch processing
    all_positions = torch.tensor(all_positions, dtype=torch.float32)  # Shape: (num_obs, num_agents)
    
    # Batch compute rewards if possible (for simple influence types)
    if infl_configs['infl_type'] in ['gaussian', 'gaussian_og', 'Jones_M']:
        r_dict = _batch_compute_rewards(all_positions, obs_keys, possible_agents, 
                                       bin_points, infl_configs, parameters, 
                                       resource_distribution, num_agents)
    else:
        # Fall back to individual computation for complex types
        r_dict = {}
        for i, observation in enumerate(possible_observation):
            reward = reward_obs_optimized(
                observations=observation,
                possible_agents=possible_agents,
                possible_positions=possible_positions,
                num_agents=num_agents,
                bin_points=bin_points,
                infl_configs=infl_configs,
                parameters=parameters,
                fixed_pa=fixed_pa,
                infl_fshift=infl_fshift,
                infl_cshift=infl_cshift,
                cshift=cshift,
                Q=Q,
                resource_distribution=resource_distribution
            )
            r_dict[obs_keys[i]] = reward
    
    # Normalize if requested
    if normalize:
        _normalize_rewards_inplace(r_dict)
    
    return r_dict


def _batch_compute_rewards(all_positions: torch.Tensor,
                          obs_keys: List[str],
                          possible_agents: List[str],
                          bin_points: torch.Tensor,
                          infl_configs: dict,
                          parameters: torch.Tensor,
                          resource_distribution: torch.Tensor,
                          num_agents: int) -> Dict[str, Dict[str, float]]:
    """
    Batch compute rewards for all observations simultaneously.
    """
    batch_size = all_positions.shape[0]
    num_bins = len(bin_points)
    
    # Expand dimensions for batch processing
    # all_positions: (batch_size, num_agents) -> (batch_size, num_agents, 1)
    # bin_points: (num_bins,) -> (1, 1, num_bins)
    positions_expanded = all_positions.unsqueeze(2)  # (batch_size, num_agents, 1)
    bins_expanded = bin_points.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)
    params_expanded = parameters.unsqueeze(0).unsqueeze(2)  # (1, num_agents, 1)
    
    # Compute influence matrix for all observations at once
    if infl_configs['infl_type'] in ['gaussian', 'gaussian_og']:
        diff_squared = (positions_expanded - bins_expanded) ** 2
        infl_matrix_batch = torch.exp(-diff_squared / (2 * params_expanded ** 2))
    elif infl_configs['infl_type'] == 'Jones_M':
        abs_diff = torch.abs(positions_expanded - bins_expanded)
        infl_matrix_batch = torch.pow(abs_diff, params_expanded)
    
    # Shape: (batch_size, num_agents, num_bins)
    
    # Compute probability matrices
    denom = torch.sum(infl_matrix_batch, dim=1, keepdim=True)  # (batch_size, 1, num_bins)
    denom = torch.clamp(denom, min=1e-10)
    prob_matrix_batch = infl_matrix_batch / denom  # (batch_size, num_agents, num_bins)
    
    # Compute rewards
    resource_expanded = resource_distribution.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)
    rewards_batch = torch.sum(prob_matrix_batch * resource_expanded, dim=2)  # (batch_size, num_agents)
    
    # Convert to dictionary format
    r_dict = {}
    for i, obs_key in enumerate(obs_keys):
        reward = {possible_agents[j]: rewards_batch[i, j].item() for j in range(num_agents)}
        r_dict[obs_key] = reward
    
    return r_dict


def _normalize_rewards_inplace(r_dict: Dict[str, Dict[str, float]]) -> None:
    """Normalize rewards in-place for better performance."""
    # Find max reward across all observations and agents
    max_reward = 0.0
    for obs_rewards in r_dict.values():
        max_reward = max(max_reward, max(obs_rewards.values()))
    
    if max_reward > 0:
        # Normalize all rewards
        for obs_rewards in r_dict.values():
            for agent in obs_rewards:
                obs_rewards[agent] = np.around(obs_rewards[agent] / max_reward, decimals=5)


def possible_observations(possible_agents: list[str],
                          num_observations: int,
                          num_agents: int) -> list[dict[str, int]]:
    """
    Generates all possible observations for the agents.

    :param possible_agents: A list of agent identifiers.
    :type possible_agents: list[str]
    :param num_observations: The number of observations in the environment.
    :type num_observations: int
    :param num_agents: The number of players in the environment.
    :type num_agents: int
    :return: A list of dictionaries representing all possible observations for the agents.
    :rtype: list[dict[str, int]]
    """
    positions_lists = positions_list(num_observations=num_observations, num_agents=num_agents)
    observations = []
    for position in positions_lists:
        observation = {possible_agents[i]: position[i] for i in range(num_agents)}
        observations.append(observation)
    return observations

def remove_tuples(input_list: list) -> list:
    """
    Removes tuples from a list, returning a new list with only integers.

    :param input_list: A list that may contain tuples and integers.
    :type input_list: list
    :return: A new list with tuples flattened into integers.
    :rtype: list
    """
    li = []
    for item in input_list:
        if isinstance(item, tuple):
            li.extend(item)
        else:
            li.append(item)
    return li


def remove_all_tuples(input_list: list, 
                      times: int = 2) -> list:
    """
    Recursively removes tuples from a list a specified number of times.

    :param input_list: A list that may contain tuples and integers.
    :type input_list: list
    :param times: The number of times to recursively remove tuples.
    :type times: int
    :return: A new list with tuples flattened into integers.
    :rtype: list
    """
    li = input_list.copy()
    for _ in range(times):
        li = remove_tuples(li)
    return li


def positions_list(num_observations: int, 
                   num_agents: int = 2) -> list:
    """
    Generates a list of all possible positions for players based on the number of observations.

    :param num_observations: The number of observations in the environment.
    :type num_observations: int
    :param num_agents: The number of players in the environment.
    :type num_agents: int
    :return: A list of all possible positions for the players.
    :rtype: list
    """
    a = [i for i in range(num_observations)]
    b = a.copy()
    for i in range(num_agents-1):
        combinations = list(itertools.product(a, b))
        b = combinations
    for i in range(len(combinations)):
        combinations[i] = list(combinations[i])

    pos_list = [remove_all_tuples(combinations[i], times=num_agents - 1) for i in range(len(combinations))]
    return pos_list

def observation_to_position(observations: dict[str, int],
                            possible_positions: list[list[int]]) -> list[list[int]]:
    r"""
    Convert observations to positions in the domain.

    :param observations: Current observations of all agents.
    :type observations: dict[str, int]
    :return: List of positions corresponding to the observations.
    :rtype: list[list[int]]
    """
    pos_list = []
    for key in observations:
        position = possible_positions[int(observations[key])]
        pos_list.append(position)
    return pos_list

# Keep original functions for backward compatibility but redirect to optimized versions
def prob_matrix(*args, **kwargs):
    """Backward compatibility wrapper."""
    return prob_matrix_optimized(*args, **kwargs)

def reward_obs(*args, **kwargs):
    """Backward compatibility wrapper."""
    return reward_obs_optimized(*args, **kwargs)

def reward_dict(*args, **kwargs):
    """Backward compatibility wrapper."""
    return reward_dict_optimized(*args, **kwargs)


