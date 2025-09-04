r"""
.. module:: sync_game
   :synopsis: Implements a synchronized multi-agent environment for influencer games.

Synchronized Multi-Agent Environment Module
===========================================

This module implements a synchronized multi-agent environment for influencer games. The environment simulates a 
1D domain where agents can move left, stay, or move right synchronously. Rewards are calculated based on a 
probability matrix and resource distribution.

Mathematical Definitions:
-------------------------
The reward for an agent :math:`i` is computed as:

.. math::
    u_i(x) = \sum_{b \in \mathbb{B}} G_i(x,b) \cdot B(b)

where:
  - :math:`G_i(x,b)` is the probability of agent :math:`i` influencing bin :math:`b`
  - :math:`B(b)` is the resource available at bin :math:`b`
  - :math:`\mathbb{B}` is the set of all bin points

The probability is calculated by the function :func:`InflGame.MARL.utils.MARL_utils.prob_matrix`.

Dependencies:
-------------
- InflGame.MARL
- ray.rllib.env.multi_agent_env

Usage:
------
The `influencer_env_sync` class provides a synchronized multi-agent environment for influencer games. It supports 
custom configurations for agents, resource distributions, and influence kernels.

Example:
--------

.. code-block:: python

    import numpy as np
    from InflGame.MARL.sync_game import influencer_env_sync

    # Define environment configuration
    config = {
        "num_agents": 3,
        "initial_position": [0.2, 0.5, 0.8],
        "bin_points": np.linspace(0, 1, 100),
        "resource_distribution": np.random.rand(100),
        "step_size": 0.01,
        "domain_type": "1d",
        "domain_bounds": [0, 1],
        "infl_configs": {"infl_type": "gaussian"},
        "parameters": [0.1, 0.1, 0.1],
        "fixed_pa": 0,
        "NUM_ITERS": 100
    }

    # Initialize the environment
    env = influencer_env_sync(config=config)

    # Reset the environment
    observations, _ = env.reset()

    # Perform a step
    actions = {"player0": env.LEFT, "player1": env.STAY, "player2": env.RIGHT}
    observations, rewards, terminated, truncated, info = env.step(actions)

    print("Observations:", observations)
    print("Rewards:", rewards)
    print("Terminated:", terminated)

"""
import gymnasium as gym
import torch
import numpy as np
import InflGame.MARL.utils.MARL_utils as MARL_utils
import gymnasium.spaces as spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class influencer_env_sync(MultiAgentEnv):
    r"""
    A synchronized multi-agent environment for influencer games.

    This environment simulates a 1D domain where agents can move left, stay, or move right synchronously.
    Rewards are calculated based on a probability matrix and resource distribution.

    Attributes:
    -----------
    - `num_agent` (int): Number of agents in the environment.
    - `initial_position` (list[float]): Initial positions of the agents.
    - `bin_points` (list[float]): Points defining bins for resource distribution.
    - `resource_distribution` (list[float]): Resource distribution across bins.
    - `step_size` (float): Step size for discretizing the domain.
    - `domain_type` (str): Type of domain ('1d', '2d', or 'simplex').
    - `domain_bounds` (list[float]): Bounds of the domain (e.g., [min, max]).
    - `infl_configs` (dict): Configuration for influence calculations.
    - `infl_type` (str): Type of influence kernel (e.g., 'gaussian', 'diri', 'jones', etc.).
    - `parameters` (list | dict): Additional parameters for influence calculations.
    - `fixed_pa` (int): Fixed parameter for Dirichlet influence kernels.
    - `NUM_ITERS` (int): Maximum number of iterations for the game.
    - `normalize_reward` (bool): Whether to normalize rewards.
    - `domain_points_num` (int): Number of discrete points in the domain.
    - `possible_positions` (np.ndarray): Array of possible positions in the domain.
    - `reward_dict` (dict): Precomputed reward dictionary for all possible agents and positions.
    - `agents` (list[str]): List of agent identifiers.
    - `possible_agents` (list[str]): List of possible agent identifiers.
    - `observation_spaces` (dict): Observation spaces for each agent.
    - `action_spaces` (dict): Action spaces for each agent.
    - `observations` (dict): Current observations of all agents.
    - `last_move` (dict | None): Placeholder for the last move taken by agents.
    - `num_moves` (int): Number of moves made in the current episode.
    - `LEFT` (int): Action identifier for moving left.
    - `STAY` (int): Action identifier for staying in the same position.
    - `RIGHT` (int): Action identifier for moving right.
    """

    def __init__(self, config: dict = None):
        r"""
        Initialize the influencer environment.

        :param config: Configuration dictionary containing:
            - **num_agents** (*int*): Number of agents in the environment.
            - **initial_position** (*list[float]*): Initial positions of the agents in the domain.
            - **bin_points** (*list[float]*): Points defining bins for resource distribution.
            - **resource_distribution** (*list[float]*): Distribution of resources across bins.
            - **step_size** (*float*): Step size for discretizing the domain.
            - **domain_type** (*str*): Type of domain (e.g., '1d', '2d', 'simplex').
            - **domain_bounds** (*list[float]*): Bounds of the domain (e.g., for 1D [min, max]).
            - **infl_configs** (*dict*): Configuration for the influence kernel.
                - **infl_type** (*str*): Type of influence kernel (e.g., 'gaussian', 'diri', 'jones', 'multi_gaussian', 'custom').
                - **custom_infl** (*function*, optional): Custom influence kernel function (if `infl_type` is 'custom').
            - **parameters** (*list | dict*): Additional parameters for the influence kernels.
            - **fixed_pa** (*int*, optional): Fixed parameter required for Dirichlet influence kernels.
            - **NUM_ITERS** (*int*): Maximum number of iterations per episode.
            - **normalize_reward** (*bool*): Whether to normalize rewards (default: True).
        """
        super().__init__()
        self.num_agent = config.get('num_agents')
        self.initial_position = config.get('initial_position')
        self.bin_points = config.get('bin_points')
        self.resource_distribution = config.get('resource_distribution')
        self.step_size = config.get('step_size')
        self.domain_type = config.get('domain_type')
        self.domain_bounds = config.get('domain_bounds')
        self.infl_configs = config.get('infl_configs')
        self.infl_type = self.infl_configs['infl_type']
        self.parameters = config.get('parameters')
        self.fixed_pa = config.get('fixed_pa')
        self.NUM_ITERS = config.get('NUM_ITERS')
        self.precalculate_reward = config.get('precalculate_reward', True)
        self.normalize_reward = config.get('normalize_reward')
        self.cshift = config.get('cshift',0)
        self.Q = config.get('Q', 1.0)  # Default value for
        self.infl_fshift = config.get('infl_fshift', False)
        self.infl_cshift = config.get('infl_cshift', False)


        self.domain_points_num = int((self.domain_bounds[1] - self.domain_bounds[0]) / self.step_size) + 1
        self.possible_positions = np.linspace(self.domain_bounds[0], self.domain_bounds[1], num=self.domain_points_num)
        self.agents = self.possible_agents = [f"player{i}" for i in range(self.num_agent)]

        # Precalculate the reward dictionary for all possible agents and positions
        if self.precalculate_reward:
            self.reward_dict = MARL_utils.reward_dict(possible_agents=self.possible_agents,
                                                    possible_positions=self.possible_positions,
                                                    num_observations=self.domain_points_num,
                                                    num_agents=self.num_agent,
                                                    bin_points=self.bin_points,
                                                    infl_configs=self.infl_configs,
                                                    parameters=self.parameters,
                                                    fixed_pa=self.fixed_pa,
                                                    infl_fshift=self.infl_fshift,
                                                    infl_cshift=self.infl_cshift,
                                                    cshift=self.cshift,
                                                    Q=self.Q,
                                                    resource_distribution=self.resource_distribution,
                                                    normalize=True)

        
        self.observation_spaces = {f"player{i}": gym.spaces.Discrete(self.domain_points_num) for i in range(self.num_agent)}
        self.action_spaces = {f"player{i}": gym.spaces.Discrete(3) for i in range(self.num_agent)}
        self.last_move = None
        self.num_moves = 0
        self.LEFT = 0
        self.STAY = 1
        self.RIGHT = 2

    def REWARD_MAP(self, observations: dict) -> dict:
        r"""
        Maps the observations to rewards for each agent via the reward dictionary.
        The reward for an agent :math:`i` is computed as:

        .. math::
            u_i(x) = \sum_{b \in \mathbb{B}} G_i(x,b) \cdot B(b)

        where:
            - :math:`G_i(x,b)` is the probability of agent :math:`i` influencing bin :math:`b`
            - :math:`B(b)` is the resource available at bin :math:`b`
            - :math:`\mathbb{B}` is the set of all bin points

        The probability is calculated by the function :function:`InflGame.MARL.utils.MARL_utils.prob_matrix`:


        :param observations: Current observations of all agents.
        :type observations: dict
        :return: Rewards for each agent.
        :rtype: dict
        """
        for key in observations.keys():
            observations[key] = int(np.array(observations[key]).item())
        if self.precalculate_reward:
            reward = self.reward_dict[str(observations)]
        else:
            reward = MARL_utils.reward_obs(observations=observations,
                                            possible_agents = self.possible_agents,
                                            possible_positions = self.possible_positions,
                                            num_agents = self.num_agent,
                                            bin_points = self.bin_points,
                                            infl_configs = self.infl_configs,
                                            parameters = self.parameters,
                                            fixed_pa = self.fixed_pa,
                                            infl_fshift = self.infl_fshift,
                                            infl_cshift = self.infl_cshift,
                                            cshift = self.cshift,
                                            Q = self.Q,
                                            resource_distribution = self.resource_distribution,)
        return reward
    

    def reset(self, *, seed=None, options=None):
        r"""
        Reset the environment to its initial state.

        :param seed: Random seed for reproducibility (optional).
        :type seed: int, optional
        :param options: Additional options for reset (optional).
        :type options: dict, optional
        :return: Initial observations and an empty info dictionary.
        :rtype: tuple
        """
        self.num_moves = 0
        observations_list = self.initial_position_to_observation()
        observations = {self.possible_agents[agent_id]: observations_list[agent_id] for agent_id in range(self.num_agent)}
        self.observations = observations
        return observations, {}

    def initial_position_to_observation(self) -> list:
        r"""
        Convert initial positions to observations. 

        :return: List of initial observations for all agents.
        :rtype: list
        """
        observations_list = []
        for pos in self.initial_position:
            idx = int(np.where(np.around(self.possible_positions, decimals=5) == pos)[0][0])
            observations_list.append(idx)
        return observations_list

    def observation_to_position(self, observations: dict) -> list:
        r"""
        Convert observations to positions in the domain.

        :param observations: Current observations of all agents.
        :type observations: dict
        :return: List of positions corresponding to the observations.
        :rtype: list
        """
        pos_list = []
        for key in observations:
            position = self.possible_positions[int(observations[key])]
            pos_list.append(position)
        return pos_list

    def observation_update(self, actions: dict, observations: dict) -> dict:
        r"""
        Update observations based on actions taken by agents.

        The observations are updated based on the actions taken by each agent via the following rules:
        - If action is LEFT, decrease the observation by 1.
        - If action is STAY, keep the observation unchanged.
        - If action is RIGHT, increase the observation by 1.
        - If the new observation is out of bounds, keep the observation unchanged.


        :param actions: Actions taken by each agent.
        :type actions: dict
        :param observations: Current observations of all agents.
        :type observations: dict
        :return: Updated observations for all agents.
        :rtype: dict
        """
        if self.domain_type == '1d':
            for key in observations:
                if actions[key] == self.LEFT:
                    new_obs = int(observations[key]) - 1
                elif actions[key] == self.STAY:
                    new_obs = int(observations[key])
                elif actions[key] == self.RIGHT:
                    new_obs = int(observations[key]) + 1
                if new_obs < 0 or new_obs > self.domain_points_num - 1:
                    new_obs = int(observations[key])
                else:
                    observations[key] = new_obs
        return observations

    def step(self, action_dict: dict):
        r"""
        Perform a single step in the environment.
        This method updates the environment based on the actions taken by all agents via the :func:`observation_update` method.
        It computes the rewards for each agent using the :func:`REWARD_MAP` method and checks for termination conditions.

        :param action_dict: Actions taken by each agent.
        :type action_dict: dict
        :return: Updated observations, rewards, termination status, truncation status, and info dictionary.
        :rtype: tuple
        """
        self.num_moves += 1
        observations = self.observation_update(actions=action_dict, observations=self.observations)
        rewards = self.REWARD_MAP(observations)
        terminateds = {"__all__": self.num_moves >= self.NUM_ITERS}
        return observations, rewards, terminateds, {}, {}

