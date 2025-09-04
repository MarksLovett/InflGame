r"""
.. module:: async_game
   :synopsis: Implements an asynchronous multi-agent environment for influencer games.

Asynchronous Multi-Agent Environment Module
===========================================

This module implements an asynchronous multi-agent environment for influencer games. The environment simulates a 
1D domain where agents can move left, stay, or move right asynchronously. Rewards are calculated based on a 
probability matrix and resource distribution.

Mathematical Definitions:
-------------------------
The reward for an agent :math:`i` is computed as:

.. math::
    u_i(x) = \sum_{b \in \mathbb{B}} G_i(b) \cdot B(b)

where:
  - :math:`G_i(b)` is the probability of agent :math:`i` influencing bin :math:`b`
  - :math:`B(b)` is the resource available at bin :math:`b`
  - :math:`\mathbb{B}` is the set of all bin points

The probability is calculated by the function :func:`InflGame.MARL.utils.MARL_utils.prob_matrix`.

Dependencies:
-------------
- InflGame.MARL
- ray.rllib

Usage:
------
The `influencer_env_async` class provides an asynchronous multi-agent environment for influencer games. It supports 
custom configurations for agents, resource distributions, and influence kernels.

Example:
--------

.. code-block:: python

    import numpy as np
    from InflGame.MARL.async_game import influencer_env_async

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
    env = influencer_env_async(config=config)

    # Reset the environment
    observations, _ = env.reset()

    # Perform a step
    actions = {"player0": env.LEFT, "player1": env.STAY, "player2": env.RIGHT}
    observations, rewards, terminated, truncated, info = env.step(actions, player="player0")

    print("Observations:", observations)
    print("Rewards:", rewards)
    print("Terminated:", terminated)
 
"""
import gymnasium as gym
import numpy as np
import torch
import gymnasium.spaces as spaces

import InflGame.MARL.utils.MARL_utils as MARL_utils
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class influencer_env_async(MultiAgentEnv):
    r"""
    An asynchronous multi-agent environment for influencer games.

    This environment simulates a 1D domain where agents can move left, stay, or move right asynchronously.
    Rewards are calculated based on a probability matrix and resource distribution.

    Attributes:
    -----------
    - `num_agents`: Number of agents in the environment.
    - `initial_position`: Initial positions of the agents.
    - `bin_points`: Points defining bins for resource distribution.
    - `resource_distribution`: Resource distribution across bins.
    - `step_size`: Step size for discretizing the domain.
    - `domain_type`: Type of domain ('1d' or other).
    - `domain_bounds`: Bounds of the domain (min, max).
    - `infl_configs`: Configuration for influence calculations.
    - `infl_type`: Type of influence.
    - `parameters`: Additional parameters for influence calculations.
    - `fixed_pa`: Whether to use fixed probabilities.
    - `NUM_ITERS`: Number of iterations for the game.
    - `domain_points_num`: Number of points in the domain.
    - `possible_positions`: Array of possible positions in the domain.
    - `agents`: List of agent identifiers.
    - `possible_agents`: List of possible agent identifiers.
    - `observation_spaces`: Observation spaces for each agent.
    - `action_spaces`: Action spaces for each agent.
    - `last_move`: Placeholder for the last move.
    - `num_moves`: Number of moves made.
    - `LEFT`: Action identifier for moving left.
    - `STAY`: Action identifier for staying.
    - `RIGHT`: Action identifier for moving right.

    Methods:
    --------
    - `REWARD_MAP`: Computes rewards for all agents based on their positions.
    - `reset`: Resets the environment to its initial state.
    - `initial_position_to_observation`: Converts initial positions to observations.
    - `observation_to_position`: Converts observations to positions in the domain.
    - `observation_update`: Updates observations based on actions taken by agents.
    - `step`: Performs a single step in the environment.
    """

    def __init__(self, config=None):
        r"""
        Initialize the influencer_env_async environment.

        :param config: Configuration dictionary containing:
            - **num_agents** (*int*): Number of agents in the environment.
            - **initial_position** (*list*): Initial positions of the agents.
            - **bin_points** (*list*): Points defining bins for resource distribution.
            - **resource_distribution** (*list*): Resource distribution across bins.
            - **step_size** (*float*): Step size for discretizing the domain.
            - **domain_type** (*str*): Type of domain (e.g., '1d', '2d', 'simplex').
            - **domain_bounds** (*list*): Bounds of the domain (e.g., for 1d [min, max]).
            - **infl_configs** (*dict*): Configuration for influencer kernel.
                - **infl_type** (*str*): Type of influence kernel (e.g., gaussian, diri, jones, multi_gaussian, custom).
                - **custom_infl** (*function*, optional): Custom influence kernel.
            - **parameters** (*dict*): Additional parameters for the influence kernels.
            - **fixed_pa** (*int*, optional): Parameter to be fixed for diri games.
            - **NUM_ITERS** (*int*): Number of iterations for the game.
        :type config: dict
        """
        super().__init__()
        self.num_agents = config.get('num_agents')
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
        self.normalize_reward = config.get('normalize_reward')

        self.domain_points_num = int((self.domain_bounds[1] - self.domain_bounds[0]) / self.step_size) + 1
        self.possible_positions = np.linspace(self.domain_bounds[0], self.domain_bounds[1], num=self.domain_points_num)
        self.agents = self.possible_agents = [f"player{i}" for i in range(self.num_agents)]
        self.observation_spaces = {f"player{i}": gym.spaces.Discrete(self.domain_points_num) for i in range(self.num_agents)}
        self.action_spaces = {f"player{i}": gym.spaces.Discrete(3) for i in range(self.num_agents)}
        self.reward_dict = MARL_utils.reward_dict(possible_agents=self.possible_agents,
                                                  possible_positions=self.possible_positions,
                                                  num_observations=self.domain_points_num,
                                                  num_agents=self.num_agents,
                                                  bin_points=self.bin_points,
                                                  infl_configs=self.infl_configs,
                                                  parameters=self.parameters,
                                                  fixed_pa=self.fixed_pa,
                                                  resource_distribution=self.resource_distribution,
                                                  normalize=self.normalize_reward)

        self.last_move = None
        self.num_moves = 0
        self.LEFT = 0
        self.STAY = 1
        self.RIGHT = 2

    def REWARD_MAP(self, observations) -> dict:
        r"""
        Compute the reward for each agent based on their positions.

        .. math::
            u_i(x) = \sum_{b \in \mathbb{B}} G_i(b) \cdot R(b)

        where:
          - :math:`G_i(b)` is the probability of agent :math:`i` influencing bin :math:`b`
          - :math:`B(b)` is the resource available at bin :math:`b`
          - :math:`\mathbb{B}` is the set of all bin points

        The probability is calculated by the function :func:`InflGame.MARL.utils.MARL_utils.prob_matrix`.

        :param observations: Current observations of all agents.
        :type observations: dict
        :return: Rewards for each agent.
        :rtype: dict
        """
        for key in observations.keys():
            observations[key] = int(np.array(observations[key]).item())
        reward = self.reward_dict[str(observations)]
        return reward

    def reset(self, *, seed: int = None, options: dict = None) -> tuple:
        r"""
        Reset the environment to its initial state.

        :param seed: Random seed for reproducibility, defaults to None.
        :type seed: int, optional
        :param options: Additional options for reset, defaults to None.
        :type options: dict, optional
        :return: Initial observations and an empty info dictionary.
        :rtype: tuple
        """
        self.num_moves = 0
        observations_list = self.initial_position_to_observation()
        observations = {self.possible_agents[agent_id]: observations_list[agent_id] for agent_id in range(self.num_agents)}
        self.observations = observations
        return observations, {}

    def initial_position_to_observation(self) -> list:
        r"""
        Convert initial positions to observations.

        :return: List of observations corresponding to initial positions.
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

        :param observations: Observations of all agents.
        :type observations: dict
        :return: List of positions corresponding to the observations.
        :rtype: list
        """
        pos_list = []
        for key in observations:
            position = self.possible_positions[int(observations[key])]
            pos_list.append(position)
        return pos_list

    def observation_update(self, actions: dict, observations: dict, key: str) -> dict:
        r"""
        Update the observation of a specific agent based on its action.

        The observations are updated based on the actions taken by the agent via the following rules:
        - If action is LEFT, decrease the observation by 1.
        - If action is STAY, keep the observation unchanged.
        - If action is RIGHT, increase the observation by 1.
        - If the new observation is out of bounds, keep the observation unchanged.

        :param actions: Actions taken by all agents.
        :type actions: dict
        :param observations: Current observations of all agents.
        :type observations: dict
        :param key: Key of the agent whose observation is being updated.
        :type key: str
        :return: Updated observations.
        :rtype: dict
        """
        if self.domain_type == '1d':
            if actions[key] == self.LEFT:
                new_obs = int(int(observations[key]) - 1)
            elif actions[key] == self.STAY:
                new_obs = int(observations[key])
            elif actions[key] == self.RIGHT:
                new_obs = int(int(observations[key]) + 1)
            if new_obs < 0 or new_obs > self.domain_points_num - 1:
                new_obs = int(observations[key])
            else:
                observations[key] = new_obs
        return observations

    def step(self, action_dict: dict, agent: str) -> tuple:
        r"""
        Perform a single step in the environment.
        This method updates the environment based on the actions taken by all agents via the :func:`observation_update` method.
        It computes the rewards for each agent using the :func:`REWARD_MAP` method and checks for termination conditions.

        Unlike the step method in synchronous environments :func:`InflGame.MARL.sync_game`
        where all agents take actions at the same time and the environment is updated synchronously,
        this method processes actions asynchronously 

        Additionally, this method requires the player whose action is being processed. 

        :param action_dict: Actions taken by all agents.
        :type action_dict: dict
        :param agent: The player whose action is being processed.
        :type agent: str
        :return: Updated observations, rewards, termination status, truncated status, and info dictionary.
        :rtype: tuple
        """
        self.num_moves += 1 / self.num_agents
        observations = self.observation_update(actions=action_dict, observations=self.observations, key=agent)
        rewards = self.REWARD_MAP(observations)
        terminateds = {"__all__": self.num_moves >= self.NUM_ITERS}
        return observations, rewards, terminateds, {}, {}

