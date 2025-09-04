r"""
.. module:: IQL_async
   :synopsis: Implements Independent Q-Learning (IQL) with asynchronous updates for multi-agent reinforcement learning in influencer games.

Independent Q-Learning with Asynchronous Updates Module
========================================================

This module implements the Independent Q-Learning (IQL) algorithm in an asynchronous multi-agent reinforcement 
learning setting. The IQL algorithm allows agents to learn independently while interacting in a shared environment. 
It supports epsilon-greedy and softmax policies for action selection and provides utilities for training agents 
over multiple epochs and episodes.

Mathematical Definitions:
-------------------------
The Q-learning update rule for an agent :math:`i` is defined as:

.. math::
    Q_i(s, a) \leftarrow (1 - \alpha) Q_i(s, a) + \alpha \left( r + \gamma \max_{a'} Q_i(s', a')\right)

where:
  - :math:`s` is the current state
  - :math:`a` is the action taken
  - :math:`r` is the reward received
  - :math:`s'` is the next state
  - :math:`\alpha` is the learning rate
  - :math:`\gamma` is the discount factor

This implementation supports both epsilon-greedy and softmax action selection policies.

Dependencies:
-------------
- InflGame.MARL

Usage:
------
The `IQL_async` class provides an implementation of the IQL algorithm with asynchronous updates. It supports 
custom configurations for learning rate, discount factor, epsilon decay, and more.

Example:
--------

.. code-block:: python

    import numpy as np
    from InflGame.MARL.async_game import influencer_env_async
    from InflGame.MARL.IQL_async import IQL_async

    # Define environment configuration
    env_config = {
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
    env = influencer_env_async(config=env_config)

    # Define IQL configuration
    iql_config = {
        "random_seed": 42,
        "env": env,
        "epsilon_configs": {"TYPE": "cosine_annealing", "epsilon_max": 1.0, "epsilon_min": 0.1},
        "gamma": 0.9,
        "alpha": 0.01,
        "epochs": 100,
        "random_initialize": True,
        "soft_max": False,
        "episode_configs": {"TYPE": "fixed", "episode_max": 10}
    }

    # Initialize and train the IQL_async agent
    iql_agent = IQL_async(config=iql_config)
    final_q_table = iql_agent.train()

    print("Training completed. Final Q-table:", final_q_table)

"""
import numpy as np
import torch
import random

import InflGame.MARL.utils.IQL_utils as IQL_utils


class IQL_async():
    r"""
    Implements Independent Q-Learning (IQL) with asynchronous updates for multi-agent reinforcement learning.

    Attributes:
    -----------
    - `configs`: Configuration dictionary containing parameters such as random seed, environment, epsilon configurations, 
      gamma, alpha, epochs, and others.
    - `Q_table`: The Q-value table for all agents.
    - `observations`: Current observations for all agents.

    Methods:
    --------
    - `Q_table_initiation`: Initializes the Q-table for all agents.
    - `observation_initialized`: Resets the environment and initializes observations.
    - `action_choice`: Chooses actions for a specific agent using epsilon-greedy or softmax policies.
    - `Q_soft_max_action`: Selects an action using a softmax policy.
    - `Q_max_action`: Selects the action with the maximum Q-value.
    - `Q_step`: Performs a single step of Q-learning for all agents asynchronously.
    - `train`: Trains the agents over multiple epochs and episodes.
    """

    def __init__(self, config=None):
        r"""
        Initializes the IQL_async class with the given configuration.

        :param config: Configuration dictionary containing parameters such as random seed, environment,
                       epsilon configurations, gamma, alpha, epochs, and others.
        :type config: dict
        """
        super().__init__()
        self.configs = config
        self.random_seed = config['random_seed']
        self.env = config['env']
        self.epsilon_configs = config['epsilon_configs']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.epochs = config['epochs']
        self.random_initialize = config['random_initialize']
        self.soft_max = config['soft_max']
        self.episode_configs = config['episode_configs']
        self.observations, _ = self.env.reset()
        self.Q_table_initiation()

    def Q_table_initiation(self):
        r"""
        Initializes the Q-table for all agents.

        If `random_initialize` is True, the Q-values are initialized randomly; otherwise, they are initialized to zero.

        .. math::
            Q(s, a) = 
            \begin{cases} 
            \text{random value in } [0, 0.3] & \text{if random initialization is enabled} \\
            0 & \text{otherwise}
            \end{cases}
        """
        self.Q_table = {}
        if self.random_initialize:
            self.Q_table = {agent: {i: {j: np.random.uniform(low=0.0, high=.3) for j in range(self.env.action_spaces[agent].n)} for i in range(self.env.observation_spaces[agent].n)} for agent in self.env.possible_agents}
        else:
            self.Q_table = {agent: {i: {j: 0 for j in range(self.env.action_spaces[agent].n)} for i in range(self.env.observation_spaces[agent].n)} for agent in self.env.possible_agents}

    def observation_initialized(self):
        r"""
        Resets the environment and initializes random observations for all agents.
        """
        self.env.reset()
        self.env.observations = {agent: self.env.observation_spaces[agent].sample() for agent in self.env.possible_agents}
        self.observations = self.env.observation.copy()

    def action_choice(self, episode: int, agent: str):
        r"""
        Chooses an action for the given player based on the epsilon-greedy policy.

        :math:`\epsilon` is adjusted based on the episode number and the number of players in the environment via the function :func:`InflGame.MARL.utils.IQL_utils.adjusted_epsilon`.

        If a random value is less than :math:`\epsilon`, a random action is chosen. Otherwise, the action is selected 
        based on the Q-values using either a softmax or max policy.

        :param episode: The current episode number.
        :type episode: int
        :param agent: The player for whom the action is being chosen.
        :type agent: str
        :return: The chosen action.
        :rtype: int
        """
        epsilon_f = IQL_utils.adjusted_epsilon(configs=self.epsilon_configs, num_agents=self.env.num_agents, episode=episode, episodes=self.episodes)
        epsilon = np.random.rand()
        action_space = self.env.action_spaces[agent]
        if epsilon < epsilon_f:
            action = action_space.sample()
        else:
            if self.soft_max:
                action = self.Q_soft_max_action(agent)
            else:
                action = self.Q_max_action(agent)
        self.action_dict[agent] = action
        return action

    def Q_soft_max_action(self, agent: str):
        r"""
        Chooses an action for the given player using a softmax policy based on the Q-values.

        .. math::
            P(a|s) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}
        
        The temperature is adjusted based on the observation and the configuration settings via 
        the function :func:`InflGame.MARL.utils.IQL_utils.adjusted_temperature`.

        where:
          - :math:`T` is the temperature parameter
          - :math:`a'` is the set of all possible actions
          - :math:`P(a|s)` is the probability of taking action :math:`a` in state :math:`s`
          - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`

        :param agent: The player for whom the action is being chosen.
        :type agent: str
        :return: The chosen action.
        :rtype: int
        """
        observation = self.observations[agent]
        self.temperature = IQL_utils.adjusted_temperature(configs=self.configs["temperature_configs"], observation=observation, observation_space_size=self.env.observation_spaces[agent].n)
        policy = np.array([np.exp(self.Q_table[agent][observation][action] / self.temperature) for action in range(self.env.action_spaces[agent].n)])
        policy = policy / policy.sum()
        action = np.random.choice(range(self.env.action_spaces[agent].n), p=policy)
        return action

    def Q_max_action(self, agent: str):
        r"""
        Chooses the action with the maximum Q-value for the given player.

        .. math::
            a^* = \arg\max_a Q(s,a)

        where:
          - :math:`s` is the current state
          - :math:`a^*` is the action with the highest Q-value
          - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`

        :param agent: The player for whom the action is being chosen.
        :type agent: str
        :return: The chosen action.
        :rtype: int
        """
        observation = self.observations[agent]
        LEFT = 0
        STAY = 1
        RIGHT = 2
        max_Q = max(self.Q_table[agent][observation][LEFT], self.Q_table[agent][observation][STAY], self.Q_table[agent][observation][RIGHT])
        if max_Q == self.Q_table[agent][observation][LEFT]:
            return LEFT
        elif max_Q == self.Q_table[agent][observation][STAY]:
            return STAY
        else:
            return RIGHT

    def Q_step(self, episode: int):
        r"""
        Performs a single step of Q-learning for all agents in the environment asynchronously where each agent updates its Q-table independently.
    
        The Q-learning update step is done using the following formula:

        .. math::
            Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') \right)

        Where actions are chosen using the `action_choice` method independently for each agent and then 
        the Q-values are updated based on the received rewards and the next state. In the next state the next agent chooses its action.
        In a loop this looks like this:

            for each agent in agent_order:
                action = action_choice(agent)
                observations, rewards, terminateds, _, _ = env.step(action)
                Q table update
        
        after this loop agent_order is shuffled and the loop is repeated until all episodes are passed through for the epoch.

        
        :param episode: The current episode number.
        :type episode: int
        :return: The updated Q-table.
        :rtype: dict
        """
        self.action_dict = {}
        agents = self.env.possible_agents
        random.shuffle(agents)
        for agent in agents:
            action_dict = self.action_choice(episode, agent)
            previous_observations = self.observations.copy()
            self.observations, self.rewards, terminateds, _, _ = self.env.step(self.action_dict, agent)
            self.Q_table[agent][previous_observations[agent]][self.action_dict[agent]] = (1 - self.alpha) * self.Q_table[agent][previous_observations[agent]][self.action_dict[agent]] + self.alpha * (self.rewards[agent] + self.gamma * self.Q_table[agent][self.observations[agent]][self.Q_max_action(agent)])
        return self.Q_table

    def train(self):
        r"""
        Trains the agents using the IQL algorithm over multiple epochs and episodes. 

        The number of episodes in an epoch is adjusted based on the configuration settings via the function
        :func:`InflGame.MARL.utils.IQL_utils.adjusted_episodes`.

        At the end of an epoch, the environment is reset and the observations are reinitialized randomly.

        :return: The final Q-table after training.
        :rtype: dict
        """
        for epoch in range(self.epochs):
            self.episodes = IQL_utils.adjusted_episodes(configs=self.episode_configs, epoch=epoch, epochs=self.epochs)
            for episode in range(self.episodes):
                self.Q_step(episode)
            self.observation_initialized()
        return self.Q_table
