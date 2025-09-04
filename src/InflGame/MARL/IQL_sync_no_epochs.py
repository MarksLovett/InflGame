r"""
.. module:: IQL_sync
   :synopsis: Implements Independent Q-Learning (IQL) with synchronized updates for multi-agent reinforcement learning in influencer games.

Independent Q-Learning with Synchronized Updates Module
========================================================

This module implements the Independent Q-Learning (IQL) algorithm with synchronized updates for multi-agent 
reinforcement learning. The IQL algorithm allows agents to learn independently while interacting in a shared 
environment. It supports epsilon-greedy and softmax policies for action selection and provides utilities for 
training agents over multiple epochs and episodes.

Mathematical Definitions:
-------------------------
The Q-learning update rule for an agent :math:`i` is defined as:

.. math::
    Q_i(s, a) \leftarrow (1 - \alpha) Q_i(s, a) + \alpha \left( r + \gamma \max_{a'} Q_i(s', a') \right)

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
- numpy
- torch
- random
- InflGame.MARL.utils.IQL_utils

Usage:
------
The `IQL_sync` class provides an implementation of the IQL algorithm with synchronized updates. It supports 
custom configurations for learning rate, discount factor, epsilon decay, and more.

Example:
--------

.. code-block:: python

    import numpy as np
    from InflGame.MARL.async_game import influencer_env_async
    from InflGame.MARL.IQL_sync import IQL_sync

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

    # Initialize and train the IQL_sync agent
    iql_agent = IQL_sync(config=iql_config)
    final_q_table = iql_agent.train()

    print("Training completed. Final Q-table:", final_q_table)

"""
import numpy as np
import torch
import random
import hickle as hkl


import InflGame.MARL.utils.IQL_utils as IQL_utils
import InflGame.MARL.utils.MARL_utils as MARL_utils
import InflGame.utils.data_management  as data_management

class IQL_sync_no_epochs():
    r"""
    Implements Independent Q-Learning (IQL) with synchronized updates for multi-agent reinforcement learning.

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
    - `action_choice`: Chooses actions for all agents using epsilon-greedy or softmax policies.
    - `Q_soft_max_action`: Selects an action using a softmax policy.
    - `Q_max_action`: Selects the action with the maximum Q-value.
    - `Q_step`: Performs a single step of Q-learning for all agents.
    - `train`: Trains the agents over multiple epochs and episodes.
    """

    def __init__(self, config=None):
        r"""
        Initializes the IQL_sync class with the given configuration.

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
        Resets the environment and initializes observations for all agents by sampling from their observation spaces.
        """
        self.env.reset()
        self.env.observations= {agent: self.env.observation_spaces[agent].sample() for agent in self.env.possible_agents}
        self.observations = self.env.observations.copy()

    def action_choice(self, episode: int):
        r"""
        Chooses actions for all agents based on the epsilon-greedy policy.

        the :math:`\epsilon` value is adjusted based on the current episode via the function :func:`InflGame.MARL.utils.IQL_utils.adjusted_epsilon`.

        If a random value is less than :math:`\epsilon`, a random action is chosen. Otherwise, the action is selected 
        based on the Q-values using either a softmax or max policy.

        :param episode: The current episode number.
        :type episode: int
        :return: A dictionary mapping each agent to its chosen action.
        :rtype: dict
        """
        action_dict = {}
        epsilon_f = IQL_utils.adjusted_epsilon(configs=self.epsilon_configs, num_agents=self.env.num_agents, episode=episode, episodes=self.episodes)
        for agent in self.env.possible_agents:
            epsilon = np.random.rand()
            action_space = self.env.action_spaces[agent]
            if epsilon < epsilon_f:
                action = action_space.sample()
            else:
                if self.soft_max:
                    action = self.Q_soft_max_action(agent)
                else:
                    action = self.Q_max_action(agent)
            action_dict[agent] = action
        return action_dict

    def Q_soft_max_action(self, agent_id: int):
        r"""
        Chooses an action for the given agent using a softmax policy.

        .. math::
            P(a|s) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}

        The temperature is adjusted based on the observation and the configuration settings via
        :func:`InflGame.MARL.utils.IQL_utils.adjusted_temperature`.

        where:
          - :math:`T` is the temperature parameter
          - :math:`a'` is the set of all possible actions
          - :math:`P(a|s)` is the probability of taking action :math:`a` in state :math:`s`
          - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`

        :param agent_id: The ID of the agent.
        :type agent_id: int
        :return: The chosen action.
        :rtype: int
        """
        observation = self.observations[agent_id]
        self.temperature = IQL_utils.adjusted_temperature(configs=self.configs["temperature_configs"], observation=observation, observation_space_size=self.env.observation_spaces[agent_id].n)
        policy = np.array([np.exp(self.Q_table[agent_id][observation][action] / self.temperature) for action in range(self.env.action_spaces[agent_id].n)])
        policy = policy / policy.sum()
        action = np.random.choice(range(self.env.action_spaces[agent_id].n), p=policy)
        return action

    def Q_max_action(self, agent: int):
        r"""
        Chooses the action with the maximum Q-value for the given player.

        .. math::
            a^* = \arg\max_a Q(s,a)

        where:
          - :math:`s` is the current state
          - :math:`a^*` is the action with the highest Q-value
          - :math:`Q(s,a)` is the Q-value for action :math:`a` in state :math:`s`

        :param agent: The ID of the player.
        :type agent: int
        :return: The action with the highest Q-value.
        :rtype: int
        """
        observation = self.observations[agent]
        LEFT = 0
        STAY = 1
        RIGHT = 2
        max_Q = max(self.Q_table[agent][observation][LEFT], self.Q_table[agent][observation][STAY], self.Q_table[agent][observation][RIGHT])
        if self.Q_table[agent][observation][RIGHT]==self.Q_table[agent][observation][LEFT] and self.Q_table[agent][observation][LEFT]==self.Q_table[agent][observation][STAY]:
            return random.choice([LEFT, STAY, RIGHT])
        else:
            if max_Q == self.Q_table[agent][observation][LEFT]:
                return LEFT
            elif max_Q == self.Q_table[agent][observation][STAY]:
                return STAY
            else:
                return RIGHT

    def Q_step(self, episode: int):
        r"""
        Performs a single step of Q-learning for all agents in the environment. Actions are chosen simultaneously, 
        and the Q-values are updated based on the received rewards.
    
        .. math::
            Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') \right)

        :param episode: The current episode number.
        :type episode: int
        :return: The updated Q-table.
        :rtype: dict
        """
        action_dict = self.action_choice(episode)
        previous_observations = self.observations.copy()
        self.observations, self.rewards, terminateds, _, _ = self.env.step(action_dict)
        for agent in self.env.possible_agents:
            self.Q_table[agent][previous_observations[agent]][action_dict[agent]] = (1 - self.alpha) * self.Q_table[agent][previous_observations[agent]][action_dict[agent]]+ self.alpha * (self.rewards[agent] + self.gamma * self.Q_table[agent][self.observations[agent]][self.Q_max_action(agent)])
        return self.Q_table

    def train(self,checkpoints=False,save_positions=False,data_parameters=None,trials=1,name_ads=None):
        r"""
        Trains the agents using the IQL algorithm over multiple epochs and episodes.
        
        Here the number of episodes is adjusted based on the epoch using the function
        :func:`InflGame.MARL.utils.IQL_utils.adjusted_episodes`.

        At the end of an all episodes the environment is reset and the observations are reinitialized randomly.

        :return: The final Q-table after training.
        :rtype: dict
        """
        if save_positions == True:
                position_array = [MARL_utils.observation_to_position(observations=self.observations.copy(), possible_positions=self.env.possible_positions)]
        
        self.episodes = IQL_utils.adjusted_episodes(configs=self.episode_configs, epoch=0, epochs=self.epochs)
        for episode in range(self.episodes):
            self.Q_step(episode)
            if save_positions == True:
                positions=MARL_utils.observation_to_position(observations=self.observations.copy(), possible_positions=self.env.possible_positions)
                position_array.append(positions)
            if checkpoints == True:
                if episode == 0 or (episode+1)%10 == 0:
                    # Save the Q_table at each epoch
                    data=self.Q_table.copy()
                    q_tables_name=data_management.data_final_name(data_parameters=data_parameters,name_ads=name_ads+["episode"+str(episode+1)])[0]
                    hkl.dump(data, q_tables_name, mode='w', compression='gzip')
        
        
        if save_positions == True:
            if trials == 1:
                hkl.dump(position_array, data_management.data_final_name(data_parameters=data_parameters,name_ads=name_ads+["positions"])[0], mode='w', compression='gzip')
                return self.Q_table
            else:
                return self.Q_table, position_array
        else:
            return self.Q_table

