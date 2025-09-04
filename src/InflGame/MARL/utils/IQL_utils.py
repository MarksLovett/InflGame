"""
.. module:: IQL_utils
   :synopsis: Utility functions for Independent Q-Learning (IQL) in multi-agent reinforcement learning (MARL).

IQL Utilities Module
====================

This module provides utility functions for Independent Q-Learning (IQL) in multi-agent reinforcement learning (MARL). 
The functions include methods for converting Q-tables to tensors, generating policies from Q-tensors, and adjusting 
parameters such as epsilon, temperature, and the number of episodes based on various scheduling strategies.

The module supports:
- Conversion of Q-tables to tensor representations for efficient computation.
- Policy generation using softmax over Q-values.
- Dynamic adjustment of epsilon, temperature, and episodes for exploration-exploitation trade-offs.
- Scheduling strategies such as cosine annealing and reverse cosine annealing.

Dependencies:
-------------
- `InflGame.utils

Usage:
------
The functions in this module can be used to preprocess Q-tables, generate policies, and dynamically adjust parameters 
for MARL algorithms.

Example:
--------

.. code-block:: python

    from InflGame.MARL.utils.IQL_utils import Q_table_to_tensor, Q_tensor_to_policy, adjusted_epsilon

    # Convert a Q-table to a tensor
    Q_table = {
        0: {0: {0: 1.0, 1: 2.0}, 1: {0: 3.0, 1: 4.0}},
        1: {0: {0: 5.0, 1: 6.0}, 1: {0: 7.0, 1: 8.0}}
    }
    Q_tensor = Q_table_to_tensor(Q_table)

    # Generate a policy from the Q-tensor
    policy = Q_tensor_to_policy(Q_tensor, temperature=0.5, agent_id=0)

    # Adjust epsilon using cosine annealing
    configs = {
        'TYPE': 'cosine_annealing',
        'epsilon_min': 0.1,
        'epsilon_max': 1.0
    }
    epsilon = adjusted_epsilon(configs, num_agents=2, episode=10, episodes=100)
    print(f"Adjusted epsilon: {epsilon}")

"""
import InflGame.utils.general as general
import torch 
import numpy as np

def Q_table_to_tensor(Q_table):
    """
    Converts a Q-table (nested dictionary) into a tensor representation.

    .. math::
    Q_{torch}[i,j,k] = Q_{dict}[p_i,s_i, a_j]

    where:
        -:math:`Q(s_i, a_j, a_k)` is the Q-value for state :math:`s_i`, actions :math:`a_j` , and :math:`p_i` the :math:`i`-th player.

    :param dict Q_table: Nested dictionary representing the Q-table.
    :return: Tensor representation of the Q-table.
    :rtype: torch.Tensor
    """
    Q_matrix=[]
    key_int=0
    for key in Q_table:
        agent_matrix=[]
        for key2 in Q_table[key]:
            action_vec=[]
            for key3 in Q_table[key][key2]:
                action_vec.append(Q_table[key][key2][key3])
            action_row=torch.tensor(action_vec)
            agent_matrix=general.matrix_builder(row_id=key2,row=action_row,matrix=agent_matrix)
        Q_matrix=general.matrix_builder(row_id=key_int,row=agent_matrix,matrix=Q_matrix)
        key_int+=1
    return Q_matrix

def Q_tensor_to_policy(q_tensor: torch.Tensor,
                       temperature: float = 0.5,
                       agent_id: int = 0) -> torch.Tensor:
    """
    Converts a Q-tensor into a policy tensor using a softmax function.

    .. math::
        \pi(a|s) = \frac{\exp(Q(s, a) / T)}{\sum_{a'} \exp(Q(s, a') / T)}

    where:
        - :math:`a` is the action.
        - :math:`a'` is the set of all possible actions.
        - :math:`s` is the state.
        - :math:`Q(s, a)` is the Q-value for state :math:`s` and action :math:`a`.
        - :math:`T` is the temperature parameter.
        - :math:`\pi(a|s)` is the policy for action :math:`a` given state :math:`s`.
        


    :param torch.Tensor q_tensor: Q-tensor for all players.
    :param float temperature: Temperature parameter for softmax.
    :param int agent_id: ID of the player for which to compute the policy.
    :return: Policy tensor for the specified player.
    :rtype: torch.Tensor
    """
    q_tensor_exp=torch.exp(q_tensor[agent_id]/temperature)
    policy=torch.div(q_tensor_exp.T,torch.sum(q_tensor_exp,1)).T
    return policy

def adjusted_epsilon(configs, num_agents, episode, episodes):
    """
    Adjusts the epsilon value based on the specified configuration and episode progress.
    
    .. list-table:: Epsilon Adjustment Strategies
        :header-rows: 1

        * - Schedule Type
            - Epsilon Formula
        * - Fixed
            - :math:`\epsilon = \epsilon_{\text{constant}}`
        * - Cosine Annealing
            - :math:`\epsilon = \epsilon_{\text{min}} + \frac{1}{2} (\epsilon_{\text{max}} - \epsilon_{\text{min}}) (1 + \cos(\frac{\pi \cdot \text{episode}}{\text{episodes}}))`
        

    :param dict configs: Configuration dictionary containing epsilon adjustment parameters.
    :param int num_agents: Number of players in the game.
    :param int episode: Current episode number.
    :param int episodes: Total number of episodes.
    :return: Adjusted epsilon value.
    :rtype: float
    """
    type=configs['TYPE']
    if type=='fixed':
        epsilon=configs['epsilon']
        epsilon_adjusted=epsilon
    else:
        e_min=configs['epsilon_max']
        e_max=configs['epsilon_min']
        if type=='cosine_annealing':
            epsilon_adjusted=e_min+1/2*(e_max-e_min)*(1+np.cos(episode/episodes*np.pi))
        elif type=='MARK':
            if num_agents>1:
                epsilon_adjusted=e_max*(np.log(10*num_agents)/np.log(episode+1)+num_agents/np.log2(episode+1))
            else:
                epsilon_adjusted=e_max*np.log(2)/np.log(episode+1)
        elif type=='FENG':
            if num_agents>1:
                epsilon_adjusted=e_max*(np.log(10*num_agents)/np.log(episode+1)+num_agents/np.log2(episode+1))
            else:
                epsilon_adjusted=e_max*np.log(2)/np.log(episode+1)
    
    return epsilon_adjusted

def adjusted_temperature(configs, observation, observation_space_size):
    """
    Adjusts the temperature value based on the specified configuration and observation.

    
    :param dict configs: Configuration dictionary containing temperature adjustment parameters.
    :param int observation: Current observation value.
    :param int observation_space_size: Size of the observation space.
    :return: Adjusted temperature value.
    :rtype: float
    """
    TYPE=configs['TYPE']
    if TYPE=="fixed":
        temperature_adjusted=configs['temperature']
    else:
        temperature_max=configs['temperature_max']
        temperature_min=configs['temperature_min']
        observation_mid=(observation_space_size-1)/2
        if TYPE=="cosine_annealing_distance":
            temperature_adjusted=cosine_annealing_distance_dependent(value_max=temperature_max,value_min=temperature_min,time=observation,time_crit=observation_mid,time_max=observation_space_size)
        if TYPE=="cosine_annealing_distance_segmented":
            temperature_local_max=configs['temperature_local_max']
            temperature_local_min=configs['temperature_local_min']
            observation_distance_from_mid=np.abs(observation-observation_mid)
            max_distance=np.floor(observation_space_size/2)
            if observation_distance_from_mid<=max_distance/2:
                temperature_adjusted=temperature_local_min+1/2*(temperature_max-temperature_local_min)*(1+np.cos(observation_distance_from_mid/(max_distance/2)*np.pi))
            else:
                observation_distance_from_local_mid=np.abs(observation-observation_mid)/2
                temperature_adjusted=temperature_min+1/2*(temperature_local_max-temperature_min)*(1+np.cos(observation_distance_from_local_mid/(max_distance/2)*np.pi))
            
    return temperature_adjusted

def adjusted_episodes(configs: dict,
                      epoch: int,
                      epochs: int) -> int:
    """
    Adjusts the number of episodes based on the specified configuration and epoch progress.

    :param dict configs: Configuration dictionary containing episode adjustment parameters TYPE, episode_max, episode_min.
        TYPE (str): schedule type
        episodes_min (int): minimum number of episodes
        episodes_max (optional,int):maximum number of episodes
    :param int epoch: Current epoch number.
    :param int epochs: Total number of epochs.
    :return: Adjusted number of episodes.
    :rtype: int
    """
    TYPE=configs['TYPE']
    episode_max=configs['episode_max']
    if TYPE=="fixed":
        episodes_adjusted=episode_max
    else:
        episode_min=configs['episode_min']
        if TYPE=="reverse_cosine_annealing":
            episodes_adjusted=reverse_cosine_annealing(value_max=episode_max,value_min=episode_min,time=epoch,time_max=epochs)
    return episodes_adjusted

def q_tables_to_q_tensors(num_runs: int,
                          q_tables: dict) -> torch.Tensor:
    """
    Converts multiple Q-tables into a stacked tensor representation.

    :param int num_runs: Number of runs (Q-tables).
    :param dict q_tables: Dictionary containing Q-tables for each run.
    :return: Stacked tensor representation of all Q-tables.
    :rtype: torch.Tensor
    """
    Q_tensors=[]
    for run in range(num_runs):
        q_table=q_tables[f'q_table_{run}']
        q_tensor=Q_table_to_tensor(q_table)
        Q_tensors.append(q_tensor)
    return torch.stack(Q_tensors)

def cosine_annealing_distance_dependent(value_max: float,
                                        value_min: float,
                                        time: int,
                                        time_crit: int,
                                        time_max: int,
                                        max_distance: int = None) -> float:
    r"""
    Computes a value using cosine annealing based on the distance from a critical time.

    .. math::
        v(t) = v_{\text{min}} + \frac{1}{2} (v_{\text{max}} - v_{\text{min}}) 
        \left(1 + \cos\left(\frac{\pi \cdot |t - t_{\text{crit}}|}{d_{\text{max}}}\right)\right)

    where:
        - :math:`v(t)` is the computed value at time :math:`t`.
        - :math:`v_{\text{max}}` is the maximum value.
        - :math:`v_{\text{min}` is the minimum value.
        - :math:`t_{\text{crit}}` is the critical time step.
        - :math:`d_{\text{max}}` is the maximum distance from the critical time, defaulting to half of :math:`t_{\text{max}}`.

    This function adjusts the value smoothly using a cosine function, depending on the distance from a critical time step.


    :param float value_max: Maximum value.
    :param float value_min: Minimum value.
    :param int time: Current time step.
    :param int time_crit: Critical time step.
    :param int time_max: Maximum time step.
    :param int max_distance: Maximum distance from the critical time. Defaults to half of time_max.
    :return: Computed value based on cosine annealing.
    :rtype: float
    """
    time_from_crit=np.abs(time-time_crit) #distance in time from a critical time
    if max_distance==None:
        max_distance=np.floor(time_max/2) # time crit is in the middle.
    value_between=value_min+1/2*(value_max-value_min)*(1+np.cos(time_from_crit/max_distance*np.pi))
    return value_between

def reverse_cosine_annealing(value_max: float,
                             value_min: float,
                             time: int,
                             time_max: int) -> float:
    """
    Reverse cosine annealing adjusts a value smoothly over time, starting from the minimum value and increasing toward the maximum value, following a cosine curve.

    .. math::
        v(t) = v_{\text{min}} + v_{\text{max}} - \left\lfloor v_{\text{min}} + \frac{1}{2} (v_{\text{max}} - v_{\text{min}}) 
        \left(1 + \cos\left(\frac{\pi \cdot t}{t_{\text{max}}}\right)\right) \right\rfloor

    where:
        - :math:`v(t)` is the computed value at time :math:`t`.
        - :math:`v_{\text{max}}` is the maximum value.
        - :math:`v_{\text{min}` is the minimum value.
        - :math:`t_{\text{max}` is the maximum time step.




    :param float value_max: Maximum value.
    :param float value_min: Minimum value.
    :param int time: Current time step.
    :param int time_max: Maximum time step.
    :return: Computed value based on reverse cosine annealing.
    :rtype: float
    """
    value_between=int(value_min+value_max-int(np.around(value_min+1/2*(value_max-value_min)*(1+np.cos(time/time_max*np.pi)),decimals=0)))
    return value_between
