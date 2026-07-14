"""
.. module:: experiments
   :synopsis: Provides utilities for running reinforcement learning experiments in the influencer games framework.

Reinforcement Learning Experiments Module
=========================================

This module contains functions and utilities for running reinforcement learning experiments in the influencer games framework. 
It supports both synchronous and asynchronous environments and provides functionality for training and saving Q-tables.

Dependencies:
-------------
- InflGame.MARL
- InflGame.utils

Usage:
------
The `run_experiment` function is the main entry point for running reinforcement learning experiments. It supports both synchronous 
and asynchronous environments and allows for customization of learning parameters, scheduling configurations, and saving results.

Examples
--------

.. code-block:: python

    import numpy as np
    from InflGame.MARL.utils.experiments import run_experiment

    # Define environment configurations
    env_configs = {
        "num_agents": 3,
        "domain_type": "1d",
        "domain_bounds": [0, 1],
        "resource_distribution": "gaussian",
        "resource_parameters": [0.5, 0.1]
    }

    # Run a synchronous experiment
    q_tensor, q_mean = run_experiment(
        action_type="sync",
        env_configs=env_configs,
        trials=10,
        gamma=0.9,
        alpha=0.01,
        epochs=1000,
        random_seed=42,
        smoothing=True,
        description="Synchronous RL experiment",
        name_ads=["sync_test"]
    )

    print("Experiment completed. Q-tensor and Q-mean saved.")
"""

import numpy as np

import random
import hickle as hkl
import torch


import InflGame.MARL.utils.IQL_utils as IQL_utils
import InflGame.utils.data_management  as data_management


from InflGame.MARL.sync_game import influencer_env_sync
from InflGame.MARL.async_game import influencer_env_async
from InflGame.MARL.IQL_sync import IQL_sync
from InflGame.MARL.IQL_async import IQL_async


def run_experiment(action_type: str = "sync",
                   env_configs: dict = None,
                   trials: int = 100,
                   gamma: float = 0.3,
                   alpha: float = 0.005,
                   epochs: int = 5000,
                   random_seed: int = 0,
                   random_initialization: bool = False,
                   smoothing: bool = True,
                   temperature_configs: dict = None,
                   epsilon_configs: dict = None,
                   episode_configs: dict = None,
                   resource_name: str = "gauss_mix_2m",
                   description: str = "Test trials",
                   algo_epoch: bool = True,
                   checkpoints: bool = False,
                   save_positions: bool = False,
                   return_positions: bool = False,
                   name_ads: list[str] = [],
                   fresh_start: bool = True) -> None:
    """
    Runs a reinforcement learning experiment using the Influencer Games framework and an independent Q-learning algorithm.

    Parameters
    ----------
    action_type : str
        Type of environment to use ("sync" or "async").
    env_configs : dict
        Configuration dictionary for the environment.
    trials : int
        Number of trials to run.
    gamma : float
        Discount factor for the Q-learning algorithm.
    alpha : float
        Learning rate for the Q-learning algorithm.
    epochs : int
        Number of epochs for training.
    random_seed : int
        Seed for random number generation.
    random_initialization : bool
        Whether to use random initialization for Q-tables.
    smoothing : bool
        Whether to apply softmax smoothing during training via temperature.
    temperature_configs : dict, optional
        Configuration for temperature scheduling. - **TYPE** (*str*): Type of schedule, e.g., 'fixed', 'cosine_annealing_distance', 'cosine_annealing_distance_segmented'. - **temperature** (*float*, optional): If TYPE == 'fixed', temperature for smoothing. - **temperature_max** (*float*, optional): If TYPE != 'fixed', maximum global temperature. - **temperature_min** (*float*, optional): If TYPE != 'fixed', minimum global temperature. - **temperature_local_max** (*float*, optional): If TYPE == 'cosine_annealing_distance_segmented', minimum for the first segment of the schedule. - **temperature_local_min** (*float*, optional): If TYPE == 'cosine_annealing_distance_segmented', maximum for the second segment of the schedule.
    epsilon_configs : dict, optional
        Configuration for epsilon annealing. - **TYPE** (*str*): Type of schedule, e.g., 'fixed', 'cosine_annealing'. - **epsilon** (*float*, optional): If TYPE == 'fixed', epsilon value. - **epsilon_max** (*float*, optional): If TYPE != 'fixed', maximum epsilon value. - **epsilon_min** (*float*, optional): If TYPE != 'fixed', minimum epsilon value.
    episode_configs : dict, optional
        Configuration for episode scheduling. - **TYPE** (*str*): Type of schedule, e.g., 'fixed', 'reverse_cosine_annealing'. - **episode_max** (*float*): If TYPE == 'fixed', max number of episodes in an epoch. - **episode_min** (*float*, optional): If TYPE == 'reverse_cosine_annealing', global minimum number of episodes in an epoch.
    resource_name : str
        Name tag used when saving experiment artifacts.
    description : str, optional
        Description of the experiment.
    algo_epoch : bool
        If True, train with epoch-based scheduling.
    checkpoints : bool
        If True, save intermediate training checkpoints.
    save_positions : bool
        If True, persist agent position trajectories.
    return_positions : bool
        If True, also return final position tensors.
    name_ads : list[str], optional
        Additional identifiers for naming saved files.
    fresh_start : bool
        If True, start without loading existing Q-tables.

    Returns
    -------
    None | tuple
        ``None`` by default. If ``trials >= 2``, returns stacked Q-tensors (and position tensors when ``return_positions`` is True).
    """

    if episode_configs==None:
      episode_configs={"TYPE":"reverse_cosine_annealing","episode_max":100,"episode_min":10}
    if temperature_configs==None and smoothing==True:
      temperature_configs={"TYPE":"cosine_annealing_distance","temperature_max":1,"temperature_min":0.5}
    if epsilon_configs==None:
      epsilon_configs={"TYPE":"cosine_annealing","epsilon_max":.8,"epsilon_min":.3}

    # Build algo_config template (env will be set per-trial)
    algo_config = {
        "random_seed": random_seed,
        "env": None,  # Will be set per trial
        "epsilon_configs": epsilon_configs,
        "gamma": gamma,
        "alpha": alpha,
        "epochs": epochs,
        "episode_configs": episode_configs,
        "random_initialize": random_initialization,
        "soft_max": smoothing,
        "temperature_configs": temperature_configs,
    }
        
    configs={"env_config_main":env_configs,
      "epsilon_configs":epsilon_configs,
      "episode_configs":episode_configs,
      "temperature_configs":temperature_configs,
      "alpha,gamma":[alpha,gamma],
      "random_seed":random_seed,
      "trials":trials,
      "description":description
    }

    config_parameter=data_management.data_parameters(configs=configs,data_type='configs',resource_type=resource_name)
    q_tables_parameter=data_management.data_parameters(configs=configs,data_type='q_tables',resource_type=resource_name)
    config_name = data_management.data_final_name(data_parameters=config_parameter,name_ads=name_ads)[0]
    q_tables_name=data_management.data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads)[0]
    hkl.dump(configs, config_name, mode='w', compression='gzip')

    # Training step
    data={}
    if save_positions == True:
        position_lis = []
        
    for trial in range(trials):
        # Seed at the start of EACH trial for reproducibility
        trial_seed = random_seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trial_seed)
        
        # Re-create env and algo with fresh seeded state
        
        if action_type == "sync":
            if fresh_start==True:
                env = influencer_env_sync(config=env_configs)
            # Seed gym spaces for reproducibility
            for agent in env.possible_agents:
                env.action_spaces[agent].seed(trial_seed)
                env.observation_spaces[agent].seed(trial_seed)
            algo_config["env"] = env
            if algo_epoch == False:
                from InflGame.MARL.IQL_sync_no_epochs import IQL_sync_no_epochs
                algo = IQL_sync_no_epochs(algo_config)
            else:
                algo = IQL_sync(algo_config)
        elif action_type == "async":
            if fresh_start==True:
                env = influencer_env_async(config=env_configs)
            # Seed gym spaces for reproducibility
            for agent in env.possible_agents:
                env.action_spaces[agent].seed(trial_seed)
                env.observation_spaces[agent].seed(trial_seed)
            algo_config["env"] = env
            algo = IQL_async(algo_config)
        
        env.reset()
        if save_positions == True and trials > 1:
            q_table, position_array = algo.train(checkpoints=checkpoints, save_positions=save_positions, data_parameters=q_tables_parameter, trials=trials, name_ads=name_ads)
            position_lis.append(position_array.copy())
        else:
            q_table,position_array=algo.train(checkpoints=checkpoints,save_positions=save_positions, trials=trials, data_parameters=q_tables_parameter,name_ads=name_ads)
        data[f'q_table_{trial}'] = q_table
        hkl.dump(data, q_tables_name, mode='w', compression='gzip')
        print(f"Trial: {trial+1}/{trials} complete")
    if save_positions == True and trials > 1:
        arr_t=torch.tensor(position_lis)
        mean_vals=arr_t.mean(axis=0)
        mad=torch.mean(torch.abs(arr_t - mean_vals), axis=0)
        hkl.dump(mean_vals.numpy(), data_management.data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads+["mean_positions"])[0],
                  mode='w', compression='gzip')
        hkl.dump(mad.numpy(), data_management.data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads+["mad_positions"])[0],
                  mode='w', compression='gzip')
    try:
        q_tensor=IQL_utils.q_tables_to_q_tensors(q_tables=data,num_runs=configs['trials'])
        if trials>=2:
            q_mean=q_tensor.mean(axis=0)
            q_tables_name_mean=data_management.data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads+['mean'])[0]
            hkl.dump(q_mean.numpy(), q_tables_name_mean, mode='w', compression='gzip')
            if return_positions == True:
                return q_tensor,q_mean, position_lis
            else:
                return q_tensor,q_mean,None
        else:
            if return_positions == True:
                return q_tensor, None, position_array
            else:
                return q_tensor,None, None
    except:
        print("Error: Q-tensor and Q-mean could not be computed")
        return None, data
    