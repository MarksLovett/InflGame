import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

import hickle as hkl
from torch.nn import Softmax

from influencer_games.utils.utilities import *
from influencer_games.domains.resource_distributions import *
from influencer_games.utils.data_mangement import *

from influencer_games.rl_dynamics.RL_visualization import *
from influencer_games.rl_dynamics.influencer_game_sync import influencer_env_sync
from influencer_games.rl_dynamics.influencer_game_async import influencer_env_async
from influencer_games.rl_dynamics.IQL_sync import IQL_sync
from influencer_games.rl_dynamics.IQL_async import IQL_async
from influencer_games.utils.rl_utils.IQL_utilities import *
from influencer_games.utils.rl_utils.MARL_utilities import *
from influencer_games.utils.rl_utils.my_parse import add_rl_example_script_args


def run_expriment(action_type:str = "sync",
                  env_configs:dict = None,
                  trials:int = 100,
                  gamma:float = 0.3,
                  alpha:float = 0.005,
                  epochs:int = 5000 ,
                  random_seed:int = 0,
                  random_itialization:bool = False,
                  smoothing:bool = True,
                  temperature_configs:dict =None,
                  epsilon_configs:dict = None,
                  episode_configs:dict = None,
                  discription:str = "Test trials",
                  name_ads:list[str]=[],
                  )->None:
    



    if episode_configs==None:
      episode_configs={"TYPE":"Reverse_Cosine_Anneling","episode_max":100,"episode_min":10}
    if temperature_configs==None:
      temperature_configs={"TYPE":"Cosine_Anneling_Distance","temperature_max":1,"temperature_min":0.5}
    if epsilon_configs==None:
      epsilon_configs={"TYPE":"Cosine_Anneling","e_max":.8,"e_min":.3}

    if action_type == "sync":
        env=influencer_env_sync(config=env_configs)
        algo_config = {"random_seed":random_seed,"env":env, "epsilon_configs":epsilon_configs,"gamma":gamma,"alpha":alpha,"epochs":epochs,"episode_configs":episode_configs,"random_initialize":random_itialization,"soft_max":smoothing,"temperature_configs":temperature_configs}
        algo=IQL_sync(algo_config)
    elif action_type=="async":
        env=influencer_env_async(config=env_configs)
        algo_config = {"random_seed":random_seed,"env":env, "epsilon_configs":epsilon_configs,"gamma":gamma,"alpha":alpha,"epochs":epochs,"episode_configs":episode_configs,"random_initialize":random_itialization,"soft_max":smoothing,"temperature_configs":temperature_configs}
        algo=IQL_async(algo_config)

        
    configs={"env_config_main":env_configs,
      "epsilon_configs":epsilon_configs,
      "episode_configs":episode_configs,
      "temperature_configs":temperature_configs,
      "alpha,gamma":[alpha,gamma],
      "random_seed":random_seed,
      "trials":trials,
      "description":discription
    }

    config_parameter=data_parmaters(configs=configs,data_type='configs',resoure_type='gauss_mix_2m')
    q_tables_parameter=data_parmaters(configs=configs,data_type='q_tables',resoure_type='gauss_mix_2m')
    config_name = data_final_name(data_parameters=config_parameter,name_ads=name_ads)[0]
    q_tables_name=data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads)[0]
    hkl.dump(configs, config_name, mode='w', compression='gzip')


    # Training step
    data={}
    random.seed(random_seed)

    for trial in range(trials):
        env.reset()
        q_table=algo.train()
        data[f'q_table_{trial}'] = q_table
        hkl.dump(data, q_tables_name, mode='w', compression='gzip')
        print(f"Trial: {trial+1}/{trials} complete")
    q_tensor=q_tables_to_q_tensors(q_tables=data,num_runs=configs['trials'])
    if trials>=2:
        q_mean=q_tensor.mean(axis=0)
        q_tables_name_mean=data_final_name(data_parameters=q_tables_parameter,name_ads=name_ads+['mean'])[0]
        hkl.dump(q_mean.numpy(), q_tables_name_mean, mode='w', compression='gzip')
        return q_tensor,q_mean
    else:
        return q_tensor