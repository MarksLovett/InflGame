import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from typing import Type, TypeVar




from influencer_games.utils.utilities import *
from influencer_games.utils.rl_utils.MARL_utilities import *
from influencer_games.utils.rl_utils.IQL_utilities import *

from influencer_games.rl_dynamics.IQL_sync import IQL_sync
from influencer_games.rl_dynamics.IQL_async import IQL_async
from influencer_games.rl_dynamics.influencer_game_sync import influencer_env_sync 
from influencer_games.rl_dynamics.influencer_game_async import influencer_env_async
from influencer_games.utils.rl_utils.my_parse import add_rl_example_script_args


def policy_histogram(q_table:dict = None,
                     q_tensor:torch.Tensor = None,
                     player_id:int = 0,
                     temperature: float = 1,
                     )->None:
    """
        Plots the Q-table into a policy via soft_max as a seaborn histogram
    Args:
        q_tensor: Q-table turned into a torch.Tensor
        player_id(int): Player's id num 
        temperature(float): A smoothness factor

    Returns:
        The altered (or newly created) parser object.   
         
    """
    
    if q_table !=None:
        q_tensor=Q_table_to_tensor(q_table)
    fig,ax=plt.subplots()
    policy=Q_tensor_to_policy(q_tensor=q_tensor,player_id=player_id,temperature=temperature)
    ax=sns.heatmap(policy,xticklabels=['left','stay','right'],yticklabels=5,cmap=sns.cubehelix_palette(as_cmap=True))
    ax.invert_yaxis()
    ax.set_box_aspect(1) 
    plt.ylabel('States')
    plt.xlabel('Actions')
    plt.title('Policy Average for Player '+str(player_id+1))
    plt.close()
    return fig,ax

def policy_determinsiticly_to_actions(env:influencer_env_async,
                                      q_table:dict = None,
                                      q_tensor:torch.Tensor = None,
                                      intial_postion:np.ndarray = np.array([0,1]),
                                      num_step:int = 10,
                                      temperature:float = 1,):
    if q_table != None:
        q_tensor=Q_table_to_tensor(q_table)
    env.intial_postion=intial_postion
    observations=env.reset()[0]
    player_id=0
    policies=[]
    for player in env.possible_agents:
        policy_player=Q_tensor_to_policy(q_tensor=q_tensor,temperature=temperature,player_id=player_id)
        max_policy=torch.max(policy_player,dim=1)[1]
        policies.append(max_policy)
        player_id+=1
    pos_matrix=[]
    reward_matrix=[]
    for step in range(num_step):
        action_dict={}
        player_id=0
        for player in env.possible_agents:
            max_policy=policies[player_id]
            postion_id=observations[player]
            action=max_policy[postion_id]
            action_dict[player]=action
            player_id+=1
        observations,rewards,_, _ , _ =env.step(action_dict)

        reward_vec=torch.tensor([rewards[player] for player in env.possible_agents])
        pos_vec=torch.tensor(env.observation_to_postion(observations))

        pos_matrix=matrix_builder(row_id=step,row=pos_vec,matrix=pos_matrix)
        reward_matrix=matrix_builder(row_id=step,row=reward_vec,matrix=reward_matrix)
    return pos_matrix, reward_matrix
    
def reward_plot(reward_matrix,possible_agents):
    fig,ax=plt.subplots()
    ax.set_box_aspect(1)
    player_id=0
    for player in possible_agents:
        ax.plot(reward_matrix[:,player_id],label=f"{player}")
        player_id+=1
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Player Reward")
    plt.legend()

def pos_plot(pos_matrix,possible_agents,bounds):
    fig,ax=plt.subplots()
    ax.set_box_aspect(1)
    player_id=0
    for player in possible_agents:
        ax.plot(pos_matrix[:,player_id],label=f"{player}")
        player_id+=1
    plt.xlabel("Steps")
    plt.ylabel("Postion")
    plt.ylim(bounds[0],bounds[1])
    plt.title("Player Position")
    plt.legend()

