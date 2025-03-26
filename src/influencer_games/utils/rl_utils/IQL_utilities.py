from influencer_games.utils.utilities import *
import torch 
import numpy as np

def Q_table_to_tensor(Q_table):
    Q_matrix=[]
    key_int=0
    for key in Q_table:
        player_matrix=[]
        for key2 in Q_table[key]:
            action_vec=[]
            for key3 in Q_table[key][key2]:
                action_vec.append(Q_table[key][key2][key3])
            action_row=torch.tensor(action_vec)
            player_matrix=matrix_builder(row_id=key2,row=action_row,matrix=player_matrix)
        Q_matrix=matrix_builder(row_id=key_int,row=player_matrix,matrix=Q_matrix)
        key_int+=1
    return Q_matrix

def Q_tensor_to_policy(q_tensor:torch.Tensor,
                      temperature:float = .5,
                      player_id: int =0,
                      )->torch.Tensor:
    q_tensor_exp=torch.exp(q_tensor[player_id]/temperature)
    policy=torch.div(q_tensor_exp.T,torch.sum(q_tensor_exp,1)).T
    return policy

def Adjusted_epsilon(configs,num_players,episode,episodes):
    type=configs['TYPE']
    if type=='Normal':
        epsilon=configs['epsilon']
        epsilon_adjusted=epsilon
    else:
        e_min=configs['e_max']
        e_max=configs['e_min']
        if type=='Cosine_Anneling':
            epsilon_adjusted=e_min+1/2*(e_max-e_min)*(1+np.cos(episode/episodes*np.pi))
        elif type=='MARK':
            if num_players>1:
                epsilon_adjusted=e_max*(np.log(10*num_players)/np.log(episode+1)+num_players/np.log2(episode+1))
            else:
                epsilon_adjusted=e_max*np.log(2)/np.log(episode+1)
        elif type=='FENG':
            if num_players>1:
                epsilon_adjusted=e_max*(np.log(10*num_players)/np.log(episode+1)+num_players/np.log2(episode+1))
            else:
                epsilon_adjusted=e_max*np.log(2)/np.log(episode+1)
    
    return epsilon_adjusted

def Adjusted_temperature(configs,observation,observation_space_size):
    TYPE=configs['TYPE']
    if TYPE=="Normal":
        temperature_adjusted=configs['temperature']
    else:
        temperature_max=configs['temperature_max']
        temperature_min=configs['temperature_min']
        observation_mid=(observation_space_size-1)/2
        if TYPE=="Cosine_Anneling_Distance":
            observation_distance_from_mid=np.abs(observation-observation_mid)
            max_distance=np.floor(observation_space_size/2)
            temperature_adjusted=temperature_min+1/2*(temperature_max-temperature_min)*(1+np.cos(observation_distance_from_mid/max_distance*np.pi))
        if TYPE=="Cosine_Anneling_Distance_segmented":
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

def Adjusted_episodes(configs,epoch,epochs):
    TYPE=configs['TYPE']
    episode_max=configs['episode_max']
    if TYPE=="Normal":
        episodes_adjusted=episode_max
    else:
        episode_min=configs['episode_min']
        if TYPE=="Reverse_Cosine_Anneling":
            episodes_adjusted=int(episode_min+episode_max-int(np.around(episode_min+1/2*(episode_max-episode_min)*(1+np.cos(epoch/epochs*np.pi)),decimals=0)))
    return episodes_adjusted

def q_tables_to_q_tensors(num_runs,q_tables):
    Q_tensors=[]
    for run in range(num_runs):
        q_table=q_tables[f'q_table_{run}']
        q_tensor=Q_table_to_tensor(q_table)
        Q_tensors.append(q_tensor)
    return torch.stack(Q_tensors)
