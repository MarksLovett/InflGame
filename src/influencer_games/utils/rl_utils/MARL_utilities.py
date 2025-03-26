 
import torch
import hickle as hkl
import random
from influencer_games.utils.utilities import *

from influencer_games.infl_kernels.gaussian_influence import *
from influencer_games.infl_kernels.Jones_influence import *
from influencer_games.infl_kernels.dirl_influence import *
from influencer_games.infl_kernels.MVG_influence import *


def prob_matrix(num_players,agents_pos,bin_points,infl_configs,parameters,fixed_pa):
        # Takes the agents' influence matrix and makes a proability matrix with each row index corresponding to the agent's ID
        # INPUTS:
        #   parameters: parameter(s) unique to your influence distribution
        # OUTPUTS:
        #   agent_prob_matrix:Matrix of proabilities of a player/agent influenceing each bin-point/resource-point

        infl_matrix=influence_matrix(num_players=num_players,agents_pos=agents_pos,bin_points=bin_points,infl_configs=infl_configs,parameters=parameters,fixed_pa=fixed_pa)
        denom=torch.sum(infl_matrix, 0)
        numer=infl_matrix
        agent_prob_matrix=numer/denom
        
        return agent_prob_matrix

def influence_matrix(num_players,agents_pos,bin_points,infl_configs,parameters,fixed_pa)->torch.Tensor:
        # Takes the agent's postion and parameters and uses the influence function to make a matrix
        # INPUTS:
        #   player_id:The current player\agent's id
        #   parameters: parameter(s) unique to your influence distribution
        # OUTPUTS: 
        #   infl_matrix: Matrix of agnets' influences over every bin_point/resource_point

        infl_type=infl_configs['infl_type']
        #For Dirlechet influence only
        if infl_type=='dirl':
            alpha_matrix=dirl_parm(num_agents=num_players,parameters=parameters,agents_pos=agents_pos,fixed_pa=fixed_pa)
        else:
            alpha_matrix=0
        
        #Assembling the influence matrix for all agents
        infl_matrix=0
        for player_id in range(num_players):
            infl_row=influence(player_id=player_id,agents_pos=agents_pos,bin_points=bin_points,infl_configs=infl_configs,parameters=parameters,alpha_matrix=alpha_matrix)

            infl_matrix=matrix_builder(row_id=player_id,row=infl_row,matrix=infl_matrix)
            
        return infl_matrix

def influence(player_id:int,
              agents_pos,
              bin_points,
              infl_configs,
              parameters:list|np.ndarray|torch.Tensor,
              alpha_matrix:torch.Tensor=0,
              )->torch.Tensor:
        # Takes the agent's postion and parameters and finds the influence wrt to the method you have given
        # INPUTS:
        #   player_id:The current player\agent's id
        #   parameters: parameter(s) unique to your influence distribution
        #   alpha_matrix:Unique to Dirlechet influence, alpha parmaters 
        # OUTPUTS:
        #   infl: influence vetor for current agent ID, each value corresponds to influence over the bin/resource-point at that location
        infl_type=infl_configs['infl_type']


        if infl_type=='gaussian_og':
            infl=np.exp(-(agents_pos[player_id]-bin_points)**2/(2*parameters[player_id]**2))

        elif infl_type=='gaussian':
            infl=gaussian_infl(agent_id=player_id,parameter_instance=parameters,agents_pos=agents_pos,bin_points=bin_points)
        
        elif infl_type=='Jones_M':
            infl=jones_infl(agent_id=player_id,parameter_instance=parameters,agents_pos=agents_pos,bin_points=bin_points)

        elif infl_type=='dirl':
            infl=dirl_infl(agent_id=player_id,bin_points=bin_points,alpha_matrix=alpha_matrix)

        elif infl_type=='multi_gaussian':
            sigma_inv=MVG_cov_matrix(parameter_instance=parameters)

            infl=MVG_infl(agent_id=player_id,agents_pos=agents_pos,bin_points=bin_points,sigma_inv=sigma_inv)
        
        elif infl_type=='custom':
                custom_influence=infl_configs['custom_influence']
                x_torch=torch.tensor(agents_pos[player_id])
                p=np.array([parameters[player_id]])
                infl=custom_influence(x_torch,bin_points=bin_points,parameter_instance=p[0])
        else:
            print('no method selected!') 
        return infl


