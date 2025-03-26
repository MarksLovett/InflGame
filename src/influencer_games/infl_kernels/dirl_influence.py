import numpy as np
import torch
from scipy.stats import dirichlet
from scipy.special import psi
from influencer_games.utils.utilities import *



def dirl_parm(num_agents:int,
              parameter_instance:list | np.ndarray | torch.Tensor,
              agents_pos:list|np.ndarray,
              fixed_pa:int,
              )->torch.Tensor:
        # This function is only for the Dirlechet influence
        # Takes the agent's postion and parameters and transforms them accodingly such that the agent's postion is the mean of a Dirlecht distribution
        # INPUTS:
        #   parameter_instance: fixed alpha_phi
        # OUTPUTS:
        #   alpha_matrix: matrix of alpha parameters for all players; row index correspond to the player ID
        alpha_matrix=0
        for agent_id in range(num_agents):
            alpha_row=[]
            for i in range(len(agents_pos[0])):
                if i!=fixed_pa:
                    alpha_instance=parameter_instance[agent_id]/agents_pos[agent_id,fixed_pa]*agents_pos[agent_id,i]
                else:
                    alpha_instance=parameter_instance[agent_id]
                if alpha_instance<=0:
                    alpha_instance=torch.tensor(.0000001)
                alpha_row.append(alpha_instance.item())
            alpha_row=torch.tensor(alpha_row)
            
            alpha_matrix=matrix_builder(row_id=agent_id,row=alpha_row,matrix=alpha_matrix)
            
        return alpha_matrix

    
def dirl_infl(  agent_id:int,
                bin_points:np.ndarray,
                alpha_matrix:torch.Tensor,
                )->int|torch.Tensor:
    # Takes the agent's postion and parameters finds the agnet's influence using Dirlechet method
    # INPUTS:
    #   agent_id:The current player\agent's id (i)
    #   bin_points:locations of the resource/bin points
    #   alpha_matrix:Parmaeters for the Dirl influence
    # OUTPUTS:
    #   infl: The agents influence using Dirlechet method

    infl=[]
    for bin_poin in bin_points:
        if  np.any(np.round(bin_poin,5)<=0):
            infl_instance=0
        else:
            infl_instance=dirichlet.pdf(bin_poin,alpha_matrix[agent_id].detach())
        infl.append(infl_instance)
    infl=torch.tensor(infl)

    return infl
    


    

def d_dirl(agent_id:int,
           agents_pos:np.ndarray,
           bin_points:np.ndarray,
           alpha_matrix:torch.Tensor,
           fixed_pa:int,
           )->int|torch.Tensor:
    # Takes the agents' postion and calcualates the d_(i,k)=d/dx_i ln(f_(i,k)) values
    # INPUTS:
    #   agent_id:The current player\agent's id (i)
    #   agents_pos: current agent postions
    #   bin_points:locations of the resource/bin points
    #   alpha_matrix:Parameters for the Dirl influence
    #   fixed_pa: fixed coordinate direction for Dirl influence    
    # OUTPUTS:
    #   d_row: values of d_(i,k)=d/dx_i ln(f_(i,k)) for all k values


        d_row=[]
        c_f=alpha_matrix[agent_id,fixed_pa].item()/agents_pos[agent_id,fixed_pa]
        for i in range(len(agents_pos[0])):
            if i==fixed_pa:
                d_instance=np.zeros(len(bin_points[:,i]))
            else:
                d_instance=c_f*(np.log(bin_points[:,i])-psi(alpha_matrix[agent_id,i].item())+psi(torch.sum(alpha_matrix[agent_id]).item()))
            d_row.append(d_instance)
        d_row=np.array(d_row)
        d_fp_t=[]
        for i in range(len(agents_pos[0])):
            if i!=fixed_pa:
                d_temp=-agents_pos[agent_id,i]/agents_pos[agent_id,fixed_pa]*d_row[i]
            else:
                d_temp=d_row[fixed_pa]
            d_fp_t.append(d_temp)
        d_fp_s=np.sum(np.array(d_fp_t),0)
        d_row[fixed_pa]=d_fp_s
        d_row=torch.from_numpy(d_row)

        return d_row


