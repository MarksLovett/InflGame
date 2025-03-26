import numpy as np
import torch
from influencer_games.utils.utilities import *

def shift_matrix_jacobian(num_agents,
                          agents_pos,
                          bin_points,
                          Q,
                          infl_matrix,
                          )->torch.Tensor:
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=-2*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-1)
            shift_row.append(shift_instance)
        shift_row=torch.tensor(shift_row)
        shift_matrix=matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def shift_matrix_jacobian_ii(num_agents,
                             agents_pos,
                             bin_points,
                             Q,
                             infl_matrix,
                             )->torch.Tensor:
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=2*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-2)
            shift_row.append(shift_instance)
        shift_row=np.array(shift_row)
        shift_row=torch.tensor(shift_row)
        shift_matrix=matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def shift_matrix_jacobian_ij(num_agents,
                             agents_pos,
                             bin_points,
                             Q,
                             infl_matrix,
                             )->torch.Tensor:
    denom=torch.sum(infl_matrix, 0)
    shift_matrix=0
    for agent_id in range(num_agents):
        shift_row=[]
        for bin_point in bin_points:
            shift_instance=4*Q*(bin_point-agents_pos[agent_id])**(2*num_agents-2)
            shift_row.append(shift_instance)
        shift_row=torch.tensor(shift_row)
        shift_matrix=matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)
    shift_matrix=shift_matrix/denom
    return shift_matrix

def dd_lnf_matrix(agent_id:int,
                  parameter_instance:list|np.ndarray|torch.Tensor,
                  infl_type
                  )->float|torch.Tensor:
    #Calculates d_dx_{i,l}^2 ln(f_{i,k}) (second derivative of ln(f) wrt to the same variable)
    
    #INPUT
    # Agent_id: the id of the agent id
    #parameter_instance: Parameters unique to your influence function

    #OUTPUT
    # dd_i: the second derivative
    if infl_type=='gaussian':
        dd_i=-1/(parameter_instance[agent_id]**2)
    elif infl_type=='dirl':
        dd_i='idk'
    return dd_i


def jacobian_off_diag(resource_distribution,
                      infl_fshift,
                        di:torch.Tensor,
                        pi:torch.Tensor,
                        dj:torch.Tensor,
                        pj:torch.Tensor,
                        shift_i:float|torch.Tensor=0,
                        shift_j:float|torch.Tensor=0,
                        shift_ij:float|torch.Tensor=0,
                        )->float:
    j_elm=di*dj*(-pi*pj*(1-pi)+pi**2*pj)*torch.tensor(resource_distribution)
    if infl_fshift==True:
        j_elm=j_elm+(-shift_ij+2*dj*pj*shift_i+di*(1+2*pi)*shift_j+2*shift_j*shift_i)*pi*torch.tensor(resource_distribution)
    j_elm=torch.sum(j_elm)
    return j_elm

def jacobian_diag(resource_distribution,
                  infl_fshift,
                  dd_i,
                  di,
                  pi,
                  shift_i=0,
                  shift_ii=0,
                  ):
    j_elm=(dd_i*pi*(1-pi)+di**2*pi*(1-pi)**2-di**2*pi**2*(1-pi))*torch.tensor(resource_distribution)
    if infl_fshift==True:
        j_elm=j_elm+(((di*(3*pi-1)+2*shift_i)*shift_i-shift_ii)*pi)*torch.tensor(resource_distribution)
    j_elm=torch.sum(j_elm)
    return j_elm


def jacobian_matrix(num_agents,
             parameter_instance,
             agents_pos,
             bin_points,
             resource_distribution,
             infl_type,
             infl_fshift,
             Q,
             infl_matrix,
             prob_matrix,
             d_lnf_matrix,
             ):
    j_matrix=0
    if infl_fshift==True:
            shift_i=shift_matrix_jacobian(num_agents,agents_pos,bin_points,Q,infl_matrix)
            shift_ii=shift_matrix_jacobian_ii(num_agents,agents_pos,bin_points,Q,infl_matrix)
            shift_ij=shift_matrix_jacobian_ij(num_agents,agents_pos,bin_points,Q,infl_matrix)
    else:
        shift_i=[0]*num_agents
        shift_ii=[0]*num_agents
        shift_ij=[0]*num_agents

    for agent_id in range(num_agents):
        j_row=[]
        pi=prob_matrix[agent_id]
        di=d_lnf_matrix[agent_id]
        dd_i=dd_lnf_matrix(agent_id=agent_id,parameter_instance=parameter_instance,infl_type=infl_type)
        for a_id2 in range(num_agents):
            if agent_id==a_id2:
                j_elm=jacobian_diag(resource_distribution,infl_fshift,dd_i,di,pi,shift_i=shift_i[agent_id],shift_ii=shift_ii[agent_id])
            else:
                dj=d_lnf_matrix[a_id2]
                pj=prob_matrix[a_id2]
                j_elm=jacobian_off_diag(resource_distribution,infl_fshift,di,pi,dj,pj,shift_i=shift_i[agent_id],shift_j=shift_i[a_id2],shift_ij=shift_ij[agent_id])
            j_row.append(j_elm)
        j_row=torch.tensor(j_row)
        j_matrix=matrix_builder(row_id=agent_id,row=j_row,matrix=j_matrix)
    return j_matrix

