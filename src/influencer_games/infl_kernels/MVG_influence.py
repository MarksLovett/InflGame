import numpy as np
import torch
from influencer_games.utils.utilities import *

def MVG_cov_matrix(parameter_instance:torch.Tensor)->torch.Tensor:
    """ 
    Takes the Multiv-Variate gaussian's covariance matrix's inverse
    INPUTS:
        Parmater intance: Cov. matrix for player i
    OUTPUTS:
        Sigma^-1: Inverse of the covariance matrix or none
    """ 
    return torch.inverse(parameter_instance.float())
        



def MVG_infl(agent_id:int,
            agents_pos:np.ndarray,
            bin_points:np.ndarray,
            sigma_inv:torch.Tensor)->torch.Tensor:
    """"
    Takes the agent's postion and parameters finds the agnet's influence using multi-variate gaussian method
    INPUTS:
        agent_id:The current player\agent's id (i)
        agents_pos: current agent postions
        bin_points:locations of the resource/bin points    
        sigma_inv:Covariance matrix inverse (Sigma^-1)
    OUTPUTS:
        infl: The agents influence using MVG method
    """
    infl=[]
    x_vec=torch.tensor((bin_points-agents_pos[agent_id])).float()
    for i in range(len(bin_points)):
        infl_val=torch.exp(-1/2*x_vec[i,:]@sigma_inv[agent_id]@x_vec.T[:,i])
        infl.append(infl_val)
    infl=torch.stack(infl)

    return infl

def d_MVG(agent_id:int,
          agents_pos:np.ndarray,
          bin_points:np.ndarray,
          sigma_inv:torch.Tensor)->int|torch.Tensor:
    """      
    Takes the agents' postion and calcualates the d_(i,k)=d/dx_i ln(f_(i,k)) values
    INPUTS:
        agent_id:The current player\agent's id (i)
        agents_pos: current agent postions
        bin_points:locations of the resource/bin points    
        sigma_inv:Covariance matrix inverse (Sigma^-1)
    OUTPUTS:
        d_row: values of d_(i,k)=Sigma^-1*(b_k-x_i).T for all k values
    """

    x_vec=torch.tensor((bin_points-agents_pos[agent_id])).T.float()
    d_row=torch.matmul(sigma_inv[agent_id],x_vec)

    return d_row

def gaussian_symmetric_stability_2d_x_star_special_case(bin_points,resource_distribution):
    x_star1=np.dot(bin_points[:,0],resource_distribution)/np.sum(resource_distribution)
    x_star2=np.dot(bin_points[:,1],resource_distribution)/np.sum(resource_distribution)
    return  [x_star1,x_star2]

def gaussian_symmetric_stability_2d(num_agents,
                                    e_values,
                                    resource_distribution):
    e_star_1=e_values[0]
    e_star_2=e_values[1]
    r_d=torch.tensor(resource_distribution)
    a_star_1=torch.sum(r_d*e_star_1**2)
    a_star_2=torch.sum(r_d*e_star_2**2)
    b_star=torch.sum(r_d*e_star_1*e_star_2)
    sigma_star_1=2*(num_agents-1)/(num_agents-2)*torch.sum(r_d)/((a_star_1+a_star_2)+torch.sqrt((a_star_1-a_star_2)**2+(2*b_star)**2))
    sigma_star_2=2*(num_agents-1)/(num_agents-2)*torch.sum(r_d)/((a_star_1+a_star_2)-torch.sqrt((a_star_1-a_star_2)**2+(2*b_star)**2))
    sigma_star_1=1/sigma_star_1
    sigma_star_2=1/sigma_star_2
    if sigma_star_1>sigma_star_2:
        return([sigma_star_1.item(),sigma_star_2.item()])
    else:
        return([sigma_star_2.item(),sigma_star_1.item()])

def gaussian_symmetric_stability_2d_test(num_agents,
                                    bin_points,
                                    resource_distribution):
    mean_0=discrete_mean(bin_points=bin_points[:,0],resource_distribution=resource_distribution)
    var_0=discrete_variance(bin_points=bin_points[:,0],resource_distribution=resource_distribution,mean=mean_0)
    mean_1=discrete_mean(bin_points=bin_points[:,1],resource_distribution=resource_distribution)
    var_1=discrete_variance(bin_points=bin_points[:,1],resource_distribution=resource_distribution,mean=mean_1)
    c_n=(num_agents-2)/(num_agents-1)
    cov=discrete_covariance(bin_points_1=bin_points[:,0],bin_points_2=bin_points[:,1],resource_distribution=resource_distribution,mean_1=mean_0,mean_2=mean_1)
    if var_0+var_1>var_0+var_1+np.sqrt((var_0-var_1)**2+4*cov**2):
        sigma_star=c_n*(var_0+var_1)/2
    else:
        sigma_star=c_n*(var_0+var_1+np.sqrt((var_0-var_1)**2+4*cov**2))/2
        return sigma_star