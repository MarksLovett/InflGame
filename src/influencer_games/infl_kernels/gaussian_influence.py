import numpy as np
import torch
 


def gaussian_infl(agent_id:int,
                   parameter_instance:list|np.ndarray|torch.Tensor,
                   agents_pos:np.ndarray,
                   bin_points:np.ndarray)->int|torch.Tensor:
        
        # Takes the agent's postion and parameters finds the agnet's influence using gaussian method
        # INPUTS:
        #   agent_id:The current player\agent's id
        #   parameter_instance: parameter(s) unique to your influence distribution
        #   agents_pos: current agent postions
        #   bin_points:locations of the resource/bin points
        # OUTPUTS:
        #   infl: The agents influence using gaussian method
        
        infl_n=np.exp(-(agents_pos[agent_id]-bin_points)**2/(2*parameter_instance[agent_id]**2))
        infl=torch.tensor(infl_n)
                
        return infl

def d_gaussian(agent_id:int,
               parameter_instance:list|np.ndarray|torch.Tensor,
               agents_pos:np.ndarray,
               bin_points:np.ndarray)->int|torch.Tensor:
        # Takes the agents' postion and calcualates the d_(i,k)=d/dx_i ln(f_(i,k)) values
        # INPUTS:
        #   agent_id:The current player\agent's id (i)
        #   parameter_instance: parameter(s) (sigma)
        #   agents_pos: current agent postions
        #   bin_points:locations of the resource/bin points    
        # OUTPUTS:
        #   d_row: values of d_(i,k)=(b_k-x_i)/sigma^2 for all k values

        d_row=-(agents_pos[agent_id]-bin_points)/(parameter_instance[agent_id]**2)
        d_row=torch.from_numpy(d_row)
        return d_row

def gaussian_symmetric_stability(num_agents,d_values,resource_distribution):
        sigma_star=torch.sqrt((num_agents-2)/(num_agents-1)*torch.sum(d_values**2*torch.tensor(resource_distribution))/torch.sum(torch.tensor(resource_distribution)))
        return sigma_star
        