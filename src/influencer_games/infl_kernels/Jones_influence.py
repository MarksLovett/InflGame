import numpy as np
import torch

def jones_infl( agent_id:int,
                parameter_instance:list|np.ndarray|torch.Tensor,
                agents_pos:np.ndarray,
                bin_points:np.ndarray)->int|torch.Tensor:
    # Takes the agent's postion and parameters finds the agnet's influence using Jones method
    # INPUTS:
    #   agent_id:The current player\agent's id
    #   parameter_instance: parameter(s) unique to your influence distribution
    #   agents_pos: current agent postions
    #   bin_points:locations of the resource/bin points 
    # OUTPUTS:
    #   infl: The agents influence using Jones method

    infl_n=1/np.abs(agents_pos[agent_id]-bin_points)**parameter_instance[agent_id][0]
    infl=torch.tensor(infl_n)

    return infl

def d_Jones(agent_id:int,
            parameter_instance:list|np.ndarray|torch.Tensor,
            agents_pos:np.ndarray,
            bin_points:np.ndarray)->int|torch.Tensor:
    # Takes the agents' postion and calcualates the d_(i,k)=d/dx_i ln(f_(i,k)) values
    # INPUTS:
    #   agent_id:The current player\agent's id (i)
    #   parameter_instance: parameter(s) p
    #   agents_pos: current agent postions
    #   bin_points:locations of the resource/bin points    
    # OUTPUTS:
    #   d_row: values of d_(i,k)=d/dx_i ln(f_(i,k)) for all k values

    d_row=[]
    for bin_point in bin_points:  
        if  agents_pos[agent_id]>bin_point:
            d_instance=-parameter_instance[agent_id][0]*(1)/(np.abs(agents_pos[agent_id]-bin_point))
        elif  agents_pos[agent_id]<bin_point:
            d_instance=-parameter_instance[agent_id][0]*(-1)/(np.abs(agents_pos[agent_id]-bin_point))
        else:
            d_instance=0
        d_row.append(d_instance)
    d_row=np.array(d_row)
    d_row=torch.from_numpy(d_row)
    return d_row