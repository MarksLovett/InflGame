import numpy as np
import torch
import os
from pathlib import Path
def flatten_list(xss):
    return [x for xs in xss for x in xs]
    
def matrix_builder(row_id:int,
                    row:torch.Tensor,
                    matrix:torch.tensor = None,
                    )->torch.Tensor:
    #This function is a utility function

    if row_id==0:
        matrix=row
    elif row_id==1:
        matrix=torch.stack((matrix,row),0)
    else:
        matrix_shape=list(matrix.size()) 
        matrix_shape[0]=1
        matrix_shape=torch.Size(matrix_shape)
        matrix=torch.cat((matrix,torch.from_numpy(np.array(row)).reshape(matrix_shape)),0)
    return matrix

def lr(iter:int,
       lr_type:str,
       learning_rate:list|np.ndarray|float,
       )->float:
    
        if lr_type=='cosine':
            lra=learning_rate[0]+1/2*(learning_rate[1]-learning_rate[0])*(1+np.cos(iter/learning_rate[2]*np.pi))
        elif lr_type=='static':
            lra=learning_rate 
        return lra

def resource_parameter_setup(resource_distribution_type:str = 'multi_modal_gaussian_distribution_1D',
                             varying_paramter_type:str = 'mean', 
                             fixed_parameters_lst:list = [[.1,.1],[1,1]],
                             alpha_st:float = 0,
                             alpha_end: float = 1,
                             alpha_num_points:int = 100,
                             )->tuple:
    #Gives the parmaeters needed for our resource distribution given a certian relationship
    #For now this function is pretty limited to certian relationships to study bifrications
    
    #INPUT
    # resource_parameters: parameters needed for a given resource distribution type
    # sig_lst: list of variances for our multi_modal_gaussian_1D distribution
    # alpha_st: reward parmater start  
    # alpha_end: reward parmater end
    # alpha_num_points: number of reward parameters that we are testing.

    #OUTPUT
    # resource distribution
    param_list=[]
    alpha_values=np.linspace(alpha_end,alpha_st,alpha_num_points)
    if resource_distribution_type=='multi_modal_gaussian_distribution_1D':
        if varying_paramter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                param_list.append([stds,[.5-alpha/2,.5+alpha/2],mode_factors])
            param_list=np.array(param_list)

    elif resource_distribution_type=='beta':

        for alpha in alpha_values:
            param_list.append([alpha,alpha])
        param_list=np.array(param_list)

    elif resource_distribution_type=="multi_modal_gaussian_distribution_2D":
        if varying_paramter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[.5-alpha/2,.5],[.5+alpha/2,.5]])
                param_list.append([stds,alpha_matrix,mode_factors])

    elif resource_distribution_type=="multi_modal_gaussian_distribution_2D_triangle":
        if varying_paramter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[0,0],[alpha,0],[1/2*alpha, 1/2*np.sqrt(3*alpha)]])
                param_list.append([stds,alpha_matrix,mode_factors])

    elif resource_distribution_type=="multi_modal_gaussian_distribution_2D_square":
        if varying_paramter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[0,0],[alpha,0],[0,alpha],[alpha, alpha]])
                param_list.append([stds,alpha_matrix,mode_factors])

    return param_list, alpha_values
        
def player_parameter_setup(num_agents,
                           infl_type,
                           setup_type,
                           reach= None,
                           reach_start = 0.01,
                           reach_end = 0.99,
                           reach_num_points = 100):
    if setup_type=="intial_symmetric_setup":
        if infl_type in ["gaussian","dirl"]:
            player_parameters=[reach]*num_agents
            player_parameters=np.array(player_parameters)
        elif infl_type=='multi_gaussian':
            player_parameters=[reach]*num_agents
            player_parameters=torch.tensor(player_parameters)
    elif setup_type=='parameter_space':
        if infl_type in ["gaussian","dirl"]:
            start=[reach_start]*num_agents
            end=[reach_end]*num_agents
            player_parameters=np.linspace(start,end,reach_num_points)
        elif infl_type == "multi_gaussian":
            start=[[[reach_start,0],[0,reach_start]]]*num_agents
            end=[[[reach_end,0],[0,reach_end]]]*num_agents
            sigmas=np.linspace(reach_start,reach_end,reach_num_points)
            player_parameters=np.linspace(start,end,reach_num_points)
    
    return player_parameters

def organize_array(arr):
    result = []
    left, right = 0, len(arr) - 1

    while left <= right:
        if left == right:
            result.append(arr[left])
        else:
            result.append(arr[left])
            result.append(arr[right])

        left += 1
        right -= 1

    return result

def player_postion_setup(num_agents,
                         setup_type,
                         domain_type,
                         domain_bounds,
                         dimensions = None,
                         bound_lower=0.1,
                         bound_upper=0.9):
    if setup_type=="intial_symmetric_setup":
        if domain_type=="1d":
            player_postions=np.linspace(bound_lower,bound_upper,num=num_agents).reshape( (num_agents, ) )
            player_postions=np.around(player_postions,decimals=2)
        if domain_type=="2d":
            x_edge_values=organize_array(np.linspace(domain_bounds[0,0],domain_bounds[0,1],int(np.ceil(num_agents/4)+1)))
            y_edge_values=organize_array(np.linspace(domain_bounds[1,0],domain_bounds[1,1],int(np.ceil(num_agents/4)+1)))
            pos_list=[]
            tracker=0
            for x_val in x_edge_values:
                for y_val in y_edge_values:
                    pos=[x_val,y_val]
                    pos_list.append(np.array(pos))
                    tracker+=1
                    if tracker==num_agents:
                        break
                if tracker==num_agents:
                        break
            player_postions=pos_list
        elif domain_type=="simplex":
            postion_element=np.linspace(.1,.9,int(np.ceil(num_agents/dimensions)))
            player_postions=[]
            player_id=0
            for element_id in range(int(np.ceil(num_agents/dimensions))):
                for dimension in range(dimensions):
                    player_pos_element=postion_element[element_id]
                    other_player_pos_elements=(1-player_pos_element)/(dimensions-1)
                    player_postion=[other_player_pos_elements]*dimensions
                    player_postion[dimension]=player_pos_element
                    player_postions.append(player_postion)
                    player_id+=1
                    if player_id==num_agents:
                        break
                if player_id==num_agents:
                        break
            player_postions=np.array(player_postions)
    return player_postions

def player_optimal_postion_setup(num_agents:int,
                                 agents_pos:np.ndarray,
                                 infl_type:str,
                                 mean:float,
                                 domain_type:str,
                                 ids:list[int],
                                 ):
    if infl_type=='gaussian':
        player_pos=[]
        for agent_id in range(num_agents):
            if agent_id in ids:
                player_pos.append(agents_pos[agent_id])
            else:
                player_pos.append(mean)
        player_pos=np.array(player_pos)
    return player_pos

def figure_directory(fig_parameters,alt_name):
    my_path = os.path.dirname(os.path.abspath(__file__))
    cwd=my_path+'\\'+'figures'
    p = Path(cwd)
    p.mkdir(exist_ok=True)

    
    file=[cwd,fig_parameters[0]]
    file_name='\\'.join([str(x) for x in file ])
    p = Path(file_name)
    p.mkdir(exist_ok=True)

    file=file+['_'+str(fig_parameters[2])+'_p']
    file_name='\\'.join([str(x) for x in file ])
    p = Path(file_name)
    p.mkdir(exist_ok=True)
    if alt_name== False:
        file=file+['_'+fig_parameters[1]]
        file_name='\\'.join([str(x) for x in file ])
        p = Path(file_name)
        p.mkdir(exist_ok=True)
    
    return file_name
        
def figure_name(fig_parameters,name_ads,save_types):
    plt_type=fig_parameters[1]
    fig_names=[]
    if plt_type=='equalibirum_bifurcation':
        fig_name=fig_parameters[0]+"_pos_bifurcation_"+str(fig_parameters[2])+'_p_'+str(fig_parameters[3])+'_alpha' 
    elif plt_type=='stability_bifurcation_plot_fast':
        fig_name=fig_parameters[0]+"_first_order_bifurcation_"+str(fig_parameters[2])+'_p'
    elif plt_type=='postional_histogram':
        fig_name=fig_parameters[0]+"_pos_hist"+str(fig_parameters[2])+'_p'
    elif plt_type=='policy_avg':   
        fig_name="Policy Average"+'_'+str(fig_parameters[2])+'_p_'+fig_parameters[3]+'_'+ fig_parameters[4]+'_'+fig_parameters[5]

    if len(name_ads)>0:
        for name_additon in name_ads:
            fig_name=fig_name+'_'+name_additon
    for save_type in save_types:
        fig_names.append(fig_name+save_type)
    return fig_names

def figure_final_name(fig_parameters,name_ads,save_types):
    if fig_parameters[1] in ['nothingrn']:
        alt=True
    else:
        alt=False
    
    fig_names=figure_name(fig_parameters=fig_parameters,name_ads=name_ads,save_types=save_types)
    file_names=[]
    for fig_name in fig_names:
        fig_direct=figure_directory(fig_parameters=fig_parameters,alt_name=alt)
        file=[fig_direct,fig_name]
        file_name='\\'.join([str(x) for x in file ])
        file_names.append(file_name)
    return file_names

def discrete_mean(bin_points,resource_distribution):
    mean=np.dot(bin_points,resource_distribution)/np.sum(resource_distribution)
    return mean

def discrete_variance(bin_points,resource_distribution,mean):
    variance=np.dot(bin_points**2,resource_distribution)/np.sum(resource_distribution)-mean**2
    return variance

def discrete_covariance(bin_points_1,bin_points_2,resource_distribution,mean_1,mean_2):
    covariance=np.dot(bin_points_1*bin_points_2,resource_distribution)/np.sum(resource_distribution)-mean_1*mean_2
    return covariance

def split_favor_bottom(num_agents,division):
    if division==0:
        return [num_agents]
    if num_agents==2.0:
        return [num_agents]
    if num_agents==1.0:
        return [1]
    if num_agents%2==0: 
        if division==1:
            total=[np.ceil(num_agents/2**division),np.floor(num_agents/2**division)]
        else:
            bottom=split_favor_bottom(np.ceil(num_agents/2),division=division-1)
            top=bottom.copy()
            top.reverse()
            total=bottom+top
    elif num_agents==3:
        total=[2.0,1.0]
    elif num_agents>3: 
        bottom=split_favor_bottom(np.ceil(num_agents/2),division=division-1)
        top=split_favor_bottom(np.floor(num_agents/2),division=division-1)
        total=bottom+top
    


    return total