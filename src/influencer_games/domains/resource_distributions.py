import numpy as np
import torch
from scipy.stats import dirichlet
from scipy.stats import beta


from influencer_games.utils.utilities import *
from influencer_games.domains.simplex.simplex_utlities import *


def resource_distribution_choice(bin_points:np.ndarray,
                          resource_type:str,
                          resource_parameters:list|np.ndarray,
                          ):
    """
    Gives the resources distritbution for a given type of resource type 
    
    Parameters:
        resource_parameters(list|np.ndarray): parameters needed for a given resource distribution type

    Returns:
        resources (np.ndarray): resource distribution
    """
    
    if resource_type=="multi_modal_gaussian_distribution_1D":
        resources=multi_modal_gaussian_distribution_1D(bin_points, stds=resource_parameters[0], means=resource_parameters[1] , mode_factors=resource_parameters[2])
    elif resource_type=="beta":
        resources=beta_distribution(bin_points,resource_parameters[0],resource_parameters[1])
    elif resource_type in ["multi_modal_gaussian_distribution_2D","multi_modal_gaussian_distribution_2D_triangle","multi_modal_gaussian_distribution_2D_square"] :
        resources=multi_modal_gaussian_distribution_2D(bin_points,resource_parameters[0],resource_parameters[1])
    elif resource_type =="dirichlet_distribution" :
        resources=dirichlet_distribution(bin_points,resource_parameters)
    else: 
        return "No known type "+resource_type
    return resources



def multi_modal_gaussian_distribution_1D(bin_points:np.ndarray|torch.Tensor,
                            stds:list[float] = [.1,.1],
                            means:list[float] = [.5,.5],
                            mode_factors = [1,1],
                            )->np.ndarray:
    """
    Gives the resources distritbution for a 1D multi-modal gaussian distribution
    
    Parameters:
        bin_points (list[float]): the points that the resources are at
        stds (list[float]): standard dievation of each mode
        means (list[float]): where the mode is centered

    Returns:
        resources (np.ndarray): resource distribution

    """
    resource_modes=[]
    for mode_id in range(len(stds)):
        mean=means[mode_id]
        std=stds[mode_id]
        mode_factor=mode_factors[mode_id]
        mode=mode_factor*np.exp(-(bin_points-mean)**2/(2*(std)**2))
        resource_modes.append(mode)
    resource_modes=np.array(resource_modes)
    resources=np.sum(resource_modes,axis=0)
    
    return resources


def multi_modal_gaussian_distribution_2D(bin_points:np.ndarray|torch.Tensor,
                            stds:torch.Tensor = torch.tensor([[[.1,0],[0,.1]],[[.1,0],[0,.1]],[[.1,0],[0,.1]]]),
                            means:torch.Tensor = torch.tensor([[0,0],[1,0],[0.5000, 0.8660]]),
                            )->np.ndarray:
    """
    Gives the resources distritbution for a 2D multi-modal gaussian distribution
    
    Parameters:
        bin_points (list[float]): the points that the resources are at
        stds (list[float]): standard dievation of each mode
        means (list[float]): where the mode is centered

    Returns:
        resources (np.ndarray): resource distribution
    
    """

    resource_modes=[]
    for mode_id in range(len(stds)):
        mean=means[mode_id]
        std=stds[mode_id]
        x_vec=torch.tensor((bin_points-mean.numpy())).float()
        sigma_inv=torch.inverse(std)
        distrubtion_values=[]
        for i in range(len(bin_points)):
            distrubtion_value=torch.exp(-1/2*x_vec[i,:]@sigma_inv@x_vec.T[:,i])
            distrubtion_values.append(distrubtion_value.item())
        mode=np.array(distrubtion_values)
        resource_modes.append(mode)
    resource_modes=np.array(resource_modes)
    resources=np.sum(resource_modes,axis=0)
    
    return resources

def beta_distribution(bin_points:np.ndarray|torch.Tensor,
                      alpha_value:float,
                      beta_value:float):
    """
    Gives the resources distritbution for a 1D Beta distribution
    
    Parameters:
        bin_points (list[float]): the points that the resources are at
        alpha_value (float): beta dist alpha value
        beta_value (float): beta dist beta value

    Returns:
        resources (np.ndarray): resource distribution

    """
    f=lambda x: beta.pdf(x, a=alpha_value, b=beta_value)
    resources=f(bin_points)
    return resources

def dirichlet_distribution(bin_points:np.ndarray|torch.Tensor,
                          alphas:list|np.ndarray):
    """
    Gives the resources distritbution for a dirlechet distribution over a simplex. 
    
    Parameters:
        bin_points (list[float]): the points that the resources are at
        alpha_value (list[float]): dirlchet dist alpha values

    Returns:
        resources (np.ndarray): resource distribution

    """
    resources=[]
    for bin_point in bin_points:
        if any(x<=0 for x in bin_point):
            bin_point=projection_onto_siplex(torch.tensor(bin_point)).numpy()[0]
            if any(x==1 for x in bin_point):
                i=np.where(bin_point==1)[0][0]
                bin_point[i]-=.001
                bin_point[i-1]+=.0005
                if i==2:
                    bin_point[i-2]+=.0005
                else:
                    bin_point[i+1]+=.0005
                resources.append(dirichlet.pdf(bin_point, alphas))
            elif any(x==0 for x in bin_point):
                i=np.where(bin_point==0)[0][0]
                bin_point[i]+=.001
                bin_point[i-1]-=.0005
                if i==2:
                    bin_point[i-2]-=.0005
                else:
                    bin_point[i+1]-=.0005
                resources.append(dirichlet.pdf(bin_point, alphas))
        else:
            resources.append(dirichlet.pdf(bin_point, alphas))
    return np.array(resources)