"""
.. module:: general
   :synopsis: Provides general utility functions for influencer games.

General Utilities Module
========================

This module provides general utility functions for influencer games. It includes functions for matrix operations, 
learning rate calculations, resource parameter setups, agent position setups, and statistical computations. 
These utilities are used across various components of the influencer games framework.


Usage:
------
The `matrix_builder` function is used to build or append rows to a matrix, while the `learning_rate` function computes 
learning rates based on iteration and type. The `agent_position_setup` function initializes agent positions in 
different domains, and the `discrete_mean` function computes the mean of a discrete distribution.


"""

import numpy as np
import torch
import os
from pathlib import Path
from typing import Union, List, Optional,Dict
import matplotlib.pyplot as plt

def flatten_list(xss):

    """
    Flattens a list of lists into a single list.

    :param xss: A list containing sublists.
    :type xss: list of lists
    :return: A single flattened list containing all elements from the sublists.
    :rtype: list
    """
    return [x for xs in xss for x in xs]

def matrix_builder(row_id: int,
                   row: torch.Tensor,
                   matrix: torch.tensor = None) -> torch.Tensor:
    r"""
    Builds or appends rows to a matrix.

    This function is used to construct a matrix by adding rows iteratively. It supports three cases:
    1. If the matrix is empty (`matrix=None`), the function initializes the matrix with the given row.
    2. If the matrix has one row, the function stacks the new row vertically to create a two-row matrix.
    3. If the matrix already has multiple rows, the function appends the new row to the existing matrix.

    **Behavior**:
    - The function ensures that the dimensions of the new row match the existing matrix.
    - The new row is reshaped and concatenated to the matrix in a way that preserves the matrix's structure.

    **Examples**:

    .. code-block:: python
    
        import torch
        import numpy as np

        # Example 1: Initialize a matrix with the first row
        row_1 = torch.tensor([1, 2, 3])
        matrix = matrix_builder(row_id=0, row=row_1)
        print(matrix)
        # Output: tensor([1, 2, 3])

        # Example 2: Add a second row to the matrix
        row_2 = torch.tensor([4, 5, 6])
        matrix = matrix_builder(row_id=1, row=row_2, matrix=matrix)
        print(matrix)
        # Output:
        # tensor([[1, 2, 3],
        #         [4, 5, 6]])

        # Example 3: Append a third row to the matrix
        row_3 = torch.tensor([7, 8, 9])
        matrix = matrix_builder(row_id=2, row=row_3, matrix=matrix)
        print(matrix)
        # Output:
        # tensor([[1, 2, 3],
        #         [4, 5, 6],
        #         [7, 8, 9]])
    

    **Edge Cases**:
    - If the row dimensions do not match the existing matrix, the function will raise an error.
    - If the matrix is `None`, the function initializes it with the given row.

    :param row_id: The index of the row being added.
    :type row_id: int
    :param row: The row to be added.
    :type row: torch.Tensor
    :param matrix: The existing matrix. Defaults to None.
    :type matrix: torch.tensor, optional
    :return: The updated matrix with the new row added.
    :rtype: torch.Tensor
    """
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

def learning_rate(iter: int,
                  learning_rate_type: str,
                  learning_rate: list | np.ndarray | float) -> float:
    r"""
    .. list-table:: Learning Rate Types
        :header-rows: 1

        * - Learning Rate Type
          - Associated String
          - Description
        * - Cosine Annealing
          - ``'cosine_annealing'``
          - Smoothly decreases the learning rate using a cosine function.
        * - Fixed
          - ``'fixed'``
          - Keeps the learning rate constant throughout.

    The learning rate is computed based on the specified type:

    - **Cosine Annealing**:

      .. math::
         \eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) 
         \left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)

      where:
        - :math:`\eta_t` is the learning rate at iteration :math:`t`.
        - :math:`\eta_{\text{min}}` is the minimum learning rate.
        - :math:`\eta_{\text{max}}` is the maximum learning rate.
        - :math:`T` is the total number of iterations.

    - **Fixed**:
      The learning rate remains constant:
      .. math::
         \eta_t = \eta_{\text{fixed}}


    :param iter: The current iteration.
    :type iter: int
    :param learning_rate_type: The type of learning rate ('cosine' or 'static').
    :type learning_rate_type: str
    :param learning_rate: Learning rate parameters.
    :type learning_rate: list, np.ndarray, or float
    :return: The computed learning rate.
    :rtype: float
    """
    if learning_rate_type=='cosine_annealing':
        lra=learning_rate[0]+1/2*(learning_rate[1]-learning_rate[0])*(1+np.cos(iter/learning_rate[2]*np.pi))
    elif learning_rate_type=='fixed':
        lra=learning_rate 
    return lra

def resource_parameter_setup(resource_distribution_type: str = 'multi_modal_gaussian_distribution_1D',
                             varying_parameter_type: str = 'mean',
                             fixed_parameters_lst: list = [[.1, .1], [1, 1]],
                             alpha_st: float = 0,
                             alpha_end: float = 1,
                             alpha_num_points: int = 100) -> tuple:
    """
    Sets up resource distribution parameters based on the specified type.

    :param resource_distribution_type: Type of resource distribution.
    :type resource_distribution_type: str
    :param varying_parameter_type: Parameter to vary ('mean' or others).
    :type varying_parameter_type: str
    :param fixed_parameters_lst: Fixed parameters for the distribution.
    :type fixed_parameters_lst: list
    :param alpha_st: Start value for alpha.
    :type alpha_st: float
    :param alpha_end: End value for alpha.
    :type alpha_end: float
    :param alpha_num_points: Number of alpha points.
    :type alpha_num_points: int
    :return: A tuple containing the parameter list and alpha values.
    :rtype: tuple
    """
    param_list=[]
    alpha_values=np.linspace(alpha_end,alpha_st,alpha_num_points)
    if resource_distribution_type=='multi_modal_gaussian_distribution_1D':
        if varying_parameter_type=='mean':
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
        if varying_parameter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[.5-alpha/2,.5],[.5+alpha/2,.5]])
                param_list.append([stds,alpha_matrix,mode_factors])

    elif resource_distribution_type=="multi_modal_gaussian_distribution_2D_triangle":
        if varying_parameter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[0,0],[alpha,0],[1/2*alpha, 1/2*np.sqrt(3*alpha)]])
                param_list.append([stds,alpha_matrix,mode_factors])

    elif resource_distribution_type=="multi_modal_gaussian_distribution_2D_square":
        if varying_parameter_type=='mean':
            stds=fixed_parameters_lst[0]
            mode_factors=fixed_parameters_lst[1]
            for alpha in alpha_values:
                alpha_matrix=torch.tensor([[0,0],[alpha,0],[0,alpha],[alpha, alpha]])
                param_list.append([stds,alpha_matrix,mode_factors])

    return param_list, alpha_values
        
def agent_parameter_setup(num_agents: int,
                           infl_type: str,
                           setup_type: str,
                           reach: float = None,
                           reach_start: float = 0.01,
                           reach_end: float = 0.99,
                           reach_num_points: int = 100):
    """
    Sets up agent parameters based on the specified setup type.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param infl_type: Influence type ('gaussian', 'dirichlet', etc.).
    :type infl_type: str
    :param setup_type: Setup type ('initial_symmetric_setup' or 'parameter_space').
    :type setup_type: str
    :param reach: Reach value for symmetric setup. Defaults to None.
    :type reach: float, optional
    :param reach_start: Start value for reach in parameter space.
    :type reach_start: float
    :param reach_end: End value for reach in parameter space.
    :type reach_end: float
    :param reach_num_points: Number of points for reach in parameter space.
    :type reach_num_points: int
    :return: agent parameters.
    :rtype: np.ndarray or torch.Tensor
    """
    
    if setup_type=="initial_symmetric_setup":
        if infl_type in ["gaussian","dirichlet"]:
            agent_parameters=[reach]*num_agents
            agent_parameters=np.array(agent_parameters)
        elif infl_type=='multi_gaussian':
            agent_parameters=[reach]*num_agents
            agent_parameters=torch.tensor(agent_parameters)
    elif setup_type=='parameter_space':
        if infl_type in ["gaussian","dirichlet"]:
            start=[reach_start]*num_agents
            end=[reach_end]*num_agents
            agent_parameters=np.linspace(start,end,reach_num_points)
        elif infl_type == "multi_gaussian":
            start=[[[reach_start,0],[0,reach_start]]]*num_agents
            end=[[[reach_end,0],[0,reach_end]]]*num_agents
            agent_parameters=np.linspace(start,end,reach_num_points)
    if torch.is_tensor(agent_parameters):
        return agent_parameters
    else:
        return torch.tensor(agent_parameters)

def organize_array(arr: list) -> list:
    """
    Organizes an array by alternating elements from the start and end.

    :param arr: Input array.
    :type arr: list
    :return: Organized array.
    :rtype: list
    """
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

def agent_position_setup(num_agents: int,
                          setup_type: str,
                          domain_type: str,
                          domain_bounds: np.ndarray,
                          dimensions: int = None,
                          bound_lower: float = 0.1,
                          bound_upper: float = 0.9):
    """
    Sets up agent/player positions based on the specified domain and setup type.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param setup_type: Setup type ('initial_symmetric_setup').
    :type setup_type: str
    :param domain_type: Domain type ('1d', '2d', or 'simplex').
    :type domain_type: str
    :param domain_bounds: Bounds of the domain.
    :type domain_bounds: np.ndarray
    :param dimensions: Number of dimensions for simplex. Defaults to None.
    :type dimensions: int, optional
    :param bound_lower: Lower bound for positions. Defaults to 0.1.
    :type bound_lower: float
    :param bound_upper: Upper bound for positions. Defaults to 0.9.
    :type bound_upper: float
    :return: agent/player positions.
    :rtype: np.ndarray or list
    """
    if setup_type=="initial_symmetric_setup":
        
        if domain_type=="1d":
            agent_positions=np.linspace(bound_lower,bound_upper,num=num_agents).reshape( (num_agents, ) )
            agent_positions=np.around(agent_positions,decimals=2)

        #simple 2d domains
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
            agent_positions=torch.tensor(np.array(pos_list))

        #unit simplex
        elif domain_type=="simplex":
            position_element=np.linspace(.1,.9,int(np.ceil(num_agents/dimensions)))
            agent_positions=[]
            agent_id=0
            for element_id in range(int(np.ceil(num_agents/dimensions))):
                for dimension in range(dimensions):
                    agent_pos_element=position_element[element_id]
                    other_agent_pos_elements=(1-agent_pos_element)/(dimensions-1)
                    agent_position=[other_agent_pos_elements]*dimensions
                    agent_position[dimension]=agent_pos_element
                    agent_positions.append(agent_position)
                    agent_id+=1
                    if agent_id==num_agents:
                        break
                if agent_id==num_agents:
                        break
            agent_positions=np.array(agent_positions)
    elif setup_type=='paper_default':
        if domain_type=="1d":
            default_agent={2: torch.tensor([.1,.9]),
                           3: torch.tensor([.1,.4,.9]),
                           4: torch.tensor([.1,.4,.7,.9]),
                           5: torch.tensor([.1,.3,.4,.7,.9]),
                           6: torch.tensor([.1,.3,.4,.6,.7,.9]),
                           7: torch.tensor([.1,.3,.4,.2,.6,.7,.9]),
                           8: torch.tensor([.1,.2,.3,.4,.6,.7,.8,.9]),
                           9: torch.tensor([.1,.25,.3,.35,.45,.65,.75,.7,.9]),
                           10: torch.tensor([.1,.25,.3,.35,.45,.6,.65,.75,.7,.9]),
                           11: torch.tensor([.1,.2,.25,.3,.35,.45,.6,.65,.75,.7,.9]),
                           12: torch.tensor([.1,.2,.25,.3,.35,.45,.6,.65,.75,.7,.8,.9]),
                           16: torch.tensor([.115,.115,.21,.21,.29,.29,.391,.391,.609,.609,.71,.71,.79,.79,.885,.885])}
            agent_positions= default_agent[num_agents]



    if torch.is_tensor(agent_positions):
        return agent_positions
    else:
        return torch.tensor(agent_positions)

def agent_optimal_position_setup(num_agents: int,
                                  agents_pos: np.ndarray,
                                  infl_type: str,
                                  mean: float,
                                  domain_type: str,
                                  ids: list[int]):
    """
    Sets up optimal agent/player positions based on influence type and domain.

    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: Current positions of agents.
    :type agents_pos: np.ndarray
    :param infl_type: Influence type.
    :type infl_type: str
    :param mean: Mean position for non-specified agents.
    :type mean: float
    :param domain_type: Domain type.
    :type domain_type: str
    :param ids: List of agent IDs to retain their positions.
    :type ids: list[int]
    :return: Optimal agent/player positions.
    :rtype: np.ndarray
    """
    if infl_type=='gaussian':
        agent_pos=[]
        for agent_id in range(num_agents):
            if agent_id in ids:
                agent_pos.append(agents_pos[agent_id])
            else:
                agent_pos.append(mean)
        agent_pos=np.array(agent_pos)
    return agent_pos

def figure_directory(fig_parameters: list,
                     alt_name: bool) -> str:
    """
    Creates a directory structure for saving figures.

    :param fig_parameters: Parameters for the figure.
    :type fig_parameters: list
    :param alt_name: Whether to use an alternative naming scheme.
    :type alt_name: bool
    :return: Path to the final directory.
    :rtype: str
    """
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
        
def figure_name(fig_parameters: list,
                name_ads: list[float],
                save_types: list[float]) -> list[float]:
    """
    Generates figure names based on parameters and save types.

    :param fig_parameters: Parameters for the figure.
    :type fig_parameters: list
    :param name_ads: Additional names to append.
    :type name_ads: list[float]
    :param save_types: File extensions for saving.
    :type save_types: list[float]
    :return: List of figure names with extensions.
    :rtype: list[float]
    """
    plt_type=fig_parameters[1]
    fig_names=[]
    if plt_type=='equilibrium_bifurcation':
        fig_name=fig_parameters[0]+"_pos_bifurcation_"+str(fig_parameters[2])+'_p_'+str(fig_parameters[3])+'_alpha' 
    elif plt_type=='stability_bifurcation_plot_fast':
        fig_name=fig_parameters[0]+"_first_order_bifurcation_"+str(fig_parameters[2])+'_p'
    elif plt_type=='positional_histogram':
        fig_name=fig_parameters[0]+"_pos_hist"+str(fig_parameters[2])+'_p'
    elif plt_type=='policy_avg':   
        fig_name="Policy Average"+'_'+str(fig_parameters[2])+'_p_'+fig_parameters[3]+'_'+ fig_parameters[4]+'_'+fig_parameters[5]

    if len(name_ads)>0:
        for name_addition in name_ads:
            fig_name=fig_name+'_'+name_addition
    for save_type in save_types:
        fig_names.append(fig_name+save_type)
    return fig_names

def figure_final_name(fig_parameters: list,
                      name_ads: list[float],
                      save_types: list[float]) -> list[float]:
    """
    Generates final file paths for figures.

    :param fig_parameters: Parameters for the figure.
    :type fig_parameters: list
    :param name_ads: Additional names to append.
    :type name_ads: list[float]
    :param save_types: File extensions for saving.
    :type save_types: list[float]
    :return: List of full file paths for the figures.
    :rtype: list
    """
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

def discrete_mean(bin_points: Union[np.ndarray, torch.Tensor],
                  resource_distribution: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Computes the mean of a discrete distribution using torch operations.

    .. math::
        \mu = \frac{\sum_{b\in \mathbb{B}} b_i \cdot B(b)}{\sum_{b\in\mathbb{B}} B(b)}

    where:
        - :math:`b` is the bin point.
        - :math:`\mathbb{B}` is the set of bin points.
        - :math:`B(b)` is the resource value at the bin point :math:`b`.


    :param bin_points: Bin points.
    :type bin_points: Union[np.ndarray, torch.Tensor]
    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[np.ndarray, torch.Tensor]
    :return: Mean of the distribution.
    :rtype: torch.Tensor
    """
    # Convert inputs to tensors if they aren't already
    bin_points_tensor = _to_tensor(bin_points, "bin_points")
    resource_distribution_tensor = _to_tensor(resource_distribution, "resource_distribution")
    
    mean = torch.dot(bin_points_tensor, resource_distribution_tensor) / torch.sum(resource_distribution_tensor)
    return mean

def discrete_variance(bin_points: np.ndarray,
                      resource_distribution: np.ndarray,
                      mean: float) -> float:
    r"""
    Computes the variance of a discrete distribution.

    .. math::
        \sigma^2 = \frac{\sum_{b \in \mathbb{B}} b^2 \cdot B(b)}{\sum_{b \in \mathbb{B}} B(b)} - \mu^2

    where:
        - :math:`b` is the bin point.
        - :math:`\mathbb{B}` is the set of bin points.
        - :math:`B(b)` is the resource value at the bin point :math:`b`.
        - :math:`\mu` is the mean of the distribution.


    :param bin_points: Bin points.
    :type bin_points: np.ndarray
    :param resource_distribution: Resource distribution.
    :type resource_distribution: np.ndarray
    :param mean: Mean of the distribution.
    :type mean: float
    :return: Variance of the distribution.
    :rtype: float
    """
    variance=torch.dot(bin_points**2,resource_distribution)/torch.sum(resource_distribution)-mean**2
    return variance

def discrete_covariance(bin_points_1: np.ndarray,
                        bin_points_2: np.ndarray,
                        resource_distribution: np.ndarray,
                        mean_1: float,
                        mean_2: float) -> float:
    r"""
     Computes the covariance of a discrete 2d distribution.

    .. math::
        \text{Cov}(b_1, b_2) = \frac{\sum_{b \in \mathbb{B}} b_1 \cdot b_2 \cdot B(b)}{\sum_{b \in \mathbb{B}} B(b)} - \mu_1 \cdot \mu_2

    where:
        - :math:`b_1` and :math:`b_2` are the bin points from two distributions.
        - :math:`\mathbb{B}` is the set of bin points.
        - :math:`B(b)` is the resource value at the bin point :math:`b`.
        - :math:`\mu_1` and :math:`\mu_2` are the means of the two distributions.


    :param bin_points_1: First set of bin points.
    :type bin_points_1: np.ndarray
    :param bin_points_2: Second set of bin points.
    :type bin_points_2: np.ndarray
    :param resource_distribution: Resource distribution.
    :type resource_distribution: np.ndarray
    :param mean_1: Mean of the first distribution.
    :type mean_1: float
    :param mean_2: Mean of the second distribution.
    :type mean_2: float
    :return: Covariance of the distribution.
    :rtype: float
    """
    covariance=torch.dot(bin_points_1*bin_points_2,resource_distribution)/torch.sum(resource_distribution)-mean_1*mean_2
    return covariance

def split_favor_bottom(num_agents: int,
                       division: int) -> list:
    r"""
    Splits a given number of agents into groups, favoring the bottom group in terms of size.

    This function recursively divides the agents into smaller groups, ensuring that the bottom group 
    (or the first group in the resulting list) has more agents when the total number of agents cannot 
    be evenly divided. The division process continues until the specified number of divisions is reached.

    **Behavior**:
    - If `division` is 0, the function returns a single group containing all agents.
    - If the number of agents is 1, the function returns a single group with one agent.
    - If the number of agents is even, the agents are split evenly between the bottom and top groups.
    - If the number of agents is odd, the bottom group gets one more agent than the top group.

    **Examples**:
    - For `num_agents=7` and `division=2`, the function will split the agents into groups like `[4, 3]`.
    - For `num_agents=8` and `division=3`, the function will recursively split into smaller groups like `[2, 2, 2, 2]`.

    **Recursive Logic**:
    - The function uses recursion to divide the agents into smaller groups. At each step, the bottom group 
      is determined first, and the remaining agents are split further into smaller groups.

    **Edge Cases**:
    - If `division=0`, the function returns a single group containing all agents.
    - If `num_agents=1`, the function returns `[1]`.
    - If `num_agents=2` and `division=1`, the function returns `[1, 1]`.

    :param num_agents: Total number of agents.
    :type num_agents: int
    :param division: Number of divisions.
    :type division: int
    :return: List of group sizes.
    :rtype: list
    """
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

def _to_tensor(value, name: str, expected_shape: Optional[tuple] = None, dtype=torch.float32) -> torch.Tensor:
        """Helper function to convert inputs to tensors with validation."""
        if value is None:
            raise ValueError(f"{name} cannot be None")
        
        if isinstance(value, (list, np.ndarray)):
            tensor = torch.tensor(value, dtype=dtype)
        elif isinstance(value, torch.Tensor):
            tensor = value.clone().detach().to(dtype)
        else:
            raise TypeError(f"{name} must be a list, np.ndarray, or torch.Tensor, got {type(value)}")
        
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tensor.shape}")
        
        return tensor


def get_color_by_index(index: int, color_scheme: str = 'default') -> str:
    """
    Return a color based on an integer input.
    
    :param index: Integer index to determine color.
    :type index: int
    :param color_scheme: Color scheme to use ('default', 'matplotlib', 'bright', 'pastel').
    :type color_scheme: str
    :return: Hex color code or matplotlib color name.
    :rtype: str
    :raises ValueError: If color_scheme is not supported.
    """
    if not isinstance(index, int):
        raise ValueError(f"Index must be an integer, got {type(index)}")
    
    if color_scheme == 'default':
        # Predefined list of distinct colors
        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
        ]
        return colors[(10-index) % len(colors)]
    
    elif color_scheme == 'matplotlib':
        # Use matplotlib's tab colors
        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]
        return colors[index % len(colors)]
    
    elif color_scheme == 'bright':
        # Bright, high-contrast colors
        colors = [
            '#FF0000',  # bright red
            '#00FF00',  # bright green
            '#0000FF',  # bright blue
            '#FFFF00',  # bright yellow
            '#FF00FF',  # bright magenta
            '#00FFFF',  # bright cyan
            '#FFA500',  # bright orange
            '#800080',  # purple
            '#FFC0CB',  # pink
            '#A52A2A',  # brown
        ]
        return colors[index % len(colors)]
    
    elif color_scheme == 'pastel':
        # Soft, pastel colors
        colors = [
            '#FFB3BA',  # pastel red
            '#BAFFC9',  # pastel green
            '#BAE1FF',  # pastel blue
            '#FFFFBA',  # pastel yellow
            '#FFDFBA',  # pastel orange
            '#E0BBE4',  # pastel purple
            '#FFC0CB',  # pastel pink
            '#B0C4DE',  # pastel blue-gray
            '#F0E68C',  # pastel gold
            '#DDA0DD',  # pastel plum
        ]
        return colors[index % len(colors)]
    
    elif color_scheme == 'colormap':
        # Generate colors using matplotlib colormap
        colormap = plt.cm.Set3  # You can change this to other colormaps
        normalized_index = (index % 12) / 12.0  # Set3 has 12 colors
        color_rgba = colormap(normalized_index)
        # Convert RGBA to hex
        return '#{:02x}{:02x}{:02x}'.format(
            int(color_rgba[0] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[2] * 255)
        )
    elif color_scheme == 'Greys':
        colormap = plt.cm.Greys
        normalized_index = (8-index % 12) / 12.0
        color_rgba = colormap(normalized_index)
        # Convert RGBA to hex
        return '#{:02x}{:02x}{:02x}'.format(
            int(color_rgba[0] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[2] * 255)
        )
    
    else:
        raise ValueError(f"Unsupported color_scheme: {color_scheme}. "
                        f"Choose from 'default', 'matplotlib', 'bright', 'pastel', 'colormap'")

def generate_color_palette(num_colors: int, color_scheme: str = 'default') -> list:
    """
    Generate a list of colors for a given number of items.
    
    :param num_colors: Number of colors to generate.
    :type num_colors: int
    :param color_scheme: Color scheme to use.
    :type color_scheme: str
    :return: List of color codes.
    :rtype: list
    :raises ValueError: If num_colors is not positive.
    """
    if not isinstance(num_colors, int) or num_colors <= 0:
        raise ValueError(f"num_colors must be a positive integer, got {num_colors}")
    
    return [get_color_by_index(i, color_scheme) for i in range(num_colors)]