"""
.. module:: general
   :synopsis: Provides general utility functions for influencer games.

General Utilities Module
=========================

This module provides general utility functions for influencer games. It includes functions for matrix operations, 
learning rate calculations, resource parameter setups, agent position setups, and statistical computations. 
These utilities are used across various components of the influencer games framework.

Dependencies:
-------------
- NumPy
- PyTorch
- Matplotlib

Usage:
------
The `matrix_builder` function is used to build or append rows to a matrix, while the `learning_rate` function computes 
learning rates based on iteration and type. The `agent_position_setup` function initializes agent positions in 
different domains, and the `discrete_mean` function computes the mean of a discrete distribution.

Example:
--------

.. code-block:: python
    
    from InflGame.utils.general import matrix_builder, learning_rate, discrete_mean
    import torch
    import numpy as np
    
    # Build a matrix incrementally
    row1 = torch.tensor([1.0, 2.0, 3.0])
    matrix = matrix_builder(row_id=0, row=row1)
    
    # Calculate learning rate with cosine annealing
    lr = learning_rate(
        iter=10,
        learning_rate_type='cosine_annealing',
        learning_rate=[0.0001, 0.01, 100]
    )
    
    # Compute discrete mean
    bin_points = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    resources = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
    mean = discrete_mean(bin_points, resources)

"""

import numpy as np
import torch
import os
from pathlib import Path
from typing import Union, List, Optional,Dict
import matplotlib.pyplot as plt

def flatten_list(xss: list) -> list:
    """
    Flattens a list of lists into a single list.
    
    This function takes a nested list structure and returns a single-level list
    containing all elements from the sublists in order.
    
    **Example**:
    
    .. code-block:: python
        
        nested = [[1, 2], [3, 4], [5]]
        result = flatten_list(nested)
        # Returns: [1, 2, 3, 4, 5]
    
    :param xss: A list containing sublists.
    :type xss: list
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
                  learning_rate: list | np.ndarray | float,
                  gradient: torch.Tensor = None) -> float:
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
        * - Trust Region
          - ``'trust_region'``
          - Adapts learning rate based on trust region radius with exponential decay.

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

    - **Trust Region**:
      The learning rate adapts based on trust region radius:
      .. math::
         \eta_t = \eta_{\text{initial}} \cdot \max\left(\eta_{\text{min\_factor}}, \exp\left(-\frac{t}{\tau}\right)\right)

      where:
        - :math:`\eta_{\text{initial}}` is the initial learning rate.
        - :math:`\eta_{\text{min\_factor}}` is the minimum learning rate factor.
        - :math:`\tau` is the decay time constant.
        - :math:`t` is the current iteration.


    :param iter: The current iteration.
    :type iter: int
    :param learning_rate_type: The type of learning rate ('cosine_annealing', 'fixed', or 'trust_region').
    :type learning_rate_type: str
    :param learning_rate: Learning rate parameters. For trust_region: [initial_lr, min_factor, decay_constant]
    :type learning_rate: list, np.ndarray, or float
    :return: The computed learning rate.
    :rtype: float
    """
    if learning_rate_type=='cosine_annealing':
        lra=learning_rate[0]+1/2*(learning_rate[1]-learning_rate[0])*(1+np.cos(iter/learning_rate[2]*np.pi))
    elif learning_rate_type=='fixed':
        lra=learning_rate
    elif learning_rate_type=='trust_region':
        # Trust region learning rate: [initial_lr, min_factor, decay_constant]
        lra = trust_region_learning_rate(
            iter=iter,
            initial_lr=learning_rate[0],
            min_factor=learning_rate[1],
            decay_constant=learning_rate[2]
        )
    elif learning_rate_type=='gradient_magnitude':
        gradient_magnitude = torch.max(torch.abs(gradient)).item()
        if gradient_magnitude == 0:
            lra = 1.0  # Default learning rate when gradient is zero
        else:
            if iter <= learning_rate[2]:
                lra = 1.0 / (gradient_magnitude)* learning_rate[0]
            else:
                lra = learning_rate[1]
    return lra

def trust_region_learning_rate(iter: int,
                              initial_lr: float,
                              min_factor: float,
                              decay_constant: float) -> float:
    """
    Compute trust region learning rate with exponential decay.
    
    This function implements a trust region-style learning rate that starts at
    an initial value and decays exponentially over time, with a minimum bound
    to prevent the learning rate from becoming too small.
    
    The learning rate is computed as:
    η_t = η_initial × max(η_min_factor, exp(-t/τ))
    
    :param iter: The current iteration.
    :type iter: int
    :param initial_lr: Initial learning rate.
    :type initial_lr: float
    :param min_factor: Minimum learning rate factor (prevents learning rate from going too small).
    :type min_factor: float
    :param decay_constant: Decay time constant (controls how fast the learning rate decays).
    :type decay_constant: float
    :return: The computed trust region learning rate.
    :rtype: float
    :raises ValueError: If parameters are invalid (negative values, etc.).
    """
    if initial_lr <= 0:
        raise ValueError(f"initial_lr must be positive, got {initial_lr}")
    if min_factor < 0 or min_factor > 1:
        raise ValueError(f"min_factor must be between 0 and 1, got {min_factor}")
    if decay_constant <= 0:
        raise ValueError(f"decay_constant must be positive, got {decay_constant}")
    if iter < 0:
        raise ValueError(f"iter must be non-negative, got {iter}")
    
    # Exponential decay with minimum bound
    decay_factor = max(min_factor, np.exp(-iter / decay_constant))
    return initial_lr * decay_factor

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
    alpha_values=np.linspace(alpha_st,alpha_end,alpha_num_points)
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
        if infl_type in ["gaussian","dirichlet","beta"]:
            agent_parameters=[reach]*num_agents
            agent_parameters=np.array(agent_parameters)
        elif infl_type=='multi_gaussian':
            agent_parameters=[reach]*num_agents
            agent_parameters=torch.tensor(agent_parameters)
    elif setup_type=='parameter_space':
        if infl_type in ["gaussian","dirichlet","beta"]:
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
    
    This function reorders the input array by alternating between elements from
    the beginning and end of the array, moving towards the center.
    
    **Example**:
    
    .. code-block:: python
        
        arr = [1, 2, 3, 4, 5]
        result = organize_array(arr)
        # Returns: [1, 5, 2, 4, 3]
    
    :param arr: Input array.
    :type arr: list
    :return: Organized array with alternating elements.
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
                          bound_upper: float = 0.9) -> Union[np.ndarray, torch.Tensor]:
    """
    Sets up agent/player positions based on the specified domain and setup type.
    
    This function initializes agent positions within the specified domain bounds.
    It supports various domain types including 1D line segments, 2D rectangles,
    and simplex domains with barycentric coordinates.
    
    **Domain Types**:
    
    - ``'1d'``: Positions agents along a line segment
    - ``'2d'``: Positions agents in a rectangular domain
    - ``'simplex'``: Positions agents in a simplex with barycentric coordinates
    
    **Setup Types**:
    
    - ``'initial_symmetric_setup'``: Distributes agents symmetrically
    - ``'paper_default'``: Uses default positions from published work
    
    **Example**:
    
    .. code-block:: python
        
        import numpy as np
        from InflGame.utils.general import agent_position_setup
        
        # Setup 3 agents in 1D domain
        positions = agent_position_setup(
            num_agents=3,
            setup_type='initial_symmetric_setup',
            domain_type='1d',
            domain_bounds=np.array([0, 1])
        )
    
    :param num_agents: Number of agents.
    :type num_agents: int
    :param setup_type: Setup type ('initial_symmetric_setup' or 'paper_default').
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
    :return: Agent/player positions as tensor.
    :rtype: Union[np.ndarray, torch.Tensor]
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
                                  ids: List[int]) -> np.ndarray:
    """
    Sets up optimal agent/player positions based on influence type and domain.
    
    This function computes optimal positions for agents given the influence
    function type and domain constraints. Some agents can retain their current
    positions while others are optimized.
    
    **Example**:
    
    .. code-block:: python
        
        import numpy as np
        from InflGame.utils.general import agent_optimal_position_setup
        
        current_pos = np.array([0.2, 0.5, 0.8])
        optimal_pos = agent_optimal_position_setup(
            num_agents=3,
            agents_pos=current_pos,
            infl_type='gaussian',
            mean=0.5,
            domain_type='1d',
            ids=[0]  # Keep first agent fixed
        )
    
    :param num_agents: Number of agents.
    :type num_agents: int
    :param agents_pos: Current positions of agents.
    :type agents_pos: np.ndarray
    :param infl_type: Influence type ('gaussian', 'dirichlet', etc.).
    :type infl_type: str
    :param mean: Mean position for non-specified agents.
    :type mean: float
    :param domain_type: Domain type ('1d', '2d', or 'simplex').
    :type domain_type: str
    :param ids: List of agent IDs to retain their positions.
    :type ids: List[int]
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

def figure_directory(fig_parameters: List,
                     alt_name: bool) -> str:
    """
    Creates a directory structure for saving figures.
    
    This function generates a hierarchical directory structure based on the
    provided figure parameters, ensuring the necessary folders exist for
    organizing saved visualizations.
    
    **Example**:
    
    .. code-block:: python
        
        from InflGame.utils.general import figure_directory
        
        fig_params = ['section_A', 'bifurcation', 3]
        dir_path = figure_directory(fig_params, alt_name=False)
    
    :param fig_parameters: Parameters for the figure (section, type, number of players).
    :type fig_parameters: List
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
        
def figure_name(fig_parameters: List,
                name_ads: List[str],
                save_types: List[str]) -> List[str]:
    """
    Generates figure names based on parameters and save types.
    
    This function creates descriptive filenames for saved figures based on
    the figure type and optional additional naming components.
    
    **Example**:
    
    .. code-block:: python
        
        from InflGame.utils.general import figure_name
        
        fig_params = ['section_A', 'equilibrium_bifurcation', 3]
        names = figure_name(
            fig_params,
            name_ads=['alpha_0.5'],
            save_types=['.png', '.svg']
        )
    
    :param fig_parameters: Parameters for the figure.
    :type fig_parameters: List
    :param name_ads: Additional names to append.
    :type name_ads: List[str]
    :param save_types: File extensions for saving.
    :type save_types: List[str]
    :return: List of figure names with extensions.
    :rtype: List[str]
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

def figure_final_name(fig_parameters: List,
                      name_ads: List[str],
                      save_types: List[str]) -> List[str]:
    """
    Generates final file paths for figures.
    
    This function combines directory paths and filenames to create complete
    file paths for saving figures.
    
    **Example**:
    
    .. code-block:: python
        
        from InflGame.utils.general import figure_final_name
        
        fig_params = ['section_A', 'equilibrium_bifurcation', 3]
        paths = figure_final_name(
            fig_params,
            name_ads=['run_1'],
            save_types=['.png', '.svg']
        )
    
    :param fig_parameters: Parameters for the figure.
    :type fig_parameters: List
    :param name_ads: Additional names to append.
    :type name_ads: List[str]
    :param save_types: File extensions for saving.
    :type save_types: List[str]
    :return: List of full file paths for the figures.
    :rtype: List[str]
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

def discrete_variance(bin_points: Union[np.ndarray, torch.Tensor],
                      resource_distribution: Union[np.ndarray, torch.Tensor],
                      mean: float) -> torch.Tensor:
    r"""
    Computes the variance of a discrete distribution.

    .. math::
        \sigma^2 = \frac{\sum_{b \in \mathbb{B}} b^2 \cdot B(b)}{\sum_{b \in \mathbb{B}} B(b)} - \mu^2

    where:
        - :math:`b` is the bin point.
        - :math:`\mathbb{B}` is the set of bin points.
        - :math:`B(b)` is the resource value at the bin point :math:`b`.
        - :math:`\mu` is the mean of the distribution.
    
    **Example**:
    
    .. code-block:: python
        
        import torch
        from InflGame.utils.general import discrete_mean, discrete_variance
        
        bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        resources = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
        mean = discrete_mean(bins, resources)
        variance = discrete_variance(bins, resources, mean)

    :param bin_points: Bin points.
    :type bin_points: Union[np.ndarray, torch.Tensor]
    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[np.ndarray, torch.Tensor]
    :param mean: Mean of the distribution.
    :type mean: float
    :return: Variance of the distribution.
    :rtype: torch.Tensor
    """
    variance=torch.dot(bin_points**2,resource_distribution)/torch.sum(resource_distribution)-mean**2
    return variance

def discrete_covariance(bin_points_1: Union[np.ndarray, torch.Tensor],
                        bin_points_2: Union[np.ndarray, torch.Tensor],
                        resource_distribution: Union[np.ndarray, torch.Tensor],
                        mean_1: float,
                        mean_2: float) -> torch.Tensor:
    r"""
    Computes the covariance of a discrete 2D distribution.

    .. math::
        \text{Cov}(b_1, b_2) = \frac{\sum_{b \in \mathbb{B}} b_1 \cdot b_2 \cdot B(b)}{\sum_{b \in \mathbb{B}} B(b)} - \mu_1 \cdot \mu_2

    where:
        - :math:`b_1` and :math:`b_2` are the bin points from two distributions.
        - :math:`\mathbb{B}` is the set of bin points.
        - :math:`B(b)` is the resource value at the bin point :math:`b`.
        - :math:`\mu_1` and :math:`\mu_2` are the means of the two distributions.
    
    **Example**:
    
    .. code-block:: python
        
        import torch
        from InflGame.utils.general import discrete_covariance
        
        bins_x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        bins_y = torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8])
        resources = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])
        cov = discrete_covariance(bins_x, bins_y, resources, 0.5, 0.5)

    :param bin_points_1: First set of bin points.
    :type bin_points_1: Union[np.ndarray, torch.Tensor]
    :param bin_points_2: Second set of bin points.
    :type bin_points_2: Union[np.ndarray, torch.Tensor]
    :param resource_distribution: Resource distribution.
    :type resource_distribution: Union[np.ndarray, torch.Tensor]
    :param mean_1: Mean of the first distribution.
    :type mean_1: float
    :param mean_2: Mean of the second distribution.
    :type mean_2: float
    :return: Covariance of the distribution.
    :rtype: torch.Tensor
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

def _to_tensor(value,
               name: str,
               expected_shape: Optional[tuple] = None,
               dtype=torch.float32) -> torch.Tensor:
    """
    Helper function to convert inputs to tensors with validation.
    
    This internal utility ensures consistent tensor conversion across the module,
    with optional shape validation.
    
    :param value: Input value to convert to tensor.
    :type value: Union[list, np.ndarray, torch.Tensor]
    :param name: Name of the parameter for error messages.
    :type name: str
    :param expected_shape: Expected shape of the tensor. Defaults to None.
    :type expected_shape: Optional[tuple]
    :param dtype: Desired data type of the tensor.
    :type dtype: torch.dtype
    :return: Converted and validated tensor.
    :rtype: torch.Tensor
    :raises ValueError: If value is None or shape doesn't match expected_shape.
    """
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
    Return a color based on an integer index.
    
    This function provides consistent color mapping for visualization purposes.
    Colors cycle through the selected scheme if the index exceeds available colors.
    
    **Available Color Schemes**:
    
    - ``'default'``: Standard color palette
    - ``'matplotlib'``: Matplotlib tab10 colors
    - ``'bright'``: High-contrast bright colors
    - ``'pastel'``: Soft pastel colors
    - ``'colormap'``: Viridis colormap
    - ``'Greys'``: Grayscale colors
    
    **Example**:
    
    .. code-block:: python
        
        from InflGame.utils.general import get_color_by_index
        
        # Get the first color in default scheme
        color = get_color_by_index(0, 'default')
        
        # Get colors for multiple agents
        colors = [get_color_by_index(i, 'bright') for i in range(3)]
    
    :param index: Integer index to determine color.
    :type index: int
    :param color_scheme: Color scheme to use.
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

def generate_color_palette(num_colors: int, color_scheme: str = 'default') -> List[str]:
    """
    Generate a list of colors for a given number of items.
    
    This function creates a color palette suitable for distinguishing multiple
    agents or data series in visualizations.
    
    **Example**:
    
    .. code-block:: python
        
        from InflGame.utils.general import generate_color_palette
        
        # Generate 5 colors from bright scheme
        palette = generate_color_palette(5, 'bright')
        
        # Use in plotting
        for i, color in enumerate(palette):
            plt.plot(data[i], color=color, label=f'Agent {i}')
    
    :param num_colors: Number of colors to generate.
    :type num_colors: int
    :param color_scheme: Color scheme to use.
    :type color_scheme: str
    :return: List of color codes.
    :rtype: List[str]
    :raises ValueError: If num_colors is not positive.
    """
    if not isinstance(num_colors, int) or num_colors <= 0:
        raise ValueError(f"num_colors must be a positive integer, got {num_colors}")
    
    return [get_color_by_index(i, color_scheme) for i in range(num_colors)]


def smoothing_zeros(tensor: torch.Tensor,
                    fill_value: Optional[float] = None,
                    inplace: bool = False) -> torch.Tensor:
    """
    Optimized function to smooth zeros at the beginning and end of a 1D tensor.
    
    Fills leading zeros with the first non-zero value and trailing zeros 
    with the last non-zero value. This is useful for cleaning up time series
    data or trajectory data with missing values at the boundaries.
    
    **Edge Cases Handled**:
    
    - Empty tensor: returns empty tensor
    - All-zero tensor: fills with fill_value or returns unchanged
    - Single non-zero element: fills entire tensor with that value
    - No leading/trailing zeros: returns original tensor
    - Single element tensor: returns unchanged
    
    **Examples**:
    
    .. code-block:: python
        
        import torch
        from InflGame.utils.general import smoothing_zeros
        
        # Basic smoothing
        result = smoothing_zeros(torch.tensor([0, 3, 2, 0]))
        # Returns: tensor([3, 3, 2, 2])
        
        # All-zero tensor with fill value
        result = smoothing_zeros(torch.tensor([0, 0, 0, 0]), fill_value=1.0)
        # Returns: tensor([1., 1., 1., 1.])
    
    :param tensor: Input 1D tensor to smooth.
    :type tensor: torch.Tensor
    :param fill_value: Value to use if tensor is all zeros. If None, returns original tensor unchanged.
    :type fill_value: Optional[float]
    :param inplace: If True, modifies the tensor in place. Default False.
    :type inplace: bool
    :return: Smoothed tensor.
    :rtype: torch.Tensor
    :raises TypeError: If tensor is not a torch.Tensor.
    :raises ValueError: If tensor is not 1D.
    """
    
    # Input validation
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")
    
    # Handle empty tensor
    if tensor.numel() == 0:
        return tensor.clone() if not inplace else tensor
    
    # Handle single element tensor
    if tensor.numel() == 1:
        return tensor.clone() if not inplace else tensor
    
    # Create result tensor (clone if not inplace)
    result = tensor if inplace else tensor.clone()
    
    # Find non-zero elements efficiently
    non_zero_mask = tensor != 0
    
    # Handle all-zero tensor
    if not non_zero_mask.any():
        if fill_value is not None:
            result.fill_(fill_value)
        return result
    
    # Find first and last non-zero indices using efficient methods
    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
    min_idx = non_zero_indices[0].item()  # First non-zero index
    max_idx = non_zero_indices[-1].item()  # Last non-zero index
    
    
    # Early return if no smoothing needed
    if min_idx == 0 and max_idx == len(tensor) - 1:
        return result
    
    # Get the boundary values
    first_non_zero_value = tensor[min_idx]
    last_non_zero_value = tensor[max_idx]
    
    # Fill leading zeros (more efficient than slicing when possible)
    if min_idx > 0:
        result[:min_idx] = first_non_zero_value
    
    # Fill trailing zeros
    if max_idx < len(tensor) - 1:
        result[max_idx + 1:] = last_non_zero_value
    
    return result

def smoothing_zeros_batch(tensor_batch: torch.Tensor,
                          fill_value: Optional[float] = None,
                          inplace: bool = False) -> torch.Tensor:
    """
    Batch version of smoothing_zeros for processing multiple 1D tensors efficiently.
    
    This function applies zero smoothing to multiple tensors simultaneously,
    which is more efficient than processing them individually. It's particularly
    useful for processing batches of agent trajectories or time series data.
    
    **Example**:
    
    .. code-block:: python
        
        import torch
        from InflGame.utils.general import smoothing_zeros_batch
        
        # Batch of 3 trajectories
        batch = torch.tensor([
            [0, 1, 2, 0],
            [0, 0, 3, 0],
            [1, 2, 3, 4]
        ])
        
        result = smoothing_zeros_batch(batch)
    
    :param tensor_batch: 2D tensor where each row is a 1D tensor to smooth.
    :type tensor_batch: torch.Tensor
    :param fill_value: Value to use for all-zero tensors.
    :type fill_value: Optional[float]
    :param inplace: If True, modifies tensors in place.
    :type inplace: bool
    :return: Batch of smoothed tensors.
    :rtype: torch.Tensor
    :raises TypeError: If tensor_batch is not a torch.Tensor.
    """
    
    if not isinstance(tensor_batch, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor_batch)}")
    
    if tensor_batch.dim() ==1:
        return smoothing_zeros(tensor_batch, fill_value=fill_value, inplace=inplace)
    
    if tensor_batch.numel() == 0:
        return tensor_batch.clone() if not inplace else tensor_batch
    
    result = tensor_batch if inplace else tensor_batch.clone()
    
    # Step 1: Find the global minimum and maximum indices where non-zero values start/end across all batches
    global_min_nonzero_idx = tensor_batch.size(1)  # Initialize to sequence length
    global_max_nonzero_idx = -1  # Initialize to -1
    
    # Find the minimum starting index and maximum ending index of non-zero values across all batches
    for i in range(tensor_batch.size(0)):
        row = tensor_batch[i]
        non_zero_mask = row != 0
        
        if non_zero_mask.any():
            # Find first and last non-zero indices for this row
            non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
            first_nonzero_idx = non_zero_indices[0].item()
            last_nonzero_idx = non_zero_indices[-1].item()
            
            # Update global minimum and maximum
            global_min_nonzero_idx = min(global_min_nonzero_idx, first_nonzero_idx)
            global_max_nonzero_idx = max(global_max_nonzero_idx, last_nonzero_idx)

    # Step 2: For each batch, fill indices before global_min_nonzero_idx and after global_max_nonzero_idx
    for i in range(tensor_batch.size(0)):
        row = result[i]
        non_zero_mask = row != 0
        
        if non_zero_mask.any():
            # Get the values at the global boundary positions for this row
            first_value = row[global_min_nonzero_idx]
            last_value = row[global_max_nonzero_idx]
            
            # Fill all positions before global_min_nonzero_idx with first value
            if global_min_nonzero_idx > 0:
                result[i, :global_min_nonzero_idx] = first_value
            
            # Fill all positions after global_max_nonzero_idx with last value
            if global_max_nonzero_idx < tensor_batch.size(1) - 1:
                result[i, global_max_nonzero_idx + 1:] = last_value
                
        elif fill_value is not None:
            # Handle all-zero rows - fill entire row with fill_value
            result[i].fill_(fill_value)
    
    # Step 3: Handle global zero intervals between non-zero values
    # Find all global non-zero positions across all rows
    global_non_zero_positions = set()
    for i in range(tensor_batch.size(0)):
        row = tensor_batch[i]
        non_zero_mask = row != 0
        if non_zero_mask.any():
            non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
            global_non_zero_positions.update(non_zero_indices.tolist())
    
    # Convert to sorted list to find global gaps
    if global_non_zero_positions:
        global_non_zero_positions = sorted(list(global_non_zero_positions))
        
        # Find global gaps between consecutive non-zero positions
        for j in range(len(global_non_zero_positions) - 1):
            left_global_idx = global_non_zero_positions[j]
            right_global_idx = global_non_zero_positions[j + 1]
            
            # Check if there's a global gap
            if right_global_idx - left_global_idx > 1:
                gap_start = left_global_idx + 1
                gap_end = right_global_idx - 1
                gap_length = gap_end - gap_start + 1
                
                # Split the global gap
                split_point = gap_start + gap_length // 2
                
                # Apply the split to each row that has this gap
                for i in range(tensor_batch.size(0)):
                    row = result[i]
                    
                    # Check if this row has zeros in the global gap region
                    gap_region = row[gap_start:gap_end + 1]
                    if (gap_region == 0).any():
                        # Get left and right values for this row
                        left_value = row[left_global_idx] if row[left_global_idx] != 0 else 0
                        right_value = row[right_global_idx] if row[right_global_idx] != 0 else 0
                        
                        # Only fill if we have valid boundary values
                        if left_value != 0 or right_value != 0:
                            # Fill bottom half with left value (if available)
                            if left_value != 0:
                                result[i, gap_start:split_point] = torch.where(
                                    result[i, gap_start:split_point] == 0, 
                                    left_value, 
                                    result[i, gap_start:split_point]
                                )
                            
                            # Fill top half with right value (if available)
                            if right_value != 0:
                                result[i, split_point:gap_end + 1] = torch.where(
                                    result[i, split_point:gap_end + 1] == 0, 
                                    right_value, 
                                    result[i, split_point:gap_end + 1]
                                )
    
    return result

