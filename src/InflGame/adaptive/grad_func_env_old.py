"""

.. module:: grad_func_env
   :synopsis: A module for handling adaptive dynamics for agents interacting in influence games.


Adaptive Environment Module
===========================

This module defines the `AdaptiveEnv` class, which represents an adaptive environment for agents interacting with a resource distribution via influence kernels.
The class models the competition of agents via their influence over the environment and computes gradients for optimization. 
It provides methods to compute influence, reward, and the gradients of rewards of each agent.

The module supports different types of influence kernels, including:
- Gaussian
- Jones
- Dirichlet
- Multi-variate Gaussian
- Custom influence kernels (user-defined)

The `AdaptiveEnv` class supports gradient ascent methods for optimizing agent positions in the environment.

Dependencies:
-------------
- InflGame.utils
- InflGame.kernels
- InflGame.domains


Usage:
------
The `AdaptiveEnv` class can be used to simulate and optimize agent interactions in an environment with resource distributions. It supports various influence kernel types and gradient ascent methods for optimization.

Example:
--------

.. code-block:: python

    from InflGame.adaptive.grad_func_env import AdaptiveEnv
    import torch
    import numpy as np

    # Initialize the environment
    env = AdaptiveEnv(
        num_agents=3,
        agents_pos=np.array([0.2, 0.5, 0.8]),
        parameters=torch.tensor([1.0, 1.0, 1.0]),
        resource_distribution=torch.tensor([10.0, 20.0, 30.0]),
        bin_points=np.array([0.1, 0.4, 0.7]),
        infl_configs={'infl_type': 'gaussian'},
        learning_rate_type='cosine',
        learning_rate=[0.0001, 0.01, 15],
        time_steps=100
        domain_type='1d',
        domain_bounds=[0, 1]
    )

    # Perform gradient ascent
    env.gradient_ascent(show_out=True)
"""


import numpy as np
import torch
from typing import Union, List, Dict, Optional, Tuple


import InflGame.utils.general as general
import InflGame.kernels.gauss as gauss
import InflGame.kernels.jones as jones
import InflGame.kernels.diric as diric
import InflGame.kernels.MV_gauss as MV_gauss

import InflGame.domains.simplex.simplex_utils as simplex_utils

class AdaptiveEnv:
    """
    The AdaptiveEnv class represents an adaptive environment for agents interacting with a resource distribution via influence kernels.
    This class models the competition of agents via their influence over the environment and computes gradients for optimization. 
    The class provides methods to compute influence, reward, and gradients based on the influence of agents on resource distribution.
    It also supports different types of influence kernels, including Gaussian, Jones, Dirichlet, and custom influence kernels.
    """

    def __init__(self,
                 num_agents: int,
                 agents_pos: Union[List[float], np.ndarray],
                 parameters: torch.Tensor,
                 resource_distribution: torch.Tensor,
                 bin_points: Union[List[float], np.ndarray],
                 infl_configs: Dict[str, Union[str, callable]] = {'infl_type': 'gaussian'},
                 learning_rate_type: str = 'cosine',
                 learning_rate: List[float] = [.0001, .01, 15],
                 time_steps: int = 100,
                 fp: Optional[int] = 0,
                 infl_cshift: bool = False,
                 cshift: int = 0,
                 infl_fshift: bool = False,
                 Q: int = 0,
                 domain_type: str = '1d',
                 domain_bounds: Union[List[float], torch.Tensor] = [0, 1],
                 tolerance: float = 10**-5,
                 tolerated_agents: Optional[int] = None,
                 ) -> None:
        """
        Initialize the AdaptiveEnv class.

        :param num_agents: Number of agents in the environment.
        :type num_agents: int
        :param agents_pos: Initial positions of the agents.
        :type agents_pos: Union[List[float], np.ndarray]
        :param parameters: Parameters for influence kernels of each agent.
        :type parameters: torch.Tensor
        :param resource_distribution: Resource distribution in the environment.
        :type resource_distribution: torch.Tensor
        :param bin_points: Points in the domain where resources are distributed.
        :type bin_points: Union[List[float], np.ndarray]
        :param infl_configs: Configuration for influence kernel type and custom influence kernels.
        :type infl_configs: Dict[str, Union[str, callable]]
        :param learning_rate_type: Type of learning rate schedule.
        :type learning_rate_type: str
        :param learning_rate: Learning rate parameters.
        :type learning_rate: List[float]
        :param time_steps: Number of time steps for gradient ascent.
        :type time_steps: int
        :param fp: Fixed parameter for the Dirichlet kernel type.
        :type fp: Optional[int]
        :param infl_cshift: Whether to apply constant shift in influence.
        :type infl_cshift: bool
        :param cshift: Value of the constant shift.
        :type cshift: int
        :param infl_fshift: Whether to apply functional shift in influence.
        :type infl_fshift: bool
        :param Q: Scaling factor for functional shift.
        :type Q: int
        :param domain_type: Type of domain ('1d', '2d', or 'simplex').
        :type domain_type: str
        :param domain_bounds: Bounds of the domain.
        :type domain_bounds: Union[List[float], torch.Tensor]
        :param tolerance: Tolerance for convergence.
        :type tolerance: float
        :param tolerated_agents: Number of agents that need to meet tolerance before breaking.
        :type tolerated_agents: Optional[int]
        """
        
        self.num_agents=num_agents
        self.agents_pos = agents_pos
        self.infl_type=infl_configs['infl_type']
        self.infl_configs=infl_configs
        self.parameters=parameters
        self.resource_distribution=resource_distribution
        self.bin_points=bin_points
        self.learning_rate=learning_rate
        self.time_steps=time_steps
        self.fixed_pa=fp
        self.learning_rate_type=learning_rate_type
        self.infl_cshift=infl_cshift
        self.cshift=cshift
        self.infl_fshift=infl_fshift
        self.Q=Q
        self.domain_type=domain_type
        self.domain_bounds=domain_bounds
        self.sigma_inv=0
        self.tolerance=tolerance
        if tolerated_agents == None:
            tolerated_agents = num_agents
        else:
            self.tolerated_agents=tolerated_agents
        

    def influence(self,
                  agent_id: int,
                  parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                  alpha_matrix: torch.Tensor = 0,
                  ) -> torch.Tensor:
        r"""
        Compute the influence of a specific agent's influence kernel over the bin points.
        
        i.e.

        .. math::
           f_{i}(x_i,b)

        Where :math:`x_i` is the position of the :math:`i` th agent and :math:`b \in \mathbb{B}` is the resource/bin points in the environment.

        There are several types of preset influence kernels, including:

        - **Gaussian influence kernel**

          .. math::
             f_i(x_i,b,\sigma) = e^{-\frac {(x_i-b)^2}{2\sigma^2}}

        - **Jones influence kernel**

          .. math::
             f_i(x_i,b,p) = |x-b|^p

        - **Dirichlet influence kernel**

          .. math::
             f_i(\mathbb{\alpha},b)=\frac{1}{\beta(\alpha)}\prod_{l=1}^{L} b_l^{(\alpha_l-1)}

        where :math:`L` is the number of dimensions and :math:`b_l` is the :math:`l` th component of the bin point :math:`b`.
        
        Here :math:`\mathbf{\alpha}` is the parameter vector for the Dirichlet influence kernel, but :math:`\alpha_\phi` is the fixed parameter such that

          .. math::
             \alpha_l=\frac{\alpha_\phi}{x_{(i,\phi)}}*x_{(i,l)}

        where :math:`x_{(i,\phi)}` is the :math:`\phi` th component of the position of the :math:`i` th agent and :math:`x_{(i,l)}` is the the :math:`l` th component of the position of the :math:`i` th agent.

        - **Multi-variate Gaussian influence kernel**

          .. math::
            f_i(\mathbf{x}_i,\mathbf{b},\Sigma) = e^{-\frac{(\mathbf{x}_i-\mathbf{b})^T \Sigma^{-1} (\mathbf{x}_i-\mathbf{b})}{2}}

        where :math:`\Sigma` is the covariance matrix of the multi-variate Gaussian influence kernel.

        - **Custom influence kernel (user-defined)**

        This influence kernel is defined by the user and can be any function that takes in the agent's position, bin points, and parameters.
        Examples of custom influence kernels are provided in the demos.

        :param agent_id: ID of the agent.
        :type agent_id: int
        :param parameter_instance: Parameters for influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param alpha_matrix: Alpha matrix for Dirichlet influence.
        :type alpha_matrix: torch.Tensor
        :return: Influence values for the agent.
        :rtype: torch.Tensor
        """

        if self.infl_cshift==True and agent_id==self.num_agents:
            infl=torch.tensor(self.cshift)
        elif self.infl_fshift==True and agent_id>=self.num_agents:
            #This part determines if we are shifting our influence matrix by a custom function (right now just takes the abstaining function)
            infl=[]
            if len(self.agents_pos.shape)>1:
                print('Not done yet')
            else:
                for bin_point in self.bin_points:
                   infl_instance=1
                   for pos in self.agents_pos:
                        infl_instance=infl_instance*(bin_point-pos)**2
                   infl_instance=self.Q*infl_instance
                   infl.append(infl_instance)
                infl=torch.tensor(infl)
        else:
            if self.infl_type=='gaussian':
                infl=gauss.influence(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)
            
            elif self.infl_type=='Jones_M':
                infl=jones.influence(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='dirichlet':
                infl=diric.influence(agent_id=agent_id,bin_points=self.bin_points,alpha_matrix=alpha_matrix)

            elif self.infl_type=='multi_gaussian':
                self.sigma_inv=MV_gauss.cov_matrix(parameter_instance=parameter_instance)

                infl=MV_gauss.influence(agent_id=agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,sigma_inv=self.sigma_inv)
            elif self.infl_type=='custom':
                custom_influence=self.infl_configs['custom_influence']
                x_torch=torch.tensor(self.agents_pos[agent_id])
                p=np.array([parameter_instance[agent_id]])
                infl=custom_influence(x_torch,bin_points=self.bin_points,parameter_instance=p[0])
            else:
                print('no method selected!') 
        return infl
    
    def influence_matrix(self,
                         parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                         ) -> torch.Tensor:
        r"""
        Computes the influence matrix for all agents in the environment. The influence matrix is a :math:`N \times K` matrix where :math:`N` is the number of agents and
        K is the number of bin/resource points. The entry :math:`\iota_{i,k}` is a the influence of the :math:`i` th agent on the :math:`k` th bin/resource point.
        i.e.

          .. math:: 
             \begin{bmatrix}
             \iota_{1,1} & \iota_{1,2} & \cdots & \iota_{1,K} \\
             \iota_{2,1} & \iota_{2,2} & \cdots & \iota_{2,K} \\
             \vdots & \vdots & \ddots & \vdots \\
             \iota_{N,1} & \iota_{N,2} & \cdots & \iota_{N,K}
             \end{bmatrix}
        
        where :math:`\iota_{i,k}=f_i(x_i,b_k)`. 

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Influence matrix.
        :rtype: torch.Tensor
        """


        
        #For dirichlet influence only
        if self.infl_type=='dirichlet':
            alpha_matrix=diric.param(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.agents_pos,fixed_pa=self.fixed_pa)
        else:
            alpha_matrix=0
        self.alpha_matrix=alpha_matrix

        #For types of shifts
        agents=self.num_agents
        if self.infl_cshift==True:
            agents=agents+1
        if self.infl_fshift==True:
            agents=agents+1
        
        #Assembling the influence matrix for all agents
        infl_matrix=0
        for agent_id in range(agents):
            infl_row=self.influence(agent_id,parameter_instance,alpha_matrix)

            infl_matrix=general.matrix_builder(row_id=agent_id,row=infl_row,matrix=infl_matrix)
            
        return infl_matrix
    

    def prob_matrix(self,
                    parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                    ) -> torch.Tensor:
        r"""
        Computes the probability matrix for agents influencing a resource based on their influence kernel :math:'f_{i}(x_i,b_k)` computed by :func:`influence` . 
        Where the probability of agent :math:`i` influencing a bin/resource point is defined as 

            .. math::
                G_{i,k}(\mathbf{x},b_k)=\frac{f_{i}(x_i,b_k)}{\sum_{j=1}^{N}f_{j}(x_j,b_k)}.

        The probability matrix is a :math:`N \times K` matrix where :math:`N` is the number of agents and K is the number of bin/resource points. 
        The entry :math:`G_{i,k}` is a the probability of the :math:`i` th agent on the :math:`k` th bin/resource point.
        i.e.

            .. math::
                \begin{bmatrix}
                G_{1,1} & G_{1,2} & \cdots & G_{1,K} \\
                G_{2,1} & G_{2,2} & \cdots & G_{2,K} \\
                \vdots & \vdots & \ddots & \vdots \\
                G_{N,1} & G_{N,2} & \cdots & G_{N,K}
                \end{bmatrix}

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Probability matrix.
        :rtype: torch.Tensor
        """


        infl_matrix=self.influence_matrix(parameter_instance)
        denom=torch.sum(infl_matrix, 0)
        numer=infl_matrix
        agent_prob_matrix=numer/denom
        
        return agent_prob_matrix
    
    def reward_F(self,
                 parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                 ) -> Union[int, torch.Tensor]:
        r"""
        Compute the expected reward for each agent given a reward distribution and all agents influence kernels. The probability of an agent influencing a point is their relative influence over the bin points.
        
        Then reward is computed as matrix multiplication of the probability matrix and the resource distribution.
        i.e.

            .. math::
                u_i&=\sum_{b\in \mathbb{B}}^{K}B(b) G_{i}(x_i,b))\\
                &=\sum_{k=1}^{K}B_k G_{i,k}(x_i,b_k)\\
                &=\begin{bmatrix}
                G_{1,1} & G_{1,2} & \cdots & G_{1,K} \\
                G_{2,1} & G_{2,2} & \cdots & G_{2,K} \\
                \vdots & \vdots & \ddots & \vdots \\
                G_{N,1} & G_{N,2} & \cdots & G_{N,K}
                \end{bmatrix}
                \begin{bmatrix}
                B_1 \\
                B_2 \\
                \vdots \\
                B_K
                \end{bmatrix}
                
            
        if :math:`\mathbb{B}=\set{b_1,b_2,\cdots,b_K}` is the set of bin points and :math:`B_k=B(b_k)` is the resource at bin point :math:`b_k`.
        
        The probability matrix is computed by the function :func:`prob_matrix` . 

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Reward values for agents.
        :rtype: Union[int, torch.Tensor]
        """


        pr_matrix=self.prob_matrix(parameter_instance)
        reward=torch.sum(pr_matrix*torch.tensor(self.resource_distribution),1)
        return reward


    def d_lnf_matrix(self,
                     parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                     ) -> Union[int, torch.Tensor]:
        r"""
        Computes the derivative of the log of the influence function matrix , i.e. 

            .. math::
                \frac{\partial}{\partial x_{(i,l)}}ln(f_{i}(x_i,b))=\frac{1}{f_{i}(x_{i},b)}\frac{\partial}{\partial x_{(i,l)}}f_{i}(x_{i},b)
        
        The derivative matrix is a :math:`N \times K` matrix where :math:`N` is the number of agents and K is the number of bin/resource points.
        The entry :math:`\frac{\partial}{\partial x_i}ln(f_{i}(x_i,b_k))` is a the derivative of the log of the influence of the :math:`i` th agent on the :math:`k` th bin/resource point.
        i.e.

            .. math::
                \mathbf{D}=\begin{bmatrix}
                \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_1)) & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_2)) & \cdots & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_K)) \\
                \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_1)) & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_2)) & \cdots & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_K)) \\
                \vdots & \vdots & \ddots & \vdots \\
                \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_1)) & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_2)) & \cdots & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_K))
                \end{bmatrix}

        This function **only** used for the **prebuilt** influence kernels from the paper in :func:`influence` where the derivatives are analytically computed.
        -**Gaussian influence kernel** 
            
            (infl_type=='gaussian')
        
        - **Jones influence kernel** 
            
            (infl_type=='Jones_M')
        
        - **Dirichlet influence kernel**  
            
            (infl_type=='dirichlet')
        
        - **Multi-variate Gaussian influence kernel** 
            
            (infl_type=='multi_gaussian')
        
        **For custom influence kernels** use :func:`d_torch`. This is automatically done if infl_type==custom_influence by the adaptive_env class.

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Derivative matrix.
        :rtype: Union[int, torch.Tensor]
        """

        d_matrix=0
        for agent_id in range(self.num_agents):
            if self.infl_type=='gaussian':
                d_row=gauss.d_ln_f(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='Jones_M':
                d_row=jones.d_ln_f(agent_id=agent_id,parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

            elif self.infl_type=='dirichlet':

                self.alpha_matrix=diric.param(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.agents_pos,fixed_pa=self.fixed_pa)

                d_row=diric.d_ln_f(agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,alpha_matrix=self.alpha_matrix,fixed_pa=self.fixed_pa)
            
            elif self.infl_type=='multi_gaussian':
                self.sigma_inv=MV_gauss.cov_matrix(parameter_instance=parameter_instance)
                
                d_row=MV_gauss.d_ln_f(agent_id=agent_id,agents_pos=self.agents_pos,bin_points=self.bin_points,sigma_inv=self.sigma_inv)
                

            d_matrix=general.matrix_builder(agent_id,d_row,d_matrix)
        
        return d_matrix 
    

    def shift_matrix(self,
                     parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                     ) -> torch.Tensor:
        r"""
        Compute the shift matrix for functional shifts in influence kerenels. This function is mostly used for abstaining voters and fixed party examples, but can also be used for 
        demonstrating how types of non-symmetry can impact agent behavior. 
        The shift matrix is a :math:`N \times K` matrix where :math:`N` is the number of agents and K is the number of bin/resource points.
        The entry :math:`s_{i,k}` is a the shift of the :math:`i` th agent on the :math:`k` th bin/resource point.
        i.e.
           
            .. math::
                \begin{bmatrix}
                s_{1,1} & s_{1,2} & \cdots & s_{1,K} \\
                s_{2,1} & s_{2,2} & \cdots & s_{2,K} \\
                \vdots & \vdots & \ddots & \vdots \\
                s_{N,1} & s_{N,2} & \cdots & s_{N,K}
                \end{bmatrix}
        
        there are different types of shifts that can be applied to the influence matrix. 
        
        - **Constant shift** (infl_cshift=True)
            
            .. math::
                s_{i,k}=cshift
        
        - **Functional shift** (infl_fshift=True)'
            An example of a functional shift is the abstaining voter model where the shift is defined as:
            
            .. math::
                s_{i,k}=-2Q\prod_{\substack{j=1\\ j\neq i}}^{N} (b_k-x_j)^2(b_k-x_i)
            
            i.e. the influence of an abstaining voter on point :math:`b_k` is

            .. math::
                s_{i}(x_i,b_k)=\prod_{i=1}^{N} Q(b_k-x_j)^2

            where :math:`x_i` is the position of the :math:`i` th agent and :math:`b_k` is the :math:`k` th bin point.
            :math:`Q` is a scaling factor for the functional shift.
        

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Shift matrix.
        :rtype: torch.Tensor
        """


        infl_matrix=self.influence_matrix(parameter_instance)
        denom=torch.sum(infl_matrix, 0)
        shift_matrix=0
        for agent_id in range(self.num_agents):
            shift_row=[]
            if agent_id==0:
                for bin_point in self.bin_points:
                    shift_instance=1
                    for pos in self.agents_pos[1:]:
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)
            elif agent_id==self.num_agents-1:
                for bin_point in self.bin_points:
                    shift_instance=1
                    for pos in self.agents_pos[:-1]:
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)  
            else:
                for bin_point in self.bin_points: 
                    shift_instance=1
                    for pos in np.concatenate((self.agents_pos[:agent_id],self.agents_pos[agent_id+1:]), axis=0):
                        shift_instance=shift_instance*(bin_point-pos)**2
                    shift_instance=-2*self.Q*shift_instance*(bin_point-self.agents_pos[agent_id])
                    shift_row.append(shift_instance)
            shift_row=torch.tensor(shift_row)
            shift_matrix=general.matrix_builder(row_id=agent_id,row=shift_row,matrix=shift_matrix)

        shift_matrix=shift_matrix/denom
        return shift_matrix
    
    def d_torch(self,
                parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                ) -> torch.Tensor:
        r"""
        Compute the gradient of the custom influence matrix using PyTorch autograd i.e.
            
            .. math::
                    \frac{\partial}{\partial x_{(i,l)}}ln(f_{i}(x_i,b))=\frac{1}{f_{i}(x_{i},b)}\frac{\partial}{\partial x_{(i,l)}}f_{i}(x_{i},b)
        
        if you using the infl_type='custom' influence kernel. This is done using PyTorch's autograd functionality, so number of bin points must be larger enough to compute the gradient ( :math:`K\sim 100` ).

        If you are using a non-custom influence kernel, use :func:`d_lnf_matrix` instead, this is automatically done by the adaptive_env class.

        The derivative matrix is a :math:`N \times K` matrix where :math:`N` is the number of agents and K is the number of bin/resource points.
        The entry :math:`\frac{\partial}{\partial x_i}ln(f_{i}(x_i,b_k))` is a the gradient of the log of the influence of the :math:`i` th agent on the :math:`k` th bin/resource point.
        i.e.
            
            .. math::
                \mathbf{D}=\begin{bmatrix}
                \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_1)) & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_2)) & \cdots & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_K)) \\
                \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_1)) & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_2)) & \cdots & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_K)) \\
                \vdots & \vdots & \ddots & \vdots \\
                \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_1)) & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_2)) & \cdots & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_K))
                \end{bmatrix}
        

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: derivative matrix.
        :rtype: torch.Tensor
        """
        d_matrix=0
        if self.domain_type=='1d':
            for agent_id in range(self.num_agents):
                x_torch=torch.tensor([self.agents_pos[agent_id]]*len(self.bin_points),requires_grad=True)
                external_grad = torch.tensor([1.]*len(self.bin_points))
                custom_influence=self.infl_configs['custom_influence']
                infl_row=torch.log(custom_influence(x_torch,self.bin_points,parameter_instance[agent_id]))
                infl_row.backward(gradient=external_grad)
                d_row=x_torch.grad
                d_matrix=general.matrix_builder(row_id=agent_id,row=d_row,matrix=d_matrix)
        else:
            for agent_id in range(self.num_agents):
                x_torch=torch.tensor(self.agents_pos[agent_id])
                x=x_torch.repeat(len(self.bin_points),1)
                x.requires_grad=True
                external_grad=torch.tensor([1.]*len(self.bin_points))
                custom_influence=self.infl_configs['custom_influence']
                infl_row=torch.log(custom_influence(x,self.bin_points,parameter_instance[agent_id]))
                infl_row.backward(gradient=external_grad)
                d_row=[]
                for dim in range(len(x_torch)):
                    d_row.append(x.grad[:,dim])
                d_row=torch.stack(d_row)
                d_matrix=general.matrix_builder(row_id=agent_id,row=d_row,matrix=d_matrix)
                
        return d_matrix



    def gradient(self,
                 parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                 ) -> torch.Tensor:
        r"""
        Compute the gradient of the reward function :math:`u_i(x)` with respect to agent positions `x_i`.
        The gradient is computed as the elment-wise product of the derivative of the log of the influence function matrix and the probability matrix dot-producted with the resource vector
        :math:`\mathbf{B}` . 
        i.e.
            
            .. math::
                \frac{\partial}{\partial x_{(i,l)}}u_i(x)=\sum_{k=1}^{K}G_{i,k}(x_i,b_k)\frac{\partial}{\partial x_{(i,l)}}ln(f_{i}(x_i,b_k))\\
                =\left(\mathbf{G}\odot\mathbf{D}\right) \cdot \vec{B}\\
                
            .. math::
                \nabla\vec{R}=\left(\begin{bmatrix}
                G_{1,1} & G_{1,2} & \cdots & G_{1,K} \\
                G_{2,1} & G_{2,2} & \cdots & G_{2,K} \\
                \vdots & \vdots & \ddots & \vdots \\
                G_{N,1} & G_{N,2} & \cdots & G_{N,K}
                \end{bmatrix}
                \odot
                \begin{bmatrix}
                \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_1)) & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_2)) & \cdots & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_K)) \\
                \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_1)) & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_2)) & \cdots & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_K)) \\
                \vdots & \vdots & \ddots & \vdots \\
                \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_1)) & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_2)) & \cdots & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_K))
                \end{bmatrix}
                \right)\cdot 
                \begin{bmatrix}
                B_1 \\
                B_2 \\
                \vdots \\
                B_K
                \end{bmatrix}
                
        The matrix :math:`\mathbf{D}` is the derivative of the log of the influence function matrix computed by :func:`d_lnf_matrix` or :func:`d_torch` .
        The probability matrix :math:`\mathbf{G}` is computed by the function :func:`prob_matrix`.

        The output :math:`\nabla\vec{R}` is a :math:`N \times L` matrix where :math:`N` is the number of agents and :math:`L` is the number of dimensions.
        The entry :math:`\nabla\vec{R}_{i,l}` is a the gradient of the reward of the :math:`i` th agent on the :math:`l` th dimension.

        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Gradient values.
        :rtype: torch.Tensor
        """


        grad=0
        pr_matrix=self.prob_matrix(parameter_instance)
        if self.infl_type=='custom':
            d_matrix=self.d_torch(parameter_instance)
        elif self.infl_type in ['multi_gaussian','gaussian','Jones_M','dirichlet']:
            d_matrix=self.d_lnf_matrix(parameter_instance)
        pr_matrix_c=1-pr_matrix
        pr_prod=pr_matrix*pr_matrix_c
        if self.domain_type=='1d':
            if self.infl_fshift==True:
                shift_matrix=self.shift_matrix(parameter_instance)
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_matrix[a_id]*pr_matrix_c[a_id]*torch.tensor(self.resource_distribution)
                if self.infl_fshift==True:
                    agent_grad=agent_grad-shift_matrix[a_id]*pr_matrix[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad)
                grad=general.matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
        else:
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_prod[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad,1)
                grad=general.matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
            
    
        return grad
    
    def mv_gradient_ascent(self,
                           show_out: bool = False,
                           grad_modify: bool = False,
                           reward: bool = True,
                           ) -> torch.Tensor:
        r"""
        Perform multi-variable gradient ascent for agents in the environment using the graident calculated by the function :func:`gradient`.


        The gradient ascent is performed by updating the agent positions based on the gradient and a learning rate. The learning rate is scheduled using the function :func:`InflGame.utils.general.learning_rate`.
        The alogrithm is performed for a fixed number of time steps or until the agents converge to a solution within a specified tolerance for the absoulte difference between the current and previous agent positions.
        The gradient ascent is performed in the following steps:

        1. Compute the gradient of the reward function using the function :func:`gradient`.
        2. Normalize the gradient if the domain type is 'simplex'.
            
            - For simplex, the gradient is normalized to ensure that the agent positions remain within the simplex.
        
        3. Update the agent positions using the gradient and the learning rate.
            
            - The learning rate is computed using the function :func:`InflGame.utils.general.learning_rate`.
            - The agent positions are updated by adding the gradient multiplied by the learning rate to the current agent positions.
            - The updated agent positions are projected onto the simplex if the domain type is 'simplex'.
        
        4. Store the agent positions, gradients, and rewards at each time step.
        5. Check for convergence by computing the absolute difference between the current and previous agent positions.
        6. If the absolute difference is less than the specified tolerance (var:tolerance) for a set number of agents (var:tolarated_agents), break the loop.
        
        i.e. a time step looks liek this:
            
            .. math::
                \mathbf{x}_{i;t+1}=\mathbf{x}_{i;t}+\eta_t\cdot\nabla\vec{R}_{i;t}\\
            
        with the stop condition:
            
            .. math::
                \sum_{i=1}^{N}||\mathbf{x}_{i;t+1}-\mathbf{x}_{i;t}||_1\leq \epsilon = E\\
            
        where :math:`\epsilon` is the tolerance and :math:`E` is the tolerated agents.
        The learning rate :math:`\eta_t` is computed using the function :func:`InflGame.utils.general.learning_rate`.

            If the domain type is 'simplex', the agent positions are projected onto the simplex so the update step looks like this:
                
                .. math::
                    \mathbf{x}_{i;t+1}=\mathbf{P}_{\Delta}(\mathbf{x}_{i;t}+\eta_t\cdot normalized(\nabla \vec{R}_{i;t}))\\
            
            using the function :func:`InflGame.domains.simplex.simplex_utils.projection_onto_simplex`.

            Due to the normalization of the gradient, the stoping condtions is slightly different:
                
                .. math::
                    \sum_{i=1}^{N}||\mathbf{x}_{i;t+5}-\mathbf{x}_{i;t}||_1\leq \epsilon = E\\

        
        :param show_out: Whether to return intermediate outputs.
        :type show_out: bool
        :param grad_modify: Whether to modify gradients.
        :type grad_modify: bool
        :param reward: Whether to compute rewards.
        :type reward: bool
        :return: Gradient ascent results.
        :rtype: torch.Tensor
        """

        self.grad_modify=grad_modify
        reward_vec=0
        pos_vec=0
        grad_vec=0
        agents_og=self.agents_pos
        
        self.agents_pos=agents_og.copy()
        reward_vec=0
        pos_vec=0
        grad_vec=0
        for time in range(self.time_steps):
            grad_vec_row=self.gradient(self.parameters)
            if self.domain_type=='simplex':
                grad=torch.nn.functional.normalize(grad_vec_row,dim=1)
            else:
                grad=grad_vec_row
            temp=torch.tensor(np.array(self.agents_pos))+general.learning_rate(iter=time,learning_rate_type=self.learning_rate_type,learning_rate=self.learning_rate)*grad
            for t_row in range(temp.size()[0]):
                if self.domain_type=='simplex':
                    temp_row=simplex_utils.projection_onto_simplex(temp[t_row])
                    if torch.all(temp_row>0):
                        self.agents_pos[t_row]=temp_row.detach().numpy()
                    else:
                        pass
                else:
                    temp_row=temp[t_row]
                    self.agents_pos[t_row]=temp_row.detach().numpy()
            pos_vec_row=torch.tensor(np.array(self.agents_pos))
            pos_vec=general.matrix_builder(row_id=time,row=pos_vec_row,matrix=pos_vec)
            grad_vec=general.matrix_builder(row_id=time,row=grad_vec_row,matrix=grad_vec)
            
            if reward==True:
                reward_vec_row=self.reward_F(self.parameters)
                reward_vec=general.matrix_builder(row_id=time,row=reward_vec_row,matrix=reward_vec)
            if time>5:
                if self.domain_type=='simplex':
                    abs_difference=torch.linalg.norm(pos_vec_row-pos_vec[-5],axis=1)
                else:
                    abs_difference=torch.linalg.norm(pos_vec_row-pos_vec[-2],axis=1)
                abs_difference_value=torch.sum(abs_difference<=self.tolerance).item()
                if abs_difference_value>=self.tolerated_agents:
                    break
        self.grad_matrix=grad_vec.clone()
        self.pos_matrix=pos_vec.clone()
        if reward==True:
            self.reward_matrix=reward_vec
        
        
        
    
    def sv_gradient_ascent(self,
                           show_out: bool = False,
                           grad_modify: bool = False,
                           reward: bool = True,
                           ) -> torch.Tensor:
        r"""
        The gradient ascent is performed by updating the agent positions based on the gradient and a learning rate. The learning rate is scheduled using the function :func:`InflGame.utils.general.learning_rate`.
        The alogrithm is performed for a fixed number of time steps or until the agents converge to a solution within a specified tolerance for the absoulte difference between the current and previous agent positions.
        The gradient ascent is performed in the following steps:

        1. Compute the gradient of the reward function using the function :func:`gradient`.
        2. Update the agent positions using the gradient and the learning rate.
            
            - The learning rate is computed using the function :func:`InflGame.utils.general.learning_rate`.
            - The agent positions are updated by adding the gradient multiplied by the learning rate to the current agent positions.
        
        4. Store the agent positions, gradients, and rewards at each time step.
        5. Check for convergence by computing the absolute difference between the current and previous agent positions.
        6. If the absolute difference is less than the specified tolerance (var:tolerance) for a set number of agents (var:tolarated_agents), break the loop.

        i.e. a time step looks liek this:
            
            .. math::
                \mathbf{x}_{i;t+1}=\mathbf{x}_{i;t}+\eta_t\cdot\nabla\vec{R}_{i;t}\\
            
        with the stop condition:
            
            .. math::
                \sum_{i=1}^{N}||\mathbf{x}_{i;t+1}-\mathbf{x}_{i;t}||_1\leq \epsilon = E\\
            
        where :math:`\epsilon` is the tolerance and :math:`E` is the tolerated agents.
        The learning rate :math:`\eta_t` is computed using the function :func:`InflGame.utils.general.learning_rate`.

        :param show_out: Whether to return intermediate outputs.
        :type show_out: bool
        :param grad_modify: Whether to modify gradients.
        :type grad_modify: bool
        :param reward: Whether to compute rewards.
        :type reward: bool
        :return: Gradient ascent results.
        :rtype: torch.Tensor
        """

        self.grad_modify=grad_modify
        reward_vec=0
        pos_vec=0
        grad_vec=0
        agents_og=self.agents_pos
        self.agents_pos=agents_og
        for time in range(self.time_steps):
            grad_vec_row=self.gradient(self.parameters)
            temp=torch.tensor(self.agents_pos)+general.learning_rate(iter=time,learning_rate_type=self.learning_rate_type,learning_rate=self.learning_rate)*grad_vec_row
            
            for t_row in range(temp.size()[0]):
                temp_row=temp[t_row]
                if torch.all(temp_row>=self.domain_bounds[0]) and torch.all(temp_row<=self.domain_bounds[1]):
                    self.agents_pos[t_row]=temp_row.detach().numpy()
                else:
                    print("passed!")
                    pass
        
            pos_vec_row=torch.tensor(self.agents_pos)       
            pos_vec=general.matrix_builder(row_id=time,row=pos_vec_row,matrix=pos_vec)
            grad_vec=general.matrix_builder(row_id=time,row=grad_vec_row,matrix=grad_vec)
            if reward==True:
                reward_vec_row=self.reward_F(self.parameters)
                reward_vec=general.matrix_builder(row_id=time,row=reward_vec_row,matrix=reward_vec)
            if time>5:
                abs_difference=torch.abs(pos_vec_row-pos_vec[-2])
                abs_difference_value=torch.sum(abs_difference<=self.tolerance).item()
                if abs_difference_value>=self.tolerated_agents:
                    break
        self.grad_matrix=grad_vec
        self.pos_matrix=pos_vec
                        

        
    def gradient_ascent(self,
                        show_out: bool = False,
                        grad_modify: bool = False,
                        reward: bool = True,
                        ) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        r"""
        This is the helper function for performing gradient ascent for agents in the environment. It calls the appropriate gradient ascent function based on the domain type.
        The gradient ascent is performed using the function :func:`sv_gradient_ascent` or :func:`mv_gradient_ascent` depending on the domain type.

        :param show_out: Whether to return intermediate outputs.
        :type show_out: bool
        :param grad_modify: Whether to modify gradients.
        :type grad_modify: bool
        :param reward: Whether to compute rewards.
        :type reward: bool
        :return: Gradient ascent results.
        :rtype: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
        """
        
        agent_og=self.agents_pos
        
        if self.domain_type=='1d':
            self.sv_gradient_ascent(show_out=show_out,grad_modify=grad_modify,reward=reward)
        else:
            self.mv_gradient_ascent(show_out=show_out,grad_modify=grad_modify,reward=reward)
            
            
        self.agents_pos=agent_og
        if show_out==True:
            if reward==True:
                return self.pos_matrix, self.grad_matrix,self.reward_matrix
            else:
                return self.pos_matrix, self.grad_matrix

    

    def gradient_function(self,
                          agents_pos: Union[List[float], np.ndarray],
                          parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                          ids: List[int] = [0, 1],
                          two_a: bool = True,
                          ) -> torch.Tensor:
        r"""
        The gradient function computes the gradient of the reward function for a given set of agent positions and parameters. 
        It is used to compute the gradient of the reward function for a specific set of agents, given a postion vector and parameters.
        The gradient is computed as the elment-wise product of the derivative of the log of the influence function matrix and the probability matrix dot-producted with the resource vector
        :math:`\mathbf{B}` .
        
        i.e.
            
            .. math::
                \frac{\partial}{\partial x_{(i,l)}}u_i(x)=\sum_{b\in \mathbb{B}}^{K}B(b)G_{i}(x_i,b)\\
                =\sum_{k=1}^{K}G_{i,k}(x_i,b_k)\frac{\partial}{\partial x_{(i,l)}}ln(f_{i}(x_i,b_k))\\
                =\left(\mathbf{G}\odot\mathbf{D}\right) \cdot \vec{B}\\
                
            .. math::

                \nabla\vec{R}=\left(\begin{bmatrix}
                G_{1,1} & G_{1,2} & \cdots & G_{1,K} \\
                G_{2,1} & G_{2,2} & \cdots & G_{2,K} \\
                \vdots & \vdots & \ddots & \vdots \\
                G_{N,1} & G_{N,2} & \cdots & G_{N,K}
                \end{bmatrix}
                \odot
                \begin{bmatrix}
                \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_1)) & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_2)) & \cdots & \frac{\partial}{\partial x_1}ln(f_{1}(x_1,b_K)) \\
                \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_1)) & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_2)) & \cdots & \frac{\partial}{\partial x_2}ln(f_{2}(x_2,b_K)) \\
                \vdots & \vdots & \ddots & \vdots \\
                \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_1)) & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_2)) & \cdots & \frac{\partial}{\partial x_N}ln(f_{N}(x_N,b_K))
                \end{bmatrix}
                \right)\cdot
                \begin{bmatrix}
                B_1 \\
                B_2 \\
                \vdots \\
                B_K
                \end{bmatrix}

        The matrix :math:`\mathbf{D}` is the derivative of the log of the influence function matrix computed by :func:`d_lnf_matrix` or :func:`d_torch` .
        The probability matrix :math:`\mathbf{G}` is computed by the function :func:`prob_matrix`.

        The output :math:`\nabla\vec{R}` is a :math:`N \times L` matrix where :math:`N` is the number of agents and :math:`L` is the number of dimensions.
        The entry :math:`\nabla\vec{R}_{i,l}` is a the gradient of the reward of the :math:`i` th agent on the :math:`l` th dimension.
        

        :param agents_pos: Positions of the agents.
        :type agents_pos: Union[List[float], np.ndarray]
        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :param ids: IDs of the agents to compute gradients for.
        :type ids: List[int]
        :param two_a: Whether to compute gradients for all agents.
        :type two_a: bool
        :return: Gradient values.
        :rtype: torch.Tensor
        """

        grad=0
        og_pos=self.agents_pos
        self.agents_pos=agents_pos
        og_alpha=self.alpha_matrix
        pr_matrix=self.prob_matrix(parameter_instance)
        d_matrix=self.d_lnf_matrix(parameter_instance)
        pr_matrix_c=1-pr_matrix
        pr_prod=pr_matrix*pr_matrix_c
        if self.domain_type=='1d':
            if self.infl_fshift==True:
                shift_matrix=self.shift_matrix(parameter_instance)
            if two_a==False:
                agents=ids
            else:
                agents=range(self.num_agents)

            for a_id in agents:
                agent_grad=d_matrix[a_id]*pr_matrix[a_id]*pr_matrix_c[a_id]*torch.tensor(self.resource_distribution)
                if self.infl_fshift==True:
                    agent_grad=agent_grad-shift_matrix[a_id]*pr_matrix[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad)
                grad=general.matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
                self.alpha_matrix=og_alpha
        else:
            for a_id in range(self.num_agents):
                agent_grad=d_matrix[a_id]*pr_prod[a_id]*torch.tensor(self.resource_distribution)
                agent_grad=torch.sum(agent_grad,1)
                grad=general.matrix_builder(row_id=a_id,row=agent_grad,matrix=grad)
                self.alpha_matrix=og_alpha
        self.agents_pos=og_pos
        return grad

