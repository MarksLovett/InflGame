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
import warnings

import InflGame.utils.general as general
import InflGame.utils.validation as validation
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
        validated = validation.validate_adaptive_config(
            num_agents=num_agents,
            agents_pos=agents_pos,
            parameters=parameters,
            resource_distribution=resource_distribution,
            bin_points=bin_points,
            infl_configs=infl_configs,
            learning_rate_type=learning_rate_type,
            learning_rate=learning_rate,
            time_steps=time_steps,
            fp=fp,
            infl_cshift=infl_cshift,
            cshift=cshift,
            infl_fshift=infl_fshift,
            Q=Q,
            domain_type=domain_type,
            domain_bounds=domain_bounds,
            tolerance=tolerance,
            tolerated_agents=tolerated_agents
        )
        self.num_agents = validated['num_agents']
        self.agents_pos = validated['agents_pos']
        self.infl_type = validated['infl_type']
        self.infl_configs = validated['infl_configs']
        self.parameters = validated['parameters']
        self.resource_distribution = validated['resource_distribution']
        self.bin_points = validated['bin_points']
        self.learning_rate = validated['learning_rate']
        self.time_steps = validated['time_steps']
        self.fp = validated['fp']
        self.learning_rate_type = validated['learning_rate_type']
        self.infl_cshift = validated['infl_cshift']
        self.cshift = validated['cshift']
        self.infl_fshift = validated['infl_fshift']
        self.Q = validated['Q']
        self.domain_type = validated['domain_type']
        self.domain_bounds = validated['domain_bounds']
        self.sigma_inv = 0
        self.tolerance = validated['tolerance']
        self.tolerated_agents = validated['tolerated_agents']
            
        

    def influence_matrix(self, parameter_instance: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute the influence matrix for all agents using vectorized operations.
        
        This function computes the influence values for all agents across all bin points,
        with optional constant and functional shifts. The function supports multiple
        influence kernel types and provides comprehensive error handling.
        
        :param parameter_instance: Parameters for the influence kernels.
        :type parameter_instance: Union[List[float], np.ndarray, torch.Tensor]
        :return: Influence matrix of shape (N, K) or (N+shifts, K) where N is number of agents,
                K is number of bin points, and shifts are additional rows for constant/functional shifts.
        :rtype: torch.Tensor
        :raises ValueError: If input parameters are invalid or incompatible.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        :raises NotImplementedError: If functional shift is requested for multi-dimensional agents.
        """
        
        try:
            # Validate parameter_instance
            if len(parameter_instance) == 0:
                raise ValueError("Parameter instance cannot be empty")

            if isinstance(parameter_instance, (list, np.ndarray)):
                    if len(parameter_instance) != self.num_agents:
                        raise ValueError(f"Parameter instance length ({len(parameter_instance)}) must match number of agents ({self.num_agents})")
            
            # Compute base influence matrix based on kernel type
            try:
                if self.infl_type == 'gaussian':
                    infl_matrix = gauss.influence_vectorized(
                        parameter_instance=parameter_instance,
                        agents_pos=self.agents_pos,
                        bin_points=self.bin_points
                    )
                elif self.infl_type == 'Jones_M':
                    infl_matrix = jones.influence_vectorized(
                        parameter_instance=parameter_instance,
                        agents_pos=self.agents_pos,
                        bin_points=self.bin_points
                    )
                elif self.infl_type == 'dirichlet':
                    # Validate Dirichlet-specific requirements
                    if not hasattr(self, 'fp'):
                        raise ValueError("Fixed parameter 'fp' is required for Dirichlet kernel")
                    
                    self.alpha_matrix = diric.param(
                        num_agents=self.num_agents,
                        parameter_instance=parameter_instance,
                        agents_pos=self.agents_pos,
                        fixed_pa=self.fp
                    )
                    infl_matrix = diric.influence_vectorized(
                        agents_pos=self.agents_pos,
                        bin_points=self.bin_points,
                        alpha_matrix=self.alpha_matrix,
                        fixed_pa=self.fp
                    )
                elif self.infl_type == 'multi_gaussian':
                    self.sigma_inv = MV_gauss.cov_matrix(parameter_instance=parameter_instance)
                    infl_matrix = MV_gauss.influence_vectorized(
                        agents_pos=self.agents_pos,
                        bin_points=self.bin_points,
                        sigma_inv=self.sigma_inv
                    )
                elif self.infl_type == 'custom':
                    # Validate custom influence configuration
                    if 'custom_influence' not in self.infl_configs:
                        raise ValueError("Custom influence function not provided in infl_configs")
                    
                    custom_influence = self.infl_configs['custom_influence']
                    if not callable(custom_influence):
                        raise TypeError("Custom influence must be a callable function")
                    
                    x_torch = torch.tensor(self.agents_pos, dtype=torch.float32)
                    bin_points_tensor = torch.tensor(self.bin_points, dtype=torch.float32)
                    
                    try:
                        infl_matrix = custom_influence(
                            x_torch,
                            bin_points=bin_points_tensor,
                            parameter_instance=parameter_instance
                        )
                    except Exception as e:
                        raise RuntimeError(f"Custom influence function failed: {str(e)}") from e
                    
            except Exception as e:
                if isinstance(e, (ValueError, RuntimeError, TypeError)):
                    raise
                else:
                    raise RuntimeError(f"Failed to compute base influence matrix: {str(e)}") from e
            
            # Validate base influence matrix
            if infl_matrix is None:
                raise RuntimeError("Base influence matrix computation returned None")
            
            if infl_matrix.shape[0] != self.num_agents:
                raise ValueError(f"Influence matrix rows ({infl_matrix.shape[0]}) must match number of agents ({self.num_agents})")
            
            if infl_matrix.shape[1] != len(self.bin_points):
                raise ValueError(f"Influence matrix columns ({infl_matrix.shape[1]}) must match bin points length ({len(self.bin_points)})")
            
            # Check for non-positive influence values
            if torch.any(infl_matrix <= 0):
                warnings.warn("Non-positive values detected in base influence matrix this may result in unpredictable behavior", UserWarning)

            # Add constant shift if enabled
            if self.infl_cshift:
                try: 
                    # Vectorized constant shift creation
                    cshift_row = torch.full((1, len(self.bin_points)), float(self.cshift), dtype=torch.float32)
                    infl_matrix = torch.cat([infl_matrix, cshift_row], dim=0)
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to add constant shift: {str(e)}") from e
            
            # Add functional shift if enabled (vectorized where possible)
            if self.infl_fshift:
                try:

                    # Vectorized functional shift computation
                    bin_points_tensor = torch.tensor(self.bin_points, dtype=torch.float32)
                    agents_pos_tensor = torch.tensor(self.agents_pos, dtype=torch.float32)
                    
                    # Broadcasting for vectorized computation
                    # bin_points: (1, K), agents_pos: (N, 1) -> diff: (N, K)
                    bin_points_expanded = bin_points_tensor.unsqueeze(0)  # Shape: (1, K)
                    agents_pos_expanded = agents_pos_tensor.unsqueeze(1)  # Shape: (N, 1)
                    
                    # Compute (bin_point - agent_pos)^2 for all combinations
                    diff_squared = (bin_points_expanded - agents_pos_expanded) ** 2  # Shape: (N, K)
                    
                    # Product across all agents for each bin point
                    fshift_values = torch.prod(diff_squared, dim=0) * self.Q  # Shape: (K,)
                    
                    # Add as new row
                    fshift_row = fshift_values.unsqueeze(0)  # Shape: (1, K)
                    infl_matrix = torch.cat([infl_matrix, fshift_row], dim=0)
                    
                except Exception as e:
                    if isinstance(e, NotImplementedError):
                        raise
                    else:
                        raise RuntimeError(f"Failed to add functional shift: {str(e)}") from e
            
            # Final validation and numerical checks
            if torch.any(torch.isnan(infl_matrix)):
                raise RuntimeError("NaN values detected in final influence matrix")
            
            if torch.any(torch.isinf(infl_matrix)):
                raise RuntimeError("Infinite values detected in final influence matrix")
            
            # Ensure consistent data type
            if infl_matrix.dtype != torch.float32:
                infl_matrix = infl_matrix.to(torch.float32)
            
            return infl_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError, NotImplementedError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in influence matrix computation: {str(e)}") from e

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
        :raises ValueError: If input dimensions are incompatible or contain invalid values.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        """
        
        try:
            # Get influence matrix with error handling
            try:
                infl_matrix = self.influence_matrix(parameter_instance)
            except Exception as e:
                raise RuntimeError(f"Failed to compute influence matrix: {str(e)}") from e
            
            
            # Vectorized probability computation with numerical stability
            # Compute column sums (denominator for each bin point)
            denom = torch.sum(infl_matrix, dim=0, keepdim=False)  # Shape: (K,)
            
            # Check for zero denominators (no influence at some bin points)
            zero_denom_mask = (denom == 0)
            if torch.any(zero_denom_mask):
                zero_bins = torch.where(zero_denom_mask)[0]
                raise RuntimeError(f"Zero total influence detected at bin points: {zero_bins.tolist()}. "
                                 f"This indicates that no agents have influence at these locations.")
            
            # Check for very small denominators (potential numerical instability)
            small_denom_threshold = 1e-12
            small_denom_mask = (denom < small_denom_threshold) & (denom > 0)
            if torch.any(small_denom_mask):
                small_bins = torch.where(small_denom_mask)[0]
                warnings.warn(f"Very small total influence detected at bin points {small_bins.tolist()}. "
                             f"This may lead to numerical instability.", UserWarning)
            
            # Efficient vectorized division with broadcasting
            # infl_matrix: (N, K), denom: (K,) -> result: (N, K)
            agent_prob_matrix = infl_matrix / denom.unsqueeze(0)  # Broadcasting: (N, K) / (1, K)
            
            
            # Ensure consistent data type
            if agent_prob_matrix.dtype != torch.float32:
                agent_prob_matrix = agent_prob_matrix.to(torch.float32)
            
            return agent_prob_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in probability matrix computation: {str(e)}") from e
    
    def reward_F(self,
                 parameter_instance: Union[List[float], np.ndarray, torch.Tensor],
                 ) -> torch.Tensor:
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
        :rtype: torch.Tensor
        :raises ValueError: If input dimensions are incompatible or contain invalid values.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        """
        
        try:
            # Convert resource distribution to tensor for efficient computation
            if not isinstance(self.resource_distribution, torch.Tensor):
                resource_tensor = torch.tensor(self.resource_distribution, dtype=torch.float32)
            else:
                resource_tensor = self.resource_distribution.clone().detach()
                if resource_tensor.dtype != torch.float32:
                    resource_tensor = resource_tensor.to(torch.float32)
            
            # Get probability matrix with error handling
            try:
                pr_matrix = self.prob_matrix(parameter_instance)
            except Exception as e:
                raise RuntimeError(f"Failed to compute probability matrix: {str(e)}") from e
            
            # Validate probability matrix
            if pr_matrix is None:
                raise RuntimeError("Probability matrix computation returned None")
            
            # Vectorized reward computation using optimized matrix multiplication
            # This is equivalent to: reward[i] = sum(pr_matrix[i, k] * resource_tensor[k] for k in range(K))
            # but much more efficient using tensor operations
            reward = torch.mv(pr_matrix, resource_tensor)  # Matrix-vector multiplication: (N, K) @ (K,) = (N,)
            
            # Validate output
            if torch.any(torch.isnan(reward)):
                raise RuntimeError("NaN values detected in computed rewards")
            
            if torch.any(torch.isinf(reward)):
                raise RuntimeError("Infinite values detected in computed rewards")
            
            # Ensure reward tensor has correct shape
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
            
            return reward
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in reward computation: {str(e)}") from e

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

        if self.infl_type=='gaussian':
            d_matirx=gauss.d_ln_f_vectorized(parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)
        elif self.infl_type=='Jones_M':
            d_matrix=jones.d_ln_f_vectorized(parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)
        elif self.infl_type=='dirichlet':
            self.alpha_matrix=diric.param(num_agents=self.num_agents,parameter_instance=parameter_instance,agents_pos=self.agents_pos,fixed_pa=self.fp)
            d_matrix=diric.d_ln_f_vectorized(agents_pos=self.agents_pos,bin_points=self.bin_points,alpha_matrix=self.alpha_matrix,fixed_pa=self.fp)
        elif self.infl_type=='multi_gaussian':
            d_matrix=MV_gauss.d_ln_f_vectorized(parameter_instance=parameter_instance,agents_pos=self.agents_pos,bin_points=self.bin_points)

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
        :raises ValueError: If input dimensions are incompatible or invalid.
        :raises RuntimeError: If computation fails due to numerical issues.
        """
        
        try:
            # Get influence matrix and compute denominator
            infl_matrix = self.influence_matrix(parameter_instance)
            denom = torch.sum(infl_matrix, dim=0)
            
            # Check for zero denominator
            if torch.any(denom == 0):
                raise RuntimeError("Zero denominator detected in influence matrix normalization")
            
            # Vectorized computation of shift matrix
            # Shape: agents_pos_tensor (N,), bin_points_tensor (K,)
            # We want to compute for each agent i and bin k:
            # s_{i,k} = -2Q * (product of (b_k - x_j)^2 for all j != i) * (b_k - x_i)
            
            num_bins = len(self.bin_points)
            
            # Expand dimensions for broadcasting
            # agents_pos: (N, 1), bin_points: (1, K)
            agents_expanded = self.agents_pos.unsqueeze(1)  # Shape: (N, 1)
            bins_expanded = self.bin_points.unsqueeze(0)    # Shape: (1, K)
            
            # Compute (b_k - x_j)^2 for all agents and bins
            # Shape: (N, K) where element [i, k] = (b_k - x_i)^2
            diff_squared = (bins_expanded - agents_expanded) ** 2
            
            # Fully vectorized approach using advanced indexing
            # Create identity matrix to mask out diagonal elements
            eye_mask = torch.eye(self.num_agents, dtype=torch.bool)
            
            # For each agent, we need the product of all other agents' squared differences
            # We'll use log-sum-exp trick for numerical stability with products
            log_diff_squared = torch.log(diff_squared + 1e-10)  # Add small epsilon to avoid log(0)
            
            # Create a mask matrix: (N, N, K) where mask[i, j, k] = True if i != j
            mask_3d = (~eye_mask).unsqueeze(2).expand(-1, -1, num_bins)
            
            # Expand log_diff_squared to (N, N, K) for broadcasting
            log_diff_expanded = log_diff_squared.unsqueeze(0).expand(self.num_agents, -1, -1)
            
            # Apply mask and sum over the j dimension (excluding i=j)
            masked_log_diff = torch.where(mask_3d, log_diff_expanded, torch.zeros_like(log_diff_expanded))
            log_product_sum = torch.sum(masked_log_diff, dim=1)  # Shape: (N, K)
            
            # Convert back from log space
            product_terms = torch.exp(log_product_sum)  # Shape: (N, K)
            
            # Compute the final shift matrix
            agent_diff = bins_expanded - agents_expanded  # Shape: (N, K)
            shift_matrix = -2 * self.Q * product_terms * agent_diff
            
            # Normalize by denominator
            shift_matrix = shift_matrix / denom.unsqueeze(0)
            
            return shift_matrix
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in shift_matrix computation: {str(e)}") from e
    
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
        :raises ValueError: If input parameters are invalid or incompatible.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        :raises NotImplementedError: If custom influence function is not properly configured.
        """
        
        try:
            
            custom_influence = self.infl_configs['custom_influence']

            # Convert to tensors for efficient computation
            agents_pos_tensor = self.agents_pos.clone().detach().requires_grad_(True)
            bin_points_tensor = self.bin_points.clone().detach().requires_grad_(True)

            # Determine dimensionality and validate domain type
            if self.domain_type == '1d':
                agent_dims = 1
            else:
                agent_dims = agents_pos_tensor.shape[1] if agents_pos_tensor.dim() > 1 else agents_pos_tensor.shape[0]
            
            # Pre-allocate result matrix for efficiency
            num_bins = len(self.bin_points)
            if self.domain_type == '1d':
                d_matrix = torch.zeros((self.num_agents, num_bins), dtype=torch.float32)
            else:
                d_matrix = torch.zeros((self.num_agents, agent_dims, num_bins), dtype=torch.float32)
            
            # Vectorized gradient computation
            try:
                if self.domain_type == '1d':
                    # Optimized 1D case with vectorized operations
                    for agent_id in range(self.num_agents):
                        # Create position tensor for this agent across all bin points
                        agent_pos = agents_pos_tensor[agent_id].item()
                        x_torch = torch.full((num_bins,), agent_pos, requires_grad=True, dtype=torch.float32)
                        try:
                            # Compute influence for all bin points at once
                            infl_values = custom_influence(x_torch, bin_points_tensor, parameter_instance[agent_id])
                            
                            # Validate influence values
                            if torch.any(infl_values <= 0):
                                raise RuntimeError(f"Non-positive influence values detected for agent {agent_id}. "
                                                 f"Log gradient computation requires positive influence values.")
                            
                            # Compute log and sum for backward pass
                            log_infl = torch.log(infl_values)
                            total_log_infl = torch.sum(log_infl)
                            
                            # Backward pass to get gradients
                            total_log_infl.backward()
                            
                            if x_torch.grad is None:
                                raise RuntimeError(f"Gradient computation failed for agent {agent_id}. "
                                                 f"Ensure custom influence function is differentiable.")
                            
                            # Store gradients
                            d_matrix[agent_id] = x_torch.grad.clone()
                            
                        except Exception as e:
                            raise RuntimeError(f"Custom influence computation failed for agent {agent_id}: {str(e)}") from e
                
                else:
                    # Multi-dimensional case with vectorized operations where possible
                    for agent_id in range(self.num_agents):
                        try:
                            # Get agent position and expand for all bin points
                            agent_pos = agents_pos_tensor[agent_id]
                            x_expanded = agent_pos.unsqueeze(0).expand(num_bins, -1)
                            x_torch = x_expanded.clone().detach().requires_grad_(True)
                            
                            # Compute influence for all bin points
                            infl_values = custom_influence(x_torch, bin_points_tensor, parameter_instance[agent_id])
                            
                            # Validate influence values
                            if torch.any(infl_values <= 0):
                                raise RuntimeError(f"Non-positive influence values detected for agent {agent_id}. "
                                                 f"Log gradient computation requires positive influence values.")
                            
                            # Compute log and sum for backward pass
                            log_infl = torch.log(infl_values)
                            total_log_infl = torch.sum(log_infl)
                            
                            # Backward pass
                            total_log_infl.backward()
                            
                            if x_torch.grad is None:
                                raise RuntimeError(f"Gradient computation failed for agent {agent_id}. "
                                                 f"Ensure custom influence function is differentiable.")
                            
                            # Store gradients for each dimension
                            for dim in range(agent_dims):
                                d_matrix[agent_id, dim] = x_torch.grad[:, dim].clone()
                                
                        except Exception as e:
                            raise RuntimeError(f"Custom influence computation failed for agent {agent_id}: {str(e)}") from e
            
            except Exception as e:
                if isinstance(e, (ValueError, RuntimeError, TypeError)):
                    raise
                else:
                    raise RuntimeError(f"Unexpected error in gradient computation: {str(e)}") from e
            
            # Final validation of output
            if torch.any(torch.isnan(d_matrix)):
                raise RuntimeError("NaN values detected in computed derivative matrix")
            
            if torch.any(torch.isinf(d_matrix)):
                raise RuntimeError("Infinite values detected in computed derivative matrix")
            
            # Reshape for 1D case to match expected output format
            if self.domain_type == '1d':
                return d_matrix  # Shape: (N, K)
            else:
                return d_matrix.transpose(1, 2)  # Shape: (N, K, dims) -> transpose to (N, dims, K)
                
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError, NotImplementedError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in d_torch computation: {str(e)}") from e

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
        :raises ValueError: If input parameters are invalid or incompatible.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        """
        
        try:
            # Convert and validate parameter_instance
            if isinstance(parameter_instance, (list, np.ndarray)):
                if len(parameter_instance) != self.num_agents:
                    raise ValueError(f"parameter_instance length ({len(parameter_instance)}) must match number of agents ({self.num_agents})")
                parameter_instance = torch.tensor(parameter_instance, dtype=torch.float32)
            elif isinstance(parameter_instance, torch.Tensor):
                parameter_instance = parameter_instance.to(torch.float32)
                if len(parameter_instance) != self.num_agents:
                    raise ValueError(f"parameter_instance length ({len(parameter_instance)}) must match number of agents ({self.num_agents})")
            else:
                raise TypeError(f"parameter_instance must be list, np.ndarray, or torch.Tensor, got {type(parameter_instance)}")
            
            try:
                pr_matrix = self.prob_matrix(parameter_instance)
            except Exception as e:
                raise RuntimeError(f"Failed to compute probability matrix: {str(e)}") from e
            
            if pr_matrix is None:
                raise RuntimeError("Probability matrix computation returned None")
            
            # Get derivative matrix based on influence type
            try:
                if self.infl_type == 'custom':
                    d_matrix = self.d_torch(parameter_instance)
                elif self.infl_type in ['multi_gaussian', 'gaussian', 'Jones_M', 'dirichlet']:
                    d_matrix = self.d_lnf_matrix(parameter_instance)
                    
            except Exception as e:
                raise RuntimeError(f"Failed to compute derivative matrix: {str(e)}") from e
            
            if d_matrix is None:
                raise RuntimeError("Derivative matrix computation returned None")
            
            # Validate matrix dimensions
            if pr_matrix.shape[0] != self.num_agents:
                raise ValueError(f"Probability matrix rows ({pr_matrix.shape[0]}) must match number of agents ({self.num_agents})")
            
            if pr_matrix.shape[1] != len(self.bin_points):
                raise ValueError(f"Probability matrix columns ({pr_matrix.shape[1]}) must match number of bin points ({len(self.bin_points)})")
            resource_tensor = self.resource_distribution.clone().detach()
            # Vectorized gradient computation based on domain type
            if self.domain_type == '1d':
                # Optimized 1D vectorized computation
                try:
                    # Compute complementary probability matrix
                    pr_matrix_c = 1 - pr_matrix  # Shape: (N, K)
                    
                    # Element-wise product: d_matrix * pr_matrix * pr_matrix_c * resource_tensor
                    # All operations are vectorized using broadcasting
                    gradient_terms = d_matrix * pr_matrix * pr_matrix_c * resource_tensor.unsqueeze(0)  # Shape: (N, K)
                    
                    # Handle functional shift if enabled
                    if self.infl_fshift:
                        try:
                            shift_matrix = self.shift_matrix(parameter_instance)
                            if shift_matrix.shape != (self.num_agents, len(self.bin_points)):
                                raise ValueError(f"Shift matrix shape {shift_matrix.shape} doesn't match expected ({self.num_agents}, {len(self.bin_points)})")
                            
                            # Subtract shift contribution (vectorized)
                            shift_terms = shift_matrix * pr_matrix * resource_tensor.unsqueeze(0)  # Shape: (N, K)
                            gradient_terms = gradient_terms - shift_terms
                            
                        except Exception as e:
                            raise RuntimeError(f"Failed to compute functional shift: {str(e)}") from e
                    
                    # Sum across bin points for each agent (vectorized)
                    grad = torch.sum(gradient_terms, dim=1)  # Shape: (N,)
                    
                except Exception as e:
                    raise RuntimeError(f"Failed in 1D gradient computation: {str(e)}") from e
            
            else:
                # Multi-dimensional vectorized computation
                try:
                    # Determine number of dimensions
                    if d_matrix.dim() == 3:  # Shape: (N, dims, K)
                        num_dims = d_matrix.shape[1]
                    elif d_matrix.dim() == 2:  # Shape: (N, K) - treat as 1D
                        num_dims = 1
                        d_matrix = d_matrix.unsqueeze(1)  # Shape: (N, 1, K)
                    else:
                        raise ValueError(f"Unexpected derivative matrix dimensions: {d_matrix.shape}")
                    
                    # Compute complementary probability and product terms
                    pr_matrix_c = 1 - pr_matrix  # Shape: (N, K)
                    pr_prod = pr_matrix * pr_matrix_c  # Shape: (N, K)
                    
                    # Expand probability product for broadcasting with multi-dimensional derivative
                    pr_prod_expanded = pr_prod.unsqueeze(1).expand(-1, num_dims, -1)  # Shape: (N, dims, K)
                    
                    # Expand resource tensor for broadcasting
                    resource_expanded = resource_tensor.unsqueeze(0).unsqueeze(0).expand(self.num_agents, num_dims, -1)  # Shape: (N, dims, K)
                    
                    # Vectorized element-wise multiplication
                    gradient_terms = d_matrix * pr_prod_expanded * resource_expanded  # Shape: (N, dims, K)
                    
                    # Sum across bin points for each agent and dimension
                    grad = torch.sum(gradient_terms, dim=2)  # Shape: (N, dims)
                    
                    # If single dimension, flatten to (N,)
                    if num_dims == 1:
                        grad = grad.squeeze(1)  # Shape: (N,)
                        
                except Exception as e:
                    raise RuntimeError(f"Failed in multi-dimensional gradient computation: {str(e)}") from e
            
            # Ensure consistent data type
            if grad.dtype != torch.float32:
                grad = grad.to(torch.float32)
            
            return grad
            
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in gradient computation: {str(e)}") from e
    
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
        :raises ValueError: If input parameters are invalid or incompatible.
        :raises RuntimeError: If computation fails due to numerical issues.
        :raises TypeError: If input types are not supported.
        """
        
        try:
            # Initialize gradient modification flag
            self.grad_modify = grad_modify
            
            # Store original agent positions for restoration
            agents_og = self.agents_pos.clone() if isinstance(self.agents_pos, torch.Tensor) else torch.tensor(self.agents_pos, dtype=torch.float32)
            
            # Pre-allocate storage tensors for efficiency
            pos_history = torch.zeros((self.time_steps, *current_positions.shape), dtype=torch.float32)
            grad_history = torch.zeros((self.time_steps, *current_positions.shape), dtype=torch.float32)
            reward_history = torch.zeros((self.time_steps, self.num_agents), dtype=torch.float32) if reward else None
            
            # Main gradient ascent loop with vectorized operations
            converged_at_step = None
            
            try:
                for time_step in range(self.time_steps):
                    # Update agent positions in environment for gradient computation
                    if isinstance(self.agents_pos, torch.Tensor):
                        self.agents_pos = current_positions.clone().detach()
                    
                    # Compute gradient with error handling
                    try:
                        grad_vec_row = self.gradient(self.parameters)
                    except Exception as e:
                        raise RuntimeError(f"Gradient computation failed at step {time_step}: {str(e)}") from e
                    
                    # Validate gradient
                    if grad_vec_row is None:
                        raise RuntimeError(f"Gradient computation returned None at step {time_step}")
                    
                    if torch.any(torch.isnan(grad_vec_row)):
                        raise RuntimeError(f"NaN values detected in gradient at step {time_step}")
                    
                    if torch.any(torch.isinf(grad_vec_row)):
                        raise RuntimeError(f"Infinite values detected in gradient at step {time_step}")
                    
                    # Process gradient based on domain type
                    if self.domain_type == 'simplex':
                        if grad_vec_row.dim() == 1:
                            # If gradient is 1D, expand for normalization
                            grad_expanded = grad_vec_row.unsqueeze(0) if grad_vec_row.shape[0] == current_positions.shape[1] else grad_vec_row.unsqueeze(1)
                            processed_grad = torch.nn.functional.normalize(grad_expanded, dim=-1)
                            if grad_vec_row.shape[0] == current_positions.shape[1]:
                                processed_grad = processed_grad.squeeze(0)
                            else:
                                processed_grad = processed_grad.squeeze(1)
                        else:
                            processed_grad = torch.nn.functional.normalize(grad_vec_row, dim=-1)
                    else:
                        processed_grad = grad_vec_row
                    
                    # Compute learning rate
                    try:
                        lr = general.learning_rate(
                            iter=time_step,
                            learning_rate_type=self.learning_rate_type,
                            learning_rate=self.learning_rate
                        )
                    except Exception as e:
                        raise RuntimeError(f"Learning rate computation failed at step {time_step}: {str(e)}") from e
                    
                    # Vectorized position update
                    updated_positions = current_positions + lr * processed_grad
                    
                    # Handle domain constraints vectorized
                    if self.domain_type == 'simplex':
                        # Vectorized simplex projection
                        try:
                            # Apply projection to each agent
                            valid_mask = torch.ones(self.num_agents, dtype=torch.bool)
                            for agent_idx in range(self.num_agents):
                                projected_pos = simplex_utils.projection_onto_simplex(updated_positions[agent_idx])
                                
                                # Check if projection is valid (all positive)
                                if torch.all(projected_pos > 0):
                                    updated_positions[agent_idx] = projected_pos
                                else:
                                    valid_mask[agent_idx] = False
                            
                            # Only update positions for valid projections
                            current_positions = torch.where(
                                valid_mask.unsqueeze(-1).expand_as(current_positions),
                                updated_positions,
                                current_positions
                            )
                            
                        except Exception as e:
                            raise RuntimeError(f"Simplex projection failed at step {time_step}: {str(e)}") from e
                    else:
                        # Direct update for non-simplex domains
                        current_positions = updated_positions
                    
                    # Store history (vectorized)
                    pos_history[time_step] = current_positions.clone()
                    grad_history[time_step] = grad_vec_row.clone()
                    
                    # Compute rewards if requested
                    if reward:
                        try:
                            # Update environment positions for reward computation
                            if isinstance(self.agents_pos, torch.Tensor):
                                self.agents_pos = current_positions.clone().detach()
                            else:
                                self.agents_pos = current_positions.clone().detach().numpy()
                            
                            reward_vec_row = self.reward_F(self.parameters)
                            reward_history[time_step] = reward_vec_row.clone()
                            
                        except Exception as e:
                            raise RuntimeError(f"Reward computation failed at step {time_step}: {str(e)}") from e
                    
                    # Vectorized convergence check
                    if time_step > 5:
                        try:
                            if self.domain_type == 'simplex':
                                # Compare with position 5 steps ago
                                comparison_step = max(0, time_step - 5)
                                position_diff = current_positions - pos_history[comparison_step]
                            else:
                                # Compare with position 2 steps ago
                                comparison_step = max(0, time_step - 2)
                                position_diff = current_positions - pos_history[comparison_step]
                            
                            # Compute L1 norm differences for each agent (vectorized)
                            abs_differences = torch.linalg.norm(position_diff, ord=1, dim=-1)
                            
                            # Count agents that have converged
                            converged_agents = torch.sum(abs_differences <= self.tolerance).item()
                            
                            if converged_agents >= self.tolerated_agents:
                                converged_at_step = time_step
                                break
                                
                        except Exception as e:
                            raise RuntimeError(f"Convergence check failed at step {time_step}: {str(e)}") from e
                
                # Trim history to actual used steps
                actual_steps = converged_at_step + 1 if converged_at_step is not None else self.time_steps
                
                # Store results efficiently
                self.grad_matrix = grad_history[:actual_steps].clone()
                self.pos_matrix = pos_history[:actual_steps].clone()
                
                if reward:
                    self.reward_matrix = reward_history[:actual_steps].clone()
                
                # Restore original agent positions
                self.agents_pos = agents_og
                
                # Final validation
                if torch.any(torch.isnan(self.pos_matrix)):
                    raise RuntimeError("NaN values detected in final position matrix")
                
                if torch.any(torch.isnan(self.grad_matrix)):
                    raise RuntimeError("NaN values detected in final gradient matrix")
                
                if reward and torch.any(torch.isnan(self.reward_matrix)):
                    raise RuntimeError("NaN values detected in final reward matrix")
                
            except Exception as e:
                # Restore original positions on any error
                self.agents_pos = agents_og
                if isinstance(e, (ValueError, RuntimeError, TypeError)):
                    raise
                else:
                    raise RuntimeError(f"Unexpected error in gradient ascent loop: {str(e)}") from e
                    
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error in mv_gradient_ascent: {str(e)}") from e
        
        
        
    
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

