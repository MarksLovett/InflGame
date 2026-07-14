r"""
.. module:: beta
   :synopsis: Implements the Beta influence kernel for modeling agent interactions with resources in 1D domains.

Beta Influence Kernel Module
============================

This module implements the Beta influence kernel and its associated computations. The Beta kernel models the 
influence of agents based on a Beta distribution, which is particularly useful for bounded domains [0,1].

Mathematical Definitions:
-------------------------
The Beta influence kernel is parameterized by mode (m) and concentration (phi):

.. math::
    f_i(x_i, b) = \frac{b^{\alpha-1} (1-b)^{\beta-1}}{B(\alpha, \beta)}

where:
  - :math:`x_i` is the position (mode) of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`\alpha = x_i(\phi - 2) + 1`
  - :math:`\beta = (1 - x_i)(\phi - 2) + 1`
  - :math:`\phi` is the concentration parameter for agent :math:`i`
  - :math:`B(\alpha, \beta)` is the Beta function

The gradient of the logarithm of the Beta influence kernel is:

.. math::
    \frac{\partial \ln f_i}{\partial x_i} = (\phi - 2) \left[\ln\frac{b}{1-b} - \psi(\alpha) + \psi(\beta)\right]

where :math:`\psi` is the digamma function.

Usage:
------
The `influence` function computes the influence of an agent at specific bin points,
while the `d_ln_f` function calculates the gradient of the logarithm of the Beta influence kernel.

New vectorized functions are available for improved performance:
- `influence_vectorized` : Compute influence for all agents simultaneously
- `d_ln_f_vectorized` : Compute gradients for all agents simultaneously

Examples
--------

.. code-block:: python

  import numpy as np
  import torch
  from InflGame.kernels.beta import influence, d_ln_f
  from InflGame.kernels.beta import influence_vectorized, d_ln_f_vectorized

  # Define parameters
  num_agents = 3
  parameter_instance = [3.0, 4.0, 5.0]  # concentration parameters (phi)
  agents_pos = np.array([0.3, 0.5, 0.7])  # mode parameters (m)
  bin_points = np.linspace(0, 1, 100)
  resource_distribution = np.random.rand(100)

  # Single agent computation (backward compatible)
  influence_values = influence(agent_id=0, parameter_instance=parameter_instance, 
                              agents_pos=agents_pos, bin_points=bin_points)
  print("Influence values:", influence_values)

  # Vectorized computation (all agents at once)
  all_influences = influence_vectorized(parameter_instance=parameter_instance, 
                                       agents_pos=agents_pos, bin_points=bin_points)
  print("All influences shape:", all_influences.shape)  # (num_agents, num_bins)

  # Single agent gradient
  gradient = d_ln_f(agent_id=0, parameter_instance=parameter_instance, 
                   agents_pos=agents_pos, bin_points=bin_points)
  print("Gradient values:", gradient)

  # Vectorized gradients (all agents at once)
  all_gradients = d_ln_f_vectorized(parameter_instance=parameter_instance,
                                   agents_pos=agents_pos, bin_points=bin_points)
  print("All gradients shape:", all_gradients.shape)  # (num_agents, num_bins)
"""

import torch
import numpy as np
from typing import Union, List
from scipy.optimize import fsolve, brentq
from scipy.special import polygamma as psi


# ========================= JIT-COMPILED HELPER FUNCTIONS =========================

@torch.jit.script
def _influence_vectorized_core(
    agents_pos: torch.Tensor,
    bin_points: torch.Tensor,
    parameter_instance: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for Beta influence.
    
    Computes the Beta distribution PDF for all agents and bin points.
    
    Parameters
    ----------
    agents_pos : torch.Tensor
        Agent positions (modes) of shape (num_agents,).
    bin_points : torch.Tensor
        Bin points of shape (num_bins,).
    parameter_instance : torch.Tensor
        Concentration parameters (phi) of shape (num_agents,).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins).
    """
    # Reshape for broadcasting: agents_pos (N, 1), bin_points (1, K), params (N, 1)
    m = agents_pos.unsqueeze(1)  # (N, 1)
    b = bin_points.unsqueeze(0)  # (1, K)
    sigma = (1.0 / parameter_instance).unsqueeze(1)  # (N, 1) - using 1/parameter_instance
    
    # Compute alpha and beta parameters
    alpha = m * (sigma - 2) + 1  # (N, 1)
    beta_param = (1 - m) * (sigma - 2) + 1  # (N, 1)
    
    # Clamp bin_points to avoid log(0) issues
    b_clamped = torch.clamp(b, min=1e-10, max=1-1e-10)
    
    # Compute Beta PDF using log-space for numerical stability
    # log(f) = (alpha-1)*log(b) + (beta-1)*log(1-b) - log(B(alpha, beta))
    log_numerator = (alpha - 1) * torch.log(b_clamped) + (beta_param - 1) * torch.log(1 - b_clamped)
    log_beta_func = torch.lgamma(alpha) + torch.lgamma(beta_param) - torch.lgamma(alpha + beta_param)
    log_pdf = log_numerator - log_beta_func
    
    influence_matrix = torch.exp(log_pdf)  # (N, K)
    
    return influence_matrix


@torch.jit.script
def _d_ln_f_vectorized_core(
    agents_pos: torch.Tensor,
    bin_points: torch.Tensor,
    parameter_instance: torch.Tensor
) -> torch.Tensor:
    """
    JIT-compiled core computation for Beta influence gradient.
    
    Computes the gradient of log(Beta PDF) with respect to agent positions.
    
    Parameters
    ----------
    agents_pos : torch.Tensor
        Agent positions (modes) of shape (num_agents,).
    bin_points : torch.Tensor
        Bin points of shape (num_bins,).
    parameter_instance : torch.Tensor
        Concentration parameters (phi) of shape (num_agents,).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_bins).
    """
    # Reshape for broadcasting
    m = agents_pos.unsqueeze(1)  # (N, 1)
    b = bin_points.unsqueeze(0)  # (1, K)
    phi = (1.0 / parameter_instance).unsqueeze(1)  # (N, 1) - using 1/parameter_instance
    
    # Compute alpha and beta parameters
    alpha = m * (phi - 2) + 1  # (N, 1)
    beta_param = (1 - m) * (phi - 2) + 1  # (N, 1)
    
    # Clamp bin_points to avoid log(0) issues
    b_clamped = torch.clamp(b, min=1e-10, max=1-1e-10)
    
    # Compute gradient: (phi - 2) * [log(b/(1-b)) - digamma(alpha) + digamma(beta)]
    log_ratio = torch.log(b_clamped / (1 - b_clamped))  # (1, K)
    digamma_alpha = torch.digamma(alpha)  # (N, 1)
    digamma_beta = torch.digamma(beta_param)  # (N, 1)
    
    gradient_matrix = (phi - 2) * (log_ratio - digamma_alpha + digamma_beta)  # (N, K)
    
    return gradient_matrix


# ========================= VECTORIZED FUNCTIONS =========================

def influence_vectorized(parameter_instance: Union[list, np.ndarray, torch.Tensor],
                         agents_pos: Union[np.ndarray, torch.Tensor],
                         bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Compute the Beta influence for all agents simultaneously using vectorized operations.
    
    This function calculates the influence matrix where each row represents an agent's
    influence across all bin points, providing significant performance improvements over
    single-agent computations.
    
    The influence is computed as:

    .. math::
        f_i(x_i,b) = \frac{b^{\alpha-1} (1-b)^{\beta-1}}{B(\alpha, \beta)}

    where :math:`\alpha = x_i(\phi - 2) + 1` and :math:`\beta = (1 - x_i)(\phi - 2) + 1`.

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Concentration parameters (:math:`\phi_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions (modes :math:`x_i`) of all agents, shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Influence matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the influence of agent i at bin point j.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.3, 0.5, 0.7])
    >>> parameters = np.array([3.0, 4.0, 5.0])
    >>> bins = np.linspace(0, 1, 50)
    >>> influences = influence_vectorized(parameters, agents_pos, bins)
    >>> print(influences.shape)
    torch.Size([3, 50])
    """
    # Convert to tensors with consistent dtype
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float64)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float64)
    if not isinstance(parameter_instance, torch.Tensor):
        parameter_instance = torch.tensor(parameter_instance, dtype=torch.float64)
    
    # Use JIT-compiled core for optimal performance
    influence_matrix = _influence_vectorized_core(agents_pos, bin_points, parameter_instance)
    return influence_matrix


def d_ln_f_vectorized(parameter_instance: Union[list, np.ndarray, torch.Tensor],
                      agents_pos: Union[np.ndarray, torch.Tensor],
                      bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Compute the gradient of the logarithm of the Beta influence kernel for all agents.
    
    This vectorized function calculates gradients for all agents simultaneously,
    providing significant performance improvements over single-agent computations.

    The gradient is calculated as:

    .. math::
        \frac{\partial \ln f_i}{\partial x_i} = (\phi - 2) \left[\ln\frac{b}{1-b} - \psi(\alpha) + \psi(\beta)\right]

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Concentration parameters (:math:`\phi_i`) for all agents, shape (num_agents,).
    agents_pos : np.ndarray | torch.Tensor
        Current positions (modes :math:`x_i`) of all agents, shape (num_agents,).
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`), shape (num_bins,).
        
    Returns
    -------
    torch.Tensor
        Gradient matrix of shape (num_agents, num_bins) where element [i,j] 
        represents the gradient of agent i at bin point j.
        
    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([0.3, 0.5, 0.7])
    >>> parameters = np.array([3.0, 4.0, 5.0])
    >>> bins = np.linspace(0, 1, 50)
    >>> gradients = d_ln_f_vectorized(parameters, agents_pos, bins)
    >>> print(gradients.shape)
    torch.Size([3, 50])
    """
    # Convert to tensors with consistent dtype
    if not isinstance(agents_pos, torch.Tensor):
        agents_pos = torch.tensor(agents_pos, dtype=torch.float64)
    if not isinstance(bin_points, torch.Tensor):
        bin_points = torch.tensor(bin_points, dtype=torch.float64)
    if not isinstance(parameter_instance, torch.Tensor):
        parameter_instance = torch.tensor(parameter_instance, dtype=torch.float64)
    
    # Use JIT-compiled core for optimal performance
    gradient_matrix = _d_ln_f_vectorized_core(agents_pos, bin_points, parameter_instance)
    
    return gradient_matrix

def left_nash(x,parameter_instance):
        """
        Compute left side of Nash equilibrium equation.
        Handles both torch tensors and floats, always returns same type as input.
        """
        if torch.is_tensor(x):
            # Convert to float for scipy.special.digamma
            x_float = x.item() if x.dim() == 0 else float(x)
            param_val = float((1.0 / parameter_instance[0]).item() if torch.is_tensor(parameter_instance) else (1.0 / parameter_instance[0]))
            result = psi(0, (1 - x_float) * (param_val - 2) + 1) - psi(0, x_float * (param_val - 2) + 1)
            return torch.tensor(result, dtype=torch.float32)
        else:
            # x is already a float
            param_val = float((1.0 / parameter_instance[0]).item() if torch.is_tensor(parameter_instance) else (1.0 / parameter_instance[0]))
            return psi(0, (1 - x) * (param_val - 2) + 1) - psi(0, x * (param_val - 2) + 1)


def equation_nash(x_float,parameter_instance,log_average):
        """Wrapper for scipy.optimize - works with floats only"""
        left = left_nash(x=x_float,parameter_instance=parameter_instance)
        return left - log_average.item()


def nash_value(parameter_instance,bin_points,resource_distribution):
    log_average = -torch.sum((torch.log(bin_points) - torch.log(1 - bin_points)) * resource_distribution) / torch.sum(resource_distribution)
    # Method: Using brentq (root finding in interval [0, 1])
    # This is more robust for finding roots in a bounded interval
    root_finder_function = lambda x: equation_nash(x,parameter_instance=parameter_instance, log_average=log_average)
    solution_brentq = brentq(root_finder_function, 0.001, 0.999)
    solution_brentq_tensor = torch.tensor(solution_brentq, dtype=torch.float32)
    nash = solution_brentq_tensor
    return nash

def x_star_left(sig, nash):
        """
        Compute left side of second equation using polygamma(1, ...).
        Handles both torch tensors and floats for sig, always returns same type as sig input.
        """
        # Extract float value from nash_equilibrium_x
        if torch.is_tensor(nash):
            x_float = nash.item() if nash.dim() == 0 else float(nash)
        else:
            x_float = float(nash)
        
        if torch.is_tensor(sig):
            # Convert to float for scipy.special.polygamma
            sig_float = sig.item() if sig.dim() == 0 else float(sig)
            result = psi(1, (1 - x_float) * (sig_float - 2) + 1) + psi(1, x_float * (sig_float - 2) + 1)
            return torch.tensor(result, dtype=torch.float32)
        else:
            # sig is already a float
            return psi(1, (1 - x_float) * (sig - 2) + 1) + psi(1, x_float * (sig - 2) + 1)

def equation_sig_star(sig_float,nash,log_std):
    """Wrapper for scipy.optimize - solves for sig given nash_equilibrium_x"""
    left = x_star_left(sig_float, nash)
    return left - log_std.item()

def sigma_star(num_agents,bin_points,resource_distribution,parameter_instance=None, nash=None):
    log_average = torch.sum((torch.log(bin_points) - torch.log(1 - bin_points)) * torch.tensor(resource_distribution)) / torch.sum(torch.tensor(resource_distribution))
    log_squared_average = torch.sum(((torch.log(torch.tensor(bin_points)) - torch.log(1 - torch.tensor(bin_points)))**2 )* torch.tensor(resource_distribution)) / torch.sum(torch.tensor(resource_distribution))
    log_std =(num_agents-2)/(num_agents-1)*(log_squared_average-log_average**2)
    if nash==None:
        nash=nash_value(parameter_instance=parameter_instance,bin_points=bin_points,resource_distribution=resource_distribution)
    root_finder_function = lambda x: equation_sig_star(x, nash=nash, log_std=log_std)
    try:
        solution_sig = brentq(root_finder_function, 0, 1000)
    except Exception as e:
            solution_sig = 2  # Fallback value if all methods fail

    solution_sig_tensor = torch.tensor(solution_sig, dtype=torch.float32)
    sigma_star = solution_sig_tensor
    return 1/sigma_star

# ================= BACKWARD COMPATIBLE FUNCTIONS =================

def influence(agent_id: int,
              parameter_instance: Union[list, np.ndarray, torch.Tensor],
              agents_pos: Union[np.ndarray, torch.Tensor],
              bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Calculates the influence of a single agent using the Beta influence kernel.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations when beneficial.

    The influence is computed as:

    .. math::
        f_i(x_i,b) = \frac{b^{\alpha-1} (1-b)^{\beta-1}}{B(\alpha, \beta)}

    where:
      - :math:`x_i` is the position (mode) of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`\alpha = x_i(\phi - 2) + 1`
      - :math:`\beta = (1 - x_i)(\phi - 2) + 1`
      - :math:`\phi` is the concentration parameter for agent :math:`i`

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID.
    parameter_instance : list | np.ndarray | torch.Tensor
        Concentration parameter(s) unique to the agent's influence distribution (:math:`\phi_i`).
    agents_pos : np.ndarray | torch.Tensor
        Current positions (modes :math:`x_i`) of all agents.
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b`).
        
    Returns
    -------
    torch.Tensor
        The agent's influence calculated using the Beta method.
        
    Notes
    -----
    For improved performance when computing influence for multiple agents,
    consider using :func:`influence_vectorized` instead.
    """
    # Use vectorized computation and extract single agent result
    influence_matrix = influence_vectorized(parameter_instance, agents_pos, bin_points)
    return influence_matrix[agent_id]


def d_ln_f(agent_id: int,
           parameter_instance: Union[list, np.ndarray, torch.Tensor],
           agents_pos: Union[np.ndarray, torch.Tensor],
           bin_points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    r"""
    Calculates the gradient of the logarithm of the Beta influence kernel for a single agent.
    
    This function provides backward compatibility while internally using optimized
    vectorized operations.

    The gradient is calculated as:

    .. math::
        \frac{\partial \ln f_i}{\partial x_i} = (\phi - 2) \left[\ln\frac{b}{1-b} - \psi(\alpha) + \psi(\beta)\right]

    where:
      - :math:`x_i` is the position (mode) of agent :math:`i`
      - :math:`b` is the bin point
      - :math:`\alpha = x_i(\phi - 2) + 1`
      - :math:`\beta = (1 - x_i)(\phi - 2) + 1`
      - :math:`\phi` is the concentration parameter
      - :math:`\psi` is the digamma function

    Parameters
    ----------
    agent_id : int
        The current player/agent's ID.
    parameter_instance : list | np.ndarray | torch.Tensor
        Concentration parameter(s) (:math:`\phi_i`) for all agents.
    agents_pos : np.ndarray | torch.Tensor
        Current positions (modes :math:`x_i`) of all agents.
    bin_points : np.ndarray | torch.Tensor
        Locations of the resource/bin points (:math:`b_k`).
        
    Returns
    -------
    torch.Tensor
        The gradient values for the specified agent.
        
    Notes
    -----
    For improved performance when computing gradients for multiple agents,
    consider using :func:`d_ln_f_vectorized` instead.
    """
    # Use vectorized computation and extract single agent result
    gradient_matrix = d_ln_f_vectorized(parameter_instance, agents_pos, bin_points)
    return gradient_matrix[agent_id]


