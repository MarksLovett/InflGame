"""
.. module:: validation
   :synopsis: Provides configuration validation utilities for adaptive influence models in influencer games.

Validation Module
=================

This module provides comprehensive validation functions for adaptive influence game configurations.
It ensures that all parameters for the adaptive environment are properly typed, within valid ranges,
and mutually compatible before simulation begins.

The module is designed to work with the `InflGame.adaptive` package and provides robust error checking
with detailed error messages to help users quickly identify and fix configuration issues.

Dependencies:
-------------
- torch
- numpy
- matplotlib.tri
- InflGame.utils.general

Usage:
------
The `validate_adaptive_config` function is the main entry point for validating all configuration
parameters before creating an `AdaptiveEnv` instance. It performs type checking, range validation,
and compatibility checks between related parameters.

Example:
--------

.. code-block:: python
    
    from InflGame.utils.validation import validate_adaptive_config
    import torch
    import numpy as np

    # Define configuration
    config = validate_adaptive_config(
        num_agents=3,
        agents_pos=np.array([0.2, 0.5, 0.8]),
        parameters=torch.tensor([1.0, 1.0, 1.0]),
        resource_distribution=torch.tensor([10.0, 20.0, 30.0]),
        bin_points=np.array([0.1, 0.4, 0.7]),
        infl_configs={'infl_type': 'gaussian'},
        learning_rate_type='cosine_annealing',
        learning_rate=[0.0001, 0.01, 15],
        time_steps=100,
        domain_type='1d',
        domain_bounds=[0, 1],
        tolerance=1e-5
    )
    
    # Use validated config to create environment
    # env = AdaptiveEnv(**config)
"""

import warnings
import torch
import numpy as np
from typing import Union, List, Optional, Dict
import matplotlib.tri as tri
from InflGame.utils.general import _to_tensor


def validate_adaptive_config(num_agents: int,
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
                            device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    Validate the configuration of an adaptive influence model.
    
    Performs comprehensive validation of all parameters required to initialize an adaptive environment
    for influencer games. This includes type checking, range validation, and compatibility checks
    between related parameters. The function converts inputs to appropriate types and returns a
    validated configuration dictionary.

    Parameters
    ----------
    num_agents : int
        Number of agents in the system. Must be a positive integer.
    agents_pos : Union[List[float], np.ndarray]
        Initial positions of the agents. Length must equal ``num_agents``.
        For 1D domains: 1D array of positions.
        For 2D domains: :math:`(N, 2)` array of (x, y) coordinates.
        For simplex domains: :math:`(N, k)` array of barycentric coordinates that sum to 1.
    parameters : torch.Tensor
        Parameters for influence kernels of each agent. Length must equal ``num_agents``.
        Must contain finite values. For Gaussian kernels, typically positive values.
        For beta kernels, concentration parameter :math:`\\phi > 0`.
    resource_distribution : torch.Tensor
        Resource distribution in the environment. Must match length of ``bin_points``.
        Should contain non-negative finite values.
    bin_points : Union[List[float], np.ndarray]
        Points in the domain where resources are distributed. Must be non-empty.
        For 1D: points on the line segment.
        For 2D/simplex: coordinate pairs.
    infl_configs : Dict[str, Union[str, callable]], optional
        Influence kernel configuration dictionary, by default ``{'infl_type': 'gaussian'}``.
        
        - ``'infl_type'`` (str): Type of influence kernel. Valid options:
          
          - ``'gaussian'``: Gaussian influence kernel
          - ``'multi_gaussian'``: Multivariate Gaussian kernel
          - ``'dirichlet'``: Dirichlet kernel for simplex domains
          - ``'diric_mode'``: Mode-parameterized Dirichlet kernel for simplex domains
          - ``'beta'``: Beta distribution kernel
          - ``'Jones_M'``: Jones mean kernel
          - ``'custom'``: User-defined custom kernel
        
        - ``'custom_influence'`` (callable): Required when ``infl_type='custom'``.
          Custom influence function.
    learning_rate_type : str, optional
        Type of learning rate schedule, by default 'cosine'. Valid options:
        
        - ``'cosine_annealing'``: Cosine annealing schedule
        - ``'fixed'``: Constant learning rate
        - ``'trust_region'``: Trust region adaptive schedule
        - ``'gradient_magnitude'``: Magnitude-based adaptive schedule
    learning_rate : List[float], optional
        Learning rate parameters, by default [.0001, .01, 15].
        For schedules: ``[min_lr, max_lr, decay_steps]``.
        For fixed: single value ``[lr]``.
    time_steps : int, optional
        Maximum number of gradient ascent iterations, by default 100.
        Must be a positive integer.
    fp : Optional[int], optional
        Fixed parameter index for Dirichlet kernel, by default 0.
        Required when ``infl_type='dirichlet'``. Must be non-negative integer
        between 0 and simplex dimension - 1.
    infl_cshift : bool, optional
        Whether to apply constant shift to influence function, by default False.
    cshift : int, optional
        Value of constant shift, by default 0. Required when ``infl_cshift=True``.
        Must be a list, array, or tensor.
    infl_fshift : bool, optional
        Whether to apply functional shift to influence, by default False.
        Not yet implemented for multi-dimensional agents.
    Q : int, optional
        Scaling factor for functional shift, by default 0.
        Required when ``infl_fshift=True``. Should be non-negative.
    domain_type : str, optional
        Type of domain, by default '1d'. Valid options:
        
        - ``'1d'``: One-dimensional line segment
        - ``'2d'``: Two-dimensional rectangular domain
        - ``'simplex'``: Probability simplex
    domain_bounds : Union[List[float], torch.Tensor], optional
        Bounds of the domain, by default [0, 1].
        
        - For ``'1d'``: ``[min, max]``
        - For ``'2d'``: ``[[xmin, xmax], [ymin, ymax]]``
        - For ``'simplex'``: ``(r2, corners, triangle, trimesh)`` tuple with:
          
          - ``r2``: 2D reference point
          - ``corners``: :math:`(3, 2)` array of triangle vertices
          - ``triangle``: matplotlib Triangulation object
          - ``trimesh``: matplotlib Triangulation mesh object
    tolerance : float, optional
        Convergence tolerance for position changes, by default :math:`10^{-5}`.
        Must be a positive number.
    tolerated_agents : Optional[int], optional
        Number of agents that must meet tolerance before stopping, by default None.
        If None, defaults to ``num_agents``. Must be between 1 and ``num_agents``.
    device : Optional[Union[str, torch.device]], optional
        Device to place tensors on, by default None (uses CPU).
        Can be a string like ``'cuda'`` or ``'cpu'``, or a ``torch.device`` object.

    Returns
    -------
    dict
        Dictionary of validated and converted parameters with all inputs converted to
        appropriate types (tensors, validated ranges, etc.). Keys match parameter names.

    Raises
    ------
    ValueError
        If any configuration parameter is invalid or out of range.
    TypeError
        If input types are incorrect.
    NotImplementedError
        If unsupported functionality is requested (e.g., functional shift for multi-dimensional agents).
        
    Warns
    -----
    UserWarning
        For potentially problematic but not invalid configurations:
        
        - Negative parameters for Gaussian kernels
        - Negative learning rates
        - Non-negative resources
        - Negative Q parameter with functional shift

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> 
    >>> # Validate a simple 1D configuration
    >>> config = validate_adaptive_config(
    ...     num_agents=2,
    ...     agents_pos=[0.3, 0.7],
    ...     parameters=torch.tensor([0.1, 0.15]),
    ...     resource_distribution=torch.tensor([1.0, 2.0, 1.5]),
    ...     bin_points=[0.0, 0.5, 1.0],
    ...     domain_type='1d',
    ...     domain_bounds=[0, 1]
    ... )
    >>> config['num_agents']
    2
    """
    
    
    
    validated = {}
    
    # 1. Validate num_agents (first parameter)
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError(f"Number of agents must be a positive integer, got {num_agents}")
    validated['num_agents'] = num_agents
    
    # 2. Validate and convert agents_pos (second parameter)
    agents_pos = _to_tensor(agents_pos, "agents_pos", device=device)
    if len(agents_pos) != num_agents:
        raise ValueError(f"agents_pos must be a tensor with {num_agents} elements, got shape {agents_pos.shape}")
    validated['agents_pos'] = agents_pos
    
    # 3. Validate and convert parameters (third parameter)
    if parameters is not None:
        parameters = _to_tensor(parameters, "parameters", device=device)
        _infl_type = infl_configs.get('infl_type')
        # Blotto uses [sigma, chi] — a fixed-length 2-element parameter vector shared
        # across all agents, so the num_agents length check is skipped.
        if _infl_type == 'blotto':
            if len(parameters) != num_agents:
                    warnings.warn(f"Blotto kernel expects parameters=[sigma, chi] with exactly 2 elements shared across all agents, got {len(parameters)} elements. Skipping length check for Blotto kernel.", UserWarning)
            if len(parameters[0]) != 2:
                raise ValueError(
                    f"Blotto kernel requires parameters=[sigma, chi] with exactly 2 elements, "
                    f"got {len(parameters[0])}"
                    )
            if float(parameters[0][0]) <= 0:
                raise ValueError(f"Blotto sigma (parameters[0]) must be positive, got {float(parameters[0][0])}")
            if float(parameters[0][1]) <= 0:
                raise ValueError(f"Blotto chi (parameters[1]) must be positive, got {float(parameters[0][1])}")
        else:
            if len(parameters) != num_agents:
                raise ValueError(f"parameters must be a tensor with {num_agents} elements")
        if not torch.all(torch.isfinite(parameters)):
            raise ValueError("parameters must contain finite values (no NaN or Inf)")
        if torch.any(parameters < 0) and _infl_type in ['gaussian', 'multi_gaussian', 'dirichlet', 'diric_mode']:
            warnings.warn("Parameters with negative values detected, this may result in unpredictable behavior", UserWarning)
        # Beta distribution requires concentration parameter phi > 2 for proper mode parameterization
        if _infl_type == 'beta' and torch.any(parameters < 0):
            raise ValueError("Beta kernel concentration parameters (phi) must be >0 for mode parameterization")
        validated['parameters'] = parameters
    
    # 4. Validate and convert resource_distribution (fourth parameter)
    resource_distribution = _to_tensor(resource_distribution, "resource_distribution", device=device)
    if not torch.all(torch.isfinite(resource_distribution)):
        raise ValueError("resource_distribution must contain finite values (no NaN or Inf)")
    if torch.any(resource_distribution < 0):
        warnings.warn("Non-negative values detected in resource_distribution, this may result in unpredictable behavior", UserWarning)
    validated['resource_distribution'] = resource_distribution
    
    # 5. Validate and convert bin_points (fifth parameter)
    bin_points = _to_tensor(bin_points, "bin_points", device=device)
    if len(bin_points) == 0:
        raise ValueError("bin_points must be a non-empty tensor")
    if len(resource_distribution) != len(bin_points):
        raise ValueError(f"resource_distribution length ({len(resource_distribution)}) must match bin_points length ({len(bin_points)})")
    validated['bin_points'] = bin_points
    
    # 6. Validate infl_configs (sixth parameter)
    if not isinstance(infl_configs, dict):
        raise TypeError("infl_configs must be a dictionary")
    
    valid_infl_types = ['gaussian', 'Jones_M', 'dirichlet', 'diric_mode', 'multi_gaussian', 'beta', 'blotto', 'custom']
    infl_type = infl_configs.get('infl_type')
    if infl_type not in valid_infl_types:
        raise ValueError(f"Invalid influence type '{infl_type}'. Supported types are {valid_infl_types}")
    validated['infl_type'] = infl_type
    
    if infl_type == 'custom':
        if 'custom_influence' not in infl_configs or not callable(infl_configs['custom_influence']):
            raise ValueError("For custom influence type, 'custom_infl_func' must be provided and callable")
        validated['custom_influence'] = infl_configs['custom_influence']
    
    validated['infl_configs'] = infl_configs
    
    # 7. Validate learning_rate_type (seventh parameter)
    valid_lr_types = ['cosine_annealing', 'fixed', 'trust_region','gradient_magnitude']
    if learning_rate_type not in valid_lr_types:
        raise ValueError(f"Invalid learning rate type '{learning_rate_type}'. Supported types are {valid_lr_types}")
    validated['learning_rate_type'] = learning_rate_type
    
    # 8. Validate and convert learning_rate (eighth parameter)
    learning_rate = _to_tensor(learning_rate, "learning_rate", device=device)
    if learning_rate.dim() != 1 or (len(learning_rate) != 3 and len(learning_rate) != 1):
        raise ValueError("learning_rate must be a 1D tensor with exactly 3 elements [min_lr, max_lr, decay_steps] or a single float for fixed learning rate")
    if torch.any(learning_rate <= 0):
        warnings.warn("Negative learning rate parameters detected, this may result in unstable training", UserWarning)
    validated['learning_rate'] = learning_rate
    
    # 9. Validate time_steps (ninth parameter)
    if not isinstance(time_steps, int) or time_steps <= 0:
        raise ValueError(f"time_steps must be a positive integer, got {time_steps}")
    validated['time_steps'] = time_steps
    
    # 10. Validate fp (tenth parameter)
    if infl_type == 'dirichlet':
        if fp is not None and (not isinstance(fp, int) or fp < 0):
            raise ValueError(f"fp ('fixed parameter') must be a non-negative integer between 0 and max(simplex dimension)-1, got {fp}")    
        else:
           fp = 0  # Default value if not provided
    validated['fp'] = fp

    # 11. Validate infl_cshift (eleventh parameter)
    if not isinstance(infl_cshift, bool):
        raise TypeError("infl_cshift (to shift the influence by a constant values) must be a boolean")
    validated['infl_cshift'] = infl_cshift
    
    # 12. Validate cshift (twelfth parameter)
    if infl_cshift and not isinstance(cshift, (list, np.ndarray, torch.Tensor)):
        raise ValueError("cshift must be a list, np.ndarray, or torch.Tensor when infl_cshift is True")
        
    # convert to a tensor
    if infl_cshift:
        cshift = _to_tensor(cshift, "cshift", device=device)
    validated['cshift'] =cshift
    
    # 13. Validate infl_fshift (thirteenth parameter)
    if not isinstance(infl_fshift, bool):
        raise TypeError("infl_fshift (to shift the influence by a functional form) must be a boolean")
    validated['infl_fshift'] = infl_fshift
    
    # 14. Validate Q (fourteenth parameter)
    if infl_fshift and (not isinstance(Q, (int, float)) or Q < 0):
        warnings.warn("A negative Q-parameter value was detected, this may result in unpredictable behavior", UserWarning)
    if infl_fshift and agents_pos.dim() > 1:
        raise NotImplementedError("Functional shift for multi-dimensional agents is not implemented yet")
    validated['Q'] = Q
    
    # 15. Validate domain_type (fifteenth parameter)
    valid_domain_types = ['1d', '2d', 'simplex']
    if domain_type not in valid_domain_types:
        raise ValueError(f"Invalid domain type '{domain_type}'. Supported types are {valid_domain_types}")
    validated['domain_type'] = domain_type
    
    # 16. Validate and convert domain_bounds (sixteenth parameter)
    if domain_type == '1d':
        domain_bounds = _to_tensor(domain_bounds, "domain_bounds", device=device)
        if domain_bounds.dim() != 1 or len(domain_bounds) != 2:
            raise ValueError("domain_bounds must be a 1D tensor with exactly 2 elements [min, max]")
        if domain_bounds[0] >= domain_bounds[1]:
            raise ValueError(f"domain_bounds must have min < max, got {domain_bounds.tolist()}")

    elif domain_type == '2d':
        domain_bounds = _to_tensor(domain_bounds, "domain_bounds", device=device)
        if domain_bounds.dim() != 2 or domain_bounds.shape[0] != 2 or domain_bounds.shape[1] != 2:
            raise ValueError("domain_bounds must be a 2D tensor with [[xmin,xmax],[ymin,ymax]] with shape [2, 2] for 2D rectangular domains")
        
    
    elif domain_type == 'simplex':
        if len(domain_bounds) != 4:
            raise ValueError(f"domain_bounds must be a tuple with exactly 4 elements for simplex domain, r2, corners, triangle, trimesh, see simplex_utils.simplex_setup")
        if len(domain_bounds[0]) != 2:
            raise ValueError(f"r2 is a 2d point, expected shape [2], got {domain_bounds[0].shape}")
        if np.shape(domain_bounds[1]) != (3, 2):
            raise ValueError(f"corners must be a 2D tensor of the 3 corners in 2d cartesian coordinates with shape [3, 2] for simplex domain, got {domain_bounds[1].shape}")
        if type(domain_bounds[2]) != tri._triangulation.Triangulation:
            raise ValueError(f"triangle must be type matplotlib.tri.Triangulation for simplex domain, got {type(domain_bounds[2])}")
        if type(domain_bounds[3]) != tri._triangulation.Triangulation:
            raise ValueError(f"trimesh must be type matplotlib.tri.Triangulation for simplex domain, got {type(domain_bounds[3])}")
    validated['domain_bounds'] = domain_bounds
    
    # Now validate spatial constraints with domain_bounds available
    if domain_type == "1d":
        if not torch.all((bin_points >= domain_bounds[0]) & (bin_points <= domain_bounds[1])):
            raise ValueError(f"bin_points must be within 1d domain bounds {domain_bounds.tolist()}")
        if not torch.all((agents_pos >= domain_bounds[0]) & (agents_pos <= domain_bounds[1])):
            raise ValueError(f"agents_pos must be within domain bounds {domain_bounds.tolist()}")
    #elif domain_type == "2d":
    #    if not torch.all((agents_pos[:, 0] >= domain_bounds[0, 0]) & (agents_pos[:, 0] <= domain_bounds[1, 0]) &
    #                     (agents_pos[:, 1] >= domain_bounds[0, 1]) & (agents_pos[:, 1] <= domain_bounds[1, 1])):
    #        raise ValueError(f"agents_pos must be within 2d domain bounds {domain_bounds.tolist()}")
    elif domain_type == "simplex":
        if infl_type == 'blotto':
            # Blotto positions lie on the budget simplex: all > 0, rows sum to chi.
            # chi is the second column of parameters (per-agent or shared).
            if parameters.dim() == 2:
                chi_vals = parameters[:, 1]
            else:
                chi_vals = parameters[1].expand(num_agents)
            row_sums = agents_pos.sum(dim=-1)
            tol = 1e-4
            if not (torch.all(agents_pos > 0) and
                    torch.all(torch.abs(row_sums - chi_vals) < tol * chi_vals.abs().clamp(min=1))):
                raise ValueError(
                    f"For blotto domain, agents_pos rows must be positive and each sum to chi "
                    f"(expected chi per agent: {chi_vals.tolist()}, got row sums: {row_sums.tolist()})"
                )
        else:
            if not torch.all((agents_pos > 0) & (agents_pos < 1) & torch.all(agents_pos.sum(dim=-1) == torch.ones(num_agents))):
                raise ValueError(f"agents_pos must be valid simplex coordinates (all values between 0 and 1, sum to 1) for simplex domain, got {agents_pos.tolist()}")

    # 17. Validate tolerance (seventeenth parameter)
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("tolerance must be a positive number")
    validated['tolerance'] = tolerance
    
    # 18. Validate tolerated_agents (eighteenth parameter)
    if tolerated_agents is not None:
        if not isinstance(tolerated_agents, int) or tolerated_agents <= 0 or tolerated_agents > num_agents:
            raise ValueError(f"tolerated_agents must be an integer between 1 and {num_agents}")
    else:
        tolerated_agents = num_agents
    validated['tolerated_agents'] = tolerated_agents
    
    # 19. Store device (nineteenth parameter)
    validated['device'] = device

    return validated



    
