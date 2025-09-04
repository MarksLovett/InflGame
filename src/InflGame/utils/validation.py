import warnings
import torch
import numpy as np
from typing import Union, List, Optional,Dict
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
                            tolerated_agents: Optional[int] = None) -> dict:
    """
    Validate the configuration of the adaptive influence model.
    
    :param infl_type: Type of influence kernel ('gaussian', 'Jones_M', 'dirichlet', 'multi_gaussian', 'custom')
    :param num_agents: Number of agents in the system
    :param bin_points: Points in the domain where resources are distributed
    :param agents_pos: Initial positions of the agents
    :param resource_distribution: Resource distribution in the environment
    :param domain_bounds: Bounds of the domain [min, max]
    :param domain_type: Type of domain ('1d', '2d', 'simplex')
    :param learning_rate_type: Type of learning rate schedule ('cosine', 'constant', 'linear')
    :param learning_rate: Learning rate parameters [min_lr, max_lr, decay_steps]
    :param fixed_pa: Fixed parameter for Dirichlet kernel (required if infl_type='dirichlet')
    :param infl_cshift: Whether to apply constant shift in influence
    :param cshift: Value of the constant shift (required if infl_cshift=True)
    :param infl_fshift: Whether to apply functional shift in influence
    :param Q: Scaling factor for functional shift (required if infl_fshift=True)
    :param tolerance: Tolerance for convergence
    :param tolerated_agents: Number of agents that need to meet tolerance before stopping
    :param parameters: Parameters for influence kernels of each agent
    :return: Dictionary of validated and converted parameters
    :raises ValueError: If any configuration is invalid
    :raises TypeError: If input types are incorrect
    :raises NotImplementedError: If unsupported functionality is requested
    """
    
    
    
    validated = {}
    
    # 1. Validate num_agents (first parameter)
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError(f"Number of agents must be a positive integer, got {num_agents}")
    validated['num_agents'] = num_agents
    
    # 2. Validate and convert agents_pos (second parameter)
    agents_pos = _to_tensor(agents_pos, "agents_pos")
    if len(agents_pos) != num_agents:
        raise ValueError(f"agents_pos must be a tensor with {num_agents} elements, got shape {agents_pos.shape}")
    validated['agents_pos'] = agents_pos
    
    # 3. Validate and convert parameters (third parameter)
    if parameters is not None:
        parameters = _to_tensor(parameters, "parameters")
        if len(parameters) != num_agents:
            raise ValueError(f"parameters must be a tensor with {num_agents} elements")
        if not torch.all(torch.isfinite(parameters)):
            raise ValueError("parameters must contain finite values (no NaN or Inf)")
        if torch.any(parameters < 0) and infl_configs.get('infl_type') in ['gaussian', 'multi_gaussian','dirichlet']:
            warnings.warn("Parameters with negative values detected, this may result in unpredictable behavior", UserWarning)
        validated['parameters'] = parameters
    
    # 4. Validate and convert resource_distribution (fourth parameter)
    resource_distribution = _to_tensor(resource_distribution, "resource_distribution")
    if not torch.all(torch.isfinite(resource_distribution)):
        raise ValueError("resource_distribution must contain finite values (no NaN or Inf)")
    if torch.any(resource_distribution < 0):
        warnings.warn("Non-negative values detected in resource_distribution, this may result in unpredictable behavior", UserWarning)
    validated['resource_distribution'] = resource_distribution
    
    # 5. Validate and convert bin_points (fifth parameter)
    bin_points = _to_tensor(bin_points, "bin_points")
    if  len(bin_points) == 0:
        raise ValueError("bin_points must be a non-empty tensor")
    if len(resource_distribution) != len(bin_points):
        raise ValueError(f"resource_distribution length ({len(resource_distribution)}) must match bin_points length ({len(bin_points)})")
    validated['bin_points'] = bin_points
    
    # 6. Validate infl_configs (sixth parameter)
    if not isinstance(infl_configs, dict):
        raise TypeError("infl_configs must be a dictionary")
    
    valid_infl_types = ['gaussian', 'Jones_M', 'dirichlet', 'multi_gaussian', 'custom']
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
    valid_lr_types = ['cosine_annealing', 'fixed']
    if learning_rate_type not in valid_lr_types:
        raise ValueError(f"Invalid learning rate type '{learning_rate_type}'. Supported types are {valid_lr_types}")
    validated['learning_rate_type'] = learning_rate_type
    
    # 8. Validate and convert learning_rate (eighth parameter)
    learning_rate = _to_tensor(learning_rate, "learning_rate")
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
        cshift = _to_tensor(cshift, "cshift")
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
        domain_bounds = _to_tensor(domain_bounds, "domain_bounds")
        if domain_bounds.dim() != 1 or len(domain_bounds) != 2:
            raise ValueError("domain_bounds must be a 1D tensor with exactly 2 elements [min, max]")
        if domain_bounds[0] >= domain_bounds[1]:
            raise ValueError(f"domain_bounds must have min < max, got {domain_bounds.tolist()}")

    elif domain_type == '2d':
        domain_bounds = _to_tensor(domain_bounds, "domain_bounds")
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
    

    return validated



    
