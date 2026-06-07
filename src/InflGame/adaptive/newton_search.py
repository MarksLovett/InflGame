"""
.. module:: root_finding
   :synopsis: Provides Newton-method-based root finding tools for locating equilibria in adaptive dynamics for influencer games.


Root Finding Module
===================

This module provides Newton's method and related numerical root-finding algorithms for locating equilibrium
points in adaptive dynamics systems. It includes trust region and adaptive step methods for robust convergence,
as well as parallel grid search capabilities for exploring multiple initial conditions.

The module is designed to work with the `AdaptiveEnv` class and provides efficient methods for finding
gradient zeros, which correspond to equilibria in influencer game environments.

Dependencies:
-------------
- InflGame.utils.validation
- InflGame.adaptive.jacobian
- NumPy, PyTorch
- concurrent.futures (for parallel processing)

Usage:
------
The `newton_method` class can be used to find equilibrium positions by solving for gradient zeros
using Newton's method with various step control strategies. It supports parallel grid searches for
systematic exploration of the equilibrium landscape.

Example:
--------

.. code-block:: python
    
    from InflGame.adaptive.root_finding import newton_method
    from InflGame.adaptive.grad_func_env import AdaptiveEnv
    import torch
    import numpy as np

    # Initialize adaptive environment
    field = AdaptiveEnv(
        num_agents=3,
        agents_pos=np.array([0.2, 0.5, 0.8]),
        parameters=torch.tensor([0.1, 0.1, 0.1]),
        resource_distribution=torch.tensor([10.0, 20.0, 30.0]),
        bin_points=np.array([0.1, 0.4, 0.7]),
        infl_configs={'infl_type': 'gaussian'},
        domain_type='1d',
        domain_bounds=[0, 1]
    )

    # Initialize Newton method
    newton = newton_method(
        field=field,
        num_agents=3,
        agents_pos=np.array([0.2, 0.5, 0.8]),
        parameters=torch.tensor([0.1, 0.1, 0.1]),
        resource_distribution=torch.tensor([10.0, 20.0, 30.0]),
        bin_points=np.array([0.1, 0.4, 0.7]),
        infl_configs={'infl_type': 'gaussian'},
        domain_type='1d',
        domain_bounds=[0, 1]
    )

    # Find equilibrium
    initial_guess = torch.tensor([0.3, 0.5, 0.7])
    result = newton.newton_root_finder(
        initial_guess=initial_guess,
        tolerance=1e-6,
        method='trust_region'
    )
    
    if result['converged']:
        print(f"Found equilibrium at: {result['final_position']}")
"""

import numpy as np
import torch
from typing import Union, List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import itertools

import InflGame.utils.validation as validation
import InflGame.adaptive.jacobian as jc


class newton_method:
    """
    Newton's method for finding gradient zeros in adaptive dynamics.
    
    This class implements Newton's method with various step control strategies (trust region,
    adaptive stepping) for robustly finding equilibrium points in adaptive dynamics systems.
    Equilibria correspond to points where the gradient of the agents' objective functions
    equals zero, which represent Nash equilibria in the influencer game framework.
    
    The class supports:
    
    - Multiple step control methods (trust region, adaptive)
    - Parallel grid search over initial conditions
    - Stagnation detection and handling
    - Bounded optimization for constrained domains
    - Comprehensive convergence diagnostics
    
    This class is designed to work with :class:`InflGame.adaptive.grad_func_env.AdaptiveEnv`
    and uses Jacobian matrices computed by :mod:`InflGame.adaptive.jacobian` for Newton steps.
    """

    def __init__(self,
                 field: object,
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
                 ignore_zero_infl: bool = False,
                 ) -> None:
        """
        Initialize the newton_method class with configuration parameters.

        :param field: Reference to an AdaptiveEnv instance containing the gradient function.
        :type field: object
        :param num_agents: Number of agents in the environment.
        :type num_agents: int
        :param agents_pos: Initial positions of agents.
        :type agents_pos: Union[List[float], np.ndarray]
        :param parameters: Parameters for the influence kernel (e.g., reach, variance).
        :type parameters: torch.Tensor
        :param resource_distribution: Distribution of resources across the domain.
        :type resource_distribution: torch.Tensor
        :param bin_points: Bin points defining resource allocation regions.
        :type bin_points: Union[List[float], np.ndarray]
        :param infl_configs: Configuration dictionary for influence kernel type and parameters.
            - ``infl_type`` (str): The type of influence kernel (e.g., "gaussian", "multi_gaussian", "dirichlet", "beta", "custom").
            - ``custom_influence`` (callable): Function for a custom influence kernel (optional).
        :type infl_configs: Dict[str, Union[str, callable]]
        :param learning_rate_type: Type of learning rate schedule (e.g., 'cosine').
        :type learning_rate_type: str
        :param learning_rate: Learning rate parameters [min_lr, max_lr, annealing_period].
        :type learning_rate: List[float]
        :param time_steps: Maximum number of gradient ascent iterations.
        :type time_steps: int
        :param fp: Whether to use fixed point analysis.
        :type fp: Optional[int]
        :param infl_cshift: Whether to apply constant shift to influence.
        :type infl_cshift: bool
        :param cshift: Constant shift value.
        :type cshift: int
        :param infl_fshift: Whether to apply frequency shift to influence.
        :type infl_fshift: bool
        :param Q: Covariance matrix parameter for multivariate Gaussian kernels.
        :type Q: int
        :param domain_type: Type of domain ('1d', '2d', or 'simplex').
        :type domain_type: str
        :param domain_bounds: Bounds of the domain.
        :type domain_bounds: Union[List[float], torch.Tensor]
        :param tolerance: Convergence tolerance for root finding.
        :type tolerance: float
        :param tolerated_agents: Number of agents allowed to violate tolerance before convergence.
        :type tolerated_agents: Optional[int]
        :param ignore_zero_infl: Whether to ignore agents with zero influence.
        :type ignore_zero_infl: bool
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
        
        self.field = field
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
        self.tolerance = validated['tolerance']
        self.tolerated_agents = validated['tolerated_agents']
        self.ignore_zero_infl = ignore_zero_infl

    def newton_root_finder(self,
                          initial_guess,
                          tolerance=None,
                          max_iter=20000,
                          verbose=True,
                          enforce_bounds=True,
                          method='trust_region',
                          tolerated_agents=None,
                          stagnation_window=50,
                          stagnation_tolerance=1e-5,
                          return_detailed_history=False,
                          tolerance_grad=1e-5
                          ) -> Dict[str, Union[torch.Tensor, bool, List]]:
        """
        Newton's method for finding gradient zeros with trust region or adaptive stepping.
        
        This method implements Newton's method to find equilibrium points by solving for positions
        where the gradient equals zero. It uses the Jacobian matrix (computed via 
        :func:`InflGame.adaptive.jacobian.jacobian_matrix`) to determine Newton steps and includes
        sophisticated step control mechanisms for robust convergence.
        
        The algorithm:
        
        1. Computes gradient at current position
        2. Computes Jacobian matrix with stability checks
        3. Solves linear system for Newton direction
        4. Applies step control (trust region or adaptive)
        5. Enforces domain bounds if specified
        6. Checks convergence criteria
        7. Detects and handles stagnation
        
        **Step Control Methods:**
        
        - **trust_region**: Limits step size to maximum radius, prevents overshooting
        - **adaptive**: Tries multiple step sizes, accepts one that reduces gradient norm
        
        :param initial_guess: Initial agent positions from which to start Newton iteration.
        :type initial_guess: torch.Tensor
        :param tolerance: Convergence tolerance for position changes (default: instance tolerance).
        :type tolerance: Optional[float]
        :param max_iter: Maximum number of Newton iterations.
        :type max_iter: int
        :param verbose: Whether to print progress information during iteration.
        :type verbose: bool
        :param enforce_bounds: Whether to clamp positions to domain bounds.
        :type enforce_bounds: bool
        :param method: Step control method ('trust_region' or 'adaptive').
        :type method: str
        :param tolerated_agents: Number of agents that must satisfy convergence criteria.
        :type tolerated_agents: Optional[int]
        :param stagnation_window: Number of iterations to check for lack of improvement.
        :type stagnation_window: int
        :param stagnation_tolerance: Minimum improvement required to avoid stagnation detection.
        :type stagnation_tolerance: float
        :param return_detailed_history: If True, return full position/gradient history; if False, save memory.
        :type return_detailed_history: bool
        :param tolerance_grad: Convergence tolerance for gradient magnitude.
        :type tolerance_grad: float
        
        :return: Dictionary containing:
                 - 'final_position': Converged position
                 - 'converged': Whether convergence criteria were met
                 - 'final_gradient_norm': Gradient norm at final position
                 - 'iterations': Number of iterations performed
                 - 'termination_reason': Why iteration stopped ('converged', 'stagnation', 'max_iterations')
                 - 'position_matrix': (if return_detailed_history=True) Full position history
                 - 'gradient_history': (if return_detailed_history=True) Full gradient history
        :rtype: Dict[str, Union[torch.Tensor, bool, List]]
        
        :raises RuntimeError: If linear solve fails during Newton step computation.
        
        Example:
        --------
        
        .. code-block:: python
        
            initial_guess = torch.tensor([0.3, 0.5, 0.7])
            result = newton.newton_root_finder(
                initial_guess=initial_guess,
                tolerance=1e-6,
                max_iter=1000,
                method='trust_region',
                verbose=True
            )
            
            if result['converged']:
                equilibrium = result['final_position']
                print(f"Equilibrium found: {equilibrium}")
                print(f"Gradient norm: {result['final_gradient_norm']:.2e}")
        """
        if tolerated_agents is None:
            tolerated_agents = self.num_agents
            
        # Preserve original state
        original_pos = self.field.agents_pos.clone()
        
        # Set initial position and tolerance
        self.field.agents_pos = initial_guess.clone()
        if tolerance is None:
            tolerance = self.tolerance
        
        # Cache domain bounds
        domain_bounds = self.field.domain_bounds
        lower_bound = domain_bounds[0] + 1e-6
        upper_bound = domain_bounds[1] - 1e-6
        
        # Pre-clamp initial position if enforcing bounds
        if enforce_bounds:
            self.field.agents_pos = torch.clamp(self.field.agents_pos, lower_bound, upper_bound)
        
        # Track positions and gradients
        position_matrix = [self.field.agents_pos.clone()]
        gradient_history = []
        
        # Stagnation detection
        gradient_norms = []
        position_changes = []
        stagnation_detected = False
        
        if verbose:
            print(f"Starting Newton root finder from: {self.field.agents_pos}")
            print(f"Method: {method}, Tolerance: {tolerance:.2e}")
            print(f"Stagnation detection: window={stagnation_window}, tolerance={stagnation_tolerance:.2e}")
            print("-" * 50)
        
        # Convergence tracking
        prev_pos = self.field.agents_pos.clone()
        converged = False
        best_grad_norm = float('inf')
        
        for i in range(max_iter):
            # Compute gradient
            current_grad = self.field.gradient(parameter_instance=self.field.parameters)
            gradient_history.append(current_grad.clone())
            grad_norm = torch.norm(current_grad).item()
            gradient_norms.append(grad_norm)
            
            
            
            # Compute Jacobian and Newton direction
            try:
                J = jc.jacobian_matrix(
                    parameters=self.field.parameters,
                    agents_pos=self.field.agents_pos,
                    infl_matrix=self.field.influence_matrix(parameter_instance=self.field.parameters),
                    prob_matrix=self.field.prob_matrix(parameter_instance=self.field.parameters),
                    d_lnf_matrix=self.field.d_lnf_matrix(parameter_instance=self.field.parameters),
                    num_agents=self.num_agents,
                    bin_points=self.bin_points,
                    resource_distribution=self.field.resource_distribution,
                    infl_type=self.field.infl_type,
                    infl_fshift=self.field.infl_fshift,
                    Q=self.field.Q
                )
                
                J_tensor = J.float() if isinstance(J, torch.Tensor) else torch.tensor(J, dtype=torch.float32)
                grad_tensor = current_grad.float()
                
                # Check condition number for numerical stability
                condition_number = torch.linalg.cond(J_tensor).item()
                
                # Solve for Newton direction with regularization if needed
                if condition_number > 1e12:
                    if verbose and i % 100 == 0:
                        print(f"  ⚠️ Poor conditioning: {condition_number:.2e}, adding regularization")
                    regularization = 1e-8 * torch.eye(J_tensor.shape[0], dtype=J_tensor.dtype)
                    newton_direction = torch.linalg.solve(J_tensor + regularization, grad_tensor)
                else:
                    newton_direction = torch.linalg.solve(J_tensor, grad_tensor)
                
            except RuntimeError as e:
                if verbose:
                    print(f"❌ Linear solve failed at iteration {i}: {e}")
                break
            
            # Compute new position with step size control
            if method == 'trust_region':
                new_pos = self._trust_region_step(self.field.agents_pos, newton_direction, current_grad)
            else:  # adaptive method
                new_pos = self._adaptive_step(self.field.agents_pos, newton_direction, current_grad)
            
            # Enforce bounds
            if enforce_bounds:
                new_pos = torch.clamp(new_pos, lower_bound, upper_bound)
            
            self.field.agents_pos = new_pos
            position_matrix.append(new_pos.clone())
            
            # Track position changes for stagnation detection
            position_change = torch.norm(new_pos - prev_pos).item()
            position_changes.append(position_change)
            
            # User's requested convergence condition
            if i > 10:
                try:
                    comparison_step = max(0, i - 10)
                    position_diff = new_pos - position_matrix[comparison_step]
                    abs_differences = torch.abs(position_diff)
                    converged_agents = torch.sum(abs_differences <= tolerance).item()
                    
                    if converged_agents >= tolerated_agents and torch.sum(torch.abs(grad_tensor) < tolerance_grad).item() >= tolerated_agents:
                        converged = True
                        if verbose:
                            print(f"✅ Converged at iteration {i}: {converged_agents}/{self.num_agents} agents within tolerance")
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Convergence check error at step {i}: {str(e)}")
            if grad_norm < best_grad_norm:
                best_grad_norm = grad_norm
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= 20 and i > 100:
                    if verbose:
                        print(f"⚠️ Stagnation detected at iteration {i}: no improvement for {stagnation_counter} steps")
                    break

            # Progress reporting
            if verbose and (i % 50 == 0 or grad_norm < tolerance * 10):
                print(f"Iter {i:4d}: grad_norm={grad_norm:.2e}, pos_change={position_change:.2e}")
                if i >= stagnation_window:
                    recent_grad_improvement = max(gradient_norms[-stagnation_window:]) - min(gradient_norms[-stagnation_window:])
                    print(f"         grad_improvement={recent_grad_improvement:.2e} (window={stagnation_window})")
            
            prev_pos = new_pos.clone()
        
        else:
            if verbose:
                print(f"❌ Max iterations ({max_iter}) reached")
        
        # Determine termination reason
        
        if converged:
            termination_reason = 'converged'
        elif stagnation_detected:
            termination_reason = 'stagnation'
        else:
            termination_reason = 'max_iterations'
        
        # Restore original state
        self.field.agents_pos = original_pos
        
        # Prepare return dictionary
        result = {
            'final_position': new_pos.clone(),
            'converged': converged,
            'final_gradient_norm': grad_norm,
            'iterations': i + 1,
            'termination_reason': termination_reason
        }
        
        # Conditionally add detailed history data
        if return_detailed_history:
            result.update({
                'position_matrix': torch.stack(position_matrix),
                'gradient_history': gradient_history,
                'gradient_norms': gradient_norms,
                'position_changes': position_changes
            })
        else:
            # Clear memory by deleting large tracking arrays
            del position_matrix
            del gradient_history
            del gradient_norms
            del position_changes
            # Force garbage collection if needed
            import gc
            gc.collect()
        
        return result

    def _trust_region_step(self, current_pos, newton_direction, current_grad, max_radius=0.1):
        """
        Compute trust region constrained Newton step.
        
        Limits the Newton step to a maximum radius to prevent overshooting and ensure
        stability, particularly when far from the solution or in poorly conditioned regions.
        
        :param current_pos: Current agent positions.
        :type current_pos: torch.Tensor
        :param newton_direction: Computed Newton direction from Jacobian solve.
        :type newton_direction: torch.Tensor
        :param current_grad: Current gradient vector.
        :type current_grad: torch.Tensor
        :param max_radius: Maximum allowed step size.
        :type max_radius: float
        
        :return: New position after trust region step.
        :rtype: torch.Tensor
        """
        step_norm = torch.norm(newton_direction).item()
        if step_norm > max_radius:
            newton_direction = newton_direction * (max_radius / step_norm)
        return current_pos - newton_direction

    def _adaptive_step(self, current_pos, newton_direction, current_grad, max_step=0.1):
        """
        Compute adaptive Newton step with backtracking line search.
        
        Tries multiple step sizes and selects the one that achieves gradient norm reduction.
        This ensures progress toward the solution at each iteration.
        
        :param current_pos: Current agent positions.
        :type current_pos: torch.Tensor
        :param newton_direction: Computed Newton direction from Jacobian solve.
        :type newton_direction: torch.Tensor
        :param current_grad: Current gradient vector.
        :type current_grad: torch.Tensor
        :param max_step: Maximum initial step size to try.
        :type max_step: float
        
        :return: New position after adaptive step.
        :rtype: torch.Tensor
        """
        original_pos = self.field.agents_pos.clone()
        current_grad_norm = torch.norm(current_grad).item()
        
        # Try different step sizes
        for step_size in [max_step, max_step/2, max_step/10]:
            new_pos = current_pos - step_size * (newton_direction / torch.norm(newton_direction))
            
            # Check if we made progress
            self.field.agents_pos = new_pos
            new_grad = self.field.gradient(parameter_instance=self.field.parameters)
            new_grad_norm = torch.norm(new_grad).item()
            
            if new_grad_norm < current_grad_norm:
                self.field.agents_pos = original_pos
                return new_pos
        
        # If no step worked, take a small step
        self.field.agents_pos = original_pos
        return current_pos - 0.001 * (newton_direction / torch.norm(newton_direction))

    def grid_search_newton_hyperplane_xyz(self,
                                      grid_points_per_dim=5,
                                      bounds=(0, 1),
                                      max_workers=None,
                                      verbose=True,
                                      position_tolerance=1e-4,
                                      **newton_kwargs) -> Dict:
        """
        Perform parallel grid search Newton optimization for symmetric N-player equilibria.
        
        This method systematically explores the equilibrium landscape by running Newton's method
        from a grid of initial positions with a symmetric structure. The position format constrains
        middle agents to have the same position, which is appropriate for symmetric influence games.
        
        **Position Format:**
        
        For N agents (N ≥ 3):
        
        - Agent 1: position x (varies independently)
        - Agents 2 to N-1: all share position y (vary together)
        - Agent N: position z (varies independently)
        
        This gives a 3D grid search over (x, y, z) regardless of the number of agents.
        
        **Algorithm:**
        
        1. Generate grid of (x, y, z) combinations
        2. For each grid point, create initial position vector (x, y, y, ..., y, z)
        3. Run Newton's method from each initial position (in parallel)
        4. Collect and deduplicate converged equilibria
        5. Compute statistics on convergence behavior
        
        :param grid_points_per_dim: Number of grid points per dimension (x, y, z).
        :type grid_points_per_dim: int
        :param bounds: Tuple of (min, max) bounds for grid coordinates.
        :type bounds: Tuple[float, float]
        :param max_workers: Number of parallel workers (defaults to CPU count).
        :type max_workers: Optional[int]
        :param verbose: Whether to print progress information.
        :type verbose: bool
        :param position_tolerance: Tolerance for considering two positions as unique.
        :type position_tolerance: float
        :param **newton_kwargs: Additional arguments passed to :meth:`newton_root_finder`.
        
        :return: Dictionary containing:
                 - 'all_results': All Newton trial results
                 - 'unique_final_positions': List of unique equilibria found
                 - 'successful_results': All converged positions (including duplicates)
                 - 'statistics': Convergence statistics (mean/std gradient norms, iterations)
                 - 'convergence_rate': Fraction of grid points that converged
                 - 'grid_info': Grid configuration details
        :rtype: Dict
        
        :raises ValueError: If num_agents < 3.
        
        Example:
        --------
        
        .. code-block:: python
        
            # Find equilibria for 4-player game
            results = newton.grid_search_newton_4player_xyz(
                grid_points_per_dim=10,
                bounds=(0, 1),
                max_workers=8,
                verbose=True,
                tolerance=1e-6,
                max_iter=1000,
                method='trust_region'
            )
            
            print(f"Found {results['unique_points']} unique equilibria")
            for i, eq in enumerate(results['unique_final_positions']):
                print(f"Equilibrium {i+1}: {eq}")
        """
        try:
            num_agents = self.num_agents
            
            if num_agents < 3:
                raise ValueError(f"This method requires at least 3 agents, got {num_agents}")
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if max_workers is None:
            max_workers = min(mp.cpu_count(), grid_points_per_dim**3)  # 3 independent variables: x, y, z
        
        min_bound, max_bound = bounds
        
        # Create grid points for each dimension
        grid_1d = np.linspace(min_bound, max_bound, grid_points_per_dim)
        
        if verbose:
            total_combinations = grid_points_per_dim**3  # Only x, y, z are independent
            print(f"🔍 Starting {num_agents}-Player (x,y,...,y,z) Grid Search Newton optimization:")
            print(f"   Grid points per dim: {grid_points_per_dim}")
            print(f"   Position format: (x, y×{num_agents-2}, z)")
            print(f"   Bounds: [{min_bound}, {max_bound}]")
            print(f"   Total grid combinations: {total_combinations}")
            print(f"   Workers: {max_workers}")
            print(f"   Position tolerance: {position_tolerance}")
            print(f"   Newton args: {newton_kwargs}")
            print("-" * 60)
        
        # Generate all grid combinations for initial positions
        initial_conditions = []
        grid_id = 0
        
        # Generate grid for (x, y, z) where positions are (x, y, y, ..., y, z)
        for x, y, z in itertools.product(grid_1d, grid_1d, grid_1d):
            # Create N-player position vector: (x, y, y, ..., y, z)
            # First agent gets x, last agent gets z, all middle agents get y
            init_pos = torch.tensor([x] + [y]*(num_agents-2) + [z], dtype=torch.float32)
            grid_coords = (x, y, z)  # Store the independent coordinates
            initial_conditions.append((grid_id, init_pos, grid_coords))
            grid_id += 1
        
        total_grid_points = len(initial_conditions)
        
        # Parallel execution
        all_results = []
        successful_results = []
        failed_count = 0
        
        start_time = time.time()
        
        if max_workers == 1 or total_grid_points == 1:
            # Sequential execution for debugging or single point
            for grid_id, init_pos, grid_coords in initial_conditions:
                try:
                    result = self._run_single_newton_trial(grid_id, init_pos, newton_kwargs, verbose and total_grid_points <= 25)
                    result['grid_coordinates'] = grid_coords
                    # Format string showing the position pattern
                    if num_agents == 3:
                        result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[2]:.3f})"
                    elif num_agents == 4:
                        result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[1]:.3f}, {grid_coords[1]:.3f}, {grid_coords[2]:.3f})"
                    else:
                        result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[1]:.3f}×{num_agents-2}, {grid_coords[2]:.3f})"
                    all_results.append(result)
                    
                    if result['converged']:
                        successful_results.append(result['final_position'])
                        
                except Exception as e:
                    failed_count += 1
                    if verbose:
                        print(f"❌ Grid point {grid_id} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        
                if verbose and (grid_id + 1) % max(1, total_grid_points // 10) == 0:
                    print(f"Progress: {grid_id + 1}/{total_grid_points} grid points completed")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_grid = {
                    executor.submit(self._run_single_newton_trial, grid_id, init_pos, newton_kwargs, False): (grid_id, grid_coords)
                    for grid_id, init_pos, grid_coords in initial_conditions
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_grid):
                    grid_id, grid_coords = future_to_grid[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        result['grid_coordinates'] = grid_coords
                        # Format string showing the position pattern
                        if num_agents == 3:
                            result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[2]:.3f})"
                        elif num_agents == 4:
                            result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[1]:.3f}, {grid_coords[1]:.3f}, {grid_coords[2]:.3f})"
                        else:
                            result['position_format'] = f"({grid_coords[0]:.3f}, {grid_coords[1]:.3f}×{num_agents-2}, {grid_coords[2]:.3f})"
                        all_results.append(result)
                        
                        if result['converged']:
                            successful_results.append(result['final_position'])
                            
                    except Exception as e:
                        failed_count += 1
                        if verbose:
                            print(f"❌ Grid point {grid_id} failed: {e}")
                    
                    # Progress reporting
                    if verbose and completed % max(1, total_grid_points // 10) == 0:
                        print(f"Progress: {completed}/{total_grid_points} grid points completed")
        
        elapsed_time = time.time() - start_time
        
        # Find unique final positions from successful results
        def is_position_unique(pos, unique_positions, tolerance):
            """Check if a position is unique compared to existing positions."""
            for existing_pos in unique_positions:
                if torch.norm(pos - existing_pos) < tolerance:
                    return False
            return True
        
        unique_final_positions = []
        for pos in successful_results:
            if is_position_unique(pos, unique_final_positions, position_tolerance):
                unique_final_positions.append(pos)
        
        # Sort unique positions by the second agent's position (index 1)
        if unique_final_positions:
            unique_final_positions.sort(key=lambda pos: pos[1].item())
        
        # Compile statistics
        converged_count = len(successful_results)
        unique_count = len(unique_final_positions)
        convergence_rate = converged_count / total_grid_points if total_grid_points > 0 else 0
        
        # Get converged results from all_results for statistics
        converged_results = [r for r in all_results if r.get('converged', False)]
        
        if converged_results:
            grad_norms = [r['final_gradient_norm'] for r in converged_results]
            iterations = [r['iterations'] for r in converged_results]
            
            stats = {
                'mean_grad_norm': np.mean(grad_norms),
                'std_grad_norm': np.std(grad_norms),
                'min_grad_norm': np.min(grad_norms),
                'max_grad_norm': np.max(grad_norms),
                'mean_iterations': np.mean(iterations),
                'std_iterations': np.std(iterations),
                'min_iterations': np.min(iterations),
                'max_iterations': np.max(iterations)
            }
        else:
            stats = {}
        
        if verbose:
            print(f"\n🎯 {num_agents}-Player (x,y,...,y,z) Grid Search Complete:")
            print(f"   Time: {elapsed_time:.2f}s")
            print(f"   Grid points: {total_grid_points}")
            print(f"   Converged: {converged_count}/{total_grid_points} ({convergence_rate:.1%})")
            print(f"   Unique solutions: {unique_count}")
            print(f"   Failed: {failed_count}")
        
        return {
            'all_results': all_results,
            'unique_final_positions': unique_final_positions,
            'successful_results': successful_results,
            'statistics': stats,
            'convergence_rate': convergence_rate,
            'total_grid_points': total_grid_points,
            'converged_points': converged_count,
            'unique_points': unique_count,
            'failed_points': failed_count,
            'elapsed_time': elapsed_time,
            'grid_info': {
                'grid_points_per_dim': grid_points_per_dim,
                'position_format': f'(x, y×{num_agents-2}, z)',
                'num_agents': num_agents,
                'bounds': bounds,
                'grid_1d': grid_1d.tolist(),
                'independent_variables': ['x', 'y', 'z'],
                'position_tolerance': position_tolerance
            }
        }


    def _run_single_newton_trial(self, trial_id, initial_pos, newton_kwargs, verbose_trial=False):
        """
        Run a single Newton trial from given initial position.
        
        This is a helper function designed for parallel execution. It wraps :meth:`newton_root_finder`
        with error handling and result formatting suitable for grid search applications.
        
        :param trial_id: Unique identifier for this trial.
        :type trial_id: int
        :param initial_pos: Initial agent positions for this trial.
        :type initial_pos: torch.Tensor
        :param newton_kwargs: Keyword arguments to pass to newton_root_finder.
        :type newton_kwargs: Dict
        :param verbose_trial: Whether to print verbose output for this trial.
        :type verbose_trial: bool
        
        :return: Dictionary with trial results including trial_id, initial_position, and convergence info.
        :rtype: Dict
        """
        try:
            # Make a copy of newton_kwargs to avoid modifying the original
            kwargs_copy = newton_kwargs.copy()
            
            # Set return_detailed_history to False by default to save memory
            if 'return_detailed_history' not in kwargs_copy:
                kwargs_copy['return_detailed_history'] = False
            
            # Filter out any kwargs that aren't valid parameters for newton_root_finder
            valid_params = {
                'tolerance', 'max_iter', 'verbose', 'enforce_bounds', 'method',
                'tolerated_agents', 'stagnation_window', 'stagnation_tolerance',
                'return_detailed_history', 'tolerance_grad'
            }
            
            filtered_kwargs = {k: v for k, v in kwargs_copy.items() if k in valid_params}
            
            result = self.newton_root_finder(initial_pos, verbose=verbose_trial, **filtered_kwargs)
            result['trial_id'] = trial_id
            result['initial_position'] = initial_pos.clone()
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return {
                'trial_id': trial_id,
                'initial_position': initial_pos.clone(),
                'converged': False,
                'error': str(e),
                'error_trace': error_trace,
                'final_gradient_norm': float('inf'),
                'iterations': 0,
                'termination_reason': 'error'
            }

    # Convenience methods
    def newton_trust_region(self, initial_guess, **kwargs):
        """
        Convenience method for trust region Newton optimization.
        
        Calls :meth:`newton_root_finder` with method='trust_region'.
        
        :param initial_guess: Initial agent positions.
        :type initial_guess: torch.Tensor
        :param **kwargs: Additional arguments passed to newton_root_finder.
        
        :return: Newton method results dictionary.
        :rtype: Dict
        """
        return self.newton_root_finder(initial_guess, method='trust_region', **kwargs)

    def newton_adaptive(self, initial_guess, **kwargs):
        """
        Convenience method for adaptive step Newton optimization.
        
        Calls :meth:`newton_root_finder` with method='adaptive'.
        
        :param initial_guess: Initial agent positions.
        :type initial_guess: torch.Tensor
        :param **kwargs: Additional arguments passed to newton_root_finder.
        
        :return: Newton method results dictionary.
        :rtype: Dict
        """
        return self.newton_root_finder(initial_guess, method='adaptive', **kwargs)

    def grid_search_newton_4player_xyz_over_reach(self,
                                                    reach_parameters: Union[List[float], np.ndarray],
                                                    tolerance: float,
                                                    tolerated_agents: int,
                                                    percentage: float = 1.0,
    ) -> None:
        """
        Grid search Newton optimization over a range of reach parameters.
        
        .. warning::
           This method is a placeholder and is not yet implemented.
        
        :param reach_parameters: Array of reach/influence parameter values to iterate over.
        :type reach_parameters: Union[List[float], np.ndarray]
        :param tolerance: Convergence tolerance for Newton method.
        :type tolerance: float
        :param tolerated_agents: Number of agents that must converge.
        :type tolerated_agents: int
        :param percentage: Percentage parameter (usage to be determined).
        :type percentage: float
        
        :return: None (placeholder)
        :rtype: None
        
        :raises NotImplementedError: This method is not yet implemented.
        """
        # Input validation
        if tolerance == None:
            tolerance = self.tolerance
        if tolerated_agents == None:
            tolerated_agents = self.tolerated_agents
            
        if not isinstance(reach_parameters, (list, np.ndarray, torch.Tensor)):
            raise ValueError(f"reach_parameters must be list, numpy array, or torch tensor, got {type(reach_parameters)}")
        
        if isinstance(reach_parameters, list):
            reach_parameters = np.array(reach_parameters)
        elif isinstance(reach_parameters, torch.Tensor):
            reach_parameters = reach_parameters.numpy()
            
        if len(reach_parameters) == 0:
            raise ValueError("reach_parameters cannot be empty")
            
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
            
        if tolerated_agents < 0:
            raise ValueError(f"tolerated_agents must be non-negative, got {tolerated_agents}")
        
        if not (0.0 < percentage <= 1.0):
            raise ValueError(f"percentage must be between 0.0 and 1.0, got {percentage}")
        
        # Validate domain type
        if self.domain_type not in ['1d', '2d', 'simplex']:
            raise ValueError(f"Unsupported domain_type: {self.domain_type}")