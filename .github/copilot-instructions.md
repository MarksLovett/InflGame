# Influencer Games Project - AI Coding Instructions

## Architecture Overview

This is a research framework for studying spatial influence in multi-player resource competition games using two main approaches:

1. **Adaptive Dynamics** (`src/InflGame/adaptive/`) - Gradient ascent optimization of agent positions
2. **Multi-Agent Reinforcement Learning** (`src/InflGame/MARL/`) - Q-learning and deep RL using RayRL

## Core Components

### Primary Classes and Workflow
- **`Shell`** (`src/InflGame/adaptive/visualization.py`) - Main user interface for adaptive dynamics experiments
  - Creates and manages `AdaptiveEnv` via `setup_adaptive_env()`
  - Provides visualization and analysis methods (bifurcation plots, histograms, vector fields)
  - Handles parallelization for performance-critical operations
- **`AdaptiveEnv`** (`src/InflGame/adaptive/grad_func_env.py`) - Core engine for gradient ascent dynamics
  - `gradient_ascent()` - Main optimization loop with convergence checking
  - `mv_gradient_ascent()` vs `sv_gradient_ascent()` - Multi-variate vs single-variate domains

### Domain Types (Critical Architecture Decision)
The system supports three domain types that fundamentally change computation:
- **`'1d'`** - Line segment, uses numpy arrays
- **`'2d'`** - Rectangle, uses torch tensors  
- **`'simplex'`** - Triangle with barycentric coordinates, special projection operations

**Key Pattern**: Always check `domain_type` before operations - different domains use different data structures and coordinate systems.

### Influence Kernels (`src/InflGame/kernels/`)
- **`gaussian`** - Most common, uses `gaussian_infl()`
- **`multi_gaussian`** - Multivariate Gaussian with covariance matrices
- **`dirichlet`** - For simplex domains specifically
- **`custom`** - User-defined functions with torch autograd integration

## Critical Development Patterns

### State Management
```python
# Always preserve original state before modification
original_pos = self.agents_pos.clone()  # Use .clone() for torch tensors
original_pos = self.agents_pos.copy()   # Use .copy() for numpy arrays

# Restore after operations
self.agents_pos = original_pos
```

### Convergence and Tolerance
- **`tolerance`** (default: `10**-5`) - Position change threshold for convergence
- **`tolerated_agents`** - Number of agents that must meet tolerance before stopping
- **`time_steps`** - Maximum iterations before forced termination

### Parallelization Strategy
- Use `ProcessPoolExecutor` for CPU-intensive gradient ascent operations
- Always provide sequential fallback due to deep copy overhead
- Include progress reporting for long-running computations
- Example in `plot_3d_fixed_diagonal_view()` method

### Data Flow Pattern
1. **Setup**: `Shell.__init__()` → `setup_adaptive_env()` → creates `AdaptiveEnv`
2. **Compute**: `field.gradient_ascent()` → stores results in `pos_matrix`, `grad_matrix`
3. **Analyze**: Access stored matrices for plotting/analysis

## Performance Optimization

### Memory Management
- Use `torch.tensor.clone()` for safe copying
- Clear large matrices with `self.field.pos_matrix = 0` between runs
- Monitor memory in batch processing operations

### Learning Rate Scheduling
```python
# Common pattern for adaptive learning rates
learning_rate = [initial_lr, min_lr, decay_steps]
lr_type = 'cosine_annealing'  # or 'cosine', 'linear'
```

## Key Files for Understanding

- `src/InflGame/adaptive/visualization.py` - Main user interface and plotting
- `src/InflGame/adaptive/grad_func_env.py` - Core gradient ascent engine
- `src/InflGame/utils/general.py` - Parameter setup utilities
- `demo/paper_kernels/` - Complete working examples
- `src/InflGame/domains/*/` - Domain-specific implementations

## Testing and Debugging

### Common Debug Patterns
```python
# Test single point before batch processing
shell.simple_diagonal_test_point(torch.tensor([0.3, 0.5, 0.7]))

# Check convergence manually
if hasattr(shell.field, 'pos_matrix') and len(shell.field.pos_matrix) > 0:
    print(f"Converged in {len(shell.field.pos_matrix)} steps")
```

### Typical Workflow
1. Set up agent positions, parameters, resource distribution
2. Configure influence kernel type and domain
3. Create `Shell` instance and call `setup_adaptive_env()`
4. Run gradient ascent with `field.gradient_ascent()`
5. Visualize results using Shell's plotting methods

## Dependencies and Environment
- **PyTorch** - Primary tensor operations and autograd
- **NumPy** - Array operations, especially for 1D domains  
- **Ray[rllib]** - MARL components
- **Matplotlib** - All visualization
- **Hickle** - Data serialization for experiments

Use proper torch tensor operations for gradients and autograd compatibility.
