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
## Influencer Games — Quick AI contributor guide

This file captures the minimal, actionable knowledge an AI coding agent needs to work in this repo.

1) Big picture (why & where)
 - Two research paths: Adaptive dynamics (gradient-ascent) in `src/InflGame/adaptive/` and MARL experiments in `src/InflGame/MARL/`.
 - Influence kernels live under `src/InflGame/kernels/`; domain-specific code is in `src/InflGame/domains/`.

2) Critical design patterns to follow
 - domain_type is authoritative: values are `'1d'`, `'2d'`, or `'simplex'`. Branch logic based on this—`'1d'` uses numpy, `'2d'` uses torch tensors, `'simplex'` uses barycentric projections.
 - Preserve and restore state when mutating agent positions: use `.clone()` for torch tensors and `.copy()` for numpy arrays.
 - Gradient-ascent API: create a `Shell`, call `setup_adaptive_env()`, then `field.gradient_ascent()`; results are stored in `field.pos_matrix` and `field.grad_matrix`.

3) Where to make changes (examples)
 - Add kernels: `src/InflGame/kernels/` (follow `gaussian_infl()` signature and autograd rules when using torch).
 - Modify dynamics: `src/InflGame/adaptive/grad_func_env.py` (mv/sv gradient-ascent routines).
 - Visualization & experiment harness: `src/InflGame/adaptive/visualization.py` (Shell). Many demos call these directly.

4) Tests, runs, and common commands
 - Install deps: `pip install -r requirements.txt` (use project venv).
 - Run small tests/examples by executing root scripts `python test_classifier2_forward_simple.py` or the matching `test_*` files. Not all tests use pytest; use `python` for single-file tests or install `pytest` and run `pytest` if preferred.
 - Many reproducible examples are in `demo/paper_kernels/` (Jupyter notebooks & .hkl experiment data).

5) Performance & parallelism
 - CPU parallelism uses `concurrent.futures.ProcessPoolExecutor`; provide a sequential fallback to avoid deep-copy overhead.
 - Memory tips: clone tensors, zero-out large matrices between runs (`field.pos_matrix = 0`) and prefer in-place ops only when safe.

6) Common snippets (copy-paste)
 - Preserve/restore positions:
   - `original = self.agents_pos.clone()` (torch) or `.copy()` (numpy)
   - `self.agents_pos = original`
 - Single-point sanity test:
   - `shell.simple_diagonal_test_point(torch.tensor([0.3, 0.5, 0.7]))`

7) Dependencies & integration notes
 - Heavy: PyTorch (autograd), NumPy, Matplotlib, Hickle (data), Ray[rllib] for MARL experiments.
 - Many demo notebooks assume a working interactive kernel; prefer running notebooks for exploratory work before refactoring code.

8) Where to look when you get stuck
 - `src/InflGame/adaptive/visualization.py` (Shell) — experiment wiring and examples
 - `src/InflGame/adaptive/grad_func_env.py` — core gradient ascent logic
 - `src/InflGame/kernels/` and `src/InflGame/domains/` — kernel math and coordinate transforms

If anything here is unclear or you'd like me to expand a section (examples, test commands, or a short checklist for PR reviewers), tell me which part and I will iterate.

