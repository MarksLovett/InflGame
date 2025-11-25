## AI Instructions for This Repository (InflGame)

This document provides guardrails and conventions for AI assistants contributing to this codebase. Follow these instructions strictly to keep changes safe, readable, and aligned with project goals.

### Scope of the Project
- **Package**: `InflGame` (Python, packaged under `src/InflGame`)
- **Domains**: Influence games with spatial influence, studied via:
  - **Adaptive dynamics**: gradient-based dynamics, bifurcation and stability analysis
  - **Multi-agent RL (MARL)**: Independent Q-learning (IQL) and Ray RLlib-based methods
- **Key Dependencies**: `numpy`, `scipy`, `torch`, `ray[rllib]`, `gymnasium`, `matplotlib`, `seaborn`, `imageio`, `hickle`, `plotly`

### Directory Overview
- `src/InflGame/`
  - `adaptive/`: gradient environments, Jacobians, bifurcation and visualization utilities
  - `domains/`: 1D, 2D, simplex domain utils and plotting
  - `kernels/`: influence kernels (Gaussian, Dirichlet, Jones, multivariate Gaussian)
  - `MARL/`: async/sync game loops, IQL, RL plots, and MARL utils
  - `utils/`: shared utilities, data IO/validation, paper/figure utilities, plotting helpers
  - `test/`: partial test utilities
- `docs/`: Sphinx docs scaffolding and generated assets
- Top-level analysis scripts and notebooks for experiments and data analysis (e.g., `gpu_text_generation.py`, `linkedin/job_analysis.ipynb`)

### When Making Code Changes
- Prefer improving and reusing existing modules under `src/InflGame`.
- Avoid breaking public APIs. If changes are required:
  - Keep backwards compatibility when practical.
  - If you must break, isolate the change and document the migration in this file and in code docstrings.
- Keep edits focused: one conceptual change per commit/edit.

### Code Style and Quality
- Python ≥ 3.9. Use type hints on public functions/classes.
- Favor explicit, descriptive names over abbreviations (e.g., `computeJacobianMatrix` not `compJac`).
- Minimize nesting; prefer early returns.
- Only catch exceptions with meaningful handling.
- Write high-verbosity, readable code; keep comments to important context only (rationale, invariants, edge cases).
- Do not reformat unrelated code.

### Testing and Validation
- If you add or modify algorithms, include minimal, targeted checks:
  - Deterministic, small-shape numerical checks for adaptive dynamics (e.g., Jacobian shape, sign or stability on known toy configs).
  - Smoke tests for MARL entry points (single short episode with tiny env and debug config).
- Place tests near their domain if no centralized test suite is present (e.g., `src/InflGame/<area>/test_*`), or extend `src/InflGame/test/` if appropriate.

### Documentation
- Add/expand docstrings for public functions/classes.
- If you add user-facing features, update `README.md` usage examples and, when feasible, Sphinx docs under `docs/` (API or examples).

### Data and Artifacts
- Do not commit new large binary artifacts. Use existing `utils/data` structure for reading assets, not writing new ones.
- Prefer programmatic generation with seeds for reproducibility.

### Performance Guidelines
- Vectorize numerics with `numpy`/`scipy` when possible.
- For `torch` code, preserve device handling and batch semantics.
- For Ray/RLlib usage, keep configs small by default and expose scale-up options via parameters.

### Safety and Determinism
- Set seeds in tests/examples where results are asserted or plotted.
- Validate inputs with clear errors; fail fast on invalid shapes/ranges.
- Avoid silent fallback paths that change algorithmic behavior.

### Module-Specific Guidance
- `adaptive/`:
  - Keep gradient/Jacobian math centralized; avoid duplicating formulae.
  - For visualization, separate heavy computation from plotting code.
- `domains/`:
  - Keep domain geometry helpers pure and well-documented.
  - Maintain consistent interfaces across `one_d`, `two_d`, and `simplex` utilities.
- `kernels/`:
  - Add new kernels with consistent API: parameter validation, probability normalization, and vectorized evaluation.
- `MARL/`:
  - Keep environment and algorithm configs explicit; expose key hyperparameters.
  - Ensure compatibility with `gymnasium` spaces and RLlib conventions.
- `utils/`:
  - Treat as shared library; avoid circular dependencies. Keep IO boundaries obvious.

### Backwards Compatibility Contract (Informal)
- Public functions/classes under `src/InflGame` should remain stable.
- Adding optional parameters is preferred over changing required signatures.
- If removing/renaming symbols, provide shims or deprecation notes where feasible.

### How to Propose Changes (AI Agent Workflow)
1. Identify target module and confirm the change is localized.
2. Make the smallest viable edit; preserve indentation style and formatting of touched files.
3. Add/adjust docstrings and minimal tests.
4. Run quick local validations (imports, basic execution or smoke tests if present).
5. Update `README.md` or docs if user-facing behavior changed.

### Non-Goals for AI Edits
- Do not restructure directories or rename top-level packages without explicit instruction.
- Do not add heavy external dependencies beyond those already listed in `pyproject.toml`/`requirements.txt`.
- Do not commit generated figures or large data.

### Quick Map of Entry Points
- Adaptive analyses: `src/InflGame/adaptive/*.py` (e.g., `jacobian.py`, `root_finding.py`, `visualization.py`)
- Domains and plots: `src/InflGame/domains/*` (e.g., `one_utils.py`, `two_utils.py`, `simplex_utils.py`)
- Kernels: `src/InflGame/kernels/*.py` (e.g., `gauss.py`, `diric.py`, `jones.py`, `MV_gauss.py`)
- MARL: `src/InflGame/MARL/*.py` and `src/InflGame/MARL/utils/*`
- Shared utilities: `src/InflGame/utils/*`

### Versioning and Packaging
- Package metadata in `pyproject.toml` (`InflGame`, version `0.1.0`).
- Keep public APIs stable across minor changes; bump version if breaking.

---
If a guideline conflicts with a maintainer’s direct instruction or an established code pattern in a target module, follow the module’s local precedent and briefly document the rationale in the PR/commit description.
