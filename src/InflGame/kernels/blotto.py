r"""
.. module:: blotto
   :synopsis: Implements the Blotto influence kernel for modeling agent interactions in discrete-battlefield Colonel Blotto games.

Blotto Influence Kernel Module
===============================

This module implements the Blotto influence kernel and its associated computations.
The Blotto kernel models each agent as allocating a fixed resource budget across
:math:`M` discrete battlefields.  The influence exerted at battlefield :math:`b` is
the power-law :math:`(x_{i,b}/\chi)^\sigma`, recovering proportional allocation at
:math:`\sigma = 1` and approaching winner-takes-all in the limit
:math:`\sigma \to \infty`.

The strategy space of each agent is the *budget simplex*

.. math::
    \Delta_\chi^{M-1} = \{x \in \mathbb{R}_{>0}^M : \textstyle\sum_{b} x_b = \chi\}

with common budget :math:`\chi > 0`.  All kernel functions accept a ``chi``
parameter and internally normalize positions to the **unit simplex** via
:math:`y_i = x_i / \chi` before any computation.  This ensures the kernel
always operates on the unit simplex regardless of the domain's budget scale.
Gradient ascent on this simplex uses a **projected gradient step**: the
Euclidean gradient is first projected onto the tangent hyperplane of the
simplex (subtract its mean), then clamped to positivity and renormalized.
This is handled automatically by the ``AdaptiveEnv`` class when
``domain_type='simplex'`` and ``infl_type='blotto'``.

Mathematical Definitions:
--------------------------
The Blotto influence kernel is defined as:

.. math::
    f_i(x_i, b, \sigma) = \left(\frac{x_{i,b}}{\chi}\right)^{\!\sigma}, \qquad b \in \mathbb{B}

where :math:`x_{i,b}` is the resource allocation of agent :math:`i` to
battlefield :math:`b`, :math:`\chi` is the common budget, and :math:`\sigma > 0`
controls the sensitivity of influence to allocation differences.

Since the kernel at battlefield :math:`b` depends only on the :math:`b`-th
coordinate of :math:`x_i`, the log-derivative and Hessian are **diagonal**:

.. math::
    \frac{\partial}{\partial y_{i,b}} \ln f_i(y_i, b, \sigma)
    = \frac{\sigma}{y_{i,b}}, \qquad y_i = x_i / \chi

.. math::
    \frac{\partial^2}{\partial y_{i,b}^2} \ln f_i(y_i, b, \sigma)
    = -\frac{\sigma}{y_{i,b}^2}

The gradient tensor returned by :func:`d_ln_f_vectorized` has shape
:math:`(N, M, M)` where element ``[i, l, b]`` equals :math:`\sigma / y_{i,l}`
when :math:`l = b` and zero otherwise (a batch of diagonal matrices).

Dependencies:
-------------
- numpy
- torch

Usage:
------
Pass ``infl_type='blotto'`` and ``domain_type='simplex'`` to
``AdaptiveEnv``.  The ``parameters`` argument is ``[sigma, chi]``.

Example:
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.kernels.blotto import influence_vectorized, d_ln_f_vectorized, hessian_vectorized

    num_agents = 3
    sigma = 1.0          # scaling parameter
    chi = 2.0            # budget: rows of agents_pos sum to chi
    # positions on the budget simplex (rows sum to chi)
    agents_pos = np.array([[0.8, 0.6, 0.6],
                            [0.4, 1.0, 0.6],
                            [0.6, 0.6, 0.8]])

    # Influence matrix: shape (N, M)  — computed on unit simplex (x/chi)
    infl = influence_vectorized(agents_pos, sigma, chi=chi)
    print("Influence shape:", infl.shape)   # (3, 3)

    # Gradient tensor: shape (N, M, M)  — diagonal per agent, w.r.t. y=x/chi
    grad = d_ln_f_vectorized(agents_pos, sigma, chi=chi)
    print("Gradient shape:", grad.shape)    # (3, 3, 3)

    # Hessian tensor: shape (N, M, M)  — diagonal per agent, w.r.t. y=x/chi
    H = hessian_vectorized(agents_pos, sigma, chi=chi)
    print("Hessian shape:", H.shape)        # (3, 3, 3)
"""

import numpy as np
import torch
from typing import Union, Optional


# ========================= JIT-COMPILED HELPER FUNCTIONS =========================

@torch.jit.script
def _influence_core(agents_pos_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    JIT-compiled core for Blotto influence computation.

    Args:
        agents_pos_tensor: Agent positions (N, M) on the budget simplex.
        sigma: Scaling parameter (sigma > 0).

    Returns:
        torch.Tensor: Influence matrix (N, M).
    """
    # Clamp to avoid x^sigma = 0 or negative issues
    pos_clamped = torch.clamp(agents_pos_tensor, min=1e-10)
    return torch.pow(pos_clamped, sigma)


@torch.jit.script
def _d_ln_f_core(agents_pos_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    JIT-compiled core for the Blotto log-gradient.

    The gradient of ln f_i at battlefield b w.r.t. x_{i,l} is:
        sigma / x_{i,b}  if l == b,  else 0

    Returns a batch of diagonal matrices:  shape (N, M, M).

    Args:
        agents_pos_tensor: Agent positions (N, M).
        sigma: Scaling parameter (sigma > 0).

    Returns:
        torch.Tensor: Gradient tensor (N, M, M).
    """
    pos_clamped = torch.clamp(agents_pos_tensor, min=1e-10)
    # diagonal values: sigma / x_{i,l}, shape (N, M)
    diag_vals = sigma / pos_clamped
    # embed as batch of diagonal matrices (N, M, M)
    return torch.diag_embed(diag_vals)


@torch.jit.script
def _hessian_core(agents_pos_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    JIT-compiled core for the Blotto log-Hessian.

    The second derivative of ln f_i at battlefield b w.r.t. x_{i,b} is:
        -sigma / x_{i,b}^2  (diagonal only; cross-terms are zero)

    Returns shape (N, M, M).

    Args:
        agents_pos_tensor: Agent positions (N, M).
        sigma: Scaling parameter (sigma > 0).

    Returns:
        torch.Tensor: Hessian tensor (N, M, M) — batch of diagonal matrices.
    """
    pos_clamped = torch.clamp(agents_pos_tensor, min=1e-10)
    # diagonal values: -sigma / x_{i,l}^2, shape (N, M)
    diag_vals = -sigma / (pos_clamped ** 2)
    return torch.diag_embed(diag_vals)


# ========================= VECTORIZED PUBLIC FUNCTIONS =========================

def influence_vectorized(agents_pos: Union[list, np.ndarray, torch.Tensor],
                         sigma: float,
                         chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the Blotto influence matrix for all agents simultaneously.

    Each element ``[i, b]`` of the returned matrix equals
    :math:`(x_{i,b} / \chi)^{\sigma}`, the influence exerted by agent
    :math:`i` at battlefield :math:`b` after normalizing to the unit simplex.

    .. note::
        Unlike continuous-domain kernels, Blotto influence does **not**
        require a separate ``bin_points`` argument — the battlefields are
        indexed by the columns of ``agents_pos``.

    Parameters
    ----------
    agents_pos : list | np.ndarray | torch.Tensor
        Resource allocations :math:`x_i` for each agent, shape
        ``(num_agents, num_battlefields)``.  Each row should be strictly
        positive and sum to :math:`\chi`.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.  At :math:`\sigma = 1` the
        influence equals the normalized allocation; larger :math:`\sigma`
        amplifies differences between allocations.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).  Positions are
        divided by :math:`\chi` before computation so that the kernel always
        operates on the **unit simplex** internally.

    Returns
    -------
    torch.Tensor
        Influence matrix of shape ``(num_agents, num_battlefields)``.

    Raises
    ------
    ValueError
        If ``sigma`` or ``chi`` is not positive.
    TypeError
        If ``agents_pos`` is not a supported type.
    RuntimeError
        If the computation encounters NaN or Inf values.

    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]])
    >>> infl = influence_vectorized(agents_pos, sigma=1.0, chi=1.0)
    >>> print(infl.shape)
    torch.Size([2, 3])
    """
    try:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if chi <= 0:
            raise ValueError(f"chi must be positive, got {chi}")

        if isinstance(agents_pos, list):
            agents_pos_tensor = torch.tensor(agents_pos, dtype=torch.float32)
        elif isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}"
            )

        # Normalize to the unit simplex before computing influence
        normalized_pos = agents_pos_tensor / float(chi)
        infl_matrix = _influence_core(normalized_pos, float(sigma))

        if torch.any(torch.isnan(infl_matrix)):
            raise RuntimeError("NaN values detected in computed Blotto influence matrix")
        if torch.any(torch.isinf(infl_matrix)):
            raise RuntimeError("Inf values detected in computed Blotto influence matrix")

        return infl_matrix

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto influence computation: {str(e)}"
        ) from e


def d_ln_f_vectorized(agents_pos: Union[np.ndarray, torch.Tensor],
                      sigma: float,
                      chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the gradient of the log-influence for all agents (Blotto kernel).

    Positions are first normalized to the unit simplex via
    :math:`y_i = x_i / \chi`, then the gradient is computed with respect to
    those normalized coordinates.  The diagonal element ``[i, l, l]`` equals
    :math:`\sigma / y_{i,l} = \sigma \chi / x_{i,l}`.

    The gradient tensor has shape ``(num_agents, M, M)`` where element
    ``[i, l, b]`` equals :math:`\sigma / y_{i,l}` when :math:`l = b` and
    zero otherwise.  Each agent's gradient is represented as an
    :math:`M \times M` diagonal matrix.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations :math:`x_i`, shape ``(num_agents, M)``.  Each
        row is expected to sum to :math:`\chi`.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).  Positions are
        divided by :math:`\chi` before computing the gradient so that the
        kernel operates on the **unit simplex** internally.

    Returns
    -------
    torch.Tensor
        Gradient tensor of shape ``(num_agents, M, M)``.

    Raises
    ------
    ValueError
        If ``sigma`` or ``chi`` is not positive.
    TypeError
        If input types are not supported.
    RuntimeError
        If computation encounters numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]])
    >>> grad = d_ln_f_vectorized(agents_pos, sigma=1.0, chi=1.0)
    >>> print(grad.shape)
    torch.Size([2, 3, 3])
    """
    try:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if chi <= 0:
            raise ValueError(f"chi must be positive, got {chi}")

        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}"
            )

        # Normalize to the unit simplex; gradient is w.r.t. normalized coords
        normalized_pos = agents_pos_tensor / float(chi)
        d_tensor = _d_ln_f_core(normalized_pos, float(sigma))

        if torch.any(torch.isnan(d_tensor)):
            raise RuntimeError("NaN values detected in computed Blotto gradient tensor")
        if torch.any(torch.isinf(d_tensor)):
            raise RuntimeError("Inf values detected in computed Blotto gradient tensor")

        return d_tensor

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto gradient computation: {str(e)}"
        ) from e


def hessian_vectorized(agents_pos: Union[np.ndarray, torch.Tensor],
                       sigma: float,
                       chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the Hessian of the log-influence for all agents (Blotto kernel).

    Positions are first normalized to the unit simplex via
    :math:`y_i = x_i / \chi`, then the Hessian is computed with respect to
    those normalized coordinates.  The diagonal element ``[i, l, l]`` equals
    :math:`-\sigma / y_{i,l}^2 = -\sigma \chi^2 / x_{i,l}^2`.

    The log-concavity is guaranteed for all :math:`\sigma > 0`: the diagonal
    Hessian is strictly negative definite on :math:`\mathbb{R}_{>0}^M`.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations :math:`x_i`, shape ``(num_agents, M)``.  Each
        row is expected to sum to :math:`\chi`.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).  Positions are
        divided by :math:`\chi` before computing the Hessian so that the
        kernel operates on the **unit simplex** internally.

    Returns
    -------
    torch.Tensor
        Hessian tensor of shape ``(num_agents, M, M)``.

    Raises
    ------
    ValueError
        If ``sigma`` or ``chi`` is not positive.
    TypeError
        If input types are not supported.
    RuntimeError
        If computation encounters numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> agents_pos = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3]])
    >>> H = hessian_vectorized(agents_pos, sigma=1.0, chi=1.0)
    >>> print(H.shape)
    torch.Size([2, 3, 3])
    """
    try:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if chi <= 0:
            raise ValueError(f"chi must be positive, got {chi}")

        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}"
            )

        # Normalize to the unit simplex; Hessian is w.r.t. normalized coords
        normalized_pos = agents_pos_tensor / float(chi)
        h_tensor = _hessian_core(normalized_pos, float(sigma))

        if torch.any(torch.isnan(h_tensor)):
            raise RuntimeError("NaN values detected in computed Blotto Hessian tensor")
        if torch.any(torch.isinf(h_tensor)):
            raise RuntimeError("Inf values detected in computed Blotto Hessian tensor")

        return h_tensor

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto Hessian computation: {str(e)}"
        ) from e


# ========================= PER-AGENT VECTORIZED FUNCTIONS =========================

def influence_vectorized_per_agent(agents_pos: Union[list, np.ndarray, torch.Tensor],
                                   sigma_vec: torch.Tensor,
                                   chi_vec: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the Blotto influence matrix where each agent has its own
    ``sigma`` and ``chi`` parameters.

    Each element ``[i, b]`` equals :math:`(x_{i,b} / \chi_i)^{\sigma_i}`.

    Parameters
    ----------
    agents_pos : list | np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, num_battlefields)``.
    sigma_vec : torch.Tensor
        Per-agent scaling parameters, shape ``(num_agents,)``.
    chi_vec : torch.Tensor
        Per-agent budget parameters, shape ``(num_agents,)``.

    Returns
    -------
    torch.Tensor
        Influence matrix of shape ``(num_agents, num_battlefields)``.
    """
    try:
        if isinstance(agents_pos, list):
            agents_pos_tensor = torch.tensor(agents_pos, dtype=torch.float32)
        elif isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}"
            )

        sigma_vec = sigma_vec.to(torch.float32)
        chi_vec = chi_vec.to(torch.float32)

        if torch.any(sigma_vec <= 0):
            raise ValueError("All sigma values must be positive")
        if torch.any(chi_vec <= 0):
            raise ValueError("All chi values must be positive")

        # Normalize each agent's position by its own chi: shape (N, M)
        normalized = agents_pos_tensor / chi_vec.unsqueeze(1)
        clamped = torch.clamp(normalized, min=1e-10)
        # Per-agent exponent: exp(sigma_i * log(y_{i,b})), shape (N, M)
        infl_matrix = torch.exp(sigma_vec.unsqueeze(1) * torch.log(clamped))

        if torch.any(torch.isnan(infl_matrix)):
            raise RuntimeError("NaN values detected in per-agent Blotto influence matrix")
        if torch.any(torch.isinf(infl_matrix)):
            raise RuntimeError("Inf values detected in per-agent Blotto influence matrix")

        return infl_matrix

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in per-agent Blotto influence computation: {str(e)}"
        ) from e


def d_ln_f_vectorized_per_agent(agents_pos: Union[np.ndarray, torch.Tensor],
                                 sigma_vec: torch.Tensor,
                                 chi_vec: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the gradient of the log-influence where each agent has its own
    ``sigma`` and ``chi`` parameters.

    The diagonal element ``[i, l, l]`` equals
    :math:`\sigma_i / y_{i,l}` where :math:`y_{i,l} = x_{i,l} / \chi_i`.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    sigma_vec : torch.Tensor
        Per-agent scaling parameters, shape ``(num_agents,)``.
    chi_vec : torch.Tensor
        Per-agent budget parameters, shape ``(num_agents,)``.

    Returns
    -------
    torch.Tensor
        Gradient tensor of shape ``(num_agents, M, M)``.
    """
    try:
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}"
            )

        sigma_vec = sigma_vec.to(torch.float32)
        chi_vec = chi_vec.to(torch.float32)

        normalized = agents_pos_tensor / chi_vec.unsqueeze(1)
        clamped = torch.clamp(normalized, min=1e-10)
        # diag[i, l] = sigma_i / y_{i,l}
        diag_vals = sigma_vec.unsqueeze(1) / clamped  # (N, M)
        d_tensor = torch.diag_embed(diag_vals)  # (N, M, M)

        if torch.any(torch.isnan(d_tensor)):
            raise RuntimeError("NaN values detected in per-agent Blotto gradient tensor")
        if torch.any(torch.isinf(d_tensor)):
            raise RuntimeError("Inf values detected in per-agent Blotto gradient tensor")

        return d_tensor

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in per-agent Blotto gradient computation: {str(e)}"
        ) from e


def hessian_vectorized_per_agent(agents_pos: Union[np.ndarray, torch.Tensor],
                                  sigma_vec: torch.Tensor,
                                  chi_vec: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the Hessian of the log-influence where each agent has its own
    ``sigma`` and ``chi`` parameters.

    The diagonal element ``[i, l, l]`` equals
    :math:`-\sigma_i / y_{i,l}^2` where :math:`y_{i,l} = x_{i,l} / \chi_i`.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    sigma_vec : torch.Tensor
        Per-agent scaling parameters, shape ``(num_agents,)``.
    chi_vec : torch.Tensor
        Per-agent budget parameters, shape ``(num_agents,)``.

    Returns
    -------
    torch.Tensor
        Hessian tensor of shape ``(num_agents, M, M)``.
    """
    try:
        if isinstance(agents_pos, np.ndarray):
            agents_pos_tensor = torch.from_numpy(agents_pos.astype(np.float32))
        elif isinstance(agents_pos, torch.Tensor):
            agents_pos_tensor = agents_pos.to(torch.float32)
        else:
            raise TypeError(
                f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}"
            )

        sigma_vec = sigma_vec.to(torch.float32)
        chi_vec = chi_vec.to(torch.float32)

        normalized = agents_pos_tensor / chi_vec.unsqueeze(1)
        clamped = torch.clamp(normalized, min=1e-10)
        diag_vals = -sigma_vec.unsqueeze(1) / (clamped ** 2)  # (N, M)
        h_tensor = torch.diag_embed(diag_vals)  # (N, M, M)

        if torch.any(torch.isnan(h_tensor)):
            raise RuntimeError("NaN values detected in per-agent Blotto Hessian tensor")
        if torch.any(torch.isinf(h_tensor)):
            raise RuntimeError("Inf values detected in per-agent Blotto Hessian tensor")

        return h_tensor

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError)):
            raise
        raise RuntimeError(
            f"Unexpected error in per-agent Blotto Hessian computation: {str(e)}"
        ) from e


def _parse_blotto_params(parameter_instance):
    r"""
    Parse Blotto ``parameter_instance`` into ``(sigma, chi, is_per_agent)``.

    Supports both a flat ``[sigma, chi]`` spec (shared across all agents) and a
    2-D ``[[sigma_0, chi_0], [sigma_1, chi_1], ...]`` spec (one row per agent).
    When the per-agent rows all happen to be identical the function falls back
    to the scalar path so that the faster JIT-compiled kernels are used.

    Parameters
    ----------
    parameter_instance : list | np.ndarray | torch.Tensor
        Either shape ``(2,)`` for shared params or ``(num_agents, 2)`` for
        per-agent params.

    Returns
    -------
    tuple
        ``(sigma, chi, is_per_agent)`` where *sigma* and *chi* are either
        Python floats (``is_per_agent=False``) or 1-D ``torch.Tensor``
        (``is_per_agent=True``).
    """
    if isinstance(parameter_instance, torch.Tensor):
        if parameter_instance.dim() == 2:
            sigma_vec = parameter_instance[:, 0].float()
            chi_vec   = parameter_instance[:, 1].float()
            # Use scalar path when all agents share the same params
            if torch.all(sigma_vec == sigma_vec[0]) and torch.all(chi_vec == chi_vec[0]):
                return float(sigma_vec[0]), float(chi_vec[0]), False
            return sigma_vec, chi_vec, True
        elif parameter_instance.dim() == 1:
            sigma = float(parameter_instance[0]) if parameter_instance.numel() >= 1 else 1.0
            chi   = float(parameter_instance[1]) if parameter_instance.numel() >= 2 else 1.0
            return sigma, chi, False
        else:
            return float(parameter_instance), 1.0, False
    elif hasattr(parameter_instance, '__len__'):
        # list or numpy array
        arr = np.asarray(parameter_instance)
        if arr.ndim == 2:
            sigma_vec = torch.tensor(arr[:, 0], dtype=torch.float32)
            chi_vec   = torch.tensor(arr[:, 1], dtype=torch.float32)
            if torch.all(sigma_vec == sigma_vec[0]) and torch.all(chi_vec == chi_vec[0]):
                return float(sigma_vec[0]), float(chi_vec[0]), False
            return sigma_vec, chi_vec, True
        else:
            sigma = float(arr[0]) if len(arr) >= 1 else 1.0
            chi   = float(arr[1]) if len(arr) >= 2 else 1.0
            return sigma, chi, False
    else:
        return float(parameter_instance), 1.0, False


# ================= BACKWARD COMPATIBLE SINGLE-AGENT WRAPPERS =================

def influence(agent_id: int,
              agents_pos: Union[np.ndarray, torch.Tensor],
              sigma: float,
              chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the Blotto influence for a single agent.

    Backward-compatible wrapper that extracts one row from
    :func:`influence_vectorized`.

    Parameters
    ----------
    agent_id : int
        Index of the agent in ``agents_pos``.
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).

    Returns
    -------
    torch.Tensor
        Influence values of shape ``(M,)`` for the specified agent.

    Raises
    ------
    TypeError
        If ``agent_id`` is not an integer.
    IndexError
        If ``agent_id`` is out of bounds.
    """
    try:
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")

        if isinstance(agents_pos, np.ndarray):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, torch.Tensor):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, list):
            num_agents = len(agents_pos)
        else:
            raise TypeError(
                f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}"
            )

        if agent_id < 0 or agent_id >= num_agents:
            raise IndexError(
                f"agent_id {agent_id} is out of bounds for {num_agents} agents"
            )

        return influence_vectorized(agents_pos, sigma, chi)[agent_id]

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto single-agent influence: {str(e)}"
        ) from e


def d_ln_f(agent_id: int,
           agents_pos: Union[np.ndarray, torch.Tensor],
           sigma: float,
           chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the log-gradient for a single agent (Blotto kernel).

    Backward-compatible wrapper that extracts one slice from
    :func:`d_ln_f_vectorized`.

    Parameters
    ----------
    agent_id : int
        Index of the agent in ``agents_pos``.
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).

    Returns
    -------
    torch.Tensor
        Gradient matrix of shape ``(M, M)`` for the specified agent.

    Raises
    ------
    TypeError
        If ``agent_id`` is not an integer.
    IndexError
        If ``agent_id`` is out of bounds.
    """
    try:
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")

        if isinstance(agents_pos, np.ndarray):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, torch.Tensor):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, list):
            num_agents = len(agents_pos)
        else:
            raise TypeError(
                f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}"
            )

        if agent_id < 0 or agent_id >= num_agents:
            raise IndexError(
                f"agent_id {agent_id} is out of bounds for {num_agents} agents"
            )

        return d_ln_f_vectorized(agents_pos, sigma, chi)[agent_id]

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto single-agent gradient: {str(e)}"
        ) from e


def hessian(agent_id: int,
            agents_pos: Union[np.ndarray, torch.Tensor],
            sigma: float,
            chi: float = 1.0) -> torch.Tensor:
    r"""
    Compute the log-Hessian for a single agent (Blotto kernel).

    Backward-compatible wrapper that extracts one slice from
    :func:`hessian_vectorized`.

    Parameters
    ----------
    agent_id : int
        Index of the agent in ``agents_pos``.
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    sigma : float
        Scaling parameter :math:`\sigma > 0`.
    chi : float
        Budget parameter :math:`\chi > 0` (default ``1.0``).

    Returns
    -------
    torch.Tensor
        Hessian matrix of shape ``(M, M)`` for the specified agent.

    Raises
    ------
    TypeError
        If ``agent_id`` is not an integer.
    IndexError
        If ``agent_id`` is out of bounds.
    """
    try:
        if not isinstance(agent_id, int):
            raise TypeError(f"agent_id must be an integer, got {type(agent_id)}")

        if isinstance(agents_pos, np.ndarray):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, torch.Tensor):
            num_agents = agents_pos.shape[0]
        elif isinstance(agents_pos, list):
            num_agents = len(agents_pos)
        else:
            raise TypeError(
                f"agents_pos must be list, np.ndarray, or torch.Tensor, got {type(agents_pos)}"
            )

        if agent_id < 0 or agent_id >= num_agents:
            raise IndexError(
                f"agent_id {agent_id} is out of bounds for {num_agents} agents"
            )

        return hessian_vectorized(agents_pos, sigma, chi)[agent_id]

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, TypeError, IndexError)):
            raise
        raise RuntimeError(
            f"Unexpected error in Blotto single-agent Hessian: {str(e)}"
        ) from e


# ================= UTILITY FUNCTIONS =================

def validate_budget_positions(agents_pos: Union[np.ndarray, torch.Tensor],
                               budget: float = 1.0,
                               tolerance: float = 1e-5) -> bool:
    r"""
    Validate that agent positions lie on the budget simplex.

    Checks that all entries are strictly positive and that each row sums to
    ``budget`` (i.e. :math:`\chi`) within the specified tolerance.

    Parameters
    ----------
    agents_pos : np.ndarray | torch.Tensor
        Resource allocations, shape ``(num_agents, M)``.
    budget : float
        Expected row sum :math:`\chi` (default 1.0).  Pass the same value
        used as ``chi`` in the kernel functions.
    tolerance : float
        Tolerance for the sum constraint check.

    Returns
    -------
    bool
        ``True`` if all positions are valid.

    Raises
    ------
    TypeError
        If ``agents_pos`` is not a supported type.
    """
    if isinstance(agents_pos, np.ndarray):
        sums = np.sum(agents_pos, axis=1)
        return bool(np.all(agents_pos > 0) and np.all(np.abs(sums - budget) < tolerance))
    elif isinstance(agents_pos, torch.Tensor):
        sums = torch.sum(agents_pos, dim=1)
        return bool(
            torch.all(agents_pos > 0) and torch.all(torch.abs(sums - budget) < tolerance)
        )
    else:
        raise TypeError(
            f"agents_pos must be np.ndarray or torch.Tensor, got {type(agents_pos)}"
        )
