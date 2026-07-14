"""
.. module:: rd
   :synopsis: Provides resource distribution functions for influencer games.

.. currentmodule:: InflGame.domains.rd


Resource Distribution Module
============================

This module provides functions to compute and select resource distributions for influencer games. 
It includes implementations for various types of resource distributions, such as beta, Dirichlet, 
and multi-modal Gaussian distributions in 1D and 2D domains.

The module is designed to work with the `InflGame` package and supports resource distribution 
evaluation over specified bin points. These distributions are used to model resource availability 
in different domains and scenarios of influencer games.

Functions:
-----------

============================================  ========================================================================================
Function                                       Description
============================================  ========================================================================================
:func:`resource_distribution_choice`          Selects and computes a resource distribution based on the specified type and parameters.
:func:`multi_modal_gaussian_distribution_1D`  Computes a 1D multi-modal Gaussian mixture distribution for resources.
:func:`multi_modal_gaussian_distribution_2D`  Computes a 2D multi-modal Gaussian mixture distribution for resources.
:func:`beta_distribution`                     Computes a beta distribution for resources on the 2-simplex.
:func:`dirichlet_distribution`                Computes a Dirichlet distribution for resources.
============================================  ========================================================================================


Usage:
------
The `resource_distribution_choice` function serves as the main entry point for selecting and computing 
a resource distribution based on the specified type and parameters.

Examples
--------

.. code-block:: python

    import numpy as np
    from InflGame.domains.rd import resource_distribution_choice

    # Define bin points and parameters
    bin_points = np.linspace(0, 1, 100)
    resource_type = "beta"
    resource_parameters = [2, 5]

    # Compute the resource distribution
    resources = resource_distribution_choice(bin_points, resource_type, resource_parameters)
    print(resources)


"""

import numpy as np
import torch
from scipy.stats import dirichlet
from scipy.stats import beta

import InflGame.domains.simplex.simplex_utils as simplex_utils




def resource_distribution_choice(bin_points: np.ndarray,
                                 resource_type: str,
                                 resource_parameters: list | np.ndarray):
    """
    Select and compute a resource distribution based on the specified type and parameters.
    
    This function serves as the main entry point for creating resource distributions in influencer games.
    It dispatches to the appropriate distribution function based on the resource_type parameter.

    Parameters
    ----------
    bin_points : np.ndarray
        Points where the resource distribution is evaluated. For 1D distributions, this is a 1D array.
        For 2D distributions, this is an :math:`(N, 2)` array of coordinate pairs.
    resource_type : str
        Type of resource distribution to compute. Available options:
        
        - ``'beta'``: 1D beta distribution on the interval (0,1); see :func:`beta_distribution`
        - ``'dirichlet_distribution'``: Dirichlet distribution for resources on a probability 3-simplex; see :func:`dirichlet_distribution`
        - ``'multi_modal_gaussian_distribution_1D'``: 1D mixture of Gaussian kernels; see :func:`multi_modal_gaussian_distribution_1D`
        - ``'multi_modal_gaussian_distribution_2D'``: 2D mixture of multivariate Gaussian kernels; see :func:`multi_modal_gaussian_distribution_2D`
        - ``'multi_modal_gaussian_distribution_2D_triangle'``: Alias for 2D multivariate Gaussian on triangular domain
        - ``'multi_modal_gaussian_distribution_2D_square'``: Alias for 2D multivariate Gaussian on rectangular domain
    resource_parameters : list | np.ndarray
        Parameters for the specified resource distribution. Format depends on resource_type:
        
        - For ``'beta'``: ``[alpha, beta]``
        - For ``'dirichlet_distribution'``: ``[alpha_1, alpha_2, ..., alpha_k]``
        - For ``'multi_modal_gaussian_distribution_1D'``: ``[stds, means, mode_factors]``
        - For ``'multi_modal_gaussian_distribution_2D'``: ``[covariance_matrices, means]``

    Returns
    -------
    np.ndarray
        Computed resource distribution values at the specified bin_points.
        
    Raises
    ------
    str
        Returns error message string if resource_type is not recognized.
    
    Examples
    --------
    >>> bin_points = np.linspace(0, 1, 100)
    >>> resources = resource_distribution_choice(bin_points, 'beta', [2, 5])
    >>> resources.shape
    (100,)
    """
    
    if resource_type == "multi_modal_gaussian_distribution_1D":
        resources = multi_modal_gaussian_distribution_1D(bin_points, stds=resource_parameters[0], means=resource_parameters[1], mode_factors=resource_parameters[2])
    elif resource_type == "beta":
        resources = beta_distribution(bin_points, resource_parameters[0], resource_parameters[1])
    elif resource_type in ["multi_modal_gaussian_distribution_2D", "multi_modal_gaussian_distribution_2D_triangle", "multi_modal_gaussian_distribution_2D_square"]:
        resources = multi_modal_gaussian_distribution_2D(bin_points, resource_parameters[0], resource_parameters[1])
    elif resource_type == "dirichlet_distribution":
        resources = dirichlet_distribution(bin_points, resource_parameters)
    else: 
        return "No known type " + resource_type
    return resources

def multi_modal_gaussian_distribution_1D(bin_points: np.ndarray | torch.Tensor,
                                         stds: list[float] = [.1, .1],
                                         means: list[float] = [.5, .5],
                                         mode_factors: list[float] = [1, 1]) -> np.ndarray:
    r"""
    Compute a 1D multi-modal Gaussian mixture distribution for resources.
    
    Creates a resource distribution as a weighted sum of Gaussian kernels, each with its own
    mean, standard deviation, and scaling factor. This allows modeling of complex multi-modal
    resource landscapes.

    The distribution is defined as:
    
    .. math::
        R(b) = \sum_{i=1}^{k} \alpha_i \cdot \exp\left(-\frac{(b - \mu_i)^2}{2\sigma_i^2}\right)
    
    where :math:`k` is the number of modes, :math:`\alpha_i` is the scaling factor for mode :math:`i`,
    :math:`\mu_i` is the mean of mode :math:`i`, and :math:`\sigma_i` is the standard deviation of mode :math:`i`.

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Points where the distribution is evaluated, typically on the interval [0, 1].
    stds : list[float], optional
        Standard deviations :math:`\sigma_i` for each Gaussian mode, by default [.1, .1].
    means : list[float], optional
        Mean values :math:`\mu_i` for each Gaussian mode, by default [.5, .5].
    mode_factors : list[float], optional
        Scaling factors :math:`\alpha_i` for each mode, by default [1, 1].

    Returns
    -------
    np.ndarray
        Computed resource distribution values at the specified bin_points.
        
    Notes
    -----
    All three parameter lists (stds, means, mode_factors) must have the same length,
    corresponding to the number of modes :math:`k` in the mixture.
    
    Examples
    --------
    >>> bin_points = np.linspace(0, 1, 100)
    >>> resources = multi_modal_gaussian_distribution_1D(
    ...     bin_points, stds=[0.1, 0.15], means=[0.3, 0.7], mode_factors=[1, 1.5]
    ... )
    """
    
    resource_modes = []
    for mode_id in range(len(stds)):
        mean = means[mode_id]
        std = stds[mode_id]
        mode_factor = mode_factors[mode_id]
        mode = mode_factor * np.exp(-(bin_points - mean)**2 / (2 * (std)**2))
        resource_modes.append(mode)
    resource_modes = np.array(resource_modes)
    resources = np.sum(resource_modes, axis=0)
    
    return resources

def multi_modal_gaussian_distribution_2D(bin_points: np.ndarray | torch.Tensor,
                                         covariance_matrices: torch.Tensor = torch.tensor([[[.1, 0], [0, .1]], [[.1, 0], [0, .1]], [[.1, 0], [0, .1]]]),
                                         means: torch.Tensor = torch.tensor([[0, 0], [1, 0], [0.5000, 0.8660]])) -> np.ndarray:
    r"""
    Compute a 2D multi-modal Gaussian mixture distribution for resources.

    Creates a resource distribution as a weighted sum of multivariate Gaussian kernels, each with its own
    mean vector and covariance matrix. This enables modeling of complex spatial resource patterns in 2D domains.

    The distribution is defined as:
    
    .. math::
        R(\mathbf{b}) = \sum_{i=1}^{k} \exp\left(-\frac{1}{2}(\mathbf{b} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{b} - \boldsymbol{\mu}_i)\right)
    
    where :math:`k` is the number of modes, :math:`\boldsymbol{\mu}_i` is the mean vector for mode :math:`i`,
    and :math:`\boldsymbol{\Sigma}_i` is the covariance matrix for mode :math:`i`.

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Points where the distribution is evaluated, shape :math:`(N, 2)` for :math:`N` 2D coordinates.
    covariance_matrices : torch.Tensor, optional
        Covariance matrices :math:`\boldsymbol{\Sigma}_i` for each Gaussian mode, shape :math:`(k, 2, 2)`
        where :math:`k` is the number of modes. By default creates 3 isotropic modes with variance 0.1.
    means : torch.Tensor, optional
        Mean vectors :math:`\boldsymbol{\mu}_i` for each Gaussian mode, shape :math:`(k, 2)`.
        By default creates 3 modes at the vertices of an equilateral triangle.

    Returns
    -------
    np.ndarray
        Computed resource distribution values at the specified bin_points, shape :math:`(N,)`.
        
    Notes
    -----
    The number of modes :math:`k` is determined by the length of the means tensor.
    The covariance_matrices tensor must have the same number of modes.
    
    Examples
    --------
    >>> bin_points = np.random.rand(100, 2)
    >>> means = torch.tensor([[0.3, 0.3], [0.7, 0.7]])
    >>> covs = torch.tensor([[[0.05, 0], [0, 0.05]], [[0.08, 0], [0, 0.08]]])
    >>> resources = multi_modal_gaussian_distribution_2D(bin_points, covs, means)
    """
    
    resource_modes = []
    for mode_id in range(len(covariance_matrices)):
        mean = means[mode_id]
        covariance = covariance_matrices[mode_id]
        x_vec = torch.tensor((bin_points - mean.numpy())).float()
        sigma_inv = torch.inverse(covariance)
        distribution_values = []
        for i in range(len(bin_points)):
            distribution_value = torch.exp(-1/2 * x_vec[i, :] @ sigma_inv @ x_vec.T[:, i])
            distribution_values.append(distribution_value.item())
        mode = np.array(distribution_values)
        resource_modes.append(mode)
    resource_modes = np.array(resource_modes)
    resources = np.sum(resource_modes, axis=0)
    
    return resources

def beta_distribution(bin_points: np.ndarray | torch.Tensor,
                      alpha_value: float,
                      beta_value: float) -> np.ndarray:
    r"""
    Compute a beta distribution for resources on the unit interval.

    Uses the beta probability density function to model resource distribution on the interval (0, 1).
    The beta distribution is useful for modeling bounded resources with various shapes controlled
    by the alpha and beta parameters.

    The distribution is defined as:
    
    .. math::
        R(b) = \frac{b^{\alpha-1}(1-b)^{\beta-1}}{B(\alpha, \beta)}
    
    where :math:`B(\alpha, \beta)` is the beta function:
    
    .. math::
        B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1} dt = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Points where the distribution is evaluated, typically on the interval (0, 1).
    alpha_value : float
        Alpha parameter :math:`\alpha` of the beta distribution. Must be positive.
        Controls the shape of the distribution at :math:`b=0`.
    beta_value : float
        Beta parameter :math:`\beta` of the beta distribution. Must be positive.
        Controls the shape of the distribution at :math:`b=1`.

    Returns
    -------
    np.ndarray
        Computed resource distribution values (probability density) at the specified bin_points.
        
    Notes
    -----
    - When :math:`\alpha = \beta = 1`, the distribution is uniform
    - When :math:`\alpha > 1` and :math:`\beta > 1`, the distribution is unimodal
    - When :math:`\alpha < 1` and :math:`\beta < 1`, the distribution is bimodal (U-shaped)
    
    Examples
    --------
    >>> bin_points = np.linspace(0.01, 0.99, 100)
    >>> resources = beta_distribution(bin_points, alpha_value=2, beta_value=5)
    """
   
    f = lambda x: beta.pdf(x, a=alpha_value, b=beta_value)
    resources = f(bin_points)
    return resources

def dirichlet_distribution(bin_points: np.ndarray | torch.Tensor,
                           alphas: list | np.ndarray) -> np.ndarray:
    r"""
    Compute a Dirichlet distribution for resources on the probability simplex.

    Uses the Dirichlet probability density function to model resource distribution on a simplex.
    The Dirichlet distribution is a multivariate generalization of the beta distribution and is
    particularly useful for modeling resources in simplex domains.

    The distribution is defined as:
    
    .. math::
        R(\mathbf{b}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^{k} b_i^{\alpha_i-1}
    
    where :math:`B(\boldsymbol{\alpha})` is the multivariate beta function:
    
    .. math::
        B(\boldsymbol{\alpha}) = \frac{\prod_{i=1}^{k} \Gamma(\alpha_i)}{\Gamma(\sum_{i=1}^{k} \alpha_i)}
    
    and :math:`\mathbf{b} = (b_1, b_2, \ldots, b_k)` with :math:`\sum_{i=1}^{k} b_i = 1` and :math:`b_i \geq 0`.

    Parameters
    ----------
    bin_points : np.ndarray | torch.Tensor
        Points on the simplex where the distribution is evaluated, shape :math:`(N, k)` where
        :math:`k` is the dimension of the simplex. Each point must satisfy simplex constraints.
    alphas : list | np.ndarray
        Concentration parameters :math:`\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_k)`
        of the Dirichlet distribution. All values must be positive.

    Returns
    -------
    np.ndarray
        Computed resource distribution values (probability density) at the specified bin_points.
        
    Notes
    -----
    This function handles edge cases where bin_points fall outside or on the boundary of the simplex
    by applying small corrections to ensure valid simplex coordinates before evaluation.
    
    - Points with any coordinate :math:`\leq 0` are projected onto the simplex
    - Points with coordinates exactly 0 or 1 are adjusted by small epsilon values to avoid
      numerical issues with the Dirichlet PDF
    
    Examples
    --------
    >>> # 3D simplex points (must sum to 1)
    >>> bin_points = np.array([[0.33, 0.33, 0.34], [0.1, 0.2, 0.7]])
    >>> resources = dirichlet_distribution(bin_points, alphas=[2, 2, 2])
    """
   
    resources = []
    for bin_point in bin_points:
        if any(x <= 0 for x in bin_point):
            bin_point = simplex_utils.projection_onto_simplex(torch.tensor(bin_point)).numpy()[0]
            if any(x == 1 for x in bin_point):
                i = np.where(bin_point == 1)[0][0]
                bin_point[i] -= .001
                bin_point[i - 1] += .0005
                if i == 2:
                    bin_point[i - 2] += .0005
                else:
                    bin_point[i + 1] += .0005
                resources.append(dirichlet.pdf(bin_point, alphas))
            elif any(x == 0 for x in bin_point):
                i = np.where(bin_point == 0)[0][0]
                bin_point[i] += .001
                bin_point[i - 1] -= .0005
                if i == 2:
                    bin_point[i - 2] -= .0005
                else:
                    bin_point[i + 1] -= .0005
                resources.append(dirichlet.pdf(bin_point, alphas))
        else:
            resources.append(dirichlet.pdf(bin_point, alphas))
    return np.array(resources)