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

Example:
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
    
    .. rubric:: Selects and computes a resource distribution based on the specified type and parameters.

    There are several included resource distributions in the InflGame package. 
        
        -"beta": A 1D beta distribution on the 2-simplex i.e. (0,1); see :func:`beta_distribution` . 
        -"dirichlet_distribution": Dirichlet Distribution for resources on a probability 3-simplex; see :func:`dirichlet_distribution` .
        -"multi_modal_gaussian_distribution_1D": A 1D mixture of Gaussian kernels; see :func:`multi_modal_gaussian_distribution_1D` .
        -"multi_modal_gaussian_distribution_2D": A 2D mixture of multi-variate Gaussian kernels; see :func:`multi_modal_gaussian_distribution_2D` .

    :param bin_points: Points where the resource distribution is evaluated.
    :type bin_points: np.ndarray
    :param resource_type: Type of resource distribution (e.g., "beta", "dirichlet_distribution",
                          "multi_modal_gaussian_distribution_1D", "multi_modal_gaussian_distribution_2D_triangle",
                          "multi_modal_gaussian_distribution_2D_square").
    :type resource_type: str
    :param resource_parameters: Parameters for the specified resource distribution.
    :type resource_parameters: list | np.ndarray
    :return: Computed resource distribution values.
    :rtype: np.ndarray
    """
    
    if resource_type=="multi_modal_gaussian_distribution_1D":
        resources=multi_modal_gaussian_distribution_1D(bin_points, stds=resource_parameters[0], means=resource_parameters[1] , mode_factors=resource_parameters[2])
    elif resource_type=="beta":
        resources=beta_distribution(bin_points,resource_parameters[0],resource_parameters[1])
    elif resource_type in ["multi_modal_gaussian_distribution_2D","multi_modal_gaussian_distribution_2D_triangle","multi_modal_gaussian_distribution_2D_square"] :
        resources=multi_modal_gaussian_distribution_2D(bin_points,resource_parameters[0],resource_parameters[1])
    elif resource_type =="dirichlet_distribution" :
        resources=dirichlet_distribution(bin_points,resource_parameters)
    else: 
        return "No known type "+resource_type
    return resources

def multi_modal_gaussian_distribution_1D(bin_points: np.ndarray | torch.Tensor,
                                         stds: list[float] = [.1, .1],
                                         means: list[float] = [.5, .5],
                                         mode_factors=[1, 1]) -> np.ndarray:
    r"""
    .. rubric:: Computes a 1D multi-modal Gaussian mixture distribution for resources.
    
    A 1D mixture of Gaussian kernels is used to model the resource distribution.
    The distribution is defined as a weighted sum of Gaussian functions, each with its own mean, standard deviation, and weight.
    I.e. 

        .. math::
            R(b) = \sum_{i=1}^{k} \alpha_i \cdot \exp\left(-\frac{(b - \mu_i)^2}{2\sigma_i^2}\right)
    
    where :math:`k` is the number of modes, :math:`\alpha_i` is the scaling factor for mode :math:`i`,
    :math:`\mu_i` is the mean of mode :math:`i`, and :math:`\sigma_i` is the standard deviation of mode :math:`i`.


    :param bin_points: Points where the distribution is evaluated.
    :type bin_points: np.ndarray | torch.Tensor
    :param stds: Standard deviations for each mode.
    :type stds: list[float]
    :param means: Means for each mode.
    :type means: list[float]
    :param mode_factors: Scaling factors for each mode.
    :type mode_factors: list
    :return: Computed resource distribution values.
    :rtype: np.ndarray
    """
    
    resource_modes=[]
    for mode_id in range(len(stds)):
        mean=means[mode_id]
        std=stds[mode_id]
        mode_factor=mode_factors[mode_id]
        mode=mode_factor*np.exp(-(bin_points-mean)**2/(2*(std)**2))
        resource_modes.append(mode)
    resource_modes=np.array(resource_modes)
    resources=np.sum(resource_modes,axis=0)
    
    return resources

def multi_modal_gaussian_distribution_2D(bin_points: np.ndarray | torch.Tensor,
                                         stds: torch.Tensor = torch.tensor([[[.1, 0], [0, .1]], [[.1, 0], [0, .1]], [[.1, 0], [0, .1]]]),
                                         means: torch.Tensor = torch.tensor([[0, 0], [1, 0], [0.5000, 0.8660]])) -> np.ndarray:
    r"""
    .. rubric:: Computes a 2D multi-modal Gaussian mixture distribution for resources.

    A 2d multi-modal Gaussian mixture distribution is used to model the resource distribution.
    The distribution is defined as a weighted sum of Gaussian functions, each with its own mean and covariance matrix.
    I.e.
    
        .. math::
            R(b) = \sum_{i=1}^{k} \alpha_i \cdot \exp\left(-\frac{1}{2}(b - \mu_i)^T \Sigma_i^{-1} (b - \mu_i)\right)
    
    where :math:`k` is the number of modes, :math:`\alpha_i` is the scaling factor for mode :math:`i`,
    :math:`\mu_i` is the mean of mode :math:`i`, and :math:`\Sigma_i` is the covariance matrix of mode :math:`i`.


    :param bin_points: Points where the distribution is evaluated.
    :type bin_points: np.ndarray | torch.Tensor
    :param stds: Covariance matrices for each mode.
    :type stds: torch.Tensor
    :param means: Means for each mode.
    :type means: torch.Tensor
    :return: Computed resource distribution values.
    :rtype: np.ndarray
    """
    

    resource_modes=[]
    for mode_id in range(len(stds)):
        mean=means[mode_id]
        std=stds[mode_id]
        x_vec=torch.tensor((bin_points-mean.numpy())).float()
        sigma_inv=torch.inverse(std)
        distribution_values=[]
        for i in range(len(bin_points)):
            distribution_value=torch.exp(-1/2*x_vec[i,:]@sigma_inv@x_vec.T[:,i])
            distribution_values.append(distribution_value.item())
        mode=np.array(distribution_values)
        resource_modes.append(mode)
    resource_modes=np.array(resource_modes)
    resources=np.sum(resource_modes,axis=0)
    
    return resources

def beta_distribution(bin_points: np.ndarray | torch.Tensor,
                      alpha_value: float,
                      beta_value: float):
    r"""
    .. rubric:: Computes a beta distribution for resources on the 2 simplex.

    A 1D beta distribution is used to model the resource distribution.
    The distribution is defined as a beta function, which is a continuous probability distribution defined on the interval (0, 1).
    I.e.

        .. math::
            R(b) = \frac{b^{\alpha-1}(1-b)^{\beta-1}}{B(\alpha, \beta)}



    :param bin_points: Points where the distribution is evaluated.
    :type bin_points: np.ndarray | torch.Tensor
    :param alpha_value: Alpha parameter of the beta distribution.
    :type alpha_value: float
    :param beta_value: Beta parameter of the beta distribution.
    :type beta_value: float
    :return: Computed resource distribution values.
    :rtype: np.ndarray
    """
   
    f=lambda x: beta.pdf(x, a=alpha_value, b=beta_value)
    resources=f(bin_points)
    return resources

def dirichlet_distribution(bin_points: np.ndarray | torch.Tensor,
                           alphas: list | np.ndarray):
    r"""
    .. rubric:: Computes a Dirichlet distribution.

    A Dirichlet distribution is used to model the resource distribution.
    The distribution is defined as a multivariate generalization of the beta distribution.
    I.e.

        .. math::
            R(b) = \frac{1}{B(\alpha)} \prod_{i=1}^{k} b_i^{\alpha_i-1}



    :param bin_points: Points where the distribution is evaluated.
    :type bin_points: np.ndarray | torch.Tensor
    :param alphas: Parameters of the Dirichlet distribution.
    :type alphas: list | np.ndarray
    :return: Computed resource distribution values.
    :rtype: np.ndarray
    """
   
    resources=[]
    for bin_point in bin_points:
        if any(x<=0 for x in bin_point):
            bin_point=simplex_utils.projection_onto_simplex(torch.tensor(bin_point)).numpy()[0]
            if any(x==1 for x in bin_point):
                i=np.where(bin_point==1)[0][0]
                bin_point[i]-=.001
                bin_point[i-1]+=.0005
                if i==2:
                    bin_point[i-2]+=.0005
                else:
                    bin_point[i+1]+=.0005
                resources.append(dirichlet.pdf(bin_point, alphas))
            elif any(x==0 for x in bin_point):
                i=np.where(bin_point==0)[0][0]
                bin_point[i]+=.001
                bin_point[i-1]-=.0005
                if i==2:
                    bin_point[i-2]-=.0005
                else:
                    bin_point[i+1]-=.0005
                resources.append(dirichlet.pdf(bin_point, alphas))
        else:
            resources.append(dirichlet.pdf(bin_point, alphas))
    return np.array(resources)