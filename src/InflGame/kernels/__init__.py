
r"""
kernels (:mod:`InflGame.kernels`)
==============================================================

.. currentmodule:: InflGame.kernels

Pre-made influence kernels from the paper including Gaussian, Multi-variate Gaussian, Dirichlet, and Matt Jones' kernel. 

==================  =============================================
Submodules          Description
==================  =============================================
`beta`              Beta influence kernels
`blotto`            Colonel Blotto influence kernels
`diric`             Dirichlet influence kernels 
`diric_mode`        Mode-parameterized Dirichlet influence kernels
`gauss`             Gaussian influence kernels
`jones`             Mathew Jones influence kernels
`MV_gauss`          Multi Variate Gaussian kernels
==================  ===============================================


Dirichlet Influence Kernel
===========================

Mathematical Definitions:
-------------------------
The Dirichlet influence kernel is defined as:

.. math::
    f_i(\alpha, b) = \frac{1}{\beta(\alpha)} \prod_{l=1}^{L} b_{l}^{\alpha_{l} - 1}

where:
  - :math:`\alpha` is the vector of parameters for the Dirichlet distribution, defined by the `param` function.
  - :math:`b` is the bin point.
  - :math:`\beta(\alpha)` is the beta function.

  
Gaussian Influence Kernel
===========================

Mathematical Definitions:
-------------------------
The Gaussian influence kernel is defined as:

.. math::
    f_i(x_i, b) = \exp\left(-\frac{(b - x_i)^2}{2\sigma_i^2}\right)

where:
  - :math:`x_i` is the position of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`\sigma_i` is the parameter for agent :math:`i`


Jones Influence Kernel
===========================

This influence kernel is from the work of Mathew Jones et al in their paper "Polarization, abstention, and the median voter
theorem" (`paper <https://www.nature.com/articles/s41599-022-01056-0.pdf>`_). 


Mathematical Definitions:
-------------------------
The Jones influence kernel is defined as:

.. math::
    f_i(x_i, b) = \frac{1}{|x_i - b|^{P_i}}

where:
  - :math:`x_i` is the position of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`P_i` is the parameter for agent :math:`i`



Multi Variate Gaussian Influence Kernel
========================================

Mathematical Definitions:
-------------------------
The multivariate Gaussian influence kernel is defined as:

.. math::
    f_i(x_i, b) = \exp\left(-\frac{1}{2} (b - x_i)^T \Sigma_i^{-1} (b - x_i)\right)

where:
  - :math:`x_i` is the position of agent :math:`i`
  - :math:`b` is the bin point
  - :math:`\Sigma_i` is the covariance matrix for agent :math:`i`


Beta Influence Kernel
===========================

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
  - :math:`\phi` is the concentration parameter for agent :math:`i` (must be > 2)
  - :math:`B(\alpha, \beta)` is the Beta function


Mode-Parameterized Dirichlet Influence Kernel
==============================================

Mathematical Definitions:
-------------------------
The mode-parameterized Dirichlet influence kernel uses the parameterization:

.. math::
    \alpha_{(i,l)} = 1 + \sigma \cdot x_{(i,l)}

where:
  - :math:`\sigma > 0` is a concentration parameter controlling spread
  - :math:`x_{(i,l)}` is the position of agent :math:`i` in dimension :math:`l`
  - The sum :math:`\alpha_0 = L + \sigma` where :math:`L` is the dimension

The gradient with respect to agent position is:

.. math::
    d_{(i,l)} = \sigma \left( \ln(b_l) - \psi_0(1 + \sigma x_{(i,l)}) \right)

The Hessian is a diagonal matrix:

.. math::
    H_{l,l} = -\sigma^2 \psi_1(1 + \sigma x_{(i,l)})

   
"""