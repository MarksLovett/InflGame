
r"""
kernels (:mod:`InflGame.kernels`)
==============================================================

.. currentmodule:: InflGame.kernels

Pre-made influence kernels from the paper including Gaussian, Multi-variate Gaussian, Dirichlet, and Matt Jones' kernel. 

==================  =============================================
Submodules          Description
==================  =============================================
`diric`             Dirichlet influence kernels 
`gauss`             Gaussian influence kernels
`jones`             Mathew Jones influence kernels
`MV_gauss`          Multi Variate Gaussian kernels
==================  =============================================


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

   
"""