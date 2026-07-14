:html_theme.sidebar_secondary.remove: true
Resource Distributions 
====================================================


.. module:: rd
   :synopsis: Provides resource distribution functions for influencer games.

.. currentmodule:: InflGame.domains.rd


(:mod:`InflGame.domains.rd`)
============================

This module provides functions to compute and select resource distributions for influencer games. 
It includes implementations for various types of resource distributions, such as beta, Dirichlet, 
and multi-modal Gaussian distributions in 1D and 2D domains.

The module is designed to work with the `InflGame` package and supports resource distribution 
evaluation over specified bin points. These distributions are used to model resource availability 
in different domains and scenarios of influencer games.

Functions:
-----------

========================================================  =========================================================================================
Function                                                  Description
========================================================  =========================================================================================
`resource_distribution_choice`                             Selects and computes a resource distribution based on the specified type and parameters.
`multi_modal_gaussian_distribution_1D`                     Computes a 1D multi-modal Gaussian mixture distribution for resources.
`multi_modal_gaussian_distribution_2D`                     Computes a 2D multi-modal Gaussian mixture distribution for resources.
`beta_distribution`                                        Computes a beta distribution for resources on the 2-simplex.
`dirichlet_distribution`                                   Computes a Dirichlet distribution for resources.
========================================================  =========================================================================================


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


.. toctree:: :maxdepth: 2
   :glob:
   :titlesonly:
   :hidden:


   resource_distribution_choice
   multi_modal_gaussian_distribution_1D
   beta_distribution
   dirichlet_distribution
   multi_modal_gaussian_distribution_2D