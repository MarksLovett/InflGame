"""
.. module:: two_utils
   :synopsis: Provides utility functions for setting up and managing 2D domains in influencer games.

2D Utility Module
=================

This module provides utility functions for setting up and managing 2D domains in influencer games. 
It includes tools for creating rectangular grids and handling domain-specific configurations.

The module is designed to work with the `InflGame` package and supports creating structured 2D environments 
for simulations involving agent dynamics and resource distributions.


Usage:
------
The `two_dimensional_rectangle_setup` function can be used to create a 2D rectangular grid within specified domain bounds.

"""


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pylab


def two_dimensional_rectangle_setup(domain_bounds: np.ndarray,
                                    domain_refinement: int = 10) -> np.ndarray:
    """
    Sets up a 2D rectangular grid within the specified domain bounds.

    :param domain_bounds: A 2x2 array specifying the bounds of the domain. The first row corresponds to the Y-axis bounds, and the second row corresponds to the X-axis bounds.
    :type domain_bounds: np.ndarray
    :param domain_refinement: The number of points along each axis for the grid. Higher values result in finer grids. Default is 10.
    :type domain_refinement: int

    :returns: A tuple containing:
              - rect_X (np.ndarray): The X-coordinates of the grid points.
              - rect_Y (np.ndarray): The Y-coordinates of the grid points.
              - rect_positions (np.ndarray): A 2D array of shape (N, 2), where N is the total number of grid points. Each row represents the (X, Y) coordinates of a grid point.
    :rtype: tuple
    """
    rect_Y, rect_X = np.mgrid[domain_bounds[0,0]:domain_bounds[0,1]:domain_refinement*1j, domain_bounds[1,0]:domain_bounds[1,1]:domain_refinement*1j]
    rect_positions = np.vstack([rect_X.ravel(), rect_Y.ravel()])
    rect_positions=rect_positions.transpose()
    return rect_X, rect_Y, rect_positions