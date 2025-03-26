import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pylab

from influencer_games.utils.utilities import *

def two_dimensional__rectangle_setup(domain_bounds:np.ndarray,
                          domain_refinement:int = 10)->np.ndarray:
        """
        Setsup the grid for the domain
        
        Parameters:
                domain_bounds (np.ndarray): The bounds for the domain in 2d
                domain_refinement (int): The number of grid points in the domain
                
        Returns:
                rect_X, rect_Y (np.ndarray) = grid points
                rect_postions (np.ndarray) = grid points as coordinates (for resource bins)
        """

        rect_Y, rect_X = np.mgrid[domain_bounds[0,0]:domain_bounds[0,1]:domain_refinement*1j, domain_bounds[1,0]:domain_bounds[1,1]:domain_refinement*1j]
        rect_positions = np.vstack([rect_X.ravel(), rect_Y.ravel()])
        rect_positions=rect_positions.transpose()
        return rect_X,rect_Y,rect_positions